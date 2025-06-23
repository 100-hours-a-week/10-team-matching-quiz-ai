import asyncio
import json
import logging
import time

import aio_pika

# Configs
from app.config import rabbitmq_config

# Schema for input
from app.api.quiz_generator.quiz_generator_schema import FollowupRequest

# API 함수 직접 import
from app.api.quiz_generator.quiz_generator_api import process_quiz_generation
from app.api.quiz_generator.quiz_generator_model import initialize_quiz_model
from app import rabbitmq_producer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger("QuizWorker")

# 모델 초기화 함수
def initialize_models():
    """워커 시작 시 모델들을 미리 GPU에 로드"""
    logger.info("워커 모델 초기화 시작...")
    
    try:
        # Quiz 생성 모델 초기화 (먼저 실행)
        logger.info("Quiz 생성 모델 로딩 중...")
        initialize_quiz_model()
        logger.info("Quiz 생성 모델 로딩 완료")
            
        # 임베딩 모델 초기화 (나중에 실행)
        logger.info("임베딩 모델 로딩 중...")
        from app.vector_db.utils import get_embedding_model
        embedding_model = get_embedding_model()
        if embedding_model:
            logger.info("임베딩 모델 로딩 완료")
        else:
            logger.error("임베딩 모델 로딩 실패")
            return False
            
        logger.info("워커 모델 초기화 완료")
        return True
        
    except Exception as e:
        logger.error(f"워커 모델 초기화 실패: {e}", exc_info=True)
        return False

async def process_quiz_generation_task(message: aio_pika.IncomingMessage):
    async with message.process(ignore_processed=True):
        try:
            logger.info(f"Received message: {message.message_id} for quiz generation.")
            data = json.loads(message.body.decode())
            req = FollowupRequest(**data)

            request_start_time = time.time()

            logger.info(f"Running quiz generation pipeline for interview_id: {req.interview_id}")
            
            # API 함수를 직접 호출 (동기 함수를 비동기로 실행)
            api_response = await asyncio.to_thread(process_quiz_generation, req)
            
            request_execution_time = time.time() - request_start_time
            
            logger.info(
                f"Performance metrics - Interview ID: {req.interview_id}, "
                f"Total time: {request_execution_time:.2f}s, "
                f"Questions processed: {len(req.question_history_list)}"
            )
            
            logger.info(f"Successfully processed quiz generation for interview: {req.interview_id}")
            
            # API에서 반환된 응답을 그대로 RabbitMQ로 전송
            if api_response:
                response_success = await rabbitmq_producer.publish_response_message(
                    message_body=api_response,
                    exchange_name=rabbitmq_config.QUIZ_RESPONSE_EXCHANGE_NAME,
                    routing_key=rabbitmq_config.QUIZ_RESPONSE_ROUTING_KEY
                )
                if response_success:
                    logger.info(f"Successfully published quiz response for interview_id: {req.interview_id}")
                else:
                    logger.error(f"Failed to publish quiz response for interview_id: {req.interview_id}")
            
            logger.info(f"Generated Quiz Response: {api_response}")

            await message.ack()
            logger.info(f"Message {message.message_id} acked.")

        except json.JSONDecodeError as e:
            logger.error(f"JSONDecodeError: {e} for message body: {message.body[:200]}", exc_info=True)
            await message.reject(requeue=False)
        except ValueError as e:
            logger.error(f"Validation error: {e}")
            await message.reject(requeue=False)
        except Exception as e:
            logger.error(f"Unhandled error processing message {message.message_id if message else 'UnknownMsg'}: {e}", exc_info=True)
            await message.reject(requeue=False)


async def main_quiz_worker():
    # 모델 초기화 추가 (RabbitMQ 연결 전에)
    logger.info("Quiz Generation Worker 시작: 모델 초기화 중...")
    if not initialize_models():
        logger.critical("모델 초기화 실패. 워커를 종료합니다.")
        return
    
    logger.info("Quiz Generation Worker: 모델 초기화 완료, RabbitMQ 연결 시도 중...")
    
    connection = None
    retry_count = 0
    max_connection_retries = getattr(rabbitmq_config, 'MAX_CONNECTION_RETRIES', 10)
    
    while retry_count < max_connection_retries:
        try:
            connection = await aio_pika.connect_robust(
                host=rabbitmq_config.RABBITMQ_HOST,
                port=rabbitmq_config.RABBITMQ_PORT,
                login=rabbitmq_config.RABBITMQ_USER,
                password=rabbitmq_config.RABBITMQ_PASSWORD,
                virtualhost=rabbitmq_config.RABBITMQ_VIRTUAL_HOST,
                timeout=10,
                client_properties={'connection_name': 'quiz_worker_connection'}
            )
            logger.info("Quiz Generation Worker: RabbitMQ 연결 성공")
            break 
        except (aio_pika.exceptions.AMQPConnectionError, ConnectionRefusedError) as e:
            retry_count += 1
            logger.error(f"RabbitMQ connection failed (attempt {retry_count}/{max_connection_retries}): {e}. Retrying in 5 seconds...")
            await asyncio.sleep(5)
        except Exception as e:
            retry_count += 1
            logger.error(f"Unexpected error during RabbitMQ connection (attempt {retry_count}/{max_connection_retries}): {e}. Retrying in 5 seconds...")
            await asyncio.sleep(5)
    
    if not connection:
        logger.critical("Failed to connect to RabbitMQ after multiple retries. Exiting.")
        return

    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=getattr(rabbitmq_config, 'PREFETCH_COUNT', 1))

        exchange_name = rabbitmq_config.SERVICE_EXCHANGE_NAME
        exchange_type = aio_pika.ExchangeType(rabbitmq_config.SERVICE_EXCHANGE_TYPE)
        
        exchange = await channel.declare_exchange(
            name=exchange_name, type=exchange_type, durable=True
        )

        queue_name = rabbitmq_config.QUIZ_QUEUE_NAME
        routing_key = rabbitmq_config.ROUTING_KEY_QUIZ_GENERATOR

        queue = await channel.declare_queue(name=queue_name, durable=True)
        await queue.bind(exchange, routing_key=routing_key)

        logger.info(f"Waiting for messages on queue '{queue_name}' with routing key '{routing_key}'.")
        
        try:
            async with queue.iterator() as queue_iter:
                async for message in queue_iter:
                    await process_quiz_generation_task(message)
        except asyncio.CancelledError:
            logger.info("Queue consumption cancelled. Shutting down.")
        except Exception as e:
            logger.error(f"Queue consumption error: {e}", exc_info=True)
        finally:
            logger.info("Shutting down Quiz Worker.")

if __name__ == "__main__":
    logger.info("Quiz Generation Worker 시작 중...")
    try:
        asyncio.run(main_quiz_worker())
    except KeyboardInterrupt:
        logger.info("Quiz Generation Worker가 사용자에 의해 중단되었습니다.")
    except Exception as e:
        logger.critical(f"Quiz Generation Worker 실행 실패: {e}", exc_info=True)