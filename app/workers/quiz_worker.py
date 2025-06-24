import asyncio
import json
import logging
import time
from typing import Optional

import aio_pika
from aio_pika.exceptions import AMQPConnectionError, AMQPChannelError

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


def initialize_models():
    """워커 시작 시 모델들을 미리 GPU에 로드"""
    logger.info("워커 모델 초기화 시작...")
    
    try:
        # Quiz 생성 모델 초기화
        logger.info("Quiz 생성 모델 로딩 중...")
        if not initialize_quiz_model():
            logger.error("Quiz 생성 모델 초기화 실패")
            return False
        logger.info("Quiz 생성 모델 로딩 완료")
            
        # 임베딩 모델 초기화
        logger.info("임베딩 모델 로딩 중...")
        try:
            from app.vector_db.utils import get_embedding_model
            embedding_model = get_embedding_model()
            if embedding_model:
                logger.info("임베딩 모델 로딩 완료")
            else:
                logger.warning("임베딩 모델 로딩 실패 - 계속 진행")
        except ImportError:
            logger.warning("벡터 DB 모듈 없음 - 임베딩 모델 스킵")
            
        logger.info("워커 모델 초기화 완료")
        return True
        
    except Exception as e:
        logger.error(f"워커 모델 초기화 실패: {e}", exc_info=True)
        return False


async def process_quiz_generation_task(message: aio_pika.IncomingMessage):
    """퀴즈 생성 작업 처리"""
    message_id = getattr(message, 'message_id', 'unknown')
    
    async with message.process(ignore_processed=True):
        try:
            logger.info(f"메시지 수신: {message_id}")
            
            # JSON 파싱
            try:
                data = json.loads(message.body.decode())
            except json.JSONDecodeError as e:
                logger.error(f"JSON 파싱 실패: {e}, body: {message.body[:200]}")
                await message.reject(requeue=False)
                return
            
            # 요청 데이터 검증
            try:
                req = FollowupRequest(**data)
            except ValueError as e:
                logger.error(f"요청 데이터 검증 실패: {e}")
                await message.reject(requeue=False)
                return

            request_start_time = time.time()
            logger.info(f"퀴즈 생성 시작: {req.interview_id}")
            
            # API 함수 호출 (비동기 처리)
            try:
                if asyncio.iscoroutinefunction(process_quiz_generation):
                    api_response = await process_quiz_generation(req)
                else:
                    api_response = await asyncio.to_thread(process_quiz_generation, req)
            except Exception as e:
                logger.error(f"퀴즈 생성 실패 ({req.interview_id}): {e}", exc_info=True)
                await message.reject(requeue=True)  # 재처리 가능하도록
                return
            
            request_execution_time = time.time() - request_start_time
            
            logger.info(
                f"퀴즈 생성 완료: {req.interview_id}, "
                f"처리 시간: {request_execution_time:.2f}초"
            )
            
            # 응답 발행
            if api_response:
                try:
                    response_success = await rabbitmq_producer.publish_response_message(
                        message_body=api_response,
                        exchange_name=rabbitmq_config.QUIZ_RESPONSE_EXCHANGE_NAME,
                        routing_key=rabbitmq_config.QUIZ_RESPONSE_ROUTING_KEY
                    )
                    
                    if response_success:
                        logger.info(f"응답 발행 성공: {req.interview_id}")
                    else:
                        logger.error(f"응답 발행 실패: {req.interview_id}")
                        await message.reject(requeue=True)
                        return
                        
                except Exception as e:
                    logger.error(f"응답 발행 중 오류: {e}", exc_info=True)
                    await message.reject(requeue=True)
                    return
            
            # 메시지 승인
            await message.ack()
            logger.info(f"메시지 처리 완료: {message_id}")

        except Exception as e:
            logger.error(f"예상치 못한 오류 ({message_id}): {e}", exc_info=True)
            await message.reject(requeue=False)


async def connect_to_rabbitmq_with_retry() -> Optional[aio_pika.Connection]:
    """RabbitMQ 연결 (개선된 재시도 로직)"""
    max_retries = 10
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            logger.info(f"RabbitMQ 연결 시도 {attempt + 1}/{max_retries}")
            
            connection = await aio_pika.connect(
                host=rabbitmq_config.RABBITMQ_HOST,
                port=rabbitmq_config.RABBITMQ_PORT,
                login=rabbitmq_config.RABBITMQ_USER,
                password=rabbitmq_config.RABBITMQ_PASSWORD,
                virtualhost=rabbitmq_config.RABBITMQ_VIRTUAL_HOST,
                timeout=30,
                client_properties={'connection_name': 'quiz_worker_connection'}
            )
            
            logger.info("RabbitMQ 연결 성공")
            return connection
            
        except asyncio.TimeoutError:
            logger.error(f"RabbitMQ 연결 타임아웃 (시도 {attempt + 1}/{max_retries})")
        except (ConnectionRefusedError, AMQPConnectionError) as e:
            logger.error(f"RabbitMQ 연결 거부 (시도 {attempt + 1}/{max_retries}): {e}")
        except Exception as e:
            logger.error(f"RabbitMQ 연결 실패 (시도 {attempt + 1}/{max_retries}): {e}")
        
        if attempt < max_retries - 1:
            logger.info(f"{retry_delay}초 후 재시도...")
            await asyncio.sleep(retry_delay)
    
    logger.critical("RabbitMQ 연결 실패: 최대 재시도 횟수 초과")
    return None


async def main_quiz_worker():
    """메인 워커 함수"""
    logger.info("Quiz Generation Worker 시작")
    
    # 모델 초기화
    logger.info("모델 초기화 중...")
    if not initialize_models():
        logger.critical("모델 초기화 실패. 워커를 종료합니다.")
        return
    
    # RabbitMQ 연결
    logger.info("RabbitMQ 연결 시도 중...")
    connection = await connect_to_rabbitmq_with_retry()
    if not connection:
        logger.critical("RabbitMQ 연결 실패. 워커를 종료합니다.")
        return

    try:
        async with connection:
            channel = await connection.channel()
            await channel.set_qos(
                prefetch_count=getattr(rabbitmq_config, 'PREFETCH_COUNT', 1)
            )

            # Exchange 선언
            exchange_name = rabbitmq_config.SERVICE_EXCHANGE_NAME
            exchange_type = aio_pika.ExchangeType(rabbitmq_config.SERVICE_EXCHANGE_TYPE)
            
            exchange = await channel.declare_exchange(
                name=exchange_name, 
                type=exchange_type, 
                durable=True
            )

            # Queue 설정
            queue_name = rabbitmq_config.QUIZ_QUEUE_NAME
            routing_key = rabbitmq_config.ROUTING_KEY_QUIZ_GENERATOR

            queue = await channel.declare_queue(name=queue_name, durable=True)
            await queue.bind(exchange, routing_key=routing_key)

            logger.info(f"큐 대기 중: '{queue_name}' (routing_key: '{routing_key}')")
            
            # 메시지 소비
            try:
                async with queue.iterator() as queue_iter:
                    async for message in queue_iter:
                        await process_quiz_generation_task(message)
            except asyncio.CancelledError:
                logger.info("큐 소비 취소됨. 종료 중...")
            except Exception as e:
                logger.error(f"큐 소비 오류: {e}", exc_info=True)
            finally:
                logger.info("Quiz Worker 종료")
                
    except Exception as e:
        logger.error(f"워커 실행 중 오류: {e}", exc_info=True)


if __name__ == "__main__":
    try:
        asyncio.run(main_quiz_worker())
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단됨")
    except Exception as e:
        logger.error(f"워커 시작 실패: {e}", exc_info=True)