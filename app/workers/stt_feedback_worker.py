import asyncio
import json
import logging
import time
from typing import Optional

import aio_pika
from aio_pika.exceptions import AMQPConnectionError

# Configs
from app.config import rabbitmq_config

# Schema for input
from app.api.stt_feedback.stt_feedback_schema import VoiceFeedbackRequest

# Core logic components
from app.api.stt_feedback.service.feedback_pipline import run_feedback_pipeline
from app.api.stt_feedback.stt_model_loader import WhisperXModel
from app import rabbitmq_producer

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("STTFeedbackWorker")


def format_feedback(feedback_dict: dict) -> str:
    """피드백 딕셔너리를 문자열로 포맷팅"""
    if not feedback_dict:
        return ""
    return (
        f"{feedback_dict.get('overall_score', '')} 점\n\n"
        f"{feedback_dict.get('detailed_analysis', '')}\n\n"
        f"잘한 점: {feedback_dict.get('good_points', '')}\n\n"
        f"개선할 점: {feedback_dict.get('areas_for_improvement', '')}"
    )


def parse_and_format_response(result) -> dict:
    """결과를 파싱하고 포맷팅하여 최종 응답 생성"""
    
    # 변환 적용: feedback dict → string
    for item in result.feedbackLists:
        item.feedback = format_feedback(item.feedback)

    # pipeline 수행 후 결과 로깅
    for idx, item in enumerate(result.feedbackLists):
        logger.info(f"[Feedback Result][{idx+1}] Segment ID: {item.segment_id}")
        logger.info(f"[Feedback Result][{idx+1}] 질문: {item.question}")
        logger.info(f"[Feedback Result][{idx+1}] 질문별 모범답안: {item.model_answer}")
        logger.info(f"[Feedback Result][{idx+1}] 질문별 피드백: {item.feedback}")
        
    # 응답 데이터 생성 (question 제외)
    response = result.model_dump()
    response['feedbackLists'] = [
        {
            "segment_id": item["segment_id"],
            "model_answer": item["model_answer"], 
            "feedback": item["feedback"]
        }
        for item in response['feedbackLists']
    ]

    return response


async def process_stt_feedback_task(message: aio_pika.IncomingMessage):
    async with message.process(ignore_processed=True):
        try:
            logger.info(f"Received message: {message.message_id} for STT feedback generation.")
            data = json.loads(message.body.decode())
            req = VoiceFeedbackRequest(**data)

            request_start_time = time.time()

            # WhisperX 모델 확인 (실제 구현에 맞게 수정)
            if WhisperXModel.model is None:  # is_loaded() 대신 직접 모델 체크
                logger.info("WhisperX model not loaded. Attempting to load...")
                WhisperXModel.ensure_loaded()
                if WhisperXModel.model is None:  # 로드 후 다시 체크
                    logger.error("Failed to load WhisperX model in worker.")
                    await message.reject(requeue=False)
                    return
                logger.info("WhisperX model loaded successfully in worker.")


            pipeline_start_time = time.time()

            logger.info(f"Running feedback pipeline for recording URL: {req.recording_url}")
            
            for idx, q in enumerate(req.question_lists):
                logger.info(f"[Worker] 질문 {idx+1}: {q.question} (start: {q.start_time}, end: {q.end_time})")
            
            # run_feedback_pipeline is synchronous, run in thread
            feedback_result = await asyncio.to_thread(
                run_feedback_pipeline,
                recording_url=str(req.recording_url),
                question_lists=req.question_lists
            )
            logger.info(f"Feedback pipeline completed for recording URL: {req.recording_url}")
            
            pipeline_execution_time = time.time() - pipeline_start_time
            request_execution_time = time.time() - request_start_time
            
            logger.info(
                f"Performance metrics - Recording URL: {req.recording_url}, "
                f"Pipeline time: {pipeline_execution_time:.2f}s, "
                f"Total time: {request_execution_time:.2f}s, "
                f"Questions processed: {len(req.question_lists)}"
            )
            
            logger.info(f"Successfully processed STT feedback for recording: {req.recording_url}")
            
            # Send response back to backend if feedback_result exists
            if feedback_result:
                try:
                    # 응답 파싱 및 포맷팅 적용
                    formatted_response = parse_and_format_response(feedback_result)
                    
                    response_success = await rabbitmq_producer.publish_response_message(
                        message_body=formatted_response,
                        exchange_name=rabbitmq_config.STT_RESPONSE_EXCHANGE_NAME,
                        routing_key=rabbitmq_config.STT_RESPONSE_ROUTING_KEY
                    )
                    if response_success:
                        logger.info(f"Successfully published STT response for recording: {req.recording_url}")
                    else:
                        logger.error(f"Failed to publish STT response for recording: {req.recording_url}")
                except Exception as e:
                    logger.error(f"Error publishing STT response for recording {req.recording_url}: {e}")
            
            logger.info(f"Generated STT Feedback: {formatted_response if 'formatted_response' in locals() else 'None'}")

            await message.ack()
            logger.info(f"Message {message.message_id} acked.")

        except json.JSONDecodeError as e:
            logger.error(f"JSONDecodeError: {e} for message body: {message.body[:200]}", exc_info=True)
            await message.reject(requeue=False)  # Do not requeue malformed messages
        except Exception as e:
            logger.error(f"Unhandled error processing message {message.message_id if message else 'UnknownMsg'}: {e}", exc_info=True)
            await message.reject(requeue=False)


async def main_stt_feedback_worker():
    # Load models once at startup
    logger.info("STT Feedback Worker: Initializing WhisperX model...")
    max_retries = getattr(rabbitmq_config, 'MODEL_LOAD_MAX_RETRIES', 3)
    
    for attempt in range(max_retries):
        try:
            WhisperXModel.ensure_loaded()
            if WhisperXModel.model is not None:  # is_loaded() 대신 직접 모델 체크
                logger.info("STT Feedback Worker: WhisperX model loaded successfully.")
                break
            else:
                logger.warning(f"Model load attempt {attempt + 1} failed.")
        except Exception as e:
            logger.error(f"Model load attempt {attempt + 1} error: {e}")
            if attempt == max_retries - 1:
                logger.critical("Failed to load model after all retries. Exiting.")
                return
            await asyncio.sleep(5)

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
                client_properties={'connection_name': 'stt_feedback_worker_connection'}
            )
            logger.info("Connected to RabbitMQ.")
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

        queue_name = rabbitmq_config.STT_FEEDBACK_QUEUE_NAME
        routing_key = rabbitmq_config.ROUTING_KEY_STT_FEEDBACK

        queue = await channel.declare_queue(name=queue_name, durable=True)
        await queue.bind(exchange, routing_key=routing_key)

        logger.info(f"Waiting for messages on queue '{queue_name}' with routing key '{routing_key}'.")
        
        try:
            async with queue.iterator() as queue_iter:
                async for message in queue_iter:
                    await process_stt_feedback_task(message)
        except asyncio.CancelledError:
            logger.info("Queue consumption cancelled. Shutting down.")
        except Exception as e:
            logger.error(f"Queue consumption error: {e}", exc_info=True)
        finally:
            logger.info("Shutting down STT Feedback Worker.")


if __name__ == "__main__":
    logger.info("Starting STT Feedback Worker...")
    try:
        asyncio.run(main_stt_feedback_worker())
    except KeyboardInterrupt:
        logger.info("STT Feedback Worker interrupted by user. Exiting.")
    except Exception as e:
        logger.critical(f"STT Feedback Worker failed to start or run: {e}", exc_info=True)