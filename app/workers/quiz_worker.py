import asyncio
import json
import logging
import time
from typing import Optional
import aio_pika
from aio_pika.exceptions import AMQPConnectionError, AMQPChannelError
from app.config import rabbitmq_config
from app.api.quiz_generator.quiz_generator_schema import FollowupRequest
from app.api.quiz_generator.quiz_generator_api import process_quiz_generation
from app.api.quiz_generator.quiz_generator_model import initialize_quiz_model
from app import rabbitmq_producer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger("QuizWorker")


def initialize_models():
    """Initialize models on worker startup (preload to GPU)"""
    logger.info("Worker model initialization started...")
    
    try:
        logger.info("Loading quiz generation model...")
        if not initialize_quiz_model():
            logger.error("Quiz generation model initialization failed")
            return False
        logger.info("Quiz generation model loaded")
            
        logger.info("Loading embedding model...")
        try:
            from app.vector_db.utils import get_embedding_model
            embedding_model = get_embedding_model()
            if embedding_model:
                logger.info("Embedding model loaded")
            else:
                logger.warning("Embedding model load failed - continuing")
        except ImportError:
            logger.warning("Vector DB module not found - skipping embedding model")
            
        logger.info("Worker model initialization complete")
        return True
        
    except Exception as e:
        logger.error(f"Worker model initialization failed: {e}", exc_info=True)
        return False


async def process_quiz_generation_task(message: aio_pika.IncomingMessage):
    """Process quiz generation task"""
    message_id = getattr(message, 'message_id', 'unknown')
    
    async with message.process(ignore_processed=True):
        try:
            logger.info(f"Received message: {message_id}")
            
            try:
                data = json.loads(message.body.decode())
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing failed: {e}, body: {message.body[:200]}")
                await message.reject(requeue=False)
                return
            
            try:
                req = FollowupRequest(**data)
            except ValueError as e:
                logger.error(f"Request data validation failed: {e}")
                await message.reject(requeue=False)
                return

            request_start_time = time.time()
            logger.info(f"Quiz generation started: {req.interview_id}")
            
            # 퀴즈 생성 API를 불러오는 코드
            try:
                if asyncio.iscoroutinefunction(process_quiz_generation):
                    api_response = await process_quiz_generation(req)
                else:
                    api_response = await asyncio.to_thread(process_quiz_generation, req)
            except Exception as e:
                logger.error(f"Quiz generation failed ({req.interview_id}): {e}", exc_info=True)
                await message.reject(requeue=True)  # allow reprocessing
                return
            
            request_execution_time = time.time() - request_start_time
            
            logger.info(
                f"Quiz generation completed: {req.interview_id}, "
                f"Processing time: {request_execution_time:.2f}s"
            )
            
            if api_response:
                try:
                    response_success = await rabbitmq_producer.publish_response_message(
                        message_body=api_response,
                        exchange_name=rabbitmq_config.QUIZ_RESPONSE_EXCHANGE_NAME,
                        routing_key=rabbitmq_config.QUIZ_RESPONSE_ROUTING_KEY
                    )
                    
                    if response_success:
                        logger.info(f"Response published successfully: {req.interview_id}")
                    else:
                        logger.error(f"Response publish failed: {req.interview_id}")
                        await message.reject(requeue=True)
                        return
                        
                except Exception as e:
                    logger.error(f"Error publishing response: {e}", exc_info=True)
                    await message.reject(requeue=True)
                    return
            
            await message.ack()
            logger.info(f"Message processed: {message_id}")

        except Exception as e:
            logger.error(f"Unexpected error ({message_id}): {e}", exc_info=True)
            await message.reject(requeue=False)


async def connect_to_rabbitmq_with_retry() -> Optional[aio_pika.Connection]:
    """Connect to RabbitMQ (improved retry logic)"""
    max_retries = 10
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            logger.info(f"RabbitMQ connection attempt {attempt + 1}/{max_retries}")
            
            connection = await aio_pika.connect(
                host=rabbitmq_config.RABBITMQ_HOST,
                port=rabbitmq_config.RABBITMQ_PORT,
                login=rabbitmq_config.RABBITMQ_USER,
                password=rabbitmq_config.RABBITMQ_PASSWORD,
                virtualhost=rabbitmq_config.RABBITMQ_VIRTUAL_HOST,
                timeout=rabbitmq_config.RABBITMQ_TIMEOUT,
                client_properties={'connection_name': 'quiz_worker_connection'}
            )
            
            logger.info("RabbitMQ connection successful")
            return connection
            
        except asyncio.TimeoutError:
            logger.error(f"RabbitMQ connection timeout (attempt {attempt + 1}/{max_retries})")
        except (ConnectionRefusedError, AMQPConnectionError) as e:
            logger.error(f"RabbitMQ connection refused (attempt {attempt + 1}/{max_retries}): {e}")
        except Exception as e:
            logger.error(f"RabbitMQ connection failed (attempt {attempt + 1}/{max_retries}): {e}")
        
        if attempt < max_retries - 1:
            logger.info(f"Retrying in {retry_delay} seconds...")
            await asyncio.sleep(retry_delay)
    
    logger.critical("RabbitMQ connection failed: maximum retry attempts exceeded")
    return None


async def main_quiz_worker():
    """Main worker function"""
    logger.info("Quiz Generation Worker started")
    
    logger.info("Initializing models...")
    if not initialize_models():
        logger.critical("Model initialization failed. Exiting worker.")
        return
    
    logger.info("Attempting RabbitMQ connection...")
    connection = await connect_to_rabbitmq_with_retry()
    if not connection:
        logger.critical("RabbitMQ connection failed. Exiting worker.")
        return

    try:
        async with connection:
            channel = await connection.channel()
            await channel.set_qos(
                prefetch_count=getattr(rabbitmq_config, 'PREFETCH_COUNT', 1)
            )

            exchange_name = rabbitmq_config.SERVICE_EXCHANGE_NAME
            exchange_type = aio_pika.ExchangeType(rabbitmq_config.SERVICE_EXCHANGE_TYPE)
            
            exchange = await channel.declare_exchange(
                name=exchange_name, 
                type=exchange_type, 
                durable=True
            )

            queue_name = rabbitmq_config.QUIZ_QUEUE_NAME
            routing_key = rabbitmq_config.ROUTING_KEY_QUIZ_GENERATOR

            queue = await channel.declare_queue(name=queue_name, durable=True)
            await queue.bind(exchange, routing_key=routing_key)

            logger.info(f"Waiting for messages on queue '{queue_name}' (routing_key: '{routing_key}')")
            
            try:
                async with queue.iterator() as queue_iter:
                    async for message in queue_iter:
                        await process_quiz_generation_task(message)
            except asyncio.CancelledError:
                logger.info("Queue consumption cancelled. Shutting down...")
            except Exception as e:
                logger.error(f"Queue consumption error: {e}", exc_info=True)
            finally:
                logger.info("Quiz Worker stopped")
                
    except Exception as e:
        logger.error(f"Worker runtime error: {e}", exc_info=True)


if __name__ == "__main__":
    try:
        asyncio.run(main_quiz_worker())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Worker startup failed: {e}", exc_info=True)