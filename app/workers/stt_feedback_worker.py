import asyncio
import json
import logging
import time

import aio_pika

# Configs
from app.config import rabbitmq_config

# Schema for input
from app.api.stt_feedback.stt_feedback_schema import VoiceFeedbackRequest
# Schema for output (if the worker produces a specific structure to be stored/sent elsewhere)
# from app.api.stt_feedback.stt_feedback_model import FeedbackResponse # Or similar

# Core logic components
from app.api.stt_feedback.service.feedback_pipline import run_feedback_pipeline
from app.api.stt_feedback.stt_model_loader import load_whisperx_model, get_whisperx_model

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("STTFeedbackWorker")

async def process_stt_feedback_task(message: aio_pika.IncomingMessage):
    async with message.process(ignore_processed=True):
        try:
            logger.info(f"Received message: {message.message_id} for STT feedback generation.")
            data = json.loads(message.body.decode())
            req = VoiceFeedbackRequest(**data)

            request_start_time = time.time()

            # Ensure WhisperX model is loaded (it's loaded on app startup in main_stt_feedback.py, 
            # but worker is separate process, so needs its own loading if not shared or pre-loaded)
            # For simplicity, assuming get_whisperx_model() checks and loads if necessary, or it was loaded at worker start.
            if get_whisperx_model() is None:
                logger.info("WhisperX model not loaded. Attempting to load...")
                load_whisperx_model() # This should ideally be done once at worker startup
                if get_whisperx_model() is None:
                    logger.error("Failed to load WhisperX model in worker.")
                    await message.reject(requeue=False) # Do not requeue if model can't load
                    return
                logger.info("WhisperX model loaded successfully in worker.")

            pipeline_start_time = time.time()

            logger.info(f"Running feedback pipeline for interview_id: {req.interview_id}")
            # run_feedback_pipeline is synchronous, run in thread
            feedback_result = await asyncio.to_thread(
                run_feedback_pipeline,
                interview_id=req.interview_id,
                recording_url=req.recording_url,
                questionLists=[q.model_dump(by_alias=True) for q in req.questionLists]
            )
            logger.info(f"Feedback pipeline completed for interview_id: {req.interview_id}")
            
            pipeline_execution_time = time.time() - pipeline_start_time

            request_execution_time = time.time() - request_start_time
            
            logger.info(f"Successfully processed STT feedback for interview_id: {req.interview_id}.")
            # Optionally, do something with feedback_result (e.g., store in DB, send notification)
            logger.info(f"Generated STT Feedback for interview_id {req.interview_id}: {feedback_result.model_dump_json(indent=2) if feedback_result else 'None'}")

            await message.ack()
            logger.info(f"Message {message.message_id} acked.")

        except json.JSONDecodeError as e:
            logger.error(f"JSONDecodeError: {e} for message body: {message.body[:200]}", exc_info=True)
            await message.reject(requeue=False) # Do not requeue malformed messages
        except Exception as e:
            logger.error(f"Unhandled error processing message {message.message_id if message else 'UnknownMsg'}: {e}", exc_info=True)
            await message.reject(requeue=False)

async def main_stt_feedback_worker():
    # Load models once at startup
    logger.info("STT Feedback Worker: Initializing WhisperX model...")
    try:
        load_whisperx_model()
        if get_whisperx_model() is not None:
            logger.info("STT Feedback Worker: WhisperX model loaded successfully.")
        else:
            logger.error("STT Feedback Worker: Failed to load WhisperX model during startup. Worker might not function correctly.")
            # Depending on severity, you might want to exit or retry
    except Exception as e:
        logger.error(f"STT Feedback Worker: Error loading WhisperX model during startup: {e}", exc_info=True)

    connection = None
    while True:
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
            logger.error(f"RabbitMQ connection failed: {e}. Retrying in 5 seconds...")
            await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"Unexpected error during RabbitMQ connection: {e}. Retrying in 5 seconds...")
            await asyncio.sleep(5)
    
    if not connection:
        logger.critical("Failed to connect to RabbitMQ after multiple retries. Exiting.")
        return

    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=rabbitmq_config.PREFETCH_COUNT or 1) # Use prefetch from config or default

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
