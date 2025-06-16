import asyncio
import json
import logging
import time

import aio_pika
from langfuse import Langfuse

# Configs
from app.config import rabbitmq_config
from app.api.quiz_generator.quiz_generator_config import (
    QUIZ_LANGFUSE_SECRET_KEY,
    QUIZ_LANGFUSE_PUBLIC_KEY,
    QUIZ_LANGFUSE_HOST,
)

# Schema for input and internal data structures
from app.api.quiz_generator.quiz_generator_schema import FollowupRequest, QuizItem, QuizData, FollowupResponse

# Core logic components
from app.vector_db.retriever import quiz_rag_retriever
from app.api.quiz_generator.quiz_generator_model import generate_quiz
from app.api.quiz_generator.quiz_generator_parser import (
    parse_response,
    filter_and_select_quizzes,
    remove_prompt_content,
)
from app import rabbitmq_producer

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("QuizWorker")

# Initialize Langfuse
langfuse_client = Langfuse(
    secret_key=QUIZ_LANGFUSE_SECRET_KEY,
    public_key=QUIZ_LANGFUSE_PUBLIC_KEY,
    host=QUIZ_LANGFUSE_HOST,
    debug=False, # Set to True for more verbose Langfuse logs if needed
)

async def process_quiz_generation_task(message: aio_pika.IncomingMessage):
    async with message.process(ignore_processed=True):
        trace = None  # Initialize trace to None
        try:
            logger.info(f"Received message: {message.message_id} for quiz generation.")
            data = json.loads(message.body.decode())
            req = FollowupRequest(**data)

            trace = langfuse_client.trace(
                name="quiz_generation_worker",
                tags=["quiz", "generate", "worker"],
                input={"question_list": req.question_history_list, "interview_id": req.interview_id},
                metadata={"message_id": str(message.message_id), "amqp_timestamp": str(message.timestamp)},
            )
            request_start_time = time.time()

            prompt_template_name = "quiz_generation"
            
            get_prompt_span = trace.span(name="langfuse_get_prompt_worker")
            try:
                # Assuming get_prompt might be blocking
                prompt_template = await asyncio.to_thread(langfuse_client.get_prompt, prompt_template_name)
            except Exception as e:
                logger.error(f"Failed to get Langfuse prompt '{prompt_template_name}': {e}", exc_info=True)
                get_prompt_span.end(output={"error": str(e)}, status="ERROR")
                trace.update(status="ERROR", output={"error": f"Failed to get prompt: {e}"})
                await message.reject(requeue=False)
                return
            
            if prompt_template is None:
                logger.error(f"Langfuse prompt '{prompt_template_name}' not found.")
                get_prompt_span.end(output={"error": f"Prompt '{prompt_template_name}' not found"}, status="ERROR")
                trace.update(status="ERROR", output={"error": f"Prompt '{prompt_template_name}' not found"})
                await message.reject(requeue=False)
                return
            get_prompt_span.end(output={"prompt_name": prompt_template_name, "type": str(type(prompt_template))})

            # RAG
            rag_span = trace.span(name="rag_retrieval_worker")
            rag_start_time = time.time()
            # Assuming quiz_rag_retriever might be blocking
            quiz_rag_results = await asyncio.to_thread(quiz_rag_retriever, req.question_history_list)
            related_questions = []
            if quiz_rag_results: # Ensure quiz_rag_results is not None
                for rag_result_item in quiz_rag_results:
                    if rag_result_item and rag_result_item.get("result"):
                        for doc in rag_result_item["result"]:
                            if doc and "content" in doc:
                                related_questions.append(doc["content"])
            rag_execution_time = time.time() - rag_start_time
            
            # RAG 결과를 더 상세하게 Langfuse에 추적
            rag_span.update(
                input={
                    "question_history_list": req.question_history_list,
                    "question_history_count": len(req.question_history_list),
                    "search_method": "quiz_rag_retriever",
                    "retrieval_type": "vector_similarity"
                },
                output={
                    "rag_results_count": len(quiz_rag_results) if quiz_rag_results else 0,
                    "related_questions": related_questions,
                    "related_questions_count": len(related_questions),
                    "retrieved_documents": quiz_rag_results if quiz_rag_results else [],
                    "retrieval_successful": quiz_rag_results is not None and len(related_questions) > 0
                },
                metadata={
                    "execution_time_seconds": rag_execution_time,
                    "collection_name": "quiz_collection",
                    "interview_id": req.interview_id,
                    "worker_type": "quiz_worker"
                }
            )
            rag_span.end()

            joined_questions = "\n".join(req.question_history_list)
            related_questions_text = "\n".join(related_questions) if related_questions else "관련 문서 없음"
            
            context_api = {
                "joined_questions": joined_questions,
                "related_questions": related_questions_text,
            }

            # Prompt Compilation
            prompt_build_span = trace.span(name="prompt_build_worker")
            prompt_build_start_time = time.time()
            
            if hasattr(prompt_template, "compile"): # Check if it's a compilable prompt object
                prompt = prompt_template.compile(**context_api)
                compilation_method = "langfuse_compile"
            else: # Fallback for string templates or other non-compilable objects
                prompt_text = prompt_template.prompt if hasattr(prompt_template, "prompt") else str(prompt_template)
                prompt = prompt_text.replace("{{joined_questions}}", joined_questions)
                prompt = prompt.replace("{{related_questions}}", related_questions_text)
                compilation_method = "string_replacement"
            
            prompt += ("\n--- END OF INSTRUCTION ---")
            prompt_build_execution_time = time.time() - prompt_build_start_time
            
            # 더 상세한 prompt 추적 (question_generator_api 스타일)
            prompt_build_span.update(
                input={
                    "context_api": context_api,
                    "joined_questions_length": len(joined_questions),
                    "related_questions_count": len(related_questions),
                    "template_type": str(type(prompt_template)),
                    "compilation_method": compilation_method
                },
                output={
                    "compiled_prompt": prompt,
                    "compiled_prompt_length": len(prompt),
                    "final_prompt_preview": prompt[:200] + "..." if len(prompt) > 200 else prompt
                },
                metadata={
                    "execution_time_seconds": prompt_build_execution_time,
                    "interview_id": req.interview_id
                }
            )
            prompt_build_span.end()

            # LLM Generation
            llm_span = trace.span(name="llm_generation_worker")
            llm_start_time = time.time()
            logger.info(f"Calling generate_quiz for interview_id: {req.interview_id}")
            
            # generate_quiz is synchronous, run in thread
            raw_output = await asyncio.to_thread(generate_quiz, prompt, use_chat_template=True)
            
            logger.info(f"generate_quiz completed for interview_id: {req.interview_id}")
            llm_execution_time = time.time() - llm_start_time
            
            # Langfuse에 상세한 input/output 추적 (question_generator_api 스타일)
            llm_span.update(
                input={
                    "prompt": prompt,
                    "prompt_length": len(prompt),
                    "use_chat_template": True,
                    "interview_id": req.interview_id,
                    "model_type": "transformers"
                },
                output={
                    "raw_response": raw_output,
                    "raw_output_length": len(raw_output or ""),
                    "generation_successful": raw_output is not None and len(raw_output.strip()) > 0
                },
                metadata={
                    "execution_time_seconds": llm_execution_time,
                    "worker_name": "quiz_worker",
                    "function_called": "generate_quiz"
                }
            )
            llm_span.end()

            cleaned_output = remove_prompt_content(raw_output)

            # Parsing
            parsing_span = trace.span(name="response_parsing_worker")
            parsing_start_time = time.time()
            parsed_list = parse_response(cleaned_output)
            parsing_execution_time = time.time() - parsing_start_time
            
            if not parsed_list:
                logger.error(f"Failed to parse any quizzes for interview_id: {req.interview_id}. Cleaned output: {cleaned_output[:500]}")
                parsing_span.end(
                    input={
                        "cleaned_output": cleaned_output,
                        "cleaned_output_length": len(cleaned_output),
                        "cleaned_output_preview": cleaned_output[:500] + "..." if len(cleaned_output) > 500 else cleaned_output
                    },
                    output={
                        "error": "No quizzes parsed", 
                        "parsed_count": 0,
                        "parsing_successful": False
                    }, 
                    metadata={
                        "execution_time_seconds": parsing_execution_time,
                        "interview_id": req.interview_id
                    }, 
                    status="ERROR"
                )
                trace.update(status="ERROR", output={"error": "Parsing failed"})
                await message.reject(requeue=False)
                return
            
            # 성공적인 파싱의 경우 상세 정보 추적
            parsing_span.update(
                input={
                    "cleaned_output": cleaned_output,
                    "cleaned_output_length": len(cleaned_output),
                    "raw_output_length": len(raw_output or "")
                },
                output={
                    "parsed_quiz_count": len(parsed_list),
                    "parsing_successful": True,
                    "parsed_quizzes_sample": parsed_list[:2] if len(parsed_list) >= 2 else parsed_list  # 처음 2개 예시
                },
                metadata={
                    "execution_time_seconds": parsing_execution_time,
                    "interview_id": req.interview_id
                }
            )
            parsing_span.end()
            
            # Filtering
            filtering_span = trace.span(name="difficulty_filtering_worker")
            filtering_start_time = time.time()
            final_quizzes = filter_and_select_quizzes(parsed_list)
            filtering_execution_time = time.time() - filtering_start_time
            
            easy_count = sum(1 for q in parsed_list if q.get("difficulty") == "하")
            medium_count = sum(1 for q in parsed_list if q.get("difficulty") == "중")
            hard_count = sum(1 for q in parsed_list if q.get("difficulty") == "상")

            if len(final_quizzes) != 10:
                error_msg = f"난이도별 문제 부족 - 최종 {len(final_quizzes)}개. 파싱된 문제 중 하:{easy_count}, 중:{medium_count}, 상:{hard_count}"
                logger.error(f"{error_msg} for interview_id: {req.interview_id}")
                filtering_span.end(
                    input={
                        "parsed_list": parsed_list,
                        "parsed_quiz_count": len(parsed_list),
                        "difficulty_distribution": {
                            "easy": easy_count,
                            "medium": medium_count,
                            "hard": hard_count
                        }
                    },
                    output={
                        "error": error_msg, 
                        "final_quiz_count": len(final_quizzes),
                        "filtering_successful": False
                    }, 
                    metadata={
                        "execution_time_seconds": filtering_execution_time,
                        "interview_id": req.interview_id,
                        "target_quiz_count": 10
                    }, 
                    status="ERROR"
                )
                trace.update(status="ERROR", output={"error": "Filtering failed - insufficient quizzes per difficulty"})
                await message.reject(requeue=False)
                return
            
            # 성공적인 필터링의 경우
            filtering_span.update(
                input={
                    "parsed_quiz_count": len(parsed_list),
                    "difficulty_distribution": {
                        "easy": easy_count,
                        "medium": medium_count,
                        "hard": hard_count
                    }
                },
                output={
                    "final_quiz_count": len(final_quizzes),
                    "filtering_successful": True,
                    "final_quizzes_sample": final_quizzes[:2] if len(final_quizzes) >= 2 else final_quizzes  # 처음 2개 예시
                },
                metadata={
                    "execution_time_seconds": filtering_execution_time,
                    "interview_id": req.interview_id,
                    "target_quiz_count": 10
                }
            )
            filtering_span.end()

            quiz_items = [QuizItem(**item) for item in final_quizzes] # Validate with Pydantic
            quiz_data_obj = QuizData(interview_id=req.interview_id, questions=quiz_items)
            
            response_obj = FollowupResponse(message="Quiz generation completed successfully.",data=quiz_data_obj)

            logger.info(f"퀴즈 응답 요약] interview_id={req.interview_id}, 퀴즈 수={len(quiz_items)}")
            logger.debug(f"[전체 응답 내용]: {response_obj}")
            
            # Publish response back to backend
            response_span = trace.span(name="response_publishing_worker")
            response_start_time = time.time()
            try:
                response_success = await rabbitmq_producer.publish_response_message(
                    message_body=response_obj.model_dump(),
                    exchange_name=rabbitmq_config.QUIZ_RESPONSE_EXCHANGE_NAME,
                    routing_key=rabbitmq_config.QUIZ_RESPONSE_ROUTING_KEY
                )
                if response_success:
                    logger.info(f"Successfully published quiz response for interview_id: {req.interview_id}")
                else:
                    logger.error(f"Failed to publish quiz response for interview_id: {req.interview_id}")
                    # Don't fail the entire task, but log the error
            except Exception as e:
                logger.error(f"Error publishing quiz response for interview_id: {req.interview_id}: {e}")
                
            response_execution_time = time.time() - response_start_time
            response_span.update(
                input={"interview_id": req.interview_id, "quiz_count": len(quiz_items)},
                output={"response_published": response_success if 'response_success' in locals() else False},
                metadata={"execution_time_seconds": response_execution_time}
            )
            response_span.end()
            
            request_execution_time = time.time() - request_start_time
            trace.update(
                output={"final_quiz_count": len(quiz_items), "interview_id_processed": quiz_data_obj.interview_id},
                metadata={"total_execution_time_seconds": request_execution_time}
            )
            logger.info(f"Successfully processed quiz generation for interview_id: {req.interview_id}. Generated {len(quiz_items)} quizzes.")
            logger.info(f"Generated Quiz Data for interview_id {req.interview_id}: {quiz_data_obj.model_dump_json(indent=2)}")

            await message.ack()
            logger.info(f"Message {message.message_id} acked.")

        except json.JSONDecodeError as e:
            logger.error(f"JSONDecodeError: {e} for message body: {message.body[:200]}", exc_info=True)
            if trace: trace.update(status="ERROR", output={"error": f"JSONDecodeError: {str(e)}"})
            await message.reject(requeue=False)
        except Exception as e:
            logger.error(f"Unhandled error processing message {message.message_id if message else 'UnknownMsg'}: {e}", exc_info=True)
            if trace: trace.update(status="ERROR", output={"error": f"Unhandled error: {str(e)}"})
            await message.reject(requeue=False)


async def main_quiz_worker():
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
                client_properties={'connection_name': 'quiz_worker_connection'}
            )
            logger.info("Connected to RabbitMQ.")
            break 
        except (aio_pika.exceptions.AMQPConnectionError, ConnectionRefusedError) as e:
            logger.error(f"RabbitMQ connection failed: {e}. Retrying in 5 seconds...")
            await asyncio.sleep(5)
        except Exception as e: # Catch any other unexpected errors during connection
            logger.error(f"Unexpected error during RabbitMQ connection: {e}. Retrying in 5 seconds...")
            await asyncio.sleep(5)

    if not connection:
        logger.critical("Failed to connect to RabbitMQ after multiple retries. Exiting.")
        return

    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=rabbitmq_config.PREFETCH_COUNT or 1)

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
    # Model initialization for quiz_generator is handled by its lazy loading mechanism
    # when generate_quiz -> get_model is called.
    logger.info("Starting Quiz Generation Worker...")
    try:
        asyncio.run(main_quiz_worker())
    except KeyboardInterrupt:
        logger.info("Quiz Generation Worker interrupted by user. Exiting.")
    except Exception as e:
        logger.critical(f"Quiz Generation Worker failed to start or run: {e}", exc_info=True)
