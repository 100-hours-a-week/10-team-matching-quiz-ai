from fastapi import APIRouter, HTTPException
from app.api.quiz_generator.quiz_generator_schema import FollowupRequest, FollowupResponse, QuizData, QuizItem
from app import rabbitmq_producer
from app.config import rabbitmq_config
import logging
import sys
import time
import json
from langfuse import Langfuse

# Core logic components
from app.vector_db.retriever import quiz_rag_retriever
from app.api.quiz_generator.quiz_generator_model import generate_quiz
from app.api.quiz_generator.quiz_generator_parser import (
    parse_response,
    filter_and_select_quizzes,
    remove_prompt_content,
)
from app.api.quiz_generator.quiz_generator_config import (
    QUIZ_LANGFUSE_SECRET_KEY,
    QUIZ_LANGFUSE_PUBLIC_KEY,
    QUIZ_LANGFUSE_HOST,
)

router = APIRouter()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Initialize Langfuse
langfuse_client = Langfuse(
    secret_key=QUIZ_LANGFUSE_SECRET_KEY,
    public_key=QUIZ_LANGFUSE_PUBLIC_KEY,
    host=QUIZ_LANGFUSE_HOST,
    debug=False,
)

def process_quiz_generation(request: FollowupRequest) -> dict:
    """Quiz 생성 로직을 독립적인 함수로 분리"""
    logger.info(f"[API 요청] 퀴즈 생성 시작: interview_id={request.interview_id}")
    logger.info(f"[API 요청] 질문 히스토리 개수: {len(request.question_history_list)}")

    # 면접 질문 없으면 바로 빈 리스트 반환
    if not request.question_history_list:
        logger.info(f"interview_id={request.interview_id} 면접 질문 히스토리 없음 → 빈 리스트 반환")
        return {
            "message": "면접 질문 히스토리 없음",
            "data": {
                "interview_id": request.interview_id,
                "questions": []
            }
        }

    trace = langfuse_client.trace(
        name="quiz_generation_api",
        tags=["quiz", "generate", "api"],
        input={"question_list": request.question_history_list, "interview_id": request.interview_id},
    )

    try:
        request_start_time = time.time()

        prompt_template_name = "quiz_generation"
        
        get_prompt_span = trace.span(name="langfuse_get_prompt_api")
        try:
            prompt_template = langfuse_client.get_prompt(prompt_template_name)
        except Exception as e:
            logger.error(f"Failed to get Langfuse prompt '{prompt_template_name}': {e}")
            get_prompt_span.end(output={"error": str(e)}, status="ERROR")
            trace.update(status="ERROR", output={"error": f"Failed to get prompt: {e}"})
            raise ValueError(f"프롬프트 템플릿을 가져올 수 없습니다: {e}")
        
        if prompt_template is None:
            logger.error(f"Langfuse prompt '{prompt_template_name}' not found.")
            get_prompt_span.end(output={"error": f"Prompt '{prompt_template_name}' not found"}, status="ERROR")
            trace.update(status="ERROR", output={"error": f"Prompt '{prompt_template_name}' not found"})
            raise ValueError(f"프롬프트 템플릿 '{prompt_template_name}'을 찾을 수 없습니다.")
        get_prompt_span.end(output={"prompt_name": prompt_template_name, "type": str(type(prompt_template))})

        # RAG
        rag_span = trace.span(name="rag_retrieval_api")
        rag_start_time = time.time()
        quiz_rag_results = quiz_rag_retriever(request.question_history_list)
        related_questions = []
        if quiz_rag_results:
            for rag_result_item in quiz_rag_results:
                if rag_result_item and rag_result_item.get("result"):
                    for doc in rag_result_item["result"]:
                        if doc and "content" in doc:
                            related_questions.append(doc["content"])
        rag_execution_time = time.time() - rag_start_time
        
        rag_span.update(
            input={
                "question_history_list": request.question_history_list,
                "question_history_count": len(request.question_history_list),
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
                "interview_id": request.interview_id,
                "api_type": "quiz_api"
            }
        )
        rag_span.end()

        joined_questions = "\n".join(request.question_history_list)
        related_questions_text = "\n".join(related_questions) if related_questions else "관련 문서 없음"
        
        context_api = {
            "joined_questions": joined_questions,
            "related_questions": related_questions_text,
        }

        # Prompt Compilation
        prompt_build_span = trace.span(name="prompt_build_api")
        prompt_build_start_time = time.time()
        
        if hasattr(prompt_template, "compile"):
            prompt = prompt_template.compile(**context_api)
            compilation_method = "langfuse_compile"
        else:
            prompt_text = prompt_template.prompt if hasattr(prompt_template, "prompt") else str(prompt_template)
            prompt = prompt_text.replace("{{joined_questions}}", joined_questions)
            prompt = prompt.replace("{{related_questions}}", related_questions_text)
            compilation_method = "string_replacement"
        
        prompt += ("\n--- END OF INSTRUCTION ---")
        prompt_build_execution_time = time.time() - prompt_build_start_time
        
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
                "interview_id": request.interview_id
            }
        )
        prompt_build_span.end()

        # LLM Generation
        llm_span = trace.span(name="llm_generation_api")
        llm_start_time = time.time()
        logger.info(f"Calling generate_quiz for interview_id: {request.interview_id}")
        
        raw_output = generate_quiz(prompt, use_chat_template=True)
        
        logger.info(f"generate_quiz completed for interview_id: {request.interview_id}")
        llm_execution_time = time.time() - llm_start_time
        
        llm_span.update(
            input={
                "prompt": prompt,
                "prompt_length": len(prompt),
                "use_chat_template": True,
                "interview_id": request.interview_id,
                "model_type": "transformers"
            },
            output={
                "raw_response": raw_output,
                "raw_output_length": len(raw_output or ""),
                "generation_successful": raw_output is not None and len(raw_output.strip()) > 0
            },
            metadata={
                "execution_time_seconds": llm_execution_time,
                "api_name": "quiz_api",
                "function_called": "generate_quiz"
            }
        )
        llm_span.end()

        cleaned_output = remove_prompt_content(raw_output)

        # Parsing
        parsing_span = trace.span(name="response_parsing_api")
        parsing_start_time = time.time()
        parsed_list = parse_response(cleaned_output)
        parsing_execution_time = time.time() - parsing_start_time
        
        if not parsed_list:
            logger.error(f"Failed to parse any quizzes for interview_id: {request.interview_id}. Cleaned output: {cleaned_output[:500]}")
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
                    "interview_id": request.interview_id
                }, 
                status="ERROR"
            )
            trace.update(status="ERROR", output={"error": "Parsing failed"})
            raise ValueError("퀴즈 파싱에 실패했습니다.")
        
        parsing_span.update(
            input={
                "cleaned_output": cleaned_output,
                "cleaned_output_length": len(cleaned_output),
                "raw_output_length": len(raw_output or "")
            },
            output={
                "parsed_quiz_count": len(parsed_list),
                "parsing_successful": True,
                "parsed_quizzes": json.dumps(parsed_list, ensure_ascii=False, indent=2)
            },
            metadata={
                "execution_time_seconds": parsing_execution_time,
                "interview_id": request.interview_id
            }
        )
        parsing_span.end()
        
        # Filtering
        filtering_span = trace.span(name="difficulty_filtering_api")
        filtering_start_time = time.time()
        final_quizzes = filter_and_select_quizzes(parsed_list)
        filtering_execution_time = time.time() - filtering_start_time
        
        easy_count = sum(1 for q in parsed_list if q.get("difficulty") == "하")
        medium_count = sum(1 for q in parsed_list if q.get("difficulty") == "중")
        hard_count = sum(1 for q in parsed_list if q.get("difficulty") == "상")

        if len(final_quizzes) != 10:
            error_msg = f"난이도별 문제 부족 - 최종 {len(final_quizzes)}개. 파싱된 문제 중 하:{easy_count}, 중:{medium_count}, 상:{hard_count}"
            logger.error(f"{error_msg} for interview_id: {request.interview_id}")
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
                    "interview_id": request.interview_id,
                    "target_quiz_count": 10
                }, 
                status="ERROR"
            )
            trace.update(status="ERROR", output={"error": "Filtering failed - insufficient quizzes per difficulty"})
            raise ValueError(f"퀴즈 필터링 실패: {error_msg}")
        
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
                "final_quizzes_sample": final_quizzes[:2] if len(final_quizzes) >= 2 else final_quizzes
            },
            metadata={
                "execution_time_seconds": filtering_execution_time,
                "interview_id": request.interview_id,
                "target_quiz_count": 10
            }
        )
        filtering_span.end()

        quiz_items = [QuizItem(**item) for item in final_quizzes]
        quiz_data_obj = QuizData(interview_id=request.interview_id, questions=quiz_items)
        
        response_obj = FollowupResponse(message="Quiz generation completed successfully.", data=quiz_data_obj)

        request_execution_time = time.time() - request_start_time
        trace.update(
            output={
                "final_quiz_count": len(quiz_items), 
                "interview_id_processed": quiz_data_obj.interview_id,
                "final_quizzes": json.dumps([item.model_dump() for item in quiz_items], ensure_ascii=False, indent=2)
            },
            metadata={"total_execution_time_seconds": request_execution_time}
        )
        
        logger.info(f"[API 처리 완료] interview_id={request.interview_id}, 퀴즈 수={len(quiz_items)}")
        
        return response_obj.model_dump()

    except Exception as e:
        logger.error(f"API 퀴즈 생성 처리 중 오류: {e}")
        if trace:
            trace.update(status="ERROR", output={"error": str(e)})
        raise


@router.post("/generate-quiz", status_code=202)
async def generate_quiz_api(req: FollowupRequest):
    """
    퀴즈 생성 요청 (비동기 처리)
    Quiz Worker로 작업을 전송하고 즉시 응답합니다.
    """
    logger.info(f"퀴즈 생성 요청 수신 (비동기 처리): interview_id={req.interview_id}")
    logger.info(f"질문 히스토리 개수: {len(req.question_history_list)}")

    # 없으면 CS 퀴즈 생성
    if req.question_history_list is None:
        logger.warning("질문 히스토리가 None입니다. 빈 리스트로 대체합니다.")
        req.question_history_list = []

    # Quiz Worker 상태 확인 (RabbitMQ 연결로 대체)
    try:
        connection = await rabbitmq_producer.get_rabbitmq_connection()
        if not connection or connection.is_closed:
            logger.error("RabbitMQ 연결이 불가능합니다.")
            raise HTTPException(
                status_code=503,
                detail="퀴즈 생성 서비스가 현재 사용할 수 없습니다. 잠시 후 다시 시도해주세요.",
            )
    except Exception as e:
        logger.error(f"RabbitMQ 연결 확인 실패: {e}")
        raise HTTPException(
            status_code=503,
            detail="퀴즈 생성 서비스 연결에 실패했습니다.",
        )

    # Quiz Worker로 작업 전송
    message_body = req.model_dump()

    try:
        success = await rabbitmq_producer.publish_message(
            routing_key=rabbitmq_config.ROUTING_KEY_QUIZ_GENERATOR,
            message_body=message_body,
        )
        
        if success:
            logger.info(f"퀴즈 생성 작업 메시지 발행 성공: interview_id={req.interview_id}")
            return {
                "message": "퀴즈 생성 작업이 Quiz Worker로 전송되었습니다.",
                "interview_id": req.interview_id,
                "status": "queued",
                "processing_mode": "worker",
                "questions_count": len(req.question_history_list),
                "note": "처리 완료 시 RabbitMQ 응답 큐를 통해 결과가 전송됩니다."
            }
        else:
            logger.error(f"퀴즈 생성 작업 메시지 발행 실패: interview_id={req.interview_id}")
            raise HTTPException(
                status_code=500, 
                detail="퀴즈 생성 요청 처리에 실패했습니다 (메시지 발행 실패)."
            )
            
    except Exception as e:
        logger.error(f"퀴즈 생성 작업 메시지 발행 중 예외 발생: {e}, interview_id={req.interview_id}")
        raise HTTPException(
            status_code=500, 
            detail="퀴즈 생성 요청 처리 중 내부 오류가 발생했습니다."
        )


@router.post("/generate-quiz-sync")
async def generate_quiz_sync(req: FollowupRequest):
    """
    퀴즈 생성 요청 (동기 처리) - 개발/테스트용
    """
    try:
        return process_quiz_generation(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"동기 API 처리 중 오류: {e}")
        raise HTTPException(status_code=500, detail="내부 서버 오류")

