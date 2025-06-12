
from fastapi import APIRouter, HTTPException, status
import asyncio
import logging
import time
import uuid # uuid 추가
from app.api.question_generator.question_generator_schema import (
    FollowupRequest,
    FollowupResponse, # 동기 응답을 위해 다시 사용
)
from app.api.question_generator.question_generator_model import (
    call_llm,
    call_openai_api,
)
from app.api.question_generator.question_generator_parser import parse_questions
from app.api.question_generator.question_generator_config import (
    LANGFUSE_CONFIG,
    API_CONFIG,
)
from langfuse import Langfuse
import os
from typing import List, Dict, Any, Optional # os 임포트가 중복되어 하나 제거, typing 추가
from dotenv import load_dotenv

# RabbitMQ 관련 import 제거
# from app import rabbitmq_producer
# from app.config.rabbitmq_config import ROUTING_KEY_QUESTION_GENERATOR

load_dotenv()

try:
    from app.vector_db.retriever import question_rag_retriever
    VECTOR_DB_AVAILABLE = True
    logging.info("Vector DB 모듈이 로드되었습니다.")
except ImportError:
    VECTOR_DB_AVAILABLE = False
    question_rag_retriever = None
    logging.warning("Vector DB 모듈을 찾을 수 없습니다. RAG 기능이 비활성화됩니다.")

router = APIRouter()
logger = logging.getLogger(__name__)

# 설정에서 Langfuse 초기화
langfuse = Langfuse(**LANGFUSE_CONFIG) if all(LANGFUSE_CONFIG.values()) else None

# 설정에서 API 구성 가져오기
GENERATE_COUNT = API_CONFIG["generate_count"]
MAX_HISTORY_QUESTIONS = API_CONFIG["max_history_questions"]

_prompt_cache = {}


def get_cached_prompt(prompt_name: str):
    """프롬프트를 캐시에서 가져오거나, 없으면 Langfuse에서 로드하여 캐시에 저장"""
    if prompt_name not in _prompt_cache:
        logger.info(f"프롬프트 캐시 미스: {prompt_name} - Langfuse에서 로드 중...")
        if langfuse:
            try:
                _prompt_cache[prompt_name] = langfuse.get_prompt(prompt_name)
                logger.info(f"프롬프트 캐시 저장 완료: {prompt_name}")
            except Exception as e:
                logger.error(f"Langfuse에서 프롬프트 로드 실패 ({prompt_name}): {e}")
                return None # 실패 시 None 반환
        else:
            logger.warning("Langfuse가 설정되지 않아 프롬프트를 캐시할 수 없습니다.")
            return None
    else:
        logger.debug(f"프롬프트 캐시 히트: {prompt_name}")
    return _prompt_cache[prompt_name]


def validate_request(req: FollowupRequest) -> None:
    """입력 요청 유효성 검사"""
    if not req.selected_question or not req.selected_question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="메인 질문은 비워둘 수 없습니다.",
        )
    if not req.interview_id or not req.interview_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="유효한 interview_id가 필요합니다.",
        )

# prepare_context_for_worker 함수 대신 원래의 prepare_context 함수 사용 (또는 유사하게 복원)
def prepare_context(req: FollowupRequest, trace) -> Dict[str, Any]:
    """요청 데이터로부터 프롬프트 컨텍스트 준비 (동기 처리용)"""
    all_used_questions = (req.passed_questions or [])[-MAX_HISTORY_QUESTIONS:]

    passed_section = ""
    if all_used_questions:
        joined = "\n".join(f"- {q}" for q in all_used_questions)
        passed_section = f"\n\n[이전 질문 목록]\n{joined}"

    retrieved_section = ""
    rag_span = None
    if trace and hasattr(trace, 'span'): # trace 객체가 span 메소드를 가지고 있는지 확인
        rag_span = trace.span(name="rag_retrieval")
    
    rag_start_time = time.time()

    if VECTOR_DB_AVAILABLE and question_rag_retriever:
        try:
            rag_results = question_rag_retriever(req.selected_question, req.keyword or "")
            rag_execution_time = time.time() - rag_start_time
            if rag_span:
                rag_span.update(
                    input={"query": req.selected_question, "keyword": req.keyword or ""},
                    output={"results": rag_results},
                    metadata={"execution_time_seconds": rag_execution_time},
                )
            retrieved_questions = [r["question"] for r in rag_results["results"]] # 수정: rag_results가 딕셔너리일 수 있음
            if retrieved_questions:
                joined_rag = "\n".join(f"- {q}" for q in retrieved_questions)
                retrieved_section = f"\n\n[유사한 기존 질문]\n{joined_rag}"
            if rag_span: rag_span.end()
        except Exception as e:
            rag_execution_time = time.time() - rag_start_time
            logging.warning(f"RAG 검색 실패: {e}")
            if rag_span:
                rag_span.end(
                    error=str(e), metadata={"execution_time_seconds": rag_execution_time}
                )
            retrieved_section = ""
    else:
        rag_execution_time = time.time() - rag_start_time
        logging.info("Vector DB 모듈이 없어 RAG 검색을 건너뜁니다.")
        if rag_span:
            rag_span.update(
                input={"status": "skipped"},
                output={"reason": "Vector DB 모듈이 없음"},
                metadata={"execution_time_seconds": rag_execution_time},
            )
            if rag_span: rag_span.end()
            
    return {
        "selected_question": req.selected_question,
        "keyword": req.keyword or "",
        "passed_questions": passed_section,
        "retrieved_questions": retrieved_section,
        "num_questions": GENERATE_COUNT,
    }


async def generate_primary_questions(prompt: str, trace) -> List[str]:
    """주 질문 생성 로직 (동기 처리용)"""
    llm_span = None
    if trace and hasattr(trace, 'span'):
        llm_span = trace.span(name="llm_call_primary")
    llm_start_time = time.time()

    try:
        # call_llm이 비동기 함수라고 가정
        raw_response = await call_llm(prompt, trace_id=trace.id if trace else None)
        llm_execution_time = time.time() - llm_start_time
        if llm_span:
            llm_span.update(
                input={"prompt": prompt},
                output={"raw_response": raw_response},
                metadata={"execution_time_seconds": llm_execution_time},
            )
            llm_span.end()

        parsing_span = None
        if trace and hasattr(trace, 'span'):
            parsing_span = trace.span(name="response_parsing_primary")
        parsing_start_time = time.time()
        questions = parse_questions(raw_response)[:GENERATE_COUNT]
        parsing_execution_time = time.time() - parsing_start_time
        if parsing_span:
            parsing_span.update(
                input={"raw_response": raw_response},
                output={"parsed_questions": questions},
                metadata={"execution_time_seconds": parsing_execution_time},
            )
            parsing_span.end()
        return questions
    except Exception as e:
        llm_execution_time = time.time() - llm_start_time
        if llm_span and not llm_span.ended: # llm_span이 None이 아니고 아직 끝나지 않았다면
            llm_span.end(
                error=str(e), metadata={"execution_time_seconds": llm_execution_time}
            )
        logger.error(f"주 질문 생성 중 오류: {e}")
        raise


async def generate_additional_questions(
    req: FollowupRequest,
    passed_section: str, # prepare_context에서 생성된 passed_section 사용
    remaining_count: int,
    trace,
    existing_questions: List[str],
) -> List[str]:
    """질문 개수가 부족할 시 추가질문을 생성 (동기 처리용)"""
    if remaining_count <= 0:
        return existing_questions

    logger.info(
        f"OpenAI API로 추가 질문 생성 시작: interview_id={req.interview_id}, remaining_count={remaining_count}"
    )

    context_api = {
        "selected_question": req.selected_question,
        "keyword": req.keyword or "",
        "passed_questions": passed_section, # 수정: passed_section 직접 사용
        "ungenerated_questions_num": remaining_count,
    }

    prompt_template_api = get_cached_prompt("followup_questions_generator_api")
    if not prompt_template_api:
        logger.error("추가 질문 생성을 위한 프롬프트 템플릿을 로드할 수 없습니다.")
        return existing_questions # 프롬프트 없으면 기존 질문만 반환

    prompt_api = prompt_template_api.compile(**context_api)

    llm_span_api = None
    if trace and hasattr(trace, 'span'):
        llm_span_api = trace.span(name="open_api_call_additional")
    llm_api_start_time = time.time()

    try:
        # call_openai_api가 비동기 함수라고 가정
        raw_response_api = await call_openai_api(prompt_api, trace_id=trace.id if trace else None)
        llm_api_execution_time = time.time() - llm_api_start_time
        if llm_span_api:
            llm_span_api.update(
                input={"prompt_api": prompt_api},
                output={"raw_response_api": raw_response_api},
                metadata={"execution_time_seconds": llm_api_execution_time},
            )
            llm_span_api.end()

        parsing_span_api = None
        if trace and hasattr(trace, 'span'):
            parsing_span_api = trace.span(name="response_parsing_additional")
        parsing_api_start_time = time.time()
        additional_questions = parse_questions(raw_response_api)
        parsing_api_execution_time = time.time() - parsing_api_start_time
        if parsing_span_api:
            parsing_span_api.update(
                input={"raw_response_api": raw_response_api},
                output={"parsed_questions": additional_questions},
                metadata={"execution_time_seconds": parsing_api_execution_time},
            )
            parsing_span_api.end()

        unique_questions = [
            q for q in additional_questions if q not in existing_questions
        ]
        
        final_questions = existing_questions.copy()
        final_questions.extend(unique_questions)

        logger.info(
            f"OpenAI API로 추가 질문 생성 완료: interview_id={req.interview_id}, generated_count={len(additional_questions)}, unique_count={len(unique_questions)}"
        )
        return final_questions[:GENERATE_COUNT] # 최종적으로 GENERATE_COUNT 만큼만 반환
    except Exception as e:
        llm_api_execution_time = time.time() - llm_api_start_time
        logger.error(
            f"OpenAI API 추가 질문 생성 실패: interview_id={req.interview_id}, error={str(e)}"
        )
        if llm_span_api and not llm_span_api.ended: # llm_span_api가 None이 아니고 아직 끝나지 않았다면
            llm_span_api.end(
                error=str(e),
                metadata={"execution_time_seconds": llm_api_execution_time},
            )
        # 추가 질문 생성 실패 시 기존 질문만 반환하거나, 예외를 발생시킬 수 있음
        return existing_questions # 실패 시 기존 질문만 반환


@router.post("/followup-questions", response_model=FollowupResponse) # response_model 복원
async def generate_followup_sync(req: FollowupRequest) -> FollowupResponse: # 함수명 변경 및 RabbitMQ 로직 제거
    validate_request(req)
    request_start_time = time.time()

    # 모델 가용성 체크는 유지
    from app.main import is_model_available
    if not is_model_available("question_generator"):
        logger.error("question_generator 모델이 사용 불가능합니다.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="꼬리질문 생성 모델이 현재 사용할 수 없습니다. 잠시 후 다시 시도해주세요.",
        )

    trace = None
    if langfuse:
        try:
            # 동기 처리이므로 전체 과정을 하나의 trace로 관리
            trace = langfuse.trace(
                name="generate_followup_questions_sync",
                input=req.dict(),
                metadata={"interview_id": req.interview_id, "async": False}, # async: False로 명시
                # trace_id=f"followup_sync_{req.interview_id}_{uuid.uuid4().hex}" # 필요시 고유 ID 사용
            )
            logger.info(f"Langfuse trace 시작 (동기): {trace.id} for interview_id: {req.interview_id}")
        except Exception as e:
            logger.warning(f"Langfuse trace 시작 실패: {e}")
            trace = None # 실패 시 trace는 None

    try:
        context = prepare_context(req, trace)
        
        prompt_template = get_cached_prompt("followup_questions_generator")
        if not prompt_template:
            raise HTTPException(status_code=500, detail="주 질문 생성을 위한 프롬프트 템플릿을 로드할 수 없습니다.")

        prompt = prompt_template.compile(**context)

        questions = await generate_primary_questions(prompt, trace)

        if len(questions) < GENERATE_COUNT:
            logger.info(f"생성된 주 질문 개수 부족 ({len(questions)}/{GENERATE_COUNT}), 추가 질문 생성 시도...")
            # passed_section을 prepare_context에서 가져오거나, generate_additional_questions 내부에서 다시 생성
            # 여기서는 prepare_context의 passed_questions를 사용한다고 가정
            questions = await generate_additional_questions(
                req,
                context["passed_questions"], # prepare_context에서 생성된 passed_questions 사용
                GENERATE_COUNT - len(questions),
                trace,
                questions, # 기존에 생성된 질문 전달
            )
        
        if not questions:
             logger.warning(f"최종 생성된 질문이 없습니다. interview_id={req.interview_id}")
             # 질문이 전혀 생성되지 않은 경우 빈 리스트 대신 오류를 발생시킬 수도 있음
             # raise HTTPException(status_code=500, detail="꼬리질문을 생성하지 못했습니다.")

        api_processing_time = time.time() - request_start_time
        logger.info(
            f"꼬리질문 생성 완료 (동기): interview_id={req.interview_id}, "
            f"count={len(questions)}, total_time={api_processing_time:.4f}s"
        )
        if trace:
            trace.update(output={"questions": questions}, metadata={"total_execution_time_seconds": api_processing_time})

        return FollowupResponse(
            message="Follow-up questions generated successfully.",
            interview_id=req.interview_id,
            questions=questions[:GENERATE_COUNT], # 최종적으로 GENERATE_COUNT 만큼만 반환
            trace_id=trace.id if trace else None
        )

    except HTTPException as http_exc: # FastAPI의 HTTPException은 그대로 전달
        if trace:
            trace.update(status="ERROR", output={"error": http_exc.detail, "status_code": http_exc.status_code})
        raise http_exc
    except Exception as e:
        api_processing_time = time.time() - request_start_time
        logger.error(f"꼬리질문 생성 중 심각한 오류 발생: interview_id={req.interview_id}, error={str(e)}", exc_info=True)
        if trace:
            trace.update(status="ERROR", output={"error": str(e)}, metadata={"total_execution_time_seconds": api_processing_time})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"꼬리질문 생성 중 서버 오류 발생: {str(e)}",
        )
