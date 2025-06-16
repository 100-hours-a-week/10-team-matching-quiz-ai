
from fastapi import APIRouter, HTTPException, status
import logging
import time
import uuid
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from app.api.question_generator.question_generator_schema import (
    FollowupRequest,
    FollowupResponse,
)
from app.api.question_generator.question_generator_model_api import (
    call_llm,
    call_openai_api,
    check_vllm_api_health,
)
from app.api.question_generator.question_generator_parser import parse_questions
from app.api.question_generator.question_generator_config import (
    LANGFUSE_CONFIG,
    API_CONFIG,
)

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

# Langfuse 초기화
langfuse = None
try:
    if all(LANGFUSE_CONFIG.values()):
        from langfuse import Langfuse
        langfuse = Langfuse(**LANGFUSE_CONFIG)
        logger.info("Langfuse가 성공적으로 초기화되었습니다.")
    else:
        logger.warning("Langfuse 설정이 불완전하여 비활성화됩니다.")
except ImportError:
    logger.warning("Langfuse 패키지를 찾을 수 없습니다.")
except Exception as e:
    logger.error(f"Langfuse 초기화 실패: {e}")

# API 설정
GENERATE_COUNT = API_CONFIG["generate_count"]
MAX_HISTORY_QUESTIONS = API_CONFIG["max_history_questions"]

# 프롬프트 캐시
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
    """vLLM API를 사용한 주 질문 생성"""
    llm_span = None
    if trace and hasattr(trace, 'span'):
        llm_span = trace.span(name="vllm_api_primary_generation")
    
    llm_start_time = time.time()

    try:
        # vLLM API 호출
        raw_response = await call_llm(prompt, trace_id=trace.id if trace else None)
        llm_execution_time = time.time() - llm_start_time
        
        if llm_span:
            llm_span.update(
                input={"prompt": prompt},
                output={"raw_response": raw_response},
                metadata={"execution_time_seconds": llm_execution_time},
            )
            llm_span.end()

        # 응답 파싱
        parsing_span = None
        if trace and hasattr(trace, 'span'):
            parsing_span = trace.span(name="response_parsing")
        
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
        
        logger.info(f"vLLM API로 {len(questions)}개의 주 질문을 생성했습니다.")
        return questions
        
    except Exception as e:
        llm_execution_time = time.time() - llm_start_time
        if llm_span:
            llm_span.end(error=str(e), metadata={"execution_time_seconds": llm_execution_time})
        logger.error(f"vLLM API 주 질문 생성 중 오류: {e}")
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


@router.post("/followup-questions", response_model=FollowupResponse)
async def generate_followup_questions(req: FollowupRequest) -> FollowupResponse:
    """
    vLLM OpenAI 호환 API를 사용한 꼬리질문 생성
    """
    validate_request(req)
    request_start_time = time.time()

    # vLLM API 서버 상태 확인
    if not await check_vllm_api_health():
        logger.error("vLLM API 서버가 응답하지 않습니다.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="vLLM API 서버가 현재 사용할 수 없습니다. 잠시 후 다시 시도해주세요.",
        )

    # Langfuse 추적 시작
    trace = None
    if langfuse:
        try:
            trace = langfuse.trace(
                name="generate_followup_questions_vllm_api",
                input=req.dict(),
                metadata={
                    "interview_id": req.interview_id,
                    "method": "vllm_openai_compatible_api"
                },
            )
            logger.info(f"Langfuse trace 시작: {trace.id} for interview_id: {req.interview_id}")
        except Exception as e:
            logger.warning(f"Langfuse trace 시작 실패: {e}")
            trace = None

    try:
        # 1. 컨텍스트 준비 (RAG 검색 포함)
        context = prepare_context(req, trace)
        
        # 2. 주 프롬프트 템플릿 로드
        prompt_template = get_cached_prompt("followup_questions_generator")
        if not prompt_template:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="주 질문 생성을 위한 프롬프트 템플릿을 로드할 수 없습니다."
            )

        prompt = prompt_template.compile(**context)

        # 3. vLLM API로 주 질문 생성
        questions = await generate_primary_questions(prompt, trace)

        # 4. 부족한 질문 개수를 OpenAI API로 보완
        if len(questions) < GENERATE_COUNT:
            remaining_count = GENERATE_COUNT - len(questions)
            logger.info(f"생성된 주 질문 개수 부족 ({len(questions)}/{GENERATE_COUNT}), OpenAI API로 {remaining_count}개 추가 생성")
            
            questions = await generate_additional_questions(
                req, context["passed_questions"], remaining_count, trace, questions
            )

        # 5. 최종 검증 및 응답
        if not questions:
            logger.warning(f"최종 생성된 질문이 없습니다. interview_id={req.interview_id}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="꼬리질문을 생성하지 못했습니다."
            )

        # 최종 질문 개수 제한
        final_questions = questions[:GENERATE_COUNT]
        
        api_processing_time = time.time() - request_start_time
        logger.info(
            f"꼬리질문 생성 완료: interview_id={req.interview_id}, "
            f"count={len(final_questions)}, total_time={api_processing_time:.4f}s"
        )

        # Langfuse 추적 완료
        if trace:
            trace.update(
                output={"questions": final_questions},
                metadata={
                    "total_execution_time_seconds": api_processing_time,
                    "questions_count": len(final_questions),
                    "success": True
                }
            )

        return FollowupResponse(
            message="Follow-up questions generated successfully using vLLM API.",
            interview_id=req.interview_id,
            followup_questions=final_questions,
        )

    except HTTPException as http_exc:
        # FastAPI HTTPException은 그대로 전달
        if trace:
            trace.update(
                status="ERROR",
                output={"error": http_exc.detail, "status_code": http_exc.status_code}
            )
        raise http_exc
        
    except Exception as e:
        api_processing_time = time.time() - request_start_time
        logger.error(
            f"꼬리질문 생성 중 예상치 못한 오류: interview_id={req.interview_id}, error={str(e)}",
            exc_info=True
        )
        
        if trace:
            trace.update(
                status="ERROR",
                output={"error": str(e)},
                metadata={
                    "total_execution_time_seconds": api_processing_time,
                    "success": False
                }
            )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"꼬리질문 생성 중 서버 오류 발생: {str(e)}",
        )
