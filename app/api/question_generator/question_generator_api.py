from fastapi import APIRouter, HTTPException, status
import logging
import time
import uuid
from typing import List, Dict, Any
from dotenv import load_dotenv

from app.api.question_generator.question_generator_schema import (
    FollowupRequest,
    FollowupResponse,
)
from app.api.question_generator.question_generator_model import (
    call_llm,
    call_openai_api,
)
from app.api.question_generator.question_generator_parser import parse_questions
from app.api.question_generator.question_generator_config import (
    LANGFUSE_CONFIG,
    API_CONFIG,
    VLLM_API_CONFIG,
)
from langfuse import Langfuse

load_dotenv()

# Vector DB 모듈 로드
try:
    from app.vector_db.retriever import question_rag_retriever
    VECTOR_DB_AVAILABLE = True
except ImportError:
    VECTOR_DB_AVAILABLE = False
    question_rag_retriever = None

router = APIRouter()
logger = logging.getLogger(__name__)

# 설정 초기화
langfuse = Langfuse(**LANGFUSE_CONFIG) if all(LANGFUSE_CONFIG.values()) else None
GENERATE_COUNT = API_CONFIG["generate_count"]
MAX_HISTORY_QUESTIONS = API_CONFIG["max_history_questions"]

# 프롬프트 캐시
_prompt_cache = {}


def get_cached_prompt(prompt_name: str):
    """프롬프트 캐시에서 가져오기"""
    if prompt_name not in _prompt_cache and langfuse:
        logger.info(f"프롬프트 로드: {prompt_name}")
        _prompt_cache[prompt_name] = langfuse.get_prompt(prompt_name)
    return _prompt_cache.get(prompt_name)


def prepare_context(req: FollowupRequest) -> Dict[str, Any]:
    """프롬프트 컨텍스트 준비"""
    # 이전 질문 섹션
    passed_section = ""
    if req.passed_questions:
        questions = req.passed_questions[-MAX_HISTORY_QUESTIONS:]
        passed_section = f"\n\n[이전 질문 목록]\n" + "\n".join(f"- {q}" for q in questions)

    # RAG 검색 섹션
    retrieved_section = ""
    if VECTOR_DB_AVAILABLE and question_rag_retriever:
        try:
            rag_results = question_rag_retriever(req.selected_question, req.keyword or "")
            retrieved_questions = [r["question"] for r in rag_results["results"]]
            if retrieved_questions:
                retrieved_section = f"\n\n[유사한 기존 질문]\n" + "\n".join(f"- {q}" for q in retrieved_questions)
        except Exception as e:
            logger.warning(f"RAG 검색 실패: {e}")

    return {
        "selected_question": req.selected_question,
        "keyword": req.keyword or "",
        "passed_questions": passed_section,
        "retrieved_questions": retrieved_section,
        "num_questions": GENERATE_COUNT,
    }


async def generate_questions_with_fallback(req: FollowupRequest, context: Dict[str, Any], trace_id: str = None) -> List[str]:
    """질문 생성 (부족시 OpenAI로 보완)"""
    # 프롬프트 준비
    prompt_template = get_cached_prompt("followup_questions_generator")
    if not prompt_template:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="프롬프트 템플릿을 로드할 수 없습니다.",
        )
    
    prompt = prompt_template.compile(**context)

    # vLLM으로 질문 생성
    try:
        raw_response = await call_llm(prompt, trace_id=trace_id)
        questions = parse_questions(raw_response)[:GENERATE_COUNT]
        
        # 질문 개수가 부족하면 OpenAI로 보완
        if len(questions) < GENERATE_COUNT:
            remaining_count = GENERATE_COUNT - len(questions)
            logger.info(f"질문 부족으로 OpenAI API 사용: {len(questions)}/{GENERATE_COUNT}")
            
            # OpenAI용 프롬프트
            api_context = {
                "selected_question": req.selected_question,
                "keyword": req.keyword or "",
                "passed_questions": context["passed_questions"],
                "ungenerated_questions_num": remaining_count,
            }
            
            api_prompt_template = get_cached_prompt("followup_questions_generator_api")
            if api_prompt_template:
                api_prompt = api_prompt_template.compile(**api_context)
                raw_response_api = await call_openai_api(api_prompt, trace_id=trace_id)
                additional_questions = parse_questions(raw_response_api)
                
                # 중복 제거 후 합치기
                unique_questions = [q for q in additional_questions if q not in questions]
                questions.extend(unique_questions)
        
        return questions[:GENERATE_COUNT]
        
    except Exception as e:
        logger.error(f"질문 생성 실패: {e}")
        raise


@router.post("/followup-questions", response_model=FollowupResponse)
async def generate_followup(req: FollowupRequest) -> FollowupResponse:
    """꼬리 질문 생성"""
    # 모델 상태 확인
    from app.main import is_model_available
    if not is_model_available("question_generator"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="질문 생성 모델이 사용할 수 없습니다.",
        )

    # 입력 검증
    if not req.selected_question or not req.selected_question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="메인 질문은 필수입니다.",
        )
    if not req.interview_id or not req.interview_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="interview_id는 필수입니다.",
        )

    logger.info(f"질문 생성 요청: {req.interview_id}")

    # Langfuse 추적
    trace_id = f"followup_{req.interview_id}_{uuid.uuid4().hex}"
    trace = None
    if langfuse:
        trace = langfuse.trace(
            id=trace_id,
            name="followup_generation",
            input={
                "interview_id": req.interview_id,
                "selected_question": req.selected_question,
                "keyword": req.keyword,
            },
        )

    start_time = time.time()
    
    try:
        # 컨텍스트 준비 및 질문 생성
        context = prepare_context(req)
        questions = await generate_questions_with_fallback(req, context, trace_id)
        
        execution_time = time.time() - start_time
        
        # 추적 완료
        if trace:
            trace.update(
                output={"followup_questions": questions},
                metadata={"execution_time": execution_time},
            )
        
        logger.info(f"질문 생성 완료: {req.interview_id}, {len(questions)}개, {execution_time:.2f}초")
        
        return FollowupResponse(
            message="followup_questions_generated",
            interview_id=req.interview_id,
            followup_questions=questions,
        )

    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = f"질문 생성 실패: {str(e)}"
        logger.error(f"{error_msg} ({req.interview_id})")
        
        if trace:
            trace.update(
                error={"message": str(e)},
                metadata={"execution_time": execution_time},
            )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg,
        )