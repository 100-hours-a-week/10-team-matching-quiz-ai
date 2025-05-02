from fastapi import APIRouter, HTTPException
from schemas import FollowupRequest, FollowupResponse
from services.llm_service import call_llm, call_openai_api
from services.backend_client import push_to_backend
# from utils.prompt_builder import build_prompt
from utils.parser import parse_questions
import asyncio
from langfuse import Langfuse
import os
import logging
import uuid
from langfuse.decorators import langfuse_context, observe

router = APIRouter()
logger = logging.getLogger(__name__)

langfuse = Langfuse(
    secret_key=os.getenv('LANGFUSE_SECRET_KEY'),
    public_key=os.getenv('LANGFUSE_PUBLIC_KEY'),
    host=os.getenv('LANGFUSE_HOST')
)


async def safe_push(interview_id: int, questions: list[str]):
    """
    백엔드 전송 중 예외를 잡아서 로깅하는 안전 래퍼
    """
    try:
        await push_to_backend(interview_id, questions)
    except Exception as e:
        logger.error(f"백엔드 전송 실패 (interview_id={interview_id}): {e}")


@router.post("/interview/followup-questions", response_model=FollowupResponse)
async def generate_followup(req: FollowupRequest):
    """
    IT 면접을 위한 새로운 꼬리 질문을 생성하는 엔드포인트 핸들러입니다.
    """
    # 입력값 검증
    if not req.selected_question or not req.selected_question.strip():
        raise HTTPException(status_code=400, detail="메인 질문은 비워둘 수 없습니다.")
    if req.interview_id <= 0:
        raise HTTPException(status_code=400, detail="유효한 interview_id가 필요합니다.")

    generate_count = 4

    all_used_questions = req.passed_questions or []

    if len(all_used_questions) > 20:
        all_used_questions = all_used_questions[-20:]
    else:
        pass

    passed_section = ""
    if all_used_questions:
        joined = "\n".join(f"- {q}" for q in all_used_questions)
        passed_section = f"\n\n[이전 질문 목록]\n{joined}"
    else:
        []

    trace_id = f"followup_{req.interview_id}_{uuid.uuid4().hex}"

    context = {
        "selected_question": req.selected_question,
        "keyword": req.keyword or "",
        "passed_questions": passed_section,
        "num_questions": generate_count,
    }

    prompt_template = langfuse.get_prompt(
        "followup_questions_generator")
    compiled = prompt_template.compile(**context)
    prompt = compiled

    trace = langfuse.trace(id=trace_id, name="followup_generation_llm", input={
        "interview_id": req.interview_id,
        "selected_question": req.selected_question,
        "keyword": req.keyword,
        "passed_questions": passed_section,
    })
    try:
        llm_span = trace.span(name="llm_call")
        raw_response = await call_llm(prompt)
        llm_span.update(input={"prompt": prompt}, output={
                        "raw_response": raw_response})
        llm_span.end()

        parsing_span = trace.span(name="response_parsing")
        generated_questions = parse_questions(raw_response)[:4]
        parsing_span.update(input={"raw_response": raw_response}, output={
                            "parsed_questions": generated_questions})
        parsing_span.end()

        if len(generated_questions) < generate_count:
            remaining_questions_count = generate_count - \
                len(generated_questions)

            context_api = {
                "selected_question": req.selected_question,
                "keyword": req.keyword or "",
                "passed_questions": passed_section,
                "ungenerated_questions_num": remaining_questions_count
            }

            prompt_template_api = langfuse.get_prompt(
                "followup_questions_generator_api")
            compiled_api = prompt_template_api.compile(**context_api)
            prompt_api = compiled_api

            try:
                llm_span_api = trace.span(name="llm_call_additional")
                raw_response_api = await call_openai_api(prompt_api)
                llm_span_api.update(input={"prompt_api": prompt_api}, output={
                                    "raw_response_api": raw_response_api})
                llm_span_api.end()

                parsing_span_api = trace.span(
                    name="response_parsing_additional")
                additional_questions = parse_questions(raw_response_api)
                parsing_span_api.update(input={"raw_response_api": raw_response_api}, output={
                                        "parsed_questions": additional_questions})
                parsing_span_api.end()

                unique_additional_questions = [
                    q for q in additional_questions if q not in generated_questions]
                generated_questions.extend(unique_additional_questions)

                generated_questions = generated_questions[:generate_count]

            except Exception as e:
                trace.update(error={"message": f"추가 질문 생성 실패: {str(e)}"})
                logger.error(f"추가 질문 생성 실패: {e}")
                raise HTTPException(
                    status_code=500, detail=f"추가 질문 생성 실패: {e}")
        else:
            pass

    except Exception as e:
        trace.update(error={"message": str(e)})
        raise HTTPException(status_code=500, detail=f"처리 실패: {e}")

    trace.update(output={"followup_questions": generated_questions})

    asyncio.create_task(safe_push(req.interview_id, generated_questions))

    return FollowupResponse(
        message="followup_questions_generated",
        interview_id=req.interview_id,
        followup_questions=generated_questions
    )
