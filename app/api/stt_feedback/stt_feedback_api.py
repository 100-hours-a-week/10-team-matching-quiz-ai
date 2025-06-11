from fastapi import APIRouter, HTTPException
from app.api.stt_feedback.stt_feedback_model import FeedbackResponse
from app.api.stt_feedback.service.feedback_pipline import run_feedback_pipeline
from app.api.stt_feedback.stt_feedback_schema import VoiceFeedbackRequest

import logging

logger = logging.getLogger("stt")

router = APIRouter()

# 피드백 생성
@router.post("/generate", response_model=FeedbackResponse)
async def receive_feedback(request: VoiceFeedbackRequest):
    
    logger.info(f"[API 요청] audio file 링크(S3): {request.recording_url}")
    
    for idx, q in enumerate(request.question_lists):
        logger.info(f"[API 요청] 질문: {idx+1}: {q.question} (start: {q.start_time}, end: {q.end_time})")


    if not request.question_lists:
        raise HTTPException(
            status_code=400,
            detail="question_lists가 비어 있습니다."
        )

    # 전체 처리: STT + 피드백 + 모범답안 생성
    result = run_feedback_pipeline(
        recording_url=request.recording_url,
        question_lists=request.question_lists
    )

    # pipeline 수행 후 결과 확인
    for idx, item in enumerate(result.feedbackLists):
        logger.info(f"[Feedback Result][{idx+1}] Segment ID: {item.segment_id}")
        logger.info(f"[Feedback Result][{idx+1}] 질문: {item.question}")
        logger.info(f"[Feedback Result][{idx+1}] 질문별 모범답안: {item.model_answer}")
        logger.info(f"[Feedback Result][{idx+1}] 질문별 피드백: {item.feedback}")


    return result