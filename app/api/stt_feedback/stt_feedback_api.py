from fastapi import APIRouter, HTTPException
from app.api.stt_feedback.stt_feedback_model import FeedbackResponse
from app.api.stt_feedback.service.feedback_pipline import run_feedback_pipeline
from app.api.stt_feedback.stt_feedback_schema import VoiceFeedbackRequest

router = APIRouter()

# 피드백 생성
@router.post("/generate", response_model=FeedbackResponse)
async def receive_feedback(request: VoiceFeedbackRequest):
    if not request.question_lists:
        raise HTTPException(
            status_code=400,
            detail="question_lists가 비어 있습니다."
        )

    # 전체 처리: STT + 피드백 + 모범답안 생성
    result = run_feedback_pipeline(
        recording_url=request.recording_url,
        question_lists=[q.dict(by_alias=True) for q in request.questionLists]
    )

    return result