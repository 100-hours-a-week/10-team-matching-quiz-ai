from fastapi import APIRouter, HTTPException, Request
from app.api.stt_feedback.stt_feedback_model import FeedbackResponse
from app.api.stt_feedback.service.feedback_pipline import run_feedback_pipeline
from app.api.stt_feedback.stt_feedback_schema import VoiceFeedbackRequest

import logging

logger = logging.getLogger("stt")

router = APIRouter(
    prefix="/feedback",
    tags=["stt-feedback"]
)

# POST를 제외한 요청 시 
@router.api_route("/generate", methods=["GET", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def block_non_post_generate(request: Request):
    raise HTTPException(
        status_code=405,
        detail=f"Method {request.method} not allowed. POST 만"
    )


def format_feedback(feedback_dict: dict) -> str:
    if not feedback_dict:
        return ""
    return (
        f"{feedback_dict.get('overall_score', '')} 점\n\n"
        f"{feedback_dict.get('detailed_analysis', '')}\n\n"
        f"잘한 점: {feedback_dict.get('good_points', '')}\n\n"
        f"개선할 점: {feedback_dict.get('areas_for_improvement', '')}"
    )


# 피드백 생성
@router.post("/generate")
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

    # 변환 적용: feedback dict → string
    for item in result.feedbackLists:
        item.feedback = format_feedback(item.feedback)

    # pipeline 수행 후 결과 확인
    for idx, item in enumerate(result.feedbackLists):
        logger.info(f"[Feedback Result][{idx+1}] Segment ID: {item.segment_id}")
        logger.info(f"[Feedback Result][{idx+1}] 질문: {item.question}")
        logger.info(f"[Feedback Result][{idx+1}] 질문별 모범답안: {item.model_answer}")
        logger.info(f"[Feedback Result][{idx+1}] 질문별 피드백: {item.feedback}")
        
    response = result.model_dump()
    response['feedbackLists'] = [
        {
            "segment_id": item["segment_id"],
            "model_answer": item["model_answer"],
            "feedback": item["feedback"]
        }
        for item in response['feedbackLists']
    ]

    return response