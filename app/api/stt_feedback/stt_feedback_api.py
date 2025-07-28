from fastapi import APIRouter, HTTPException, Request
from fastapi import APIRouter, HTTPException, Request
from app.api.stt_feedback.stt_feedback_model import FeedbackResponse
from app.api.stt_feedback.service.feedback_pipline import run_feedback_pipeline
from app.api.stt_feedback.stt_feedback_schema import VoiceFeedbackRequest

import logging

logger = logging.getLogger("stt")

router = APIRouter(
    tags=["stt-feedback"]
)

# POST를 제외한 요청 시 
@router.api_route("/generate", methods=["GET", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def block_non_post_generate(request: Request):
    raise HTTPException(
        status_code=405,
        detail=f"Method {request.method} not allowed. POST 만"
    )


# def format_feedback(feedback_dict: dict) -> str:
#     if not feedback_dict:
#         return ""
#     
#     # 점수 추출
#     score = feedback_dict.get('overall_score', 0)
#     
#     # 상세 분석 추출
#     detailed_analysis = feedback_dict.get('detailed_analysis', '')
#     
#     # 좋은 점과 개선점을 하나로 통합
#     good_points = feedback_dict.get('good_points', '')
#     areas_for_improvement = feedback_dict.get('areas_for_improvement', '')
#     
#     # 통합된 피드백 생성
#     integrated_feedback = f"{score}점\n\n{detailed_analysis}"
#     
#     # 좋은 점과 개선점이 있는 경우에만 추가
#     if good_points and areas_for_improvement:
#         integrated_feedback += f"\n\n잘한 점: {good_points}\n\n개선할 점: {areas_for_improvement}"
#     elif good_points:
#         integrated_feedback += f"\n\n잘한 점: {good_points}"
#     elif areas_for_improvement:
#         integrated_feedback += f"\n\n개선할 점: {areas_for_improvement}"
#     
#     return integrated_feedback


# 피드백 생성
def process_feedback_request(request: VoiceFeedbackRequest) -> dict:
    """API 로직을 독립적인 함수로 분리"""
    logger.info(f"[API 요청] audio file 링크(S3): {request.recording_url}")
    
    for idx, q in enumerate(request.question_lists):
        logger.info(f"[API 요청] 질문: {idx+1}: {q.question} (start: {q.start_time}, end: {q.end_time})")

    if not request.question_lists:
        raise ValueError("question_lists가 비어 있습니다.")

    # 전체 처리: STT + 피드백 + 모범답안 생성
    result = run_feedback_pipeline(
        recording_url=request.recording_url,
        question_lists=request.question_lists
    )

    # 변환 적용: feedback dict → string (주석 처리됨)
    # for item in result.feedbackLists:
    #     item.feedback = format_feedback(item.feedback)

    # pipeline 수행 후 결과 확인
    for idx, item in enumerate(result.feedbackLists):
        logger.info(f"[Feedback Result][{idx+1}] Segment ID: {item.segment_id}")
        logger.info(f"[Feedback Result][{idx+1}] 질문: {item.question}")
        logger.info(f"[Feedback Result][{idx+1}] 질문별 모범답안: {item.model_answer}")
        logger.info(f"[Feedback Result][{idx+1}] 질문별 피드백 점수: {item.feedback.get('overall_score', 0)}점")
        logger.info(f"[Feedback Result][{idx+1}] 질문별 피드백 상세분석: {item.feedback.get('detailed_analysis', '')[:100]}...")
        
    response = result.model_dump()
    response['feedback_lists'] = [
        {
            "segment_id": item["segment_id"],
            "model_answer": item["model_answer"],
            "feedback": {
                "overall_score": item["feedback"].get("overall_score", 0),
                "detailed_analysis": item["feedback"].get("detailed_analysis", ""),
                "good_points": item["feedback"].get("good_points", ""),
                "areas_for_improvement": item["feedback"].get("areas_for_improvement", "")
            }
        }
        for item in response['feedbackLists']
    ]

    return response

# 피드백 생성
@router.post("/generate")
async def receive_feedback(request: VoiceFeedbackRequest):
    try:
        return process_feedback_request(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"API 처리 중 오류: {e}")
        raise HTTPException(status_code=500, detail="내부 서버 오류")