from fastapi import APIRouter, HTTPException
from app.api.stt_feedback.stt_feedback_schema import VoiceFeedbackRequest
from app import rabbitmq_producer
from app.config import rabbitmq_config
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

# 피드백 생성
@router.post("/generate", status_code=202)
async def receive_feedback(request: VoiceFeedbackRequest):
    logger.info(f"STT 피드백 생성 요청 수신 (비동기 처리): interview_id={request.interview_id}")
    if not request.questionLists:
        logger.warning(f"STT 피드백 요청에 questionLists가 비어 있습니다: interview_id={request.interview_id}")
        raise HTTPException(
            status_code=400,
            detail="questionLists가 비어 있습니다."
        )

    message_body = request.model_dump() 

    try:
        success = await rabbitmq_producer.publish_message(
            routing_key=rabbitmq_config.ROUTING_KEY_STT_FEEDBACK,
            message_body=message_body
        )
        if success:
            logger.info(f"STT 피드백 작업 메시지 발행 성공: interview_id={request.interview_id}")
            return {"message": "STT 피드백 생성 작업이 요청되었습니다.", "interview_id": request.interview_id}
        else:
            logger.error(f"STT 피드백 작업 메시지 발행 실패: interview_id={request.interview_id}")
            raise HTTPException(status_code=500, detail="STT 피드백 요청 처리에 실패했습니다 (메시지 발행 실패).")
    except Exception as e:
        logger.error(f"STT 피드백 작업 메시지 발행 중 예외 발생: {e}, interview_id={request.interview_id}")
        raise HTTPException(status_code=500, detail="STT 피드백 요청 처리 중 내부 오류가 발생했습니다.")