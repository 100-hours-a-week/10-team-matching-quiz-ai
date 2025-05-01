from app.stt.schema import STTSubmitRequest, STTFeedbackRequest
import logging

logger = logging.getLogger("stt")

async def process_stt_task(payload: STTSubmitRequest):
    logger.info(f"[STT] 처리 요청 수신: {payload.task_id}")
    # TODO: STT Task Queue 등록 등 비즈니스 로직

async def save_stt_feedback(payload: STTFeedbackRequest):
    logger.info(f"[STT] 피드백 저장 완료: {payload.task_id}")
    # TODO: DB 저장, 후처리 로직
