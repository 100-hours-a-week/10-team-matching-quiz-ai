from fastapi import APIRouter, HTTPException, Response
from app.api.quiz_generator.quiz_generator_schema import (
    FollowupRequest,
    FollowupResponse,
)
from app import rabbitmq_producer
from app.config import rabbitmq_config
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/generate-quiz", status_code=202)
async def generate_quiz_api(req: FollowupRequest):
    logger.info(f"퀴즈 생성 요청 수신 (비동기 처리): interview_id={req.interview_id}")

    # 모델 사용 가능 여부 체크 (import inside function to avoid circular import)
    from app.main import is_model_available

    if not is_model_available("quiz_generator"):
        logger.error("quiz_generator 모델이 사용 불가능합니다.")
        raise HTTPException(
            status_code=503,
            detail="퀴즈 생성 모델이 현재 사용할 수 없습니다. 잠시 후 다시 시도해주세요.",
        )

    message_body = req.model_dump()  # Using Pydantic v2 model_dump(), or req.dict() for Pydantic v1

    try:
        success = await rabbitmq_producer.publish_message(
            routing_key=rabbitmq_config.ROUTING_KEY_QUIZ_GENERATOR,
            message_body=message_body,
        )
        if success:
            logger.info(f"퀴즈 생성 작업 메시지 발행 성공: interview_id={req.interview_id}")
            # Return a 202 Accepted response
            return {"message": "퀴즈 생성 작업이 요청되었습니다.", "interview_id": req.interview_id}
        else:
            logger.error(f"퀴즈 생성 작업 메시지 발행 실패: interview_id={req.interview_id}")
            raise HTTPException(status_code=500, detail="퀴즈 생성 요청 처리에 실패했습니다 (메시지 발행 실패).")
    except Exception as e:
        logger.error(f"퀴즈 생성 작업 메시지 발행 중 예외 발생: {e}, interview_id={req.interview_id}")
        raise HTTPException(status_code=500, detail="퀴즈 생성 요청 처리 중 내부 오류가 발생했습니다.")
