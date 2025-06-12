from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api.stt_feedback.stt_feedback_api import router as feedback_router
from app.api.stt_feedback.stt_model_loader import load_whisperx_model, get_whisperx_model
from app import rabbitmq_producer
import logging

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """STT Feedback 애플리케이션 라이프사이클 관리"""
    logger.info("STT Feedback 애플리케이션 시작...")

    # RabbitMQ 연결 초기화
    logger.info("STT Feedback: RabbitMQ 연결 초기화 시작...")
    try:
        await rabbitmq_producer.get_rabbitmq_connection()
        await rabbitmq_producer.get_rabbitmq_channel()
        logger.info("STT Feedback: RabbitMQ 연결 초기화 완료.")
    except Exception as e:
        logger.error(f"STT Feedback: RabbitMQ 연결 초기화 실패: {e}")
        # 필요시 애플리케이션 중단 처리

    # WhisperX 모델 로드
    logger.info("STT Feedback: WhisperX 모델 로드 시작...")
    try:
        load_whisperx_model() # 기존 모델 로더 호출
        if get_whisperx_model() is not None: # 모델이 성공적으로 로드되었는지 확인 (예시)
            logger.info("STT Feedback: WhisperX 모델 로드 완료.")
        else:
            logger.error("STT Feedback: WhisperX 모델 로드 실패 (모델이 None임).")
    except Exception as e:
        logger.error(f"STT Feedback: WhisperX 모델 로드 중 오류 발생: {e}")

    yield

    logger.info("STT Feedback 애플리케이션 종료: 리소스 정리 중...")
    # WhisperX 모델 정리 (필요한 경우)
    # 예: if hasattr(get_whisperx_model(), 'cleanup'): get_whisperx_model().cleanup()

    # RabbitMQ 연결 종료
    logger.info("STT Feedback: RabbitMQ 연결 종료 중...")
    try:
        await rabbitmq_producer.close_rabbitmq_connection()
        logger.info("STT Feedback: RabbitMQ 연결 종료 완료.")
    except Exception as e:
        logger.error(f"STT Feedback: RabbitMQ 연결 종료 실패: {e}")

    logger.info("STT Feedback: 리소스 정리 완료.")


app = FastAPI(
    title="STT Feedback Service",
    description="STT 결과를 위한 피드백 생성 서비스",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(feedback_router, prefix="/feedback", tags=["stt-feedback"])