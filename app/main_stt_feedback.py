from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api.stt_feedback.stt_feedback_api import router as feedback_router
from app.api.stt_feedback.stt_model_loader import WhisperXModel
from app.core.rabbitmq import RabbitMQConnection
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 라이프사이클 관리"""
    logger.info("STT Feedback 애플리케이션 시작")

    # WhisperX 모델 초기화
    logger.info("WhisperX 모델 초기화 시작...")
    try:
        WhisperXModel.ensure_loaded()
        logger.info("WhisperX 모델 초기화 완료")
    except Exception as e:
        logger.error(f"WhisperX 모델 초기화 실패: {e}")

    # RabbitMQ 연결 초기화
    logger.info("RabbitMQ 연결 초기화 시작...")
    try:
        await RabbitMQConnection.connect()
        logger.info("RabbitMQ 연결 초기화 완료")
    except Exception as e:
        logger.error(f"RabbitMQ 연결 초기화 실패: {e}")

    yield

    logger.info("애플리케이션 종료: 리소스 정리 중...")
    
    # RabbitMQ 연결 종료
    logger.info("RabbitMQ 연결 종료 중...")
    try:
        await RabbitMQConnection.close()
        logger.info("RabbitMQ 연결 종료 완료")
    except Exception as e:
        logger.error(f"RabbitMQ 연결 종료 실패: {e}")

    logger.info("리소스 정리 완료")


app = FastAPI(
    title="STT Feedback Service",
    description="Speech-to-Text 피드백 서비스",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(feedback_router, prefix="/feedback", tags=["stt-feedback"])


@app.get("/health")
async def health_check():
    """STT 서비스 상태 확인"""
    try:
        whisper_status = WhisperXModel.model is not None
        
        return {
            "status": "healthy" if whisper_status else "degraded",
            "whisper_model": whisper_status,
            "service": "stt-feedback"
        }
    except Exception as e:
        logger.error(f"Health check 실패: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "service": "stt-feedback"
        }
