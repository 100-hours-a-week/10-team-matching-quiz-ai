from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.config.model_config import ENVIRONMENT, ENABLED_MODELS
import logging
from datetime import datetime
from app.api.question_generator.question_generator_api import router as generate_router

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

logger.info(f"감지된 환경: {ENVIRONMENT}")
logger.info(f"활성화된 모델들: {ENABLED_MODELS}")


class ModelManager:
    """간소화된 모델 관리자"""

    def __init__(self):
        self._question_generator_ready = False
        self._vllm_client = None

    def initialize_models(self) -> bool:
        """활성화된 모델들 초기화"""
        if "question_generator" not in ENABLED_MODELS:
            return True

        try:
            from app.api.question_generator.question_generator_model import initialize_llm, get_llm_engine

            logger.info("question_generator (vLLM API) 초기화 시도...")
            
            if initialize_llm():
                self._vllm_client = get_llm_engine()
                if self._vllm_client:
                    self._question_generator_ready = True
                    logger.info("question_generator API 클라이언트 초기화 완료")
                    return True

            logger.error("question_generator 초기화 실패")
            return False

        except Exception as e:
            logger.error(f"question_generator 초기화 오류: {e}")
            return False

    def is_model_available(self, model_name: str) -> bool:
        """모델 사용 가능 여부 확인"""
        if model_name == "question_generator":
            return self._question_generator_ready
        return False

    def get_available_models(self) -> list:
        """사용 가능한 모델 목록"""
        available = []
        if self._question_generator_ready:
            available.append("question_generator")
        return available

    def cleanup(self):
        """리소스 정리"""
        if self._vllm_client:
            logger.info("question_generator API 클라이언트 정리 완료")
            self._vllm_client = None
            self._question_generator_ready = False


# 전역 모델 매니저
model_manager = ModelManager()


def initialize_vector_db():
    """벡터 데이터베이스 초기화"""
    try:
        from app.vector_db.init_data import init_all_vector_stores
        init_all_vector_stores()
        logger.info("벡터 데이터베이스 초기화 완료")
        return True
    except Exception as e:
        logger.error(f"벡터 데이터베이스 초기화 실패: {e}")
        return False


async def get_system_status():
    """시스템 상태 확인 (health check용)"""
    available_models = model_manager.get_available_models()
    
    # vLLM API 상태 확인
    vllm_api_status = False
    try:
        from app.api.question_generator.question_generator_model import check_vllm_health
        vllm_api_status = await check_vllm_health()
    except Exception as e:
        logger.error(f"vLLM API 상태 확인 실패: {e}")

    # 벡터 데이터베이스 상태 확인
    vector_db_status = {}
    try:
        from app.vector_db.chroma_client import follow_up_collection, quiz_collection
        vector_db_status = {
            "follow_up_questions": follow_up_collection.count() > 0,
            "quiz_data": quiz_collection.count() > 0,
            "follow_up_count": follow_up_collection.count(),
            "quiz_count": quiz_collection.count(),
        }
    except Exception as e:
        logger.error(f"벡터 데이터베이스 상태 확인 실패: {e}")
        vector_db_status = {"error": str(e)}

    # 시스템 상태 결정
    enabled_count = len(ENABLED_MODELS)
    available_count = len(available_models)
    
    if available_count == 0:
        system_status = "unhealthy"
    elif available_count < enabled_count:
        system_status = "degraded"
    else:
        system_status = "healthy"

    return {
        "status": system_status,
        "timestamp": datetime.now().isoformat(),
        "environment": ENVIRONMENT,
        "enabled_models": ENABLED_MODELS,
        "available_models": available_models,
        "vllm_api_status": vllm_api_status,
        "vector_db_status": vector_db_status,
        "system_info": {
            "total_enabled": enabled_count,
            "total_available": available_count,
            "availability_ratio": f"{available_count}/{enabled_count}",
        },
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 라이프사이클 관리"""
    logger.info(f"애플리케이션 시작 (환경: {ENVIRONMENT})")

    # 벡터 데이터베이스 초기화
    logger.info("벡터 데이터베이스 초기화 시작...")
    initialize_vector_db()

    # 모델 초기화
    logger.info("모델 초기화 시작...")
    initialization_success = model_manager.initialize_models()

    if initialization_success:
        logger.info("모델 초기화 성공")
    else:
        logger.warning("일부 모델 초기화 실패")

    available_models = model_manager.get_available_models()
    logger.info(f"사용 가능한 모델들: {available_models}")

    yield

    logger.info("애플리케이션 종료: 리소스 정리 중...")
    model_manager.cleanup()
    logger.info("리소스 정리 완료")


# FastAPI 앱 생성
app = FastAPI(
    title="Team Matching Quiz AI",
    description=f"AI 기반 팀 매칭 퀴즈 시스템\n환경: {ENVIRONMENT}\n활성화된 모델: {', '.join(ENABLED_MODELS)}",
    version="1.0.0",
    lifespan=lifespan,
)


# 유틸리티 함수들 (API에서 사용)
def is_model_available(model_name: str) -> bool:
    """모델 사용 가능 여부 확인"""
    return model_manager.is_model_available(model_name)


def get_available_models():
    """사용 가능한 모델 목록"""
    return model_manager.get_available_models()


# 라우터 등록
app.include_router(generate_router, prefix="/interview", tags=["question-generator"])


@app.get("/health")
async def health_check():
    """시스템 상태 확인"""
    return await get_system_status()