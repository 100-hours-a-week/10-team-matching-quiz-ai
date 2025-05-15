from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api import router
from app.api.question_generator.question_generator_model import (
    initialize_llm,
    llm as global_llm_engine,
)
from vector_db.utils import get_model
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("애플리케이션 시작 - 초기화 시작...")

    logger.info("LLM 초기화 시도 중...")
    try:
        initialize_llm()
        if global_llm_engine:
            logger.info("LLM 초기화 성공.")
        else:
            logger.error(
                "LLM 초기화 후 global_llm_engine이 None입니다. 초기화 실패 가능성이 있습니다."
            )
    except Exception as e:
        logger.error(f"LLM 초기화 중 심각한 오류 발생: {e}", exc_info=True)

    logger.info("Vector DB 관련 모델 초기화 시도 중 (get_model)...")
    try:
        get_model()
        logger.info("Vector DB 관련 모델 (get_model) 초기화 성공.")
    except Exception as e:
        logger.error(
            f"Vector DB 관련 모델 (get_model) 초기화 중 오류 발생: {e}", exc_info=True
        )

    logger.info("모든 초기화 단계 완료. 애플리케이션 준비 완료.")
    yield
    logger.info("애플리케이션 종료 - 리소스 정리 시작...")

    if global_llm_engine and hasattr(global_llm_engine, "shutdown_background_loop"):
        try:
            logger.info("AsyncLLMEngine 백그라운드 루프 종료 시도...")
            global_llm_engine.shutdown_background_loop()
            logger.info("AsyncLLMEngine 백그라운드 루프가 성공적으로 종료되었습니다.")
        except Exception as e:
            logger.error(
                f"AsyncLLMEngine 백그라운드 루프 종료 중 오류 발생: {e}", exc_info=True
            )
    else:
        logger.info(
            "AsyncLLMEngine이 초기화되지 않았거나 shutdown_background_loop 메소드가 없어 종료를 건너뜁니다."
        )

    logger.info("애플리케이션 종료 - 리소스 정리 완료.")


app = FastAPI(lifespan=lifespan)
app.include_router(router)
