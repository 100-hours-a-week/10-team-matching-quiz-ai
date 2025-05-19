from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api import router
from app.api.question_generator.question_generator_model import (
    initialize_llm,
    llm as global_llm_engine,
)
import logging
import os
import sys

# 현재 파일(main.py)의 디렉터리 (app/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 프로젝트 루트 디렉터리 (app/의 부모 디렉터리)
project_root = os.path.dirname(current_dir)

# 프로젝트 루트를 sys.path에 추가
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

VECTOR_DB_AVAILABLE = False
try:
    from vector_db.utils import get_embedding_model, get_keyword_model

    VECTOR_DB_AVAILABLE = True
    logger.info(
        "Vector DB 모듈(vector_db.utils.get_embedding/keyword_model)이 성공적으로 로드되었습니다."
    )
except ImportError:
    logger.warning(
        "Vector DB 모듈(vector_db.utils.get_embedding/keyword_model)을 찾을 수 없습니다. "
        "Vector DB 관련 기능이 비활성화될 수 있습니다."
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("애플리케이션 라이프사이클 시작: 초기화 진행...")

    # 1. LLM 초기화
    logger.info("LLM 초기화를 시도합니다...")
    try:
        initialize_llm()
        if global_llm_engine:
            logger.info("LLM 초기화가 성공적으로 완료되었습니다.")
        else:
            logger.error(
                "LLM 초기화 후 'global_llm_engine'이 None입니다. "
                "초기화 과정에 문제가 있을 수 있습니다."
            )

    except Exception as e:
        logger.error(f"LLM 초기화 중 심각한 오류 발생: {e}", exc_info=True)

    if VECTOR_DB_AVAILABLE:
        logger.info("Vector DB 관련 모델(get_model) 초기화를 시도합니다...")
        try:
            get_embedding_model()  # 동기 함수로 가정, 비동기라면 await get_embedding_model()
            get_keyword_model()
            logger.info(
                "Vector DB 관련 모델(get_embedding/keyword_model) 초기화가 성공적으로 완료되었습니다."
            )
        except Exception as e:
            logger.error(
                f"Vector DB 관련 모델(get_model) 초기화 중 오류 발생: {e}",
                exc_info=True,
            )

    else:
        logger.info("Vector DB 모듈이 로드되지 않아, 관련 모델 초기화를 건너뜁니다.")

    logger.info("모든 초기화 단계가 완료되었습니다. 애플리케이션이 준비되었습니다.")
    yield
    logger.info("애플리케이션 라이프사이클 종료: 리소스 정리 시작...")

    if global_llm_engine and hasattr(global_llm_engine, "shutdown_background_loop"):
        try:
            logger.info("AsyncLLMEngine 백그라운드 루프 종료를 시도합니다...")
            global_llm_engine.shutdown_background_loop()
            logger.info("AsyncLLMEngine 백그라운드 루프가 성공적으로 종료되었습니다.")
        except Exception as e:
            logger.error(
                f"AsyncLLMEngine 백그라운드 루프 종료 중 오류 발생: {e}", exc_info=True
            )
    else:
        logger.info(
            "AsyncLLMEngine이 초기화되지 않았거나 'shutdown_background_loop' 메소드가 없어 "
            "종료 처리를 건너뜁니다."
        )

    logger.info("애플리케이션 리소스 정리가 완료되었습니다.")


app = FastAPI(lifespan=lifespan)
app.include_router(router)
