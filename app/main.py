from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api import router
from app.api.question_generator.question_generator_model import (
    initialize_llm,
    llm as global_llm_engine,
)
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
    
async def initialize_llm_engine() -> None:
    """LLM 엔진을 초기화"""
    logger.info("LLM 초기화를 시도합니다...")
    try:
        initialize_llm()
        if global_llm_engine:
            logger.info("LLM 초기화가 성공적으로 완료되었습니다.")
        else:
            logger.error(
                "LLM 초기화 후 'global_llm_engine'이 None입니다. "
            )
    except Exception as e:
        logger.error(f"LLM 초기화 중 심각한 오류 발생: {e}", exc_info=True)

async def initialize_vector_db() -> None:
    """VectorDB 관련 모델을 초기화""" 
    try:
        from app.vector_db.utils import get_embedding_model, get_keyword_model
        logger.info("Vector DB 관련 모델 초기화를 시도합니다...")
        try:
            get_embedding_model()
            get_keyword_model()
            logger.info("Vector DB 관련 모델 초기화가 성공적으로 완료되었습니다.")
        except Exception as e:
            logger.error(f"Vector DB 관련 모델 초기화 중 오류 발생: {e}", exc_info=True)
    except ImportError:
        logger.info("Vector DB 모듈이 로드되지 않아, 관련 모델 초기화를 건너뜁니다.")

async def cleanup_llm_engine() -> None:
    """LLM 엔진 리소스를 정리합니다."""
    if global_llm_engine and hasattr(global_llm_engine, "shutdown_background_loop"):
        try:
            logger.info("AsyncLLMEngine 백그라운드 루프 종료를 시도합니다...")
            global_llm_engine.shutdown_background_loop()
            logger.info("AsyncLLMEngine 백그라운드 루프가 성공적으로 종료되었습니다.")
        except Exception as e:
            logger.error(f"AsyncLLMEngine 백그라운드 루프 종료 중 오류 발생: {e}", exc_info=True)
    else:
        logger.info(
            "AsyncLLMEngine이 초기화되지 않았거나 'shutdown_background_loop' 메소드가 없어 "
            "종료 처리를 건너뜁니다."
        )

@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 라이프사이클을 관리"""
    logger.info("애플리케이션 라이프사이클 시작: 초기화 진행...")
    
    # 모듈 초기화
    await initialize_llm_engine()
    
    await initialize_vector_db()
    
    logger.info("모든 초기화 단계가 완료되었습니다. 애플리케이션이 준비되었습니다.")
    
    yield
    
    logger.info("애플리케이션 라이프사이클 종료: 리소스 정리 시작...")
    await cleanup_llm_engine()
    logger.info("애플리케이션 리소스 정리가 완료되었습니다.")

app = FastAPI(
    title="Team Matching Quiz AI",
    description="팀 매칭을 위한 퀴즈 AI API",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(router)