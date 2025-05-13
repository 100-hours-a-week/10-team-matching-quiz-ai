from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from app.api import router  # __init__.py에서 통합된 router
from app.api.question_generator.question_generator_model import initialize_llm, llm as global_llm_engine
# from vector_db.init_data import init_vector_store_from_csv
import logging

# 로거 설정 (이미 설정되어 있다면 이 부분은 생략 가능)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("애플리케이션 시작 - LLM 초기화 시작...")
    try:
        initialize_llm()  # initialize_llm은 동기 함수입니다.
        if global_llm_engine:
            logger.info("LLM 초기화 성공.")
        else:
            # initialize_llm 내부에서 예외가 발생하지 않고 llm 객체가 None일 수 있는 경우
            logger.error(
                "LLM 초기화 후 global_llm_engine이 None입니다. 초기화 실패 가능성이 있습니다.")
    except Exception as e:
        logger.error(f"LLM 초기화 중 심각한 오류 발생: {e}")
        # 필요에 따라 애플리케이션을 시작하지 않도록 처리할 수 있습니다.
        # raise # 주석 해제 시 애플리케이션 시작 중단

    yield

    logger.info("애플리케이션 종료 - 리소스 정리 (필요시)...")
    # 필요한 경우 LLM 엔진 종료 로직 추가
    # 예: if global_llm_engine and hasattr(global_llm_engine, 'shutdown'):
    #         await global_llm_engine.shutdown() # vLLM 엔진에 shutdown 메소드가 있다면

app = FastAPI(lifespan=lifespan)
app.include_router(router)
