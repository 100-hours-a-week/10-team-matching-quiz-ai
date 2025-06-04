# from fastapi import FastAPI
# from contextlib import asynccontextmanager
# from app.api import router
# from app.config.model_config import ModelConfig
# import logging
# import os

# import sys
# logger_sys = logging.getLogger("sys_path_check")
# logger_sys.info(f"Current sys.path: {sys.path}")

# logging.basicConfig(
#     level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# )
# logger = logging.getLogger(__name__)

# # 환경에 따른 자동 모델 선택
# ENVIRONMENT = ModelConfig.get_environment()
# ENABLED_MODELS = ModelConfig.get_enabled_models()
# MEMORY_LIMIT = ModelConfig.get_model_memory_limit()

# logger.info(f"감지된 환경: {ENVIRONMENT}")
# logger.info(f"활성화된 모델들: {ENABLED_MODELS}")
# logger.info(f"메모리 제한: {MEMORY_LIMIT}")

# # 글로벌 모델 저장소
# GLOBAL_MODELS = {
#     "question_generator": None,
#     "quiz_generator": None,
# }

# VECTOR_DB_AVAILABLE = False
# try:
#     from app.vector_db.utils import get_embedding_model, get_keyword_model
#     VECTOR_DB_AVAILABLE = True
#     logger.info("Vector DB 모듈이 성공적으로 로드되었습니다.")
# except ImportError:
#     logger.warning("Vector DB 모듈을 찾을 수 없습니다.")


# def is_gcp_environment() -> bool:
#     """GCP 환경인지 확인"""
#     return ENVIRONMENT.startswith("gcp-")


# def initialize_question_generator():
#     """Question Generator (vLLM) 초기화"""
#     try:
#         # GCP 환경에서는 메모리 사용량 제한
#         if is_gcp_environment():
#             os.environ["VLLM_GPU_MEMORY_UTILIZATION"] = "0.7"
#             os.environ["VLLM_MAX_MODEL_LEN"] = "2048"
        
#         from app.api.question_generator.question_generator_model import (
#             initialize_llm,
#             llm as global_llm_engine,
#         )
        
#         logger.info("Question Generator (vLLM) 초기화를 시도합니다...")
#         initialize_llm()
#         if global_llm_engine:
#             GLOBAL_MODELS["question_generator"] = global_llm_engine
#             logger.info("Question Generator 초기화가 성공적으로 완료되었습니다.")
#             return True
#         else:
#             logger.error("Question Generator 초기화 후 'global_llm_engine'이 None입니다.")
#             return False
#     except Exception as e:
#         logger.error(f"Question Generator 초기화 중 오류 발생: {e}", exc_info=True)
#         return False


# def initialize_quiz_generator():
#     """Quiz Generator (Transformers) 초기화"""
#     try:
#         from app.api.quiz_generator.quiz_generator_model import initialize_quiz_model
        
#         logger.info("Quiz Generator 초기화를 시도합니다...")
        
#         # GCP 환경에서는 가벼운 설정
#         if is_gcp_environment():
#             # 메모리 효율적인 설정
#             os.environ["TORCH_DTYPE"] = "float16" if ENVIRONMENT == "gcp-gke" else "float32"
#             os.environ["LOW_CPU_MEM_USAGE"] = "true"
        
#         model, tokenizer = initialize_quiz_model()
#         GLOBAL_MODELS["quiz_generator"] = {
#             "model": model,
#             "tokenizer": tokenizer,
#             "type": "transformers"
#         }
#         logger.info("Quiz Generator 초기화가 성공적으로 완료되었습니다.")
#         return True
        
#     except Exception as e:
#         logger.error(f"Quiz Generator 초기화 중 오류 발생: {e}", exc_info=True)
#         return False


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     logger.info(f"애플리케이션 라이프사이클 시작 (환경: {ENVIRONMENT})")

#     # GCP 환경 최적화
#     if is_gcp_environment():
#         logger.info("GCP 환경에서 최적화된 설정을 적용합니다.")
        
#         # CPU 코어 수에 따른 워커 수 조정
#         import multiprocessing
#         cpu_count = multiprocessing.cpu_count()
#         logger.info(f"사용 가능한 CPU 코어: {cpu_count}")

#     # 선택적 모델 초기화
#     initialization_success = False
    
#     for model_name in ENABLED_MODELS:
#         model_name = model_name.strip()
#         logger.info(f"{model_name} 모델 초기화를 시도합니다...")
        
#         try:
#             if model_name == "question_generator":
#                 success = initialize_question_generator()
#             elif model_name == "quiz_generator":
#                 success = initialize_quiz_generator()
#             else:
#                 logger.warning(f"알 수 없는 모델 이름: {model_name}")
#                 continue
                
#             if success:
#                 initialization_success = True
#                 logger.info(f"{model_name} 초기화 성공")
#             else:
#                 logger.warning(f"{model_name} 초기화 실패")
                
#         except Exception as e:
#             logger.error(f"{model_name} 초기화 중 예외 발생: {e}")

#     # 최소 하나의 모델이라도 초기화되어야 함
#     if not initialization_success:
#         logger.error("어떤 모델도 초기화되지 않았습니다. 폴백 모드로 전환합니다.")
#         # 폴백: quiz_generator 강제 초기화 시도
#         try:
#             initialize_quiz_generator()
#         except Exception as e:
#             logger.critical(f"폴백 모델 초기화도 실패했습니다: {e}")

#     # Vector DB 초기화 (question_generator가 활성화된 경우에만)
#     if VECTOR_DB_AVAILABLE and "question_generator" in ENABLED_MODELS:
#         logger.info("Vector DB 관련 모델 초기화를 시도합니다...")
#         try:
#             get_embedding_model()
#             get_keyword_model()
#             logger.info("Vector DB 관련 모델 초기화가 성공적으로 완료되었습니다.")
#         except Exception as e:
#             logger.error(f"Vector DB 관련 모델 초기화 중 오류 발생: {e}", exc_info=True)

#     # 초기화된 모델 상태 로깅
#     available_models = [name for name, model in GLOBAL_MODELS.items() if model is not None]
#     logger.info(f"사용 가능한 모델들: {available_models}")
#     logger.info("모든 초기화 단계가 완료되었습니다. 애플리케이션이 준비되었습니다.")
    
#     yield
    
#     # 정리 단계
#     logger.info("애플리케이션 라이프사이클 종료: 리소스 정리 시작...")

#     # Question Generator (vLLM) 정리
#     if GLOBAL_MODELS["question_generator"]:
#         try:
#             if hasattr(GLOBAL_MODELS["question_generator"], "shutdown_background_loop"):
#                 GLOBAL_MODELS["question_generator"].shutdown_background_loop()
#             logger.info("Question Generator 정리 완료")
#         except Exception as e:
#             logger.error(f"Question Generator 정리 중 오류: {e}")

#     # Quiz Generator 정리
#     if GLOBAL_MODELS["quiz_generator"]:
#         try:
#             import torch
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
#             elif torch.backends.mps.is_available():
#                 torch.mps.empty_cache()
#             logger.info("Quiz Generator 정리 완료")
#         except Exception as e:
#             logger.error(f"Quiz Generator 정리 중 오류: {e}")

#     logger.info("애플리케이션 리소스 정리가 완료되었습니다.")


# app = FastAPI(
#     title="Team Matching Quiz AI",
#     description=f"Environment: {ENVIRONMENT}, Models: {ENABLED_MODELS}",
#     lifespan=lifespan
# )
# app.include_router(router)


# # 모델 접근을 위한 유틸리티 함수들
# def get_model(model_name: str):
#     """특정 모델 반환"""
#     return GLOBAL_MODELS.get(model_name)

# def is_model_available(model_name: str) -> bool:
#     """모델 사용 가능 여부 확인"""
#     return GLOBAL_MODELS.get(model_name) is not None

# def get_available_models():
#     """사용 가능한 모델 목록 반환"""
#     return [name for name, model in GLOBAL_MODELS.items() if model is not None]

# # 헬스체크 엔드포인트
# @app.get("/health")
# async def health_check():
#     return {
#         "status": "healthy",
#         "environment": ENVIRONMENT,
#         "enabled_models": ENABLED_MODELS,
#         "available_models": get_available_models(),
#         "memory_limit": MEMORY_LIMIT
#     }


from fastapi import FastAPI
from app.api.quiz_generator.quiz_generator_api import router as quiz_router

app = FastAPI()
app.include_router(quiz_router, prefix="/quiz", tags=["quiz"])