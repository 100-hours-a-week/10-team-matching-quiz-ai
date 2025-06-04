# from fastapi import FastAPI
# from contextlib import asynccontextmanager
# from app.api import router
# from app.config.model_config import ModelConfig
# import logging
# import os
# import asyncio
# import gc
# import threading
# from typing import Optional, Dict, Any
# from datetime import datetime, timedelta
# import time

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

# # 개선된 글로벌 모델 저장소 - 자동 언로딩 지원
# GLOBAL_MODELS = {
#     "question_generator": None,  # vLLM - 상시 유지
#     "quiz_generator": {
#         "instance": None,  # Transformers 모델
#         "status": "unloaded",  # unloaded, loading, loaded, unloading
#         "last_used": None,
#         "auto_unload_timer": None,
#         "lock": threading.Lock(),
#         "auto_unload_timeout": 3600,  # 1시간 (3600초)
#         "loading_start_time": None,
#         "total_loading_time": 0
#     },
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


# def get_memory_info():
#     """메모리 사용량 정보"""
#     try:
#         import torch
#         if torch.cuda.is_available():
#             allocated = torch.cuda.memory_allocated() / 1024**3
#             reserved = torch.cuda.memory_reserved() / 1024**3
#             total = torch.cuda.get_device_properties(0).total_memory / 1024**3
#             return {
#                 "allocated_gb": round(allocated, 2),
#                 "reserved_gb": round(reserved, 2),
#                 "total_gb": round(total, 2),
#                 "free_gb": round(total - reserved, 2),
#                 "utilization_percent": round(reserved / total * 100, 1)
#             }
#         elif torch.backends.mps.is_available():
#             return {"backend": "mps", "memory_monitoring": "limited"}
#     except Exception as e:
#         logger.warning(f"메모리 정보 조회 실패: {e}")
    
#     return {"backend": "cpu", "unlimited": True}


# async def cleanup_memory():
#     """메모리 정리"""
#     try:
#         import torch
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#             torch.cuda.synchronize()
#         elif torch.backends.mps.is_available():
#             torch.mps.empty_cache()
        
#         gc.collect()
#         logger.info("메모리 정리 완료")
        
#     except Exception as e:
#         logger.error(f"메모리 정리 중 오류: {e}")


# def initialize_question_generator():
#     """Question Generator (vLLM) 초기화 - 상시 유지"""
#     try:
#         # GCP 환경에서는 메모리 사용량 제한
#         if is_gcp_environment():
#             os.environ["VLLM_GPU_MEMORY_UTILIZATION"] = "0.5"  # Quiz Generator 고려
#             os.environ["VLLM_MAX_MODEL_LEN"] = "2048"
#         else:
#             os.environ["VLLM_GPU_MEMORY_UTILIZATION"] = "0.6"
        
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


# async def load_quiz_generator():
#     """Quiz Generator 온디맨드 로딩 (로딩 시간 측정 포함)"""
#     quiz_info = GLOBAL_MODELS["quiz_generator"]
    
#     with quiz_info["lock"]:
#         # 이미 로드된 경우
#         if quiz_info["status"] == "loaded" and quiz_info["instance"]:
#             quiz_info["last_used"] = datetime.now()
#             await schedule_auto_unload()
#             logger.info("Quiz Generator 이미 로드됨 (last_used 업데이트)")
#             return quiz_info["instance"]
        
#         # 로딩 중인 경우
#         if quiz_info["status"] == "loading":
#             logger.info("Quiz Generator 로딩 중... 대기")
#             return None
        
#         try:
#             quiz_info["status"] = "loading"
#             quiz_info["loading_start_time"] = time.time()
            
#             logger.info("Quiz Generator (Transformers) 온디맨드 로딩 시작...")
#             memory_before = get_memory_info()
#             logger.info(f"로딩 전 메모리: {memory_before}")
            
#             # 메모리 정리
#             await cleanup_memory()
            
#             from app.api.quiz_generator.quiz_generator_model import initialize_quiz_model
            
#             # GCP 환경에서는 메모리 효율적 설정
#             if is_gcp_environment():
#                 os.environ["TORCH_DTYPE"] = "float16"
#                 os.environ["LOW_CPU_MEM_USAGE"] = "true"
            
#             model, tokenizer = initialize_quiz_model()
            
#             if model and tokenizer:
#                 loading_time = time.time() - quiz_info["loading_start_time"]
#                 quiz_info["total_loading_time"] = loading_time
                
#                 quiz_info["instance"] = {
#                     "model": model,
#                     "tokenizer": tokenizer,
#                     "type": "transformers"
#                 }
#                 quiz_info["status"] = "loaded"
#                 quiz_info["last_used"] = datetime.now()
#                 quiz_info["loading_start_time"] = None
                
#                 memory_after = get_memory_info()
#                 logger.info(f"로딩 후 메모리: {memory_after}")
#                 logger.info(f"Quiz Generator 로딩 완료 (소요시간: {loading_time:.1f}초)")
                
#                 # 자동 언로딩 스케줄링
#                 await schedule_auto_unload()
                
#                 return quiz_info["instance"]
#             else:
#                 raise Exception("모델 또는 토크나이저 로딩 실패")
                
#         except Exception as e:
#             logger.error(f"Quiz Generator 로딩 실패: {e}", exc_info=True)
#             quiz_info["status"] = "unloaded"
#             quiz_info["instance"] = None
#             quiz_info["loading_start_time"] = None
            
#             await cleanup_memory()
#             return None


# async def unload_quiz_generator():
#     """Quiz Generator 언로딩"""
#     quiz_info = GLOBAL_MODELS["quiz_generator"]
    
#     with quiz_info["lock"]:
#         if quiz_info["status"] != "loaded" or not quiz_info["instance"]:
#             logger.info("Quiz Generator 이미 언로드됨")
#             return True
        
#         try:
#             logger.info("Quiz Generator 언로딩 시작...")
#             quiz_info["status"] = "unloading"
            
#             # 자동 언로딩 타이머 취소
#             if quiz_info["auto_unload_timer"]:
#                 quiz_info["auto_unload_timer"].cancel()
#                 quiz_info["auto_unload_timer"] = None
            
#             # 모델 참조 해제
#             quiz_info["instance"] = None
#             quiz_info["status"] = "unloaded"
#             quiz_info["last_used"] = None
            
#             # 메모리 정리
#             await cleanup_memory()
            
#             memory_after = get_memory_info()
#             logger.info(f"언로딩 후 메모리: {memory_after}")
#             logger.info("Quiz Generator 언로딩 완료")
#             return True
            
#         except Exception as e:
#             logger.error(f"Quiz Generator 언로딩 중 오류: {e}")
#             quiz_info["status"] = "unloaded"  # 강제 상태 변경
#             return False


# async def schedule_auto_unload():
#     """Quiz Generator 1시간 후 자동 언로딩 스케줄링"""
#     quiz_info = GLOBAL_MODELS["quiz_generator"]
    
#     # 기존 타이머 취소
#     if quiz_info["auto_unload_timer"]:
#         quiz_info["auto_unload_timer"].cancel()
    
#     async def auto_unload_task():
#         try:
#             timeout = quiz_info["auto_unload_timeout"]
#             await asyncio.sleep(timeout)
            
#             # 마지막 사용 시간 체크 (double-check)
#             if quiz_info["last_used"]:
#                 time_since_last_use = (datetime.now() - quiz_info["last_used"]).total_seconds()
#                 if time_since_last_use >= timeout:
#                     logger.info(f"Quiz Generator 자동 언로딩 실행 ({time_since_last_use/60:.1f}분 미사용)")
#                     await unload_quiz_generator()
#                 else:
#                     logger.info(f"Quiz Generator 자동 언로딩 취소 (최근 사용: {time_since_last_use/60:.1f}분 전)")
                    
#         except asyncio.CancelledError:
#             logger.debug("Quiz Generator 자동 언로딩 타이머 취소됨")
#         except Exception as e:
#             logger.error(f"자동 언로딩 중 오류: {e}")
    
#     quiz_info["auto_unload_timer"] = asyncio.create_task(auto_unload_task())
#     logger.info(f"Quiz Generator 자동 언로딩 예약됨 ({quiz_info['auto_unload_timeout']/60}분 후)")


# def initialize_quiz_generator():
#     """Quiz Generator 초기화 (startup용) - 실제로는 온디맨드 설정만"""
#     try:
#         # 실제 로딩은 하지 않고 설정만 준비
#         quiz_info = GLOBAL_MODELS["quiz_generator"]
#         quiz_info["status"] = "unloaded"
        
#         logger.info("Quiz Generator 온디맨드 모드로 설정됨 (1시간 자동 언로딩)")
#         return True
        
#     except Exception as e:
#         logger.error(f"Quiz Generator 설정 중 오류: {e}")
#         return False


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     logger.info(f"애플리케이션 라이프사이클 시작 (환경: {ENVIRONMENT})")

#     # GCP 환경 최적화
#     if is_gcp_environment():
#         logger.info("GCP 환경에서 최적화된 설정을 적용합니다.")
        
#         import multiprocessing
#         cpu_count = multiprocessing.cpu_count()
#         logger.info(f"사용 가능한 CPU 코어: {cpu_count}")

#     # 선택적 모델 초기화
#     initialization_success = False
    
#     # Question Generator 우선 초기화 (상시 유지)
#     if "question_generator" in ENABLED_MODELS:
#         try:
#             success = initialize_question_generator()
#             if success:
#                 initialization_success = True
#                 logger.info("Question Generator 초기화 성공")
#         except Exception as e:
#             logger.error(f"Question Generator 초기화 중 예외: {e}")

#     # Quiz Generator 온디맨드 설정
#     if "quiz_generator" in ENABLED_MODELS:
#         try:
#             success = initialize_quiz_generator()
#             if success:
#                 initialization_success = True
#                 logger.info("Quiz Generator 온디맨드 설정 완료")
#         except Exception as e:
#             logger.error(f"Quiz Generator 설정 중 예외: {e}")

#     # Vector DB 초기화
#     if VECTOR_DB_AVAILABLE and "question_generator" in ENABLED_MODELS:
#         logger.info("Vector DB 관련 모델 초기화를 시도합니다...")
#         try:
#             get_embedding_model()
#             get_keyword_model()
#             logger.info("Vector DB 관련 모델 초기화가 성공적으로 완료되었습니다.")
#         except Exception as e:
#             logger.error(f"Vector DB 관련 모델 초기화 중 오류 발생: {e}", exc_info=True)

#     # 초기화된 모델 상태 로깅
#     available_models = get_available_models()
#     logger.info(f"사용 가능한 모델들: {available_models}")
#     logger.info("모든 초기화 단계가 완료되었습니다. 애플리케이션이 준비되었습니다.")
    
#     yield
    
#     # 정리 단계
#     logger.info("애플리케이션 라이프사이클 종료: 리소스 정리 시작...")

#     # Quiz Generator 언로딩
#     await unload_quiz_generator()

#     # Question Generator (vLLM) 정리
#     if GLOBAL_MODELS["question_generator"]:
#         try:
#             if hasattr(GLOBAL_MODELS["question_generator"], "shutdown_background_loop"):
#                 GLOBAL_MODELS["question_generator"].shutdown_background_loop()
#             logger.info("Question Generator 정리 완료")
#         except Exception as e:
#             logger.error(f"Question Generator 정리 중 오류: {e}")

#     await cleanup_memory()
#     logger.info("애플리케이션 리소스 정리가 완료되었습니다.")


# app = FastAPI(
#     title="Team Matching Quiz AI - Auto Unload",
#     description=f"Quiz Generator 1시간 자동 언로딩 (환경: {ENVIRONMENT})",
#     lifespan=lifespan
# )
# app.include_router(router)


# # 개선된 모델 접근 함수들
# def get_question_generator():
#     """Question Generator 반환 (상시 로드됨)"""
#     return GLOBAL_MODELS.get("question_generator")


# async def get_quiz_generator():
#     """Quiz Generator 반환 (온디맨드 로딩, 1시간 자동 언로딩)"""
#     return await load_quiz_generator()


# async def release_quiz_generator():
#     """Quiz Generator 수동 해제"""
#     return await unload_quiz_generator()


# def get_model(model_name: str):
#     """특정 모델 반환 (동기 버전)"""
#     if model_name == "question_generator":
#         return GLOBAL_MODELS.get("question_generator")
#     elif model_name == "quiz_generator":
#         quiz_info = GLOBAL_MODELS["quiz_generator"]
#         return quiz_info["instance"] if quiz_info["status"] == "loaded" else None
#     return None


# def is_model_available(model_name: str) -> bool:
#     """모델 사용 가능 여부 확인"""
#     if model_name == "question_generator":
#         return GLOBAL_MODELS.get("question_generator") is not None
#     elif model_name == "quiz_generator":
#         return model_name in ENABLED_MODELS
#     return False


# def get_available_models():
#     """사용 가능한 모델 목록 반환"""
#     available = []
#     if GLOBAL_MODELS.get("question_generator"):
#         available.append("question_generator")
#     if "quiz_generator" in ENABLED_MODELS:
#         available.append("quiz_generator")
#     return available


# def get_model_status():
#     """모델 상태 상세 정보"""
#     qg = GLOBAL_MODELS.get("question_generator")
#     quiz_info = GLOBAL_MODELS["quiz_generator"]
    
#     # Quiz Generator 상태 계산
#     quiz_status = {
#         "status": quiz_info["status"],
#         "type": "transformers",
#         "persistent": False,
#         "auto_unload_timeout_minutes": quiz_info["auto_unload_timeout"] / 60,
#         "last_used": quiz_info["last_used"].isoformat() if quiz_info["last_used"] else None,
#         "loading_time_seconds": quiz_info["total_loading_time"],
#         "auto_unload_scheduled": quiz_info["auto_unload_timer"] is not None
#     }
    
#     # 마지막 사용으로부터 경과 시간 계산
#     if quiz_info["last_used"]:
#         elapsed = (datetime.now() - quiz_info["last_used"]).total_seconds()
#         quiz_status["minutes_since_last_use"] = round(elapsed / 60, 1)
#         quiz_status["minutes_until_auto_unload"] = max(0, round((quiz_info["auto_unload_timeout"] - elapsed) / 60, 1))
    
#     return {
#         "question_generator": {
#             "status": "loaded" if qg else "unloaded",
#             "type": "vllm",
#             "persistent": True,
#             "framework": "vLLM"
#         },
#         "quiz_generator": quiz_status
#     }


# # 헬스체크 엔드포인트
# @app.get("/health")
# async def health_check():
#     return {
#         "status": "healthy",
#         "environment": ENVIRONMENT,
#         "enabled_models": ENABLED_MODELS,
#         "available_models": get_available_models(),
#         "model_status": get_model_status(),
#         "memory_info": get_memory_info(),
#         "features": {
#             "quiz_generator_auto_unload": True,
#             "auto_unload_timeout_minutes": GLOBAL_MODELS["quiz_generator"]["auto_unload_timeout"] / 60
#         }
#     }


# # 관리 엔드포인트
# @app.post("/admin/quiz-generator/load")
# async def admin_load_quiz():
#     """Quiz Generator 수동 로딩"""
#     start_time = time.time()
#     result = await load_quiz_generator()
#     loading_time = time.time() - start_time
    
#     return {
#         "success": result is not None,
#         "loading_time_seconds": round(loading_time, 1),
#         "memory_info": get_memory_info(),
#         "model_status": get_model_status()
#     }


# @app.post("/admin/quiz-generator/unload")
# async def admin_unload_quiz():
#     """Quiz Generator 수동 언로딩"""
#     success = await unload_quiz_generator()
#     return {
#         "success": success,
#         "memory_info": get_memory_info(),
#         "model_status": get_model_status()
#     }


# @app.post("/admin/quiz-generator/extend-timer")
# async def admin_extend_timer():
#     """Quiz Generator 자동 언로딩 타이머 연장"""
#     quiz_info = GLOBAL_MODELS["quiz_generator"]
    
#     if quiz_info["status"] == "loaded":
#         quiz_info["last_used"] = datetime.now()
#         await schedule_auto_unload()
#         return {
#             "success": True,
#             "message": "타이머가 1시간 연장되었습니다",
#             "model_status": get_model_status()
#         }
#     else:
#         return {
#             "success": False,
#             "message": "Quiz Generator가 로드되지 않음",
#             "model_status": get_model_status()
#         }


# @app.get("/admin/models/status")
# async def admin_models_status():
#     """모델 상태 상세 조회"""
#     return {
#         "model_status": get_model_status(),
#         "memory_info": get_memory_info(),
#         "available_models": get_available_models(),
#         "quiz_generator_details": {
#             "auto_unload_timeout": GLOBAL_MODELS["quiz_generator"]["auto_unload_timeout"],
#             "status": GLOBAL_MODELS["quiz_generator"]["status"],
#             "timer_active": GLOBAL_MODELS["quiz_generator"]["auto_unload_timer"] is not None
#         }
#     }


# @app.post("/admin/memory/cleanup")
# async def admin_cleanup_memory():
#     """메모리 수동 정리"""
#     await cleanup_memory()
#     return {
#         "status": "cleaned",
#         "memory_info": get_memory_info()
#     }