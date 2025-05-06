import os
import httpx
import logging
from typing import List, Optional, Dict, Any
import asyncio

# 모듈별 로거 설정
logger = logging.getLogger(__name__)

# 환경 변수 설정
BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "")
SKIP_BACKEND_PUSH = os.getenv("SKIP_BACKEND_PUSH", "true").lower() == "true"
REQUEST_TIMEOUT = float(os.getenv("BACKEND_REQUEST_TIMEOUT", "5.0"))
MAX_RETRIES = int(os.getenv("BACKEND_MAX_RETRIES", "2"))


async def push_to_backend(interview_id: int, questions: List[str],
                          timeout: float = REQUEST_TIMEOUT) -> bool:
    """
    생성된 꼬리 질문을 백엔드 서버로 전송하는 함수

    Parameters:
        interview_id (int): 질문이 생성된 면접 ID
        questions (List[str]): 생성된 꼬리 질문 문자열 목록
        timeout (float, optional): 요청 타임아웃 (초 단위), 기본값은 환경변수에서 가져옴

    Returns:
        bool: 전송 성공 시 True, 실패 또는 스킵 시 False

    Notes:
        - SKIP_BACKEND_PUSH 환경변수가 "true"로 설정되어 있거나 
          BACKEND_BASE_URL이 설정되지 않은 경우 실제 전송을 스킵
        - 비동기 HTTP 클라이언트(httpx)를 사용하여 POST 요청을 전송
        - 최대 MAX_RETRIES 횟수만큼 재시도 (기본값: 2회)
    """
    if SKIP_BACKEND_PUSH or not BACKEND_BASE_URL:
        logger.info(
            f"[push_to_backend] Skipped (test mode): interview={interview_id}, qs={questions}")
        return False

    url = f"{BACKEND_BASE_URL}/interview/{interview_id}/question/create"
    payload = {"followup_questions": questions}

    for attempt in range(MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, json=payload, timeout=timeout)
                resp.raise_for_status()
                logger.info(
                    f"[push_to_backend] Successfully pushed questions to BE: interview={interview_id}")
                return True
        except httpx.HTTPStatusError as e:
            # HTTP 응답 코드 에러 (4xx, 5xx)
            logger.error(
                f"[push_to_backend] HTTP error: {e.response.status_code} - {e.response.text} "
                f"for interview_id={interview_id}, attempt {attempt+1}/{MAX_RETRIES+1}"
            )
            if attempt < MAX_RETRIES:
                await asyncio.sleep(1 * (attempt + 1))  # 지수 백오프
            else:
                break
        except (httpx.RequestError, asyncio.TimeoutError) as e:
            # 연결 관련 에러 (네트워크 문제, 타임아웃 등)
            logger.error(
                f"[push_to_backend] Network error: {str(e)} "
                f"for interview_id={interview_id}, attempt {attempt+1}/{MAX_RETRIES+1}"
            )
            if attempt < MAX_RETRIES:
                await asyncio.sleep(1 * (attempt + 1))  # 지수 백오프
            else:
                break
        except Exception as e:
            # 기타 예상치 못한 예외
            logger.error(
                f"[push_to_backend] Unexpected error: {str(e)} for interview_id={interview_id}")
            break

    return False
