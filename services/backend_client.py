import os
import httpx
import logging

BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "")

# 이 변수가 true면 BE로 푸시를 스킵 -> 우선 백엔드 서버에 연결되지 않은 상태에서 테스트하기 위해 skip을 함
SKIP_BACKEND_PUSH = os.getenv("SKIP_BACKEND_PUSH", "true").lower() == "true"


async def push_to_backend(interview_id: int, questions: list):
    """
    생성된 꼬리 질문을 백엔드 서버로 전송하는 함수 

    Parameters:
        interview_id (int): 질문이 생성된 면접 ID
        questions (list): 생성된 꼬리 질문 목록

    Returns:
        None

    Notes:
        - SKIP_BACKEND_PUSH 환경변수가 "true"로 설정되어 있거나 
          BACKEND_BASE_URL이 설정되지 않은 경우 실제 전송을 스킵
        - 비동기 HTTP 클라이언트(httpx)를 사용하여 POST 요청을 전송합니다. => 동시처리를 위해서 
        - 요청 실패 시 오류를 로깅하지만 예외는 발생시키지 않습니다.
    """
    if SKIP_BACKEND_PUSH or not BACKEND_BASE_URL:
        logging.info(
            f"[push_to_backend] Skipped (test mode): interview={interview_id}, qs={questions}")
        return

    url = f"{BACKEND_BASE_URL}/interview/{interview_id}/question/create"
    payload = {"followup_questions": questions}
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload, timeout=5.0)
            resp.raise_for_status()
    except Exception as e:
        logging.error(f"[push_to_backend] Failed to push to BE: {e}")
