from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, HttpUrl
from typing import Dict

router = APIRouter()

# 요청 본문 스키마
class STTSubmitRequest(BaseModel):
    task_id: str
    audio_gcs_uri: HttpUrl
    callback_url: HttpUrl

class STTFeedbackRequest(BaseModel):
    task_id: str
    audio_url: HttpUrl
    feedback: Dict[str, str]
    status: str


# STT 처리 요청 (BE → AI)
@router.post("/stt/submit", status_code=202)
async def handle_stt_submit(payload: STTSubmitRequest):
    # STT 및 피드백 처리 비동기 로직 (ex. task queue 등록 등)
    print(f"[STT] 처리 요청: {payload.task_id}")
    return {"message": "STT processing started", "task_id": payload.task_id}


# Webhook 응답 처리 (AI → BE)
@router.post("/stt/feedback", status_code=204)
async def handle_stt_feedback(payload: STTFeedbackRequest):
    # BE에 피드백 전달하는 Webhook 응답 처리
    print(f"[Webhook] 피드백 전송 완료: {payload.task_id}")
    return None  # 204 No Content