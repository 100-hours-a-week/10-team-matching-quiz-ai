from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import Dict

router = APIRouter()

class STTSubmitRequest(BaseModel):
    task_id: str
    audio_gcs_uri: HttpUrl
    callback_url: HttpUrl

class STTFeedbackRequest(BaseModel):
    task_id: str
    audio_url: HttpUrl
    feedback: Dict[str, str]
    status: str

@router.post("/submit", status_code=202)
async def handle_stt_submit(payload: STTSubmitRequest):
    print(f"[STT] 요청 처리: {payload.task_id}")
    return {"message": "STT processing started", "task_id": payload.task_id}

@router.post("/feedback", status_code=204)
async def handle_stt_feedback(payload: STTFeedbackRequest):
    print(f"[Webhook] 피드백 전달 완료: {payload.task_id}")
    return None