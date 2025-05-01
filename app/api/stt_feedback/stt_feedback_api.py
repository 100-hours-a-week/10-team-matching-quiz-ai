from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import Dict
from http import httpx
import asyncio


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

# mock 처리용 함수들 (실제 서비스에서는 구현 필요)
def run_stt_engine(audio_uri: str) -> str:
    if "fail-stt" in audio_uri:
        raise RuntimeError("STT engine error")
    return "안녕하세요. 저는 ~ ... (300자)"

def generate_feedback(transcription: str) -> Dict[str, str]:
    if "fail-llm" in transcription:
        raise RuntimeError("LLM generation error")
    return {"transcription": transcription}

# === POST /stt/submit ===
@router.post("/submit", status_code=202)
async def handle_stt_submit(payload: STTSubmitRequest):
    # 1. 필드 검증 (추가적인 manual validation)
    if not payload.task_id or not payload.audio_gcs_uri or not payload.callback_url:
        raise HTTPException(status_code=400, detail="Invalid request payload")

    # 2. STT 처리
    try:
        transcription = run_stt_engine(str(payload.audio_gcs_uri))
    except RuntimeError as e:
        if "STT engine" in str(e):
            raise HTTPException(status_code=500, detail="STT engine error")
        raise HTTPException(status_code=500, detail="Failed to access AWS bucket audio file")

    # 3. 피드백 생성
    try:
        feedback = generate_feedback(transcription)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail="LLM generation error")

     # 4. Webhook 콜백 전송 (비동기 HTTP 처리)
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                payload.callback_url,
                json={
                    "task_id": payload.task_id,
                    "audio_url": str(payload.audio_gcs_uri),
                    "feedback": feedback,
                    "status": "completed"
                }
            )
        if response.status_code >= 400:
            raise Exception("Callback failed")
    except Exception:
        raise HTTPException(status_code=502, detail="Callback server error")

    return {"message": "STT processing started", "task_id": payload.task_id}

# === POST /stt/feedback ===
@router.post("/feedback", status_code=204)
async def handle_stt_feedback(payload: STTFeedbackRequest):
    # 실제 서비스에서는 이 데이터를 DB에 기록하거나 로그 남길 수 있음
    print(f"[Webhook] 피드백 수신 완료: {payload.task_id}")
    return None