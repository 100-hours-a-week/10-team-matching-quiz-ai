import logging
import tempfile
import whisper
import google.generativeai as genai
import os

from app.api.stt_feedback.stt_feedback_schema import STTSubmitRequest, STTFeedbackRequest
from app.api.stt_feedback.stt_feedback_model import FeedbackResponse
from pydantic import HttpUrl
from typing import Tuple

logger = logging.getLogger("stt")

# Whisper 모델 초기화
whisper_model = whisper.load_model("large")

# Gemini API 키 설정 (환경변수에서 로드)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def transcribe_audio(audio_bytes: bytes) -> str:
    """오디오 바이트를 Whisper로 텍스트로 변환"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        result = whisper_model.transcribe(tmp.name, language="ko")
        return result["text"]

def generate_feedback(transcript: str) -> str:
    """전사된 텍스트를 기반으로 Gemini 모델을 통해 피드백 생성"""
    prompt = f"""
    너는 취업 면접 전문가야. 아래 면접자의 답변을 읽고 피드백을 작성해줘.

    면접자의 답변:
    \"\"\"{transcript}\"\"\"

    피드백:
    """

    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text.strip()

def run_feedback_pipeline(audio_bytes: bytes) -> FeedbackResponse:
    """오디오 입력을 받아 전사 및 피드백 생성까지 수행하는 파이프라인"""
    transcript = transcribe_audio(audio_bytes)
    feedback = generate_feedback(transcript)
    return FeedbackResponse(transcript=transcript, feedback=feedback)

async def process_stt_task(payload: STTSubmitRequest):
    logger.info(f"[STT] 처리 요청 수신: {payload.task_id}")
    # TODO: GCS에서 오디오 파일 가져오기 → run_feedback_pipeline 호출 → 결과 callback_url로 전송
    pass

async def save_stt_feedback(payload: STTFeedbackRequest):
    logger.info(f"[STT] 피드백 저장 완료: {payload.task_id}")
    # TODO: DB 저장, 후처리 로직
    pass
