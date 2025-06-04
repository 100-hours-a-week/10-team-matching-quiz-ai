import logging
import tempfile
import openai
from pydub import AudioSegment
import os

logger = logging.getLogger("stt")
openai.api_key = os.getenv("WISPER_API_KEY")

# STT Whisper X 모델 서빙 예정
def transcribe_whisperx(segment: AudioSegment) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        segment.export(tmp.name, format="mp3")
        logger.info(f"[STT] WhisperX 전사 시작: {tmp.name}")
        with open(tmp.name, "rb") as f:
            result = openai.Audio.transcribe("whisper-1", f)
            transcript = result["text"]
            logger.info(f"[STT] 전사 결과: {transcript[:50]}...")
            return transcript