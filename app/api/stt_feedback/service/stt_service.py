import whisperx
import tempfile
import logging
from pydub import AudioSegment
from app.api.stt_feedback.stt_model_loader import WhisperXModel  # 전역 모델 import
"""
from app.api.stt_feedback.stt_model_loader import (
    WhisperXModel, alignment_model, alignment_metadata
)

aligned_result = whisperx.align(result["segments"], alignment_model, alignment_metadata, tmp.name, device)
"""

logger = logging.getLogger("stt")

def transcribe_whisperx(segment: AudioSegment) -> str:
    try:
        # 오디오를 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            segment.export(tmp.name, format="mp3")
            logger.info(f"[STT] WhisperX 전사 시작: {tmp.name}")

            # Whisper X audio 로딩
            audio = whisperx.load_audio(tmp.name)

            # WhisperX 3.3.0 기준: detect_speech 제거 → transcribe 직접 호출
            result = WhisperXModel.model.transcribe(audio)

            logger.info(f"[STT] WhisperX 반환 타입: {type(result)}")
            
            # segments 리스트에서 text 추출
            segments = result['segments']
            text = " ".join([seg['text'] for seg in segments])

            # (혹시 리스트라면 바로 리스트 구조 출력)
            if isinstance(segments, list):
                logger.info(f"[STT] WhisperX 반환 결과 (샘플): {segments[:1]}")
            else:
                logger.info(f"[STT] WhisperX 반환결과: {segments}")

            # # Alignment 처리 (현재 미사용)
            # model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
            # aligned_result = whisperx.align(result["segments"], model_a, metadata, tmp.name, device)

            # logger.info(f"[STT] VAD 전사 완료: {text[:50]}...")  # 앞 50자만 로그
            logger.info(f"[STT] VAD 전사 전체 결과:\n{text}")
            return text

    except Exception as e:
        logger.error(f"[STT] WhisperX 전사 실패: {e}")
        raise