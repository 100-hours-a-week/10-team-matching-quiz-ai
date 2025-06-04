import logging
import tempfile
import requests
from pydub import AudioSegment

logger = logging.getLogger("stt")


def download_audio(url: str) -> str:
    logger.info(f"[STT] 음성 다운로드 시작: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp.write(response.content)
        tmp.flush()
        logger.info(f"[STT] 다운로드 완료 → {tmp.name}")
        return tmp.name
    except Exception as e:
        logger.error(f"[STT] 다운로드 실패: {e}")
        raise


def cut_audio(audio_path: str, start_sec: int, end_sec: int) -> AudioSegment:
    logger.info(f"[STT] 오디오 자르기: {start_sec}s ~ {end_sec}s")
    audio = AudioSegment.from_file(audio_path)
    return audio[start_sec * 1000:end_sec * 1000]