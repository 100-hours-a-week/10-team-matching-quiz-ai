import logging
import tempfile
import requests
from pydub import AudioSegment

logger = logging.getLogger("stt")


def download_audio(url: str) -> str:
    """
    S3 URL에서 오디오 파일을 다운로드하여 임시 파일로 저장
    확장자는 URL에서 자동으로 추출하여 suffix에 붙여 저장하도록 작성
    """
    url = str(url)  # HttpUrl -> str 변환

    logger.info(f"[STT] 음성 다운로드 시작: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()

        # URL에서 확장자 추출 (예: ".webm", ".mp3")
        filename = url.split("?")[0].split("/")[-1]
        suffix = "." + filename.split(".")[-1]

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(response.content)
        tmp.flush()

        logger.info(f"[STT] 다운로드 완료 → {tmp.name}")
        return tmp.name
    except Exception as e:
        logger.error(f"[STT] 다운로드 실패: {e}")
        raise

def cut_audio(audio_path: str, start_time: int, end_time: int) -> AudioSegment:
    """
    전체 오디오 파일에서 지정된 초 단위(start_time ~ end_time)로 오디오를 잘라 반환
    WhisperX 모델에 전달하기 위한 오디오 클립 생성
    """
    logger.info(f"[STT] 오디오 자르기: {start_time}s ~ {end_time}s")
    try:
        audio = AudioSegment.from_file(audio_path)
        return audio[start_time * 1000:end_time * 1000]
    except Exception as e:
        logger.error(f"[STT] 오디오 자르기 실패: {e}")
        raise