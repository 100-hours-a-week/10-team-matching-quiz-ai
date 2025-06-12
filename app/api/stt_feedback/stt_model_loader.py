import whisperx
import torch
import logging

logger = logging.getLogger("stt")

device = "cuda" if torch.cuda.is_available() else "cpu"

# WhisperX 모델 관리 클래스 선언
class WhisperXModel:
    model = None
    # alignment_model = None
    # alignment_metadata = None

    @classmethod
    def load_model(cls):
        try:
            logger.info("[WhisperX] 모델 로딩 시작")
            cls.model = whisperx.load_model("small", device=device, compute_type="int8")
            # cls.model = whisperx.load_model("large-v2", device="cuda", compute_type="auto")
            logger.info("[WhisperX] 모델 로딩 완료")

            """
            Alignment 모델 로딩은 미사용 주석 처리

            logger.info("[WhisperX] Alignment 모델 로딩 시작 (ko)")
            cls.alignment_model, cls.alignment_metadata = whisperx.load_align_model("ko", device)
            logger.info("[WhisperX] Alignment 모델 로딩 완료"
            """
        except Exception as e:
            logger.error(f"[WhisperX] 모델 로딩 실패: {e}")
            raise