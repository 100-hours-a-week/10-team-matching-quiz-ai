import whisperx
import torch
import logging

logger = logging.getLogger("stt")

device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8" # Recommended compute types

# 모델 선언
whisper_model = None
# alignment_model = None
# alignment_metadata = None

def load_whisperx_model(model_name: str = "base"): # Added model_name parameter, default to "base"
    global whisper_model
    if whisper_model is None: # Load only if not already loaded
        try:
            logger.info(f"[WhisperX] 모델 '{model_name}' 로딩 시작 (device: {device}, compute_type: {compute_type})")
            # Correctly load the model
            whisper_model = whisperx.load_model(model_name, device, compute_type=compute_type)
            logger.info(f"[WhisperX] 모델 '{model_name}' 로딩 완료")

            """
            Alignment 모델 로딩은 미사용 주석 처리

            logger.info("[WhisperX] Alignment 모델 로딩 시작 (ko)")
            alignment_model, alignment_metadata = whisperx.load_align_model("ko", device)
            logger.info("[WhisperX] Alignment 모델 로딩 완료")
            """
        except Exception as e:
            logger.error(f"[WhisperX] 모델 로딩 실패: {e}")
            raise
    return whisper_model # Return the loaded model

def get_whisperx_model():
    global whisper_model
    if whisper_model is None:
        # Attempt to load if not already loaded, or raise an error/warning
        logger.warning("[WhisperX] get_whisperx_model() 호출되었으나 모델이 로드되지 않았습니다. load_whisperx_model()을 먼저 호출해야 합니다.")
        # Alternatively, automatically load it:
        # load_whisperx_model()
    return whisper_model