import os
from dotenv import load_dotenv
from app.config.model_config import TRANSFORMERS_CONFIG

load_dotenv()
# 오류시 주석 해제하여 디버깅
# load_dotenv(verbose=True)

# ModelConfig를 통해 설정 가져오기
QUIZ_MODEL_NAME = TRANSFORMERS_CONFIG["model_name"]
QUIZ_HF_TOKEN = TRANSFORMERS_CONFIG["hf_token"]

QUIZ_LANGFUSE_SECRET_KEY = os.getenv("QUIZ_LANGFUSE_SECRET_KEY")
QUIZ_LANGFUSE_PUBLIC_KEY = os.getenv("QUIZ_LANGFUSE_PUBLIC_KEY")
QUIZ_LANGFUSE_HOST = os.getenv("LANGFUSE_HOST")