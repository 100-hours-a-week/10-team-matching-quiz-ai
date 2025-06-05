import os
from dotenv import load_dotenv
from app.config.model_config import ModelConfig

load_dotenv()

# ModelConfig를 통해 설정 가져오기
transformers_config = ModelConfig.get_transformers_config()

QUIZ_MODEL_NAME = transformers_config["model_name"]
QUIZ_HF_TOKEN = transformers_config["hf_token"]

QUIZ_LANGFUSE_SECRET_KEY = os.getenv("QUIZ_LANGFUSE_SECRET_KEY")
QUIZ_LANGFUSE_PUBLIC_KEY = os.getenv("QUIZ_LANGFUSE_PUBLIC_KEY")
QUIZ_LANGFUSE_HOST = os.getenv("LANGFUSE_HOST")
