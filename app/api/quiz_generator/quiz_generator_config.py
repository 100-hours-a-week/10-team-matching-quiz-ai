import os
from dotenv import load_dotenv

load_dotenv()

QUIZ_MODEL_NAME = os.getenv("QUIZ_MODEL_NAME")
QUIZ_LANGFUSE_SECRET_KEY = os.getenv("QUIZ_LANGFUSE_SECRET_KEY")
QUIZ_LANGFUSE_PUBLIC_KEY = os.getenv("QUIZ_LANGFUSE_PUBLIC_KEY")
QUIZ_LANGFUSE_HOST = os.getenv("LANGFUSE_HOST")
QUIZ_HF_TOKEN = os.getenv("QUIZ_HF_TOKEN")