from dotenv import load_dotenv # .env 파일 추가
from fastapi import FastAPI
from app.api.stt_feedback.stt_feedback_api import router as feedback_router
from app.api.stt_feedback.stt_model_loader import WhisperXModel
import logging

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)

app = FastAPI()
app.include_router(feedback_router, prefix="/feedback", tags=["stt-feedback"])


@app.on_event("startup")
async def startup_event():
    WhisperXModel.ensure_loaded()