from fastapi import FastAPI
from app.api.stt_feedback.stt_feedback_api import router as feedback_router
from app.api.stt_feedback.stt_model_loader import load_whisperx_model

app = FastAPI()
app.include_router(feedback_router, prefix="/feedback", tags=["stt-feedback"])


@app.on_event("startup")
async def startup_event():
    load_whisperx_model