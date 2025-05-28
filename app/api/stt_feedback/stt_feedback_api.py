from fastapi import APIRouter, UploadFile, File
from app.api.stt_feedback.stt_feedback_model import FeedbackResponse
from app.api.stt_feedback.stt_feedback_service import transcribe_audio, generate_feedback

router = APIRouter()

#TODO: file 가져오기 test 필요 (방법 추가 확인인)
@router.post("/feedback", response_model=FeedbackResponse)
async def feedback_from_audio(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    transcript = transcribe_audio(audio_bytes)
    feedback = generate_feedback(transcript)
    return FeedbackResponse(transcript=transcript, feedback=feedback)