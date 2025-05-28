from fastapi import APIRouter
from app.api.question_generator.question_generator_api import router as generate_router
from app.api.stt_feedback.stt_feedback_api import router as stt_feedback_router

router = APIRouter()
router.include_router(generate_router, prefix="/interview",
                      tags=["question-generator"])
router.include_router(stt_feedback_router, prefix="/stt",
                      tags=["stt-feedback"])