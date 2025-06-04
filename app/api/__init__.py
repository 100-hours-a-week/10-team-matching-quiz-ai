from fastapi import APIRouter
from app.api.question_generator.question_generator_api import router as generate_router
from app.api.quiz_generator.quiz_generator_api import router as quiz_router

router = APIRouter()
router.include_router(generate_router, prefix="/interview", tags=["question-generator"])
router.include_router(quiz_router, prefix="/quiz", tags=["quiz"])

