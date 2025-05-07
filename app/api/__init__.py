from fastapi import APIRouter
from app.api.question_generator.question_generator_api import router as generate_router

router = APIRouter()
router.include_router(generate_router, prefix="/interview",
                      tags=["question-generator"])
