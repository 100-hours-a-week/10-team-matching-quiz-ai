from fastapi import APIRouter
from app.api.question_generator.question_generator_api import router as generate_router

router = APIRouter()
router.include_router(generate_router, prefix="/interview",
                      tags=["question-generator"])

try:
    from app.api.quiz_generator.quiz_generator_api import router as quiz_router
    router.include_router(quiz_router, prefix="/quiz", tags=["quiz"])
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning("Quiz 라우터를 찾을 수 없습니다. Quiz 기능이 비활성화됩니다.")