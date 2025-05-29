from fastapi import FastAPI
from app.api.quiz_generator.quiz_generator_api import router as quiz_router

app = FastAPI()
app.include_router(quiz_router, prefix="/quiz", tags=["quiz"])
