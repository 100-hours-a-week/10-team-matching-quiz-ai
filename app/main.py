from fastapi import FastAPI

# 각 기능 라우터 import
from app.api import stt_feedback, quiz_generator, interview_router

app = FastAPI()

# TODO: 라우터 등록 - 기능별로 라우터 추가해서 변경해주세요
app.include_router(quiz_generator.router)
app.include_router(interview_router.router)
app.include_router(stt_feedback.router, prefix="/stt")