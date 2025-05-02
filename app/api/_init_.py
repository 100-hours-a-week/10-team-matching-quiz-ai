from fastapi import APIRouter
from app.stt_feedback import api as stt_api

router = APIRouter()
router.include_router(stt_api.router, prefix="/stt", tags=["stt"])


# main.py
from fastapi import FastAPI
from app.api import router

app = FastAPI()
app.include_router(router)