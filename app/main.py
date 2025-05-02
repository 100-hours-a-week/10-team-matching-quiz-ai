from fastapi import FastAPI
from app.api import router  # __init__.py에서 통합된 router

app = FastAPI()
app.include_router(router)