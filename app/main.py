from fastapi import FastAPI, Request
from app.api import router  # __init__.py에서 통합된 router
from app.api.question_generator.question_generator_model import initialize_llm
# from vector_db.init_data import init_vector_store_from_csv
import logging

app = FastAPI()
app.include_router(router)


@app.on_event("startup")
async def startup_event():
    initialize_llm()
