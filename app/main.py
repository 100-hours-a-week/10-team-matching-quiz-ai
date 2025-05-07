from fastapi import FastAPI
from app.api import router  # __init__.py에서 통합된 router
from app.api.question_generator.question_generator_model import initialize_llm

app = FastAPI()
app.include_router(router)

@app.on_event("startup")
async def startup_event():
    initialize_llm()