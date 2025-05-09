from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from app.api import router  # __init__.py에서 통합된 router
from app.api.question_generator.question_generator_model import initialize_llm
import logging

app = FastAPI()
app.include_router(router)

# logger 객체 생성
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = await request.body()
    logger.warning(
        f"[422 Error] Validation failed: {exc.errors()} | Body: {body.decode()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()}
    )


@app.on_event("startup")
async def startup_event():
    initialize_llm()
