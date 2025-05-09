import logging
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api import router
from app.api.question_generator.question_generator_model import initialize_llm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()
app.include_router(router)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = await request.body()
    logger.warning(f"[422 Error] Validation failed: {exc.errors()} | Body: {body.decode()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()}
    )

@app.on_event("startup")
async def startup_event():
    initialize_llm()
