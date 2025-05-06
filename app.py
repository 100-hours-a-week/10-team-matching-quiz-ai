from fastapi import FastAPI
from routers.followup_generate import router as generate_router

app = FastAPI()

app.include_router(generate_router)
