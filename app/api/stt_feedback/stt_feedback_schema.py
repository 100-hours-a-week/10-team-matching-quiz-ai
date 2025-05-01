from pydantic import BaseModel, HttpUrl, Field
from enum import Enum
from typing import Dict

class FeedbackStatus(str, Enum):
    success = "success"
    failed = "failed"
    pending = "pending"

class STTSubmitRequest(BaseModel):
    task_id: str = Field(..., description="작업 ID")
    audio_gcs_uri: HttpUrl
    callback_url: HttpUrl

class STTFeedbackRequest(BaseModel):
    task_id: str
    audio_url: HttpUrl
    feedback: Dict[str, str]
    status: FeedbackStatus