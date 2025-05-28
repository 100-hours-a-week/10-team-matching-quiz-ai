from pydantic import BaseModel

class FeedbackResponse(BaseModel):
    transcript: str
    feedback: str