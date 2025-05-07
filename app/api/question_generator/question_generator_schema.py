from typing import List, Optional
from pydantic import BaseModel


class FollowupRequest(BaseModel):
    interview_id: int
    selected_question: str
    keyword: Optional[str] = None
    passed_questions: List[str] = []


class FollowupResponse(BaseModel):
    message: str
    interview_id: int
    followup_questions: List[str]
