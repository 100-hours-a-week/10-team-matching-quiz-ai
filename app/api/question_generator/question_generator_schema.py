from typing import List, Optional
from pydantic import BaseModel


class FollowupRequest(BaseModel):
    interview_id: str
    selected_question: str
    keyword: Optional[str] = None
    passed_questions: List[str] = []


class FollowupResponse(BaseModel):
    message: str
    interview_id: str
    followup_questions: List[str]
