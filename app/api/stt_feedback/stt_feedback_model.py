from typing import List
from pydantic import BaseModel

# 인터프리터 언어에 특징으로 인해 먼저 FeedbackItem 먼저 선언
class FeedbackItem(BaseModel):
    question: str
    model_answer: str
    feedback: str
class FeedbackResponse(BaseModel):
    interview_id: str
    feedbackLists: List[FeedbackItem]