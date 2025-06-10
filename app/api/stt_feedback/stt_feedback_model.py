from typing import List
from pydantic import BaseModel

# 인터프리터 언어 특징으로 인해 먼저 FeedbackItem 먼저 선언
class FeedbackItem(BaseModel):
    question: str
    model_answer: str
    feedback: str

# 질문별로 질문, 모범답안, AI 피드백 제공
class FeedbackResponse(BaseModel):
    interview_id: str
    feedbackLists: List[FeedbackItem]