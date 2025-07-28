from typing import List
from pydantic import BaseModel

# 인터프리터 언어 특징으로 인해 먼저 FeedbackItem 먼저 선언
from typing import Dict, Any

class FeedbackItem(BaseModel):
    segment_id: str
    question: str
    model_answer: str
    feedback: Dict[str, Any]  # 4가지 피드백 항목을 포함하는 딕셔너리
    
# 질문별로 질문, 모범답안, AI 피드백 제공
class FeedbackResponse(BaseModel):
    feedbackLists: List[FeedbackItem]