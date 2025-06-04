from typing import List, Literal, Optional
from pydantic import BaseModel


class FollowupRequest(BaseModel):
    interview_id: str
    question_history_list: List[str]


class QuizItem(BaseModel):
    difficulty: Literal["상", "중", "하"]
    question: str
    options: List[str]
    answer_index: int
    explanation: str

class QuizData(BaseModel):
    user_id: str
    questions: List[QuizItem]

class FollowupResponse(BaseModel):
    message: str
    data: QuizData