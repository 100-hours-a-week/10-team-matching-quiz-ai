from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, HttpUrl

# 요청 스키마 (Request)

# 질문 별 상세 내용 : 시작(초), 종료(초), 질문별 Interview id, 질문 내용
class QuestionItem(BaseModel):
    start_time: int 
    end_time: int 
    interview_id: str 
    question: str 

# audio 음성(S3), 질문 리스트
class VoiceFeedbackRequest(BaseModel):
    recording_url: HttpUrl = Field(..., description="S3에 저장된 전체 음성 URL")
    questionLists: List[QuestionItem]


# 응답 스키마 (Response)
class StandardResponse(BaseModel):
    message: str
    data: Optional[Any] = None


# 400 - 요청 형식 오류
class InvalidRequestResponse(BaseModel):
    message: str = "invalid_request"
    data: Dict[str, str]  # 예: { "reason": "questionLists가 비어 있음" }

# 401 - 토큰 만료
class TokenExpiredResponse(BaseModel):
    message: str = "token_expired"
    data: Optional[None] = None

# 409 - 중복 제출
class AlreadySubmittedResponse(BaseModel):
    message: str = "already_submit"
    data: Optional[None] = None

# 500 - 서버 오류
class InternalServerErrorResponse(BaseModel):
    message: str = "internal_server_error"
    data: Optional[None] = None