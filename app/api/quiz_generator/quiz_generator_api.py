from fastapi import APIRouter, HTTPException, Response
from app.api.quiz_generator.quiz_generator_schema import (
    FollowupRequest,
    FollowupResponse,
)
from app import rabbitmq_producer
from app.config import rabbitmq_config
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/generate-quiz", status_code=202)
async def generate_quiz_api(req: FollowupRequest):
    """
    퀴즈 생성 요청 (비동기 처리)
    Quiz Worker로 작업을 전송하고 즉시 응답합니다.
    """
    logger.info(f"퀴즈 생성 요청 수신 (비동기 처리): interview_id={req.interview_id}")
    logger.info(f"질문 히스토리 개수: {len(req.question_history_list)}")

    # 입력 검증
    if not req.question_history_list:
        raise HTTPException(
            status_code=400,
            detail="question_history_list가 비어 있습니다."
        )

    if len(req.question_history_list) < 1:
        raise HTTPException(
            status_code=400,
            detail="최소 1개 이상의 질문이 필요합니다."
        )

    # Quiz Worker 상태 확인 (RabbitMQ 연결로 대체)
    try:
        connection = await rabbitmq_producer.get_rabbitmq_connection()
        if not connection or connection.is_closed:
            logger.error("RabbitMQ 연결이 불가능합니다.")
            raise HTTPException(
                status_code=503,
                detail="퀴즈 생성 서비스가 현재 사용할 수 없습니다. 잠시 후 다시 시도해주세요.",
            )
    except Exception as e:
        logger.error(f"RabbitMQ 연결 확인 실패: {e}")
        raise HTTPException(
            status_code=503,
            detail="퀴즈 생성 서비스 연결에 실패했습니다.",
        )

    # Quiz Worker로 작업 전송
    message_body = req.model_dump()

    try:
        success = await rabbitmq_producer.publish_message(
            routing_key=rabbitmq_config.ROUTING_KEY_QUIZ_GENERATOR,
            message_body=message_body,
        )
        
        if success:
            logger.info(f"퀴즈 생성 작업 메시지 발행 성공: interview_id={req.interview_id}")
            return {
                "message": "퀴즈 생성 작업이 Quiz Worker로 전송되었습니다.",
                "interview_id": req.interview_id,
                "status": "queued",
                "processing_mode": "worker",
                "questions_count": len(req.question_history_list),
                "note": "처리 완료 시 RabbitMQ 응답 큐를 통해 결과가 전송됩니다."
            }
        else:
            logger.error(f"퀴즈 생성 작업 메시지 발행 실패: interview_id={req.interview_id}")
            raise HTTPException(
                status_code=500, 
                detail="퀴즈 생성 요청 처리에 실패했습니다 (메시지 발행 실패)."
            )
            
    except Exception as e:
        logger.error(f"퀴즈 생성 작업 메시지 발행 중 예외 발생: {e}, interview_id={req.interview_id}")
        raise HTTPException(
            status_code=500, 
            detail="퀴즈 생성 요청 처리 중 내부 오류가 발생했습니다."
        )


@router.get("/health")
async def quiz_service_health():
    """Quiz 서비스 상태 확인 (Worker 모드)"""
    try:
        connection = await rabbitmq_producer.get_rabbitmq_connection()
        rabbitmq_status = not (connection is None or connection.is_closed)
        
        return {
            "service": "quiz-generator",
            "mode": "worker",
            "status": "healthy" if rabbitmq_status else "degraded",
            "rabbitmq_connection": rabbitmq_status,
            "processing": "asynchronous",
            "note": "Quiz generation is handled by dedicated worker process",
            "capabilities": {
                "vector_db_retrieval": True,
                "batch_quiz_generation": True,
                "difficulty_levels": ["상", "중", "하"],
                "max_quiz_count": 10
            }
        }
    except Exception as e:
        logger.error(f"Quiz 서비스 상태 확인 실패: {e}")
        return {
            "service": "quiz-generator", 
            "mode": "worker",
            "status": "unhealthy",
            "error": str(e)
        }


@router.get("/status/{interview_id}")
async def get_quiz_status(interview_id: str):
    """
    특정 면접의 퀴즈 생성 상태 조회 (선택사항)
    실제 구현 시 Redis나 DB에서 상태 조회
    """
    try:
        # 실제로는 Redis나 DB에서 상태 조회
        # 현재는 RabbitMQ 큐 상태만 확인
        connection = await rabbitmq_producer.get_rabbitmq_connection()
        if not connection or connection.is_closed:
            return {
                "interview_id": interview_id,
                "status": "unknown",
                "message": "Worker 상태를 확인할 수 없습니다."
            }
        
        return {
            "interview_id": interview_id,
            "status": "processing",  # 실제로는 DB/Redis에서 조회
            "message": "퀴즈 생성이 진행 중입니다.",
            "estimated_completion": "1-2분 예상"
        }
        
    except Exception as e:
        logger.error(f"퀴즈 상태 조회 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail="퀴즈 상태 조회 중 오류가 발생했습니다."
        )