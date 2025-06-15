from fastapi import APIRouter, HTTPException
from app.api.stt_feedback.stt_feedback_schema import VoiceFeedbackRequest
from app import rabbitmq_producer
from app.config import rabbitmq_config
import logging

logger = logging.getLogger("stt")

router = APIRouter()


@router.post("/generate", status_code=202)
async def receive_feedback(request: VoiceFeedbackRequest):
    """
    STT 피드백 생성 요청 (비동기 처리)
    Worker로 작업을 전송하고 즉시 응답합니다.
    """
    logger.info(f"[API 요청] STT 피드백 생성 요청 수신 (비동기 처리)")
    logger.info(f"[API 요청] audio file 링크(S3): {request.recording_url}")
    
    for idx, q in enumerate(request.question_lists):
        logger.info(f"[API 요청] 질문 {idx+1}: {q.question} (start: {q.start_time}, end: {q.end_time})")

    if not request.question_lists:
        raise HTTPException(
            status_code=400,
            detail="question_lists가 비어 있습니다."
        )

    # STT Worker 상태 확인 (RabbitMQ 연결로 대체)
    try:
        connection = await rabbitmq_producer.get_rabbitmq_connection()
        if not connection or connection.is_closed:
            logger.error("RabbitMQ 연결이 불가능합니다.")
            raise HTTPException(
                status_code=503,
                detail="STT 피드백 서비스가 현재 사용할 수 없습니다. 잠시 후 다시 시도해주세요.",
            )
    except Exception as e:
        logger.error(f"RabbitMQ 연결 확인 실패: {e}")
        raise HTTPException(
            status_code=503,
            detail="STT 피드백 서비스 연결에 실패했습니다.",
        )

    # Worker로 작업 전송
    message_body = request.model_dump()
    
    try:
        success = await rabbitmq_producer.publish_message(
            routing_key=rabbitmq_config.STT_FEEDBACK_ROUTING_KEY,
            message_body=message_body,
        )
        
        if success:
            logger.info(f"STT 피드백 작업 메시지 발행 성공: {request.recording_url}")
            return {
                "message": "STT 피드백 생성 작업이 STT Worker로 전송되었습니다.",
                "recording_url": str(request.recording_url),
                "status": "queued",
                "processing_mode": "worker",
                "questions_count": len(request.question_lists),
                "note": "처리 완료 시 RabbitMQ 응답 큐를 통해 결과가 전송됩니다."
            }
        else:
            logger.error(f"STT 피드백 작업 메시지 발행 실패: {request.recording_url}")
            raise HTTPException(
                status_code=500, 
                detail="STT 피드백 요청 처리에 실패했습니다 (메시지 발행 실패)."
            )
            
    except Exception as e:
        logger.error(f"STT 피드백 작업 메시지 발행 중 예외 발생: {e}")
        raise HTTPException(
            status_code=500, 
            detail="STT 피드백 요청 처리 중 내부 오류가 발생했습니다."
        )


@router.get("/health")
async def stt_service_health():
    """STT 서비스 상태 확인 (Worker 모드)"""
    try:
        connection = await rabbitmq_producer.get_rabbitmq_connection()
        rabbitmq_status = not (connection is None or connection.is_closed)
        
        return {
            "service": "stt-feedback",
            "mode": "worker",
            "status": "healthy" if rabbitmq_status else "degraded",
            "rabbitmq_connection": rabbitmq_status,
            "processing": "asynchronous",
            "note": "STT feedback processing is handled by dedicated worker process"
        }
    except Exception as e:
        logger.error(f"STT 서비스 상태 확인 실패: {e}")
        return {
            "service": "stt-feedback", 
            "mode": "worker",
            "status": "unhealthy",
            "error": str(e)
        }


# 유틸리티 함수는 Worker에서만 사용하므로 제거 또는 별도 파일로 이동
def format_feedback(feedback_dict: dict) -> str:
    """
    피드백 포맷팅 함수 (Worker에서 사용)
    이 함수는 실제로는 Worker 코드로 이동되어야 합니다.
    """
    if not feedback_dict:
        return ""
    return (
        f"{feedback_dict.get('overall_score', '')} 점\n\n"
        f"{feedback_dict.get('detailed_analysis', '')}\n\n"
        f"잘한 점: {feedback_dict.get('good_points', '')}\n\n"
        f"개선할 점: {feedback_dict.get('areas_for_improvement', '')}"
    )