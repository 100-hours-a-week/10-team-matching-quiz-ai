import logging
from typing import List
from app.api.stt_feedback.service.audio_service import download_audio, cut_audio
from app.api.stt_feedback.service.stt_service import transcribe_whisperx
from app.api.stt_feedback.service.generate_feedback_service import generate_feedback_gemini
from app.api.stt_feedback.stt_feedback_model import FeedbackResponse, FeedbackItem

logger = logging.getLogger("stt")

# feedback 생성 파이프라인
def run_feedback_pipeline(
    interview_id: str,
    recording_url: str,
    questionLists: List[dict]
) -> FeedbackResponse:
    logger.info(f"[PIPELINE] 인터뷰 ID: {interview_id}")
    feedback_items = []

    try:
        local_path = download_audio(recording_url)

        for i, q in enumerate(questionLists):
            try:
                logger.info(f"[{i+1}/{len(questionLists)}] 질문 처리 중: '{q['question']}'")
                segment = cut_audio(local_path, q["from"], q["to"])
                transcript = transcribe_whisperx(segment)
                result = generate_feedback_gemini(q["question"], transcript)

                feedback_items.append(FeedbackItem(
                    question=q["question"],
                    model_answer=result["model_answer"],
                    feedback=result["feedback"]
                ))

            except Exception as item_error:
                logger.error(
                    f"[PIPELINE][{i+1}/{len(questionLists)}] 질문 처리 실패 - 질문: '{q['question']}', 오류: {item_error}"
                )
                continue  # 다음 질문으로 넘어감

        if not feedback_items:
            logger.warning("[PIPELINE] 모든 질문 처리 실패 → 빈 피드백 반환")
        
        logger.info(f"[PIPELINE] 총 {len(feedback_items)}개 질문 처리 완료")

        return FeedbackResponse(
            interview_id=interview_id,
            feedbackLists=feedback_items
        )

    except Exception as e:
        logger.critical(f"[PIPELINE] 전체 파이프라인 오류 - 인터뷰 ID: {interview_id}, 오류: {e}")
        raise