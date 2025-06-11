import logging
from typing import List
from app.api.stt_feedback.service.audio_service import download_audio, cut_audio
from app.api.stt_feedback.service.stt_service import transcribe_whisperx
from app.api.stt_feedback.service.generate_feedback_service import generate_feedback_gemini
from app.api.stt_feedback.stt_feedback_model import FeedbackResponse, FeedbackItem
from app.api.stt_feedback.stt_feedback_schema import QuestionItem # 면접 질문 정보 가져오기

logger = logging.getLogger("stt")

# feedback 생성 파이프라인
def run_feedback_pipeline(
    recording_url: str,
    question_lists: List[QuestionItem]
) -> FeedbackResponse:
    feedback_items = []

    try:
        # 1.전체 audio 파일 다운로드 (S3) - 1명
        local_path = download_audio(recording_url)

        # 2. 각 질문별로 STT 및 Feedback 처리
        for i, q in enumerate(question_lists):
            try:
                logger.info(f"[{i+1}/{len(question_lists)}] 질문 처리 중: '{q.question}'")
                # 2-1. 질문 시작, 종료 시간 기준으로 오디올 자르기
                segment = cut_audio(local_path, q.start_time, q.end_time)
                # 2-2. WhisperX를 활용한 오디오 전사(with VAD)
                transcript = transcribe_whisperx(segment)
                # 2-3. Gemini API로 피드백 및 모범답안 생성
                result = generate_feedback_gemini(q.question, transcript)

                # 2-4. 결과 리스트에 질문, 모범답안, 피드백 추가
                feedback_items.append(FeedbackItem(
                    segment_id=q.segment_id,
                    question=q.question,
                    model_answer=result["model_answer"],
                    feedback=result["feedback"]
                ))

            except Exception as item_error:
                # 개별 질문 처리 실패시 로그만 찍고 계속 진행
                logger.error(
                    f"[PIPELINE][{i+1}/{len(question_lists)}] 질문 처리 실패 - 질문: '{q.question}', 오류: {item_error}"
                )
                continue  # 다음 질문으로 넘어감

        # 3. 전체 질문 처리 실패 시에는 경고 로그 출력 
        if not feedback_items:
            logger.warning("[PIPELINE] 모든 질문 처리 실패 → 빈 피드백 반환")
        
        logger.info(f"[PIPELINE] 총 {len(feedback_items)}개 질문 처리 완료")

        # 4. 결과 반환
        return FeedbackResponse(
            feedbackLists=feedback_items
        )

    except Exception as e:
        # 전체 프로세스 실패 (다운로드 실패, 예상치 못한 에러, 등)
        logger.critical(f"[PIPELINE] 전체 파이프라인 오류 - 오류: {e}")
        raise