import logging
import os
import json
from typing import List
from app.api.stt_feedback.service.audio_service import download_audio, cut_audio
from app.api.stt_feedback.service.stt_service import transcribe_whisperx
from app.api.stt_feedback.service.generate_feedback_service import generate_feedback_gemini
from app.api.stt_feedback.stt_feedback_model import FeedbackResponse, FeedbackItem
from app.api.stt_feedback.stt_feedback_schema import QuestionItem

logger = logging.getLogger("stt")

# 잘린 오디오 파일 저장 경로
SAVE_SEGMENT_DIR = "./tmp_segments"
os.makedirs(SAVE_SEGMENT_DIR, exist_ok=True)

# feedback 생성 파이프라인
def run_feedback_pipeline(
    recording_url: str,
    question_lists: List[QuestionItem]
) -> FeedbackResponse:
    feedback_items = []

    try:
        # 1. 전체 audio 파일 다운로드 (S3 등에서)
        logger.info(f"[PIPELINE] 오디오 다운로드 시작 - URL: {recording_url}")
        local_path = download_audio(recording_url)
        logger.info(f"[PIPELINE] 오디오 다운로드 완료 - 저장 위치: {local_path}")

        # 2. 각 질문별로 STT 및 Feedback 처리
        for i, q in enumerate(question_lists):
            try:
                logger.info(f"[{i+1}/{len(question_lists)}] 질문 처리 시작: '{q.question}'")

                # 2-1. 오디오 자르기
                segment = cut_audio(local_path, q.start_time, q.end_time)
                logger.info(f"[{i+1}/{len(question_lists)}] 오디오 자르기 완료: {q.start_time}s ~ {q.end_time}s")

                
                # (잘린 오디오 파일 저장)
                segment_filename = f"interview-q{i+1}-{q.segment_id}.mp3"
                segment_path = os.path.join(SAVE_SEGMENT_DIR, segment_filename)
                segment.export(segment_path, format='mp3')
                logger.info(f"[{i+1}/{len(question_lists)}] 자른 오디오 저장 완료: {segment_path}")
            
                
                # 2-2. WhisperX STT 수행
                transcript = transcribe_whisperx(segment)
                logger.info(f"[{i+1}/{len(question_lists)}] STT 결과: '{transcript}'")

                # 2-3. Gemini LLM으로 모범답안 + 피드백 생성
                result = generate_feedback_gemini(q.question, transcript)
                logger.info(f"[{i+1}/{len(question_lists)}] Gemini 응답 - 모범답안: '{result['model_answer']}', 피드백: '{result['feedback']}'")

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