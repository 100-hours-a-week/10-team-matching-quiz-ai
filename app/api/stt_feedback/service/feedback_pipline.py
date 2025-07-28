import logging
import os
from dotenv import load_dotenv
from typing import List
from app.api.stt_feedback.service.audio_service import download_audio, cut_audio
from app.api.stt_feedback.service.stt_service import transcribe_whisperx
from app.api.stt_feedback.service.generate_feedback_service import generate_feedback_gemini
from app.api.stt_feedback.stt_feedback_model import FeedbackResponse, FeedbackItem
from app.api.stt_feedback.stt_feedback_schema import QuestionItem

# Langfuse 연동 추가
from langfuse import Langfuse

load_dotenv()
LANGFUSE_PUBLIC_KEY = os.getenv("STT_LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("STT_LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("STT_LANGFUSE_HOST")

if not (LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY and LANGFUSE_HOST):
    logging.warning("[Langfuse] 환경 변수 미설정: 추적이 기록되지 않습니다.")
    langfuse = None
else:
    langfuse = Langfuse(
        public_key=LANGFUSE_PUBLIC_KEY,
        secret_key=LANGFUSE_SECRET_KEY,
        host=LANGFUSE_HOST,
    )

logger = logging.getLogger("stt")

SAVE_SEGMENT_DIR = "./tmp_segments"
os.makedirs(SAVE_SEGMENT_DIR, exist_ok=True)

def run_feedback_pipeline(
    recording_url: str,
    question_lists: List[QuestionItem]
) -> FeedbackResponse:
    feedback_items = []

    trace = None
    if langfuse:
        trace = langfuse.trace(
            name="run_feedback_pipeline",
            input={
                "recording_url": recording_url,
                "question_lists": [q.model_dump() for q in question_lists]
            }
        )

    try:
        try:
            download_span = trace.span(name="download_audio", input={"url": recording_url}) if trace else None

            logger.info(f"[PIPELINE] 오디오 다운로드 시작 - URL: {recording_url}")
            local_path = download_audio(recording_url)
            logger.info(f"[PIPELINE] 오디오 다운로드 완료 - 저장 위치: {local_path}")

            if download_span:
                download_span.output = {"local_path": local_path}
                download_span.end()

        except Exception as e:
            logger.error(f"[STT] 오디오 다운로드 실패: {e}")
            if download_span:
                download_span.error(str(e))
                download_span.output = {"error": str(e)}
                download_span.end()
            raise e


        for i, q in enumerate(question_lists):
            try:
                trace.span("cut_audio", input={"audio_path": local_path, "start_time": q.start_time, "end_time": q.end_time}).end()
                logger.info(f"[{i+1}/{len(question_lists)}] 질문 처리 시작: '{q.question}'")

                segment = cut_audio(local_path, q.start_time, q.end_time)
                logger.info(f"[{i+1}/{len(question_lists)}] 오디오 자르기 완료: {q.start_time}s ~ {q.end_time}s")

                segment_filename = f"interview-q{i+1}-{q.segment_id}.mp3"
                segment_path = os.path.join(SAVE_SEGMENT_DIR, segment_filename)
                segment.export(segment_path, format='mp3')
                logger.info(f"[{i+1}/{len(question_lists)}] 자른 오디오 저장 완료: {segment_path}")

                trace.span("transcribe_whisperx", input={"segment_path": segment_path}).end()
                transcript = transcribe_whisperx(segment)
                logger.info(f"[{i+1}/{len(question_lists)}] STT 결과: '{transcript}'")
                trace.span("transcription_result", output={"transcription": transcript}).end()

                trace.span("generate_feedback_gemini", input={"question": q.question, "answer": transcript}).end()
                result = generate_feedback_gemini(q.question, transcript)
                trace.span("feedback_result", output={"feedback": result})
                logger.info(f"[{i+1}/{len(question_lists)}] Gemini 응답 - 모범답안: '{result['model_answer']}'")
                logger.info(f"[{i+1}/{len(question_lists)}] 피드백 점수: {result['feedback'].get('overall_score', 0)}점")
                logger.info(f"[{i+1}/{len(question_lists)}] 피드백 상세분석: {result['feedback'].get('detailed_analysis', '')[:100]}...")
                logger.info(f"[{i+1}/{len(question_lists)}] 좋은 점: {result['feedback'].get('good_points', '')[:50]}...")
                logger.info(f"[{i+1}/{len(question_lists)}] 개선점: {result['feedback'].get('areas_for_improvement', '')[:50]}...")

                feedback_items.append(FeedbackItem(
                    segment_id=q.segment_id,
                    question=q.question,
                    model_answer=result["model_answer"],
                    feedback=result["feedback"]
                ))

            except Exception as item_error:
                trace.error(f"[PIPELINE][{i+1}/{len(question_lists)}] 질문 처리 실패 - 질문: '{q.question}', 오류: {item_error}")
                logger.error(
                    f"[PIPELINE][{i+1}/{len(question_lists)}] 질문 처리 실패 - 질문: '{q.question}', 오류: {item_error}"
                )
                continue

        if not feedback_items:
            logger.warning("[PIPELINE] 모든 질문 처리 실패 → 빈 피드백 반환")

        logger.info(f"[PIPELINE] 총 {len(feedback_items)}개 질문 처리 완료")

        trace.span(name="pipeline_complete", output={...})
        trace.update(output={"feedback_items": [item.model_dump() for item in feedback_items]})

        return FeedbackResponse(
            feedbackLists=feedback_items
        )

    except Exception as e:
        trace.error(f"[PIPELINE] 전체 파이프라인 오류 - 오류: {e}")
        logger.critical(f"[PIPELINE] 전체 파이프라인 오류 - 오류: {e}")
        raise
    finally:
        pass