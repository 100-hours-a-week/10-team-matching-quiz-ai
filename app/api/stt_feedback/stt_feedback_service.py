import logging
import tempfile
import requests
import os
from typing import List
from pydub import AudioSegment
import openai
import google.generativeai as genai
from app.api.stt_feedback.stt_feedback_model import FeedbackResponse, FeedbackItem

# API 선언
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
openai.api_key = os.getenv("WISPER_API_KEY")

# 로깅 설정
logger = logging.getLogger("stt")
logger.setLevel(logging.INFO)

# BE에서 전달받은 S3에서 audio download
def download_audio(url: str) -> str:
    logger.info(f"[STT] 음성 다운로드 시작: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp.write(response.content)
        tmp.flush()
        logger.info(f"[STT] 다운로드 완료 → {tmp.name}")
        return tmp.name
    except Exception as e:
        logger.error(f"[STT] 다운로드 실패: {e}")
        raise

# 전달받은 시간 기준으로 audio 파일 자르기
def cut_audio(audio_path: str, start_sec: int, end_sec: int) -> AudioSegment:
    logger.info(f"[STT] 오디오 자르기: {start_sec}s ~ {end_sec}s")
    audio = AudioSegment.from_file(audio_path)
    return audio[start_sec * 1000:end_sec * 1000]

# whisper로 audio 파일 STT
def transcribe_whisperx(segment: AudioSegment) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        segment.export(tmp.name, format="mp3")
        logger.info(f"[STT] WhisperX 전사 시작: {tmp.name}")
        with open(tmp.name, "rb") as f:
            result = openai.Audio.transcribe("whisper-1", f)
            transcript = result["text"]
            logger.info(f"[STT] 전사 결과: {transcript[:50]}...")  # 앞 50자만 로그
            return transcript
        
# TODO: 모델 서빙 시 VAD 처리

# Gemini로 모범답안, 피드백 생성
def generate_feedback_gemini(question: str, answer: str) -> dict:
    prompt = f"""
    당신은 10년차 백엔드 개발자입니다. 아래 면접 질문과 면접자의 답변을 보고, 두 가지를 생성하세요:

    1. 해당 질문에 대해 **이상적인 모범답안 (model_answer)** 를 제시해줘. 지원자에게 '이렇게 대답하면 좋다'고 제안할 수 있는 형식으로.
    2. 면접자의 실제 답변에 대해 **개선 피드백 (feedback)** 도 작성해줘. 어떤 점이 좋았고, 무엇이 부족했는지 구체적으로.

    [질문]
    {question}

    [면접자의 실제 답변]
    {answer}

    [응답 포맷 예시]
    model_answer: ...
    feedback: ...
    """

    logger.info(f"[LLM] Gemini 피드백 + 모범답안 생성 시작")
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    logger.info(f"[LLM] 생성 완료")

    # 파싱
    text = response.text.strip()
    try:
        # naive한 라인 분리 방식 (예: "model_answer: ..." 줄, "feedback: ..." 줄)
        lines = text.splitlines()
        model_answer = ""
        feedback = ""
        for line in lines:
            if line.lower().startswith("model_answer"):
                model_answer = line.split(":", 1)[1].strip()
            elif line.lower().startswith("feedback"):
                feedback = line.split(":", 1)[1].strip()

        return {
            "model_answer": model_answer,
            "feedback": feedback
        }

    except Exception as e:
        logger.warning(f"[LLM] 응답 파싱 실패, 전체 응답 사용: {e}")
        return {
            "model_answer": "(모범답안 추출 실패)",
            "feedback": text
        }

# 모범답안, 피드백 생성 파이프라인
def run_feedback_pipeline(
    interview_id: str,
    recording_url: str,
    questionLists: List[dict]
) -> FeedbackResponse:
    logger.info(f"[PIPELINE] 인터뷰 ID: {interview_id}") #interview id 확인
    feedback_items = []
    try:
        local_path = download_audio(recording_url)

        for i, q in enumerate(questionLists):
            logger.info(f"[{i+1}/{len(questionLists)}] 질문 처리 중: '{q['question']}'")
            segment = cut_audio(local_path, q["from"], q["to"]) # 잘라낸 음성
            transcript = transcribe_whisperx(segment) # STT 결과
            result = generate_feedback_gemini(q["question"], transcript) # 질문 - STT 결과 대응해서 피드백 생성

            feedback_items.append(FeedbackItem(
                question=q["question"],
                model_answer=result["model_answer"],
                feedback=result["feedback"]
            ))

        logger.info(f"[PIPELINE] 총 {len(feedback_items)}개 질문 처리 완료")
        return FeedbackResponse(
            interview_id=interview_id,
            feedbackLists=feedback_items
        )

    except Exception as e:
        logger.error(f"[PIPELINE] 처리 중 오류 발생: {e}")
        raise