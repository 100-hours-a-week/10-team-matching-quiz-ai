import logging
import os
import google.generativeai as genai

logger = logging.getLogger("stt")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

#TODO: LLM 프롬프트 고도화 필요(현재 Gemini 피드백)
# 아래 프롬프트 기반으로 AI 피드백 생성
def generate_feedback_gemini(question: str, answer: str) -> dict:
    prompt = f"""
    당신은 10년차 백엔드 개발자입니다. 아래 면접 질문과 면접자의 답변을 보고, 두 가지를 생성하세요:

    1. 해당 질문에 대해 **이상적인 모범답안 (model_answer)** 를 제시해줘.
    2. 면접자의 실제 답변에 대해 **좋았던 점과 개선 피드백 (feedback)** 도 작성해줘.

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

    text = response.text.strip()
    try:
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
        logger.warning(f"[LLM] 응답 파싱 실패: {e}")
        return {
            "model_answer": "(모범답안 추출 실패)",
            "feedback": text
        }
