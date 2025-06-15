import logging
import os
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import json # JSON 응답 파싱을 위해 import

logger = logging.getLogger("stt")

# Gemini API 키 설정
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")

# 최신 버전용 초기화 (0.8.x 이상)
genai.configure(api_key=GEMINI_API_KEY)

# LLM 프롬프트 고도화 및 Gemini 피드백 생성
def generate_feedback_gemini(question: str, answer: str) -> dict:
    # 프롬프트: '신입 백엔드 개발자' 면접관 역할을 명확히 하고, 상세한 JSON 출력을 요구합니다.
    prompt = f"""
    당신은 10년차 시니어 백엔드 개발자이자, 개발자 면접관입니다.
    아래 면접 질문과 면접자의 답변을 보고, 다음 항목들을 JSON 형식으로 생성하세요.

    [생성 항목]
    1.  **model_answer (string)**: 해당 질문에 대한 이상적인 모범답안을 제시해주세요.
        -   간결하고 핵심적이며, 기술적인 정확성과 실무적인 관점을 모두 담아야 합니다.
        -   예상 질문과 맥락을 파악하여 답변에 필요한 모든 핵심 내용을 포함해야 합니다.

    2.  **feedback (object)**: 면접자의 실제 답변에 대한 상세 피드백을 제공합니다.
        a.  **good_points (string)**: 면접자가 잘 답변했거나 잠재력이 보였던 긍정적인 측면을 언급합니다.
        b.  **areas_for_improvement (string)**: 면접자가 놓쳤거나 더 발전시킬 수 있는 부분을 명확하고 건설적으로 제시합니다.
            -   기술적 내용, 문제 해결 능력, 커뮤니케이션 스킬 등 답변에서 드러나는 다양한 측면을 고려합니다.
            -   구체적인 예시나 개선 방안을 포함하면 좋습니다.
        c.  **overall_score (integer)**: 1점에서 5점 사이의 면접 답변 점수 (1: 매우 미흡, 5: 매우 우수).
        d.  **detailed_analysis (string)**: 면접자의 답변에 대한 전반적인 심층 분석을 제공합니다.
            -   왜 해당 점수를 주었는지에 대한 근거를 포함합니다.

    [질문]
    {question}

    [면접자의 실제 답변]
    {answer}

    [응답 포맷 (JSON)]
    {{
        "model_answer": "...",
        "feedback": {{
            "good_points": "...",
            "areas_for_improvement": "...",
            "overall_score": 0,
            "detailed_analysis": "..."
        }}
    }}
    """

    logger.info(f"[LLM] Gemini 피드백 + 모범답안 생성 시작 (모델: gemini-1.5-pro)")
    try:
        model = genai.GenerativeModel(
            "gemini-1.5-pro", # 또는 'gemini-1.0-pro' 사용 가능 
            generation_config=genai.GenerationConfig(
                temperature=0.7, # 창의성 조절. 피드백에 맞춤
                response_mime_type="application/json", # JSON 형식 강제
            ),
            safety_settings={ # 안전 설정, 필요에 따라 조정 가능
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )

        response = model.generate_content(prompt)
        logger.info(f"[LLM] 생성 완료")

        # JSON 응답 파싱
        json_response = json.loads(response.text)

        # 필요한 정보 추출 및 반환
        model_answer = json_response.get("model_answer", "모범답안 생성 실패")
        feedback_data = json_response.get("feedback", {})

        return {
            "model_answer": model_answer,
            "feedback": {
                "good_points": feedback_data.get("good_points", "좋았던 점 없음"),
                "areas_for_improvement": feedback_data.get("areas_for_improvement", "개선 방안 없음"),
                "overall_score": feedback_data.get("overall_score", 0), # 기본값 0 또는 적절한 초기값
                "detailed_analysis": feedback_data.get("detailed_analysis", "상세 분석 없음")
            }
        }

    except json.JSONDecodeError as e:
        logger.error(f"[LLM] 응답 JSON 파싱 실패: {e}\n원본 응답: {response.text if 'response' in locals() else '응답 없음'}")
        return {
            "model_answer": "모범답안 생성 실패 (JSON 파싱 오류)",
            "feedback": {
                "good_points": "오류 발생",
                "areas_for_improvement": "JSON 응답 파싱 오류 발생",
                "overall_score": 0,
                "detailed_analysis": response.text if 'response' in locals() else "JSON 응답 파싱 오류 발생"
            }
        }
    except Exception as e:
        logger.error(f"[LLM] Gemini API 호출 또는 처리 중 오류 발생: {e}")
        return {
            "model_answer": "모범답안 생성 실패",
            "feedback": {
                "good_points": "오류 발생",
                "areas_for_improvement": "API 호출 또는 처리 중 오류 발생",
                "overall_score": 0,
                "detailed_analysis": f"오류 메시지: {e}"
            }
        }
"""
# --- 테스트 코드 예시 ---
if __name__ == "__main__":
    if not os.getenv("GEMINI_API_KEY"):
        print("경고: 'GEMINI_API_KEY' 환경 변수가 설정되지 않았습니다.")
        print("API 키를 설정한 후 다시 실행해주세요. (예: export GEMINI_API_KEY='YOUR_API_KEY')")
        exit()

    logging.basicConfig(level=logging.INFO)

    test_question_1 = "데이터베이스 트랜잭션의 ACID 속성에 대해 설명하고, 각 속성이 왜 중요한지 예시를 들어 설명해주세요."
    test_answer_1 = "ACID는 데이터베이스 트랜잭션의 안정성을 보장하는 속성입니다. Atomicity는 트랜잭션이 전부 실행되거나 전혀 실행되지 않는 것을 의미합니다. Consistency는 트랜잭션 실행 후에도 데이터베이스의 일관성이 유지되는 것입니다. Isolation은 여러 트랜잭션이 동시에 실행될 때 서로 영향을 주지 않는 것이고, Durability는 트랜잭션이 성공적으로 완료되면 그 결과가 영구적으로 반영되는 것입니다."

    print("--- 첫 번째 테스트 ---")
    feedback_result_1 = generate_feedback_gemini(test_question_1, test_answer_1)
    print("\n[모범 답안]:")
    print(feedback_result_1["model_answer"])
    print("\n[피드백]:")
    print(json.dumps(feedback_result_1["feedback"], indent=2, ensure_ascii=False)) # JSON 예쁘게 출력

    print("\n" + "="*50 + "\n")

    test_question_2 = "Spring Boot에서 @Transactional 어노테이션의 동작 방식과 주의할 점에 대해 설명해주세요."
    test_answer_2 = "Transactional 어노테이션은 메서드나 클래스에 붙여서 트랜잭션을 관리합니다. 예외가 발생하면 롤백되고, 아니면 커밋됩니다. 주의할 점은 자기 호출에서는 작동하지 않는다는 것입니다. 그리고 readOnly를 설정하면 성능이 좋아집니다."

    print("--- 두 번째 테스트 ---")
    feedback_result_2 = generate_feedback_gemini(test_question_2, test_answer_2)
    print("\n[모범 답안]:")
    print(feedback_result_2["model_answer"])
    print("\n[피드백]:")
    print(json.dumps(feedback_result_2["feedback"], indent=2, ensure_ascii=False))
"""