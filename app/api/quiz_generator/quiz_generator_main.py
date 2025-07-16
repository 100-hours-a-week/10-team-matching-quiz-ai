# main.py
from app.api.quiz_generator.quiz_generator_schema import FollowupRequest
from app.api.quiz_generator.quiz_generator_api import process_quiz_generation

if __name__ == "__main__":
    req = FollowupRequest(
        interview_id="local-test-001",
        question_history_list=[
            "Python의 특징은 무엇인가요?",
            "프로그래밍을 할 때 주로 어떤 언어를 사용하시나요?",
            "머신러닝과 딥러닝의 차이는 무엇인가요?",
            "머신러닝에서 precision은 무엇인가요?",
            "프로젝트에서 RAG를 사용해보신 적이 있나요?",
            "RAG는 데이터를 Embedding하고 VectorDB를 활용한다."
        ]
    )

    process_quiz_generation(req)