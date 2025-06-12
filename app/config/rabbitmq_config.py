import os
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

RABBITMQ_HOST = "localhost"  # GCP VM 내부에서 실행 시 localhost, 외부 브로커 사용 시 해당 주소
RABBITMQ_PORT = 5672
RABBITMQ_USER = "guest"  # 실제 환경에 맞게 변경
RABBITMQ_PASSWORD = "guest"  # 실제 환경에 맞게 변경
RABBITMQ_VIRTUAL_HOST = "/"

# 공통으로 사용할 Exchange 설정
SERVICE_EXCHANGE_NAME = "service_tasks_exchange"
SERVICE_EXCHANGE_TYPE = "direct" # 라우팅 키 기반으로 특정 큐에 직접 전달

# 각 서비스별 라우팅 키 (예시)
# 실제 서비스 이름에 맞게 정의하여 사용합니다.
ROUTING_KEY_QUIZ_GENERATOR = "quiz_generator.tasks"
ROUTING_KEY_STT_FEEDBACK = "stt_feedback.tasks" # STT Feedback 서비스용 라우팅 키 추가
# 필요에 따라 더 많은 라우팅 키를 추가할 수 있습니다.

# 워커 설정을 위한 추가 값
PREFETCH_COUNT = 1  # 각 워커가 한 번에 가져올 메시지 수 (환경에 맞게 조절)

# 각 서비스별 큐 이름
QUIZ_QUEUE_NAME = "quiz_generation_queue"
STT_FEEDBACK_QUEUE_NAME = "stt_feedback_queue"

