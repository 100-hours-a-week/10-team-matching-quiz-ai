import os
from dotenv import load_dotenv

load_dotenv()

# RabbitMQ Server Details (AWS)
RABBITMQ_HOST = "43.203.77.116"
RABBITMQ_PORT = 5672
RABBITMQ_USER = os.getenv('RABBITMQ_USER')  
RABBITMQ_PASSWORD = os.getenv('RABBITMQ_PASSWORD')  
RABBITMQ_VIRTUAL_HOST = "/" # Usually "/" unless specified otherwise

# --- Quiz Generation Flow ---

# BE -> AIServer (Request for Quiz Generation)
QUIZ_REQUEST_EXCHANGE_NAME = "quiz.request.exchange"
QUIZ_REQUEST_EXCHANGE_TYPE = "direct"  # Or your preferred exchange type
QUIZ_REQUEST_ROUTING_KEY = "quiz.request.routingKey"
# Queue for the AIServer worker to consume quiz requests from
QUIZ_PROCESSING_QUEUE_NAME = "quiz.processing.queue" # Worker consumes from this

# AIServer -> BE (Response with Generated Quiz)
QUIZ_RESPONSE_EXCHANGE_NAME = "quiz.response.exchange" # Can be the same as request exchange
QUIZ_RESPONSE_EXCHANGE_TYPE = "direct" # Or your preferred exchange type
QUIZ_RESPONSE_ROUTING_KEY = "quiz.response.routingKey" # Routing key for the response
# Queue for the BE to consume quiz responses from
QUIZ_RESPONSE_QUEUE_NAME = "quiz.response.queue"    # BE consumes from this

# --- STT Feedback Flow (Keeping existing for now, can be updated similarly if needed) ---
STT_FEEDBACK_EXCHANGE_NAME = "stt_feedback_exchange" # Example, update as needed
STT_FEEDBACK_EXCHANGE_TYPE = "direct"
STT_FEEDBACK_ROUTING_KEY = "stt_feedback.tasks"
STT_FEEDBACK_QUEUE_NAME = "stt_feedback_queue"

# Common Worker Settings
PREFETCH_COUNT = 1  # Each worker fetches one message at a time

