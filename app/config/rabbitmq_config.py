import os
from dotenv import load_dotenv
load_dotenv()

# RabbitMQ Server Details (AWS)
RABBITMQ_HOST = "43.203.77.116"
RABBITMQ_PORT = 5672
RABBITMQ_USER = os.getenv('RABBITMQ_USER')  
RABBITMQ_PASSWORD = os.getenv('RABBITMQ_PASSWORD')  
RABBITMQ_VIRTUAL_HOST = "/"

# --- Quiz Generation Flow ---

# BE -> AIServer (Request for Quiz Generation)
QUIZ_REQUEST_EXCHANGE_NAME = "quiz.request.exchange"
QUIZ_REQUEST_EXCHANGE_TYPE = "direct"  # Or your preferred exchange type
QUIZ_REQUEST_ROUTING_KEY = "quiz.request.routingKey"
QUIZ_REQUEST_QUEUE_NAME = "quiz.request.queue"    # BE consumes from this

# AIServer -> BE (Response with Generated Quiz)
QUIZ_RESPONSE_EXCHANGE_NAME = "quiz.response.exchange" # Can be the same as request exchange
QUIZ_RESPONSE_EXCHANGE_TYPE = "direct" # Or your preferred exchange type
QUIZ_RESPONSE_ROUTING_KEY = "quiz.response.routingKey" # Routing key for the response
# Queue for the BE to consume quiz responses from
QUIZ_RESPONSE_QUEUE_NAME = "quiz.response.queue"    # BE consumes from this

# --- STT Feedback Flow ---
STT_FEEDBACK_EXCHANGE_NAME = "stt.feedback.exchange" 
STT_FEEDBACK_EXCHANGE_TYPE = "direct"
STT_FEEDBACK_ROUTING_KEY = "stt_feedback.tasks"
STT_FEEDBACK_QUEUE_NAME = "stt.feedback.queue"

# STT Response Flow (AI Server -> BE)
STT_RESPONSE_EXCHANGE_NAME = "stt.response.exchange"
STT_RESPONSE_EXCHANGE_TYPE = "direct"
STT_RESPONSE_ROUTING_KEY = "stt.response.routingKey"
STT_RESPONSE_QUEUE_NAME = "stt.response.queue"

# --- Missing Constants Referenced by Workers ---
# Use quiz.request.exchange as the main service exchange for now
SERVICE_EXCHANGE_NAME = QUIZ_REQUEST_EXCHANGE_NAME
SERVICE_EXCHANGE_TYPE = QUIZ_REQUEST_EXCHANGE_TYPE

# Routing keys for workers (aliases for consistency)
ROUTING_KEY_QUIZ_GENERATOR = QUIZ_REQUEST_ROUTING_KEY
ROUTING_KEY_STT_FEEDBACK = STT_FEEDBACK_ROUTING_KEY
QUIZ_QUEUE_NAME = QUIZ_REQUEST_QUEUE_NAME

# Common Worker Settings
PREFETCH_COUNT = 1  # Each worker fetches one message at a time

