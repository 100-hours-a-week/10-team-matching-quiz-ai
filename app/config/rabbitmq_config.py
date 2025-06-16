import os
from dotenv import load_dotenv
load_dotenv()

# RabbitMQ Server Details (AWS)
RABBITMQ_HOST = os.getenv('RABBITMQ_HOST')     
RABBITMQ_PORT = os.getenv('RABBITMQ_PORT')
RABBITMQ_USER = os.getenv('RABBITMQ_USER')  
RABBITMQ_PASSWORD = os.getenv('RABBITMQ_PASSWORD')  
RABBITMQ_VIRTUAL_HOST = "/" 

# --- Quiz Generation Flow ---

# BE -> AIServer 
QUIZ_REQUEST_EXCHANGE_NAME = "quiz.request.exchange"
QUIZ_REQUEST_EXCHANGE_TYPE = "topic"  
QUIZ_REQUEST_ROUTING_KEY = "quiz.request.routingKey"
QUIZ_REQUEST_QUEUE_NAME = "quiz.request.queue"    

# AIServer -> BE 
QUIZ_RESPONSE_EXCHANGE_NAME = "quiz.response.exchange" 
QUIZ_RESPONSE_EXCHANGE_TYPE = "topic" 
QUIZ_RESPONSE_ROUTING_KEY = "quiz.response.routingKey" 
QUIZ_RESPONSE_QUEUE_NAME = "quiz.response.queue"    

# --- STT Feedback Flow ---
STT_FEEDBACK_EXCHANGE_NAME = "feed.request.exchange"
STT_FEEDBACK_EXCHANGE_TYPE = "topic" 
STT_FEEDBACK_ROUTING_KEY = "feedback.request.routingKey" 
STT_FEEDBACK_QUEUE_NAME = "feedback.request.queue" 

# --STT Response Flow (AI Server -> BE) --
STT_RESPONSE_EXCHANGE_NAME = "feedback.response.exchange" 
STT_RESPONSE_EXCHANGE_TYPE = "topic"
STT_RESPONSE_ROUTING_KEY = "feedback.response.routingKey" 
STT_RESPONSE_QUEUE_NAME = "feedback.response.queue"


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

