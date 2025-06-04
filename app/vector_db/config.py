import os
from dotenv import load_dotenv

load_dotenv()

# Model Config
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "Alibaba-NLP/gte-multilingual-base")
KEYBERT_MODEL_NAME = os.getenv("KEYBERT_MODEL_NAME")

# vector DB Config
DB_PATH = os.getenv("DB_PATH", "./chroma_db")
FOLLOW_UP_COLLECTION_NAME = os.getenv("FOLLOW_UP_COLLECTION_NAME", "questions")
QUIZ_COLLECTION_NAME = os.getenv("QUIZ_COLLECTION_NAME", "quiz-generation")

# RAG Parameters
RAG_TOP_K = int(os.getenv("RAG_TOP_K", 4))
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", 0.6))
RAG_DIVERSITY = float(os.getenv("Rag_diversity", 0.1)) 