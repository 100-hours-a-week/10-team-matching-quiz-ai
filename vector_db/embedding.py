# vector_db/embedding.py

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('intfloat/e5-small-v2')

def embed_texts(texts):
    return model.encode(texts, normalize_embeddings=True).tolist()
