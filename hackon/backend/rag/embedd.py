import os, numpy as np
from sentence_transformers import SentenceTransformer
from backend.config import EMBED_MODEL

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model

def embed_texts(texts):
    model = get_model()
    vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=32)
    return np.asarray(vecs, dtype="float32")
