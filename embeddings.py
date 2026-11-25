"""
embeddings.py
Generates sentence embeddings using Sentence Transformers and normalizes them for FAISS.
"""

import numpy as np
from sentence_transformers import SentenceTransformer


# Global cached model instance to avoid reloading weights repeatedly
_MODEL = None
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


def get_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    """
    Returns the cached SentenceTransformer model, loading it if not already loaded.
    Uses 'all-MiniLM-L6-v2' (lightweight, 384 dimensions, runs quickly on CPU).
    """
    global _MODEL
    if _MODEL is None:
        print(f"[Embeddings] Loading SentenceTransformer model: {model_name}...")
        _MODEL = SentenceTransformer(model_name)
        print("[Embeddings] Model loaded successfully.")
    return _MODEL


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    Normalizes vectors to unit length (L2 norm = 1.0).
    When normalized, Inner Product (IndexFlatIP) in FAISS equals Cosine Similarity!
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0.0] = 1.0
    return (vectors / norms).astype(np.float32)


def generate_embeddings(texts: list[str]) -> np.ndarray:
    """
    Generates normalized 2D embeddings (shape: [N, 384]) for a list of complaint texts.
    """
    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    model = get_embedding_model()
    raw_embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    # Ensure 2D float32 array
    if raw_embeddings.ndim == 1:
        raw_embeddings = np.expand_dims(raw_embeddings, axis=0)

    raw_embeddings = raw_embeddings.astype(np.float32)
    return normalize_vectors(raw_embeddings)


def generate_single_embedding(text: str) -> np.ndarray:
    """
    Generates a normalized 2D embedding (shape: [1, 384]) for a single complaint text.
    """
    return generate_embeddings([text])


if __name__ == "__main__":
    print("Testing embeddings.py...")
    test_texts = ["Fan not working in room 107", "AC blowing hot air in room 204"]
    embs = generate_embeddings(test_texts)
    print(f"Generated embeddings of shape: {embs.shape}")
    print(f"L2 norm of first vector: {np.linalg.norm(embs[0]):.4f} (should be 1.0000)")
