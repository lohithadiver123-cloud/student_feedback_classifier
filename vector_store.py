"""
vector_store.py
Handles FAISS index operations (IndexFlatIP for cosine similarity)
and vector-to-category metadata mapping.
"""

import json
import os
import faiss
import numpy as np


INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "vector_metadata.json"
VECTOR_DIMENSION = 384  # Matches sentence-transformers 'all-MiniLM-L6-v2'


def create_or_load_index(index_path: str = INDEX_FILE, dimension: int = VECTOR_DIMENSION) -> faiss.IndexFlatIP:
    """
    Loads an existing FAISS IndexFlatIP from disk, or creates a new one if not found.
    IndexFlatIP computes Inner Product (which equals Cosine Similarity when vectors are L2-normalized).
    """
    if os.path.exists(index_path):
        try:
            index = faiss.read_index(index_path)
            print(f"[FAISS] Loaded existing index from '{index_path}' with {index.ntotal} vectors.")
            return index
        except Exception as e:
            print(f"[FAISS Warning] Failed to read '{index_path}': {e}. Creating a new index.")

    # Create fresh IndexFlatIP for cosine similarity
    index = faiss.IndexFlatIP(dimension)
    print(f"[FAISS] Initialized fresh IndexFlatIP with dimension {dimension}.")
    return index


def load_vector_metadata(metadata_path: str = METADATA_FILE) -> dict:
    """
    Loads metadata mapping vector index (e.g. "0", "1") to complaint details:
    {
        "0": {
            "category_id": "C1",
            "room_number": "107",
            "complaint_text": "Fan not working in room 107"
        }
    }
    """
    if not os.path.exists(metadata_path):
        return {}

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[FAISS Warning] Could not load metadata from '{metadata_path}': {e}. Starting fresh.")
        return {}


def save_vector_store(
    index: faiss.IndexFlatIP,
    metadata: dict,
    index_path: str = INDEX_FILE,
    metadata_path: str = METADATA_FILE
) -> None:
    """
    Persists the FAISS binary index and metadata JSON to disk.
    """
    try:
        faiss.write_index(index, index_path)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"[FAISS] Saved index ({index.ntotal} vectors) to '{index_path}' and metadata to '{metadata_path}'.")
    except Exception as e:
        print(f"[FAISS Error] Failed to save vector store: {e}")


def add_complaint_vector(
    index: faiss.IndexFlatIP,
    metadata: dict,
    embedding: np.ndarray,
    category_id: str,
    room_number: str,
    complaint_text: str
) -> int:
    """
    Adds a single normalized embedding to FAISS and records its metadata mapping.
    Returns the vector_id assigned by FAISS.
    """
    # Ensure 2D float32
    if embedding.ndim == 1:
        embedding = np.expand_dims(embedding, axis=0)
    embedding = embedding.astype(np.float32)

    vector_id = index.ntotal
    index.add(embedding)

    # Store mapping
    metadata[str(vector_id)] = {
        "vector_id": vector_id,
        "category_id": category_id,
        "room_number": str(room_number).strip(),
        "complaint_text": complaint_text.strip()
    }

    return vector_id


def search_nearest_complaint(
    index: faiss.IndexFlatIP,
    metadata: dict,
    query_embedding: np.ndarray
) -> dict | None:
    """
    Searches the FAISS index for the single closest existing complaint.

    Returns:
    {
        "vector_id": int,
        "similarity": float, # Cosine similarity between 0.0 and 1.0
        "category_id": str,
        "room_number": str,
        "complaint_text": str
    }
    or None if the index is empty.
    """
    if index.ntotal == 0:
        return None

    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)
    query_embedding = query_embedding.astype(np.float32)

    # Search for top-1 nearest neighbor
    distances, indices = index.search(query_embedding, k=1)

    nearest_idx = int(indices[0][0])
    similarity_score = float(distances[0][0])

    meta = metadata.get(str(nearest_idx), {})

    return {
        "vector_id": nearest_idx,
        "similarity": similarity_score,
        "category_id": meta.get("category_id", "Unknown"),
        "room_number": meta.get("room_number", "Unknown"),
        "complaint_text": meta.get("complaint_text", "")
    }


if __name__ == "__main__":
    print("Testing vector_store.py...")
    test_index = create_or_load_index("test_faiss.bin")
    test_meta = load_vector_metadata("test_meta.json")

    dummy_vec = np.random.randn(1, 384).astype(np.float32)
    dummy_vec /= np.linalg.norm(dummy_vec)

    add_complaint_vector(test_index, test_meta, dummy_vec, "C1", "107", "Sample fan complaint")
    result = search_nearest_complaint(test_index, test_meta, dummy_vec)
    print("Search result:", result)

    # Cleanup test files
    if os.path.exists("test_faiss.bin"):
        os.remove("test_faiss.bin")
    if os.path.exists("test_meta.json"):
        os.remove("test_meta.json")
