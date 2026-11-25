"""
classify_single.py
Interactive single-complaint classifier to see the exact step-by-step logic in action!

Usage:
  python classify_single.py
  python classify_single.py "Fan making noise in room 107"
"""

import sys
import os
from dotenv import load_dotenv

import data_clean
import category_manager
import embeddings
import vector_store
import llm_pass

load_dotenv()
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.80"))
REGISTRY_JSON = "category_registry.json"
FAISS_INDEX_FILE = "faiss_index.bin"
VECTOR_METADATA_FILE = "vector_metadata.json"


def classify_single(text: str, room: str = None):
    print("\n" + "=" * 70)
    print("           STEP-BY-STEP COMPLAINT CLASSIFICATION")
    print("=" * 70)

    # STEP 1: Text Cleaning & Room Extraction
    print("\n[Step 1] Text Normalization & Room Extraction:")
    cleaned_text = data_clean.clean_text(text)
    detected_room = data_clean.extract_room_number(cleaned_text, default_room=room or "")
    print(f"  • Original Input: '{text}'")
    print(f"  • Cleaned Text:   '{cleaned_text}'")
    print(f"  • Room Detected:  '{detected_room}'")

    # STEP 2: Load Index and Registry
    print("\n[Step 2] Loading FAISS Vector Store & Category Registry:")
    registry = category_manager.load_registry(REGISTRY_JSON)
    faiss_index = vector_store.create_or_load_index(FAISS_INDEX_FILE)
    vector_metadata = vector_store.load_vector_metadata(VECTOR_METADATA_FILE)
    print(f"  • Registry Categories Loaded: {len(registry)}")
    print(f"  • FAISS Vectors in Index:    {faiss_index.ntotal}")

    if faiss_index.ntotal == 0:
        print("  ! FAISS index is empty. Please run 'python main.py' first to build the base index.")
        return

    # STEP 3: Generate Embedding
    print("\n[Step 3] Generating L2-Normalized Embedding Vector:")
    query_emb = embeddings.generate_single_embedding(cleaned_text)
    print(f"  • Embedding Shape: {query_emb.shape} (Dimension: 384)")

    # STEP 4: FAISS Similarity Search
    print("\n[Step 4] Searching FAISS for Nearest Existing Complaint (Cosine Similarity):")
    search_res = vector_store.search_nearest_complaint(faiss_index, vector_metadata, query_emb)
    nearest_sim = search_res["similarity"]
    nearest_cat = search_res["category_id"]
    nearest_room = search_res["room_number"]
    nearest_text = search_res["complaint_text"]

    print(f"  • Closest Vector ID:        {search_res['vector_id']}")
    print(f"  • Nearest Complaint Text:   '{nearest_text}'")
    print(f"  • Nearest Room:             '{nearest_room}'")
    print(f"  • Existing Category ID:     '{nearest_cat}'")
    print(f"  • Calculated Cosine Sim:    {nearest_sim:.4f} (Threshold is {SIMILARITY_THRESHOLD})")

    # STEP 5: Verification & Decision Logic
    print("\n[Step 5] Evaluating Match & Room Separation:")
    is_room_match = (str(nearest_room).strip().lower() == str(detected_room).strip().lower())
    is_sim_sufficient = (nearest_sim >= SIMILARITY_THRESHOLD)

    assigned_cat_id = None
    decision_reason = None

    if is_sim_sufficient and is_room_match:
        print("  ✓ PASS: Cosine similarity is high AND room numbers match perfectly!")
        assigned_cat_id = nearest_cat
        decision_reason = f"FAISS Direct Match (Sim: {nearest_sim:.4f} >= {SIMILARITY_THRESHOLD})"
    else:
        if not is_room_match:
            print(f"  ✗ FAIL: Room mismatch! Query Room ('{detected_room}') != Nearest Room ('{nearest_room}').")
            print("    Strict Rule: Complaints from different rooms must NEVER be merged into the same category.")
        if not is_sim_sufficient:
            print(f"  ✗ FAIL: Cosine similarity ({nearest_sim:.4f}) is below threshold ({SIMILARITY_THRESHOLD}).")

        # STEP 6: LLM Evaluation
        print("\n[Step 6] Routing to LLM for Room-Specific Decision:")
        llm_decision = llm_pass.classify_or_propose_category(text, detected_room, registry)
        action = llm_decision.get("action")

        if action == "MATCH" and llm_decision.get("category_id") in registry:
            assigned_cat_id = llm_decision["category_id"]
            decision_reason = f"LLM Matched existing category for Room {detected_room}"
            print(f"  • LLM Action: MATCH to existing category {assigned_cat_id}")
        else:
            new_name = llm_decision.get("proposed_name", f"Issue in Room {detected_room}")
            new_desc = llm_decision.get("proposed_description", f"Complaints regarding Room {detected_room}")
            assigned_cat_id = category_manager.add_category(
                registry=registry,
                category_name=new_name,
                room_number=detected_room,
                description=new_desc,
                example_complaint=text,
                filepath=REGISTRY_JSON
            )
            decision_reason = f"LLM Proposes NEW Category for Room {detected_room}"
            print(f"  • LLM Action: CREATE NEW category [{assigned_cat_id}]: '{new_name}'")

    # STEP 7: Update Counts & FAISS Index
    print("\n[Step 7] Updating Category Counts & Vector Store:")
    if decision_reason.startswith("FAISS Direct Match") or "Matched existing" in decision_reason:
        category_manager.increment_category_count(registry, assigned_cat_id, text, REGISTRY_JSON)
        print(f"  • Incremented complaint count for [{assigned_cat_id}].")

    vector_store.add_complaint_vector(
        index=faiss_index,
        metadata=vector_metadata,
        embedding=query_emb,
        category_id=assigned_cat_id,
        room_number=detected_room,
        complaint_text=text
    )
    vector_store.save_vector_store(faiss_index, vector_metadata, FAISS_INDEX_FILE, VECTOR_METADATA_FILE)
    print(f"  • Saved new vector to FAISS index. Total vectors now: {faiss_index.ntotal}")

    # Summary
    cat_details = registry.get(assigned_cat_id, {})
    print("\n" + "=" * 70)
    print("                       FINAL CLASSIFICATION")
    print("=" * 70)
    print(f"Category ID:   {assigned_cat_id}")
    print(f"Category Name: {cat_details.get('category_name')}")
    print(f"Room:          {cat_details.get('room_number')}")
    print(f"Total Count:   {cat_details.get('complaint_count')}")
    print(f"Method:        {decision_reason}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        complaint_arg = " ".join(sys.argv[1:])
        classify_single(complaint_arg)
    else:
        # Interactive prompt
        print("Enter a student complaint to test classification (or press Ctrl+C to exit):")
        try:
            user_input = input("Complaint text > ").strip()
            if user_input:
                classify_single(user_input)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
