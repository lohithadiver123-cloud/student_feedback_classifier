"""
main.py
AI-Powered Student Complaint Classification System (Buffered Batch Hybrid Architecture)

Speed & Cost Optimization:
1. FAISS Check First:
   - If Cosine Similarity >= Threshold AND Room matches:
     --> INSTANT FAISS HIT (Fast Path ⚡, 0 LLM calls, <5ms)
2. Buffered Batching for FAISS Misses:
   - If a complaint misses FAISS, it is added to a buffer.
   - When the buffer reaches 10 complaints (or at the end of the dataset):
     --> SINGLE LLM BATCH CALL (Processes all 10 in ONE network request! 🧠⚡)
3. Persists categories and vectors across runs (use '--reset' flag to wipe and start fresh).
"""

import sys
import os
from dotenv import load_dotenv

import data_clean
import category_manager
import embeddings
import vector_store
import llm_pass

# Load configuration
load_dotenv()
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.80"))
UNMATCHED_BATCH_SIZE = 10  # Waits until 10 misses accumulate to call LLM once

REGISTRY_JSON = "category_registry.json"
FAISS_INDEX_FILE = "faiss_index.bin"
VECTOR_METADATA_FILE = "vector_metadata.json"

RESET_INDEX = "--reset" in sys.argv

# Determine dataset path
DATASET_CSV = None
for arg in sys.argv[1:]:
    if arg.endswith(".csv"):
        DATASET_CSV = arg
        break

if not DATASET_CSV:
    DATASET_CSV = os.getenv("DATASET_PATH", "complaints.csv")


def process_unmatched_batch(
    unmatched_buffer: list[tuple[dict, any]],
    registry: dict,
    faiss_index,
    vector_metadata: dict
) -> int:
    """
    Sends a batch of up to 10 unmatched complaints to the LLM in ONE single API call.
    Assigns/creates categories and indexes all vectors into FAISS.
    Returns the number of categories created/updated.
    """
    if not unmatched_buffer:
        return 0

    batch_complaints = [item[0] for item in unmatched_buffer]
    batch_embeddings = [item[1] for item in unmatched_buffer]

    print("\n" + "-" * 75)
    print(f"🧠 BATCH LLM DISCOVERY: Sending {len(batch_complaints)} unmatched complaints to LLM in a SINGLE call...")
    print("-" * 75)

    decisions = llm_pass.discover_categories_batch(batch_complaints, registry)

    # Process LLM decisions
    for i, decision in enumerate(decisions):
        comp = batch_complaints[i]
        emb = batch_embeddings[i]
        cid = comp["complaint_id"]
        raw_text = comp["complaint_text"]
        room = comp["room_number"]

        action = decision.get("action")
        matched_id = decision.get("matched_category_id")
        proposed_name = decision.get("proposed_name", f"Issue in {room}")
        proposed_desc = decision.get("proposed_description", f"Complaints regarding {room}")

        assigned_cat_id = None
        method = None

        # Check if LLM matched an existing category
        if action == "MATCH" and matched_id and matched_id in registry:
            existing_room = registry[matched_id].get("room_number")
            if str(existing_room).strip().lower() == str(room).strip().lower():
                assigned_cat_id = matched_id
                method = f"🧠 LLM MATCH [{assigned_cat_id}]"
                category_manager.increment_category_count(registry, assigned_cat_id, raw_text, REGISTRY_JSON)

        # Check registry directly to avoid duplicate categories within same batch
        if not assigned_cat_id:
            direct_match = category_manager.find_category_by_room_and_issue(registry, room, proposed_name)
            if direct_match:
                assigned_cat_id = direct_match
                method = f"🧠 BATCH MATCH [{assigned_cat_id}]"
                category_manager.increment_category_count(registry, assigned_cat_id, raw_text, REGISTRY_JSON)
            else:
                # Create brand new category
                assigned_cat_id = category_manager.add_category(
                    registry=registry,
                    category_name=proposed_name,
                    room_number=room,
                    description=proposed_desc,
                    example_complaint=raw_text,
                    filepath=REGISTRY_JSON
                )
                method = f"🧠 LLM NEW CATEGORY [{assigned_cat_id}]"

        # Index vector into FAISS
        vector_store.add_complaint_vector(
            index=faiss_index,
            metadata=vector_metadata,
            embedding=emb,
            category_id=assigned_cat_id,
            room_number=room,
            complaint_text=raw_text
        )

        print(f"  • [{cid}] {method}: '{registry[assigned_cat_id]['category_name']}'")

    print("-" * 75)
    return len(decisions)


def run_pipeline():
    print("=" * 80)
    print("   AI-POWERED STUDENT COMPLAINT CLASSIFICATION SYSTEM (BUFFERED BATCH)")
    print(f"   Target Dataset:          {DATASET_CSV}")
    print(f"   Cosine Sim Threshold:    {SIMILARITY_THRESHOLD}")
    print(f"   Buffer Batch Size:       {UNMATCHED_BATCH_SIZE} (Waits for 10 misses before calling LLM)")
    print(f"   Reset Mode:              {'ENABLED (Clean slate)' if RESET_INDEX else 'DISABLED (Persistent)'}")
    print("=" * 80)

    # Reset if requested
    if RESET_INDEX:
        print("\n[Notice] '--reset' flag detected. Resetting FAISS index and category registry...")
        for stale_file in [REGISTRY_JSON, FAISS_INDEX_FILE, VECTOR_METADATA_FILE]:
            if os.path.exists(stale_file):
                try:
                    os.remove(stale_file)
                except Exception:
                    pass

    # Load persistent store
    registry = category_manager.load_registry(REGISTRY_JSON)
    faiss_index = vector_store.create_or_load_index(FAISS_INDEX_FILE)
    vector_metadata = vector_store.load_vector_metadata(VECTOR_METADATA_FILE)

    print(f"[State] Currently loaded: {len(registry)} categories, {faiss_index.ntotal} FAISS vectors.")

    # Load dataset
    print(f"\n[STAGE 1] Loading and cleaning dataset from '{DATASET_CSV}'...")
    if not os.path.exists(DATASET_CSV):
        print(f"[Error] Dataset file '{DATASET_CSV}' not found! Please check the file path.")
        return

    complaints = data_clean.load_and_clean_complaints(DATASET_CSV)
    print(f"✓ Successfully loaded and cleaned {len(complaints)} complaints.")

    if not complaints:
        print("[Error] No valid complaints found to process. Exiting.")
        return

    # -------------------------------------------------------------
    # STAGE 2: Fast FAISS-First with Buffered Batch Fallback
    # -------------------------------------------------------------
    print("\n[STAGE 2] Processing Complaints (FAISS Immediate Hit -> Buffer Misses in Batches of 10)...")
    
    faiss_hit_count = 0
    llm_batch_call_count = 0
    unmatched_buffer = []

    for idx, item in enumerate(complaints, start=1):
        cid = item["complaint_id"]
        raw_text = item["complaint_text"]
        clean_text = item["clean_text"]
        room = item["room_number"]

        # Step 2.1: Generate Embedding Vector
        query_emb = embeddings.generate_single_embedding(clean_text)

        faiss_matched = False

        # Step 2.2: Check FAISS Index FIRST
        if faiss_index.ntotal > 0:
            search_res = vector_store.search_nearest_complaint(faiss_index, vector_metadata, query_emb)
            nearest_sim = search_res["similarity"]
            nearest_cat = search_res["category_id"]
            nearest_room = search_res["room_number"]

            is_room_match = (str(nearest_room).strip().lower() == str(room).strip().lower())
            is_sim_sufficient = (nearest_sim >= SIMILARITY_THRESHOLD)

            if is_sim_sufficient and is_room_match:
                # FAST PATH: Instant FAISS Hit!
                faiss_matched = True
                faiss_hit_count += 1
                category_manager.increment_category_count(registry, nearest_cat, raw_text, REGISTRY_JSON)
                
                # Add vector to FAISS
                vector_store.add_complaint_vector(
                    index=faiss_index,
                    metadata=vector_metadata,
                    embedding=query_emb,
                    category_id=nearest_cat,
                    room_number=room,
                    complaint_text=raw_text
                )

                print(f"[{idx}/{len(complaints)}] [{cid}] ⚡ FAISS HIT ({nearest_sim:.4f} >= {SIMILARITY_THRESHOLD}): [{nearest_cat}] '{registry[nearest_cat]['category_name']}'")

        # Step 2.3: If FAISS missed, buffer the complaint
        if not faiss_matched:
            unmatched_buffer.append((item, query_emb))
            print(f"[{idx}/{len(complaints)}] [{cid}] ⏳ FAISS MISS: Buffered ({len(unmatched_buffer)}/{UNMATCHED_BATCH_SIZE}) - '{raw_text[:50]}...' ({room})")

            # When buffer reaches batch size, call LLM ONCE!
            if len(unmatched_buffer) >= UNMATCHED_BATCH_SIZE:
                llm_batch_call_count += 1
                process_unmatched_batch(unmatched_buffer, registry, faiss_index, vector_metadata)
                unmatched_buffer.clear()

    # Step 2.4: Flush any remaining unmatched complaints at the end
    if len(unmatched_buffer) > 0:
        llm_batch_call_count += 1
        print(f"\n[Flushing Buffer] Processing final {len(unmatched_buffer)} unmatched complaints with LLM...")
        process_unmatched_batch(unmatched_buffer, registry, faiss_index, vector_metadata)
        unmatched_buffer.clear()

    # Save final FAISS index & metadata
    vector_store.save_vector_store(faiss_index, vector_metadata, FAISS_INDEX_FILE, VECTOR_METADATA_FILE)

    # -------------------------------------------------------------
    # STAGE 3: Performance & Efficiency Statistics
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("                     EFFICIENCY & PERFORMANCE SUMMARY")
    print("=" * 80)
    total_processed = len(complaints)
    total_misses = total_processed - faiss_hit_count
    faiss_pct = (faiss_hit_count / total_processed * 100) if total_processed else 0

    print(f"Total Complaints Processed:        {total_processed}")
    print(f"⚡ Classified via FAISS (Fast):     {faiss_hit_count} ({faiss_pct:.1f}%) [0 LLM calls, <5ms each]")
    print(f"⏳ Total FAISS Misses:              {total_misses}")
    print(f"🧠 Total LLM Batch API Calls:      {llm_batch_call_count}  (Grouped in batches of {UNMATCHED_BATCH_SIZE}!)")
    if total_misses > 0 and llm_batch_call_count > 0:
        calls_saved = total_misses - llm_batch_call_count
        print(f"🚀 API Calls Saved by Batching:    {calls_saved} calls saved ({calls_saved/total_misses*100:.1f}% reduction in network requests!)")
    print(f"Total Categories in Registry:      {len(registry)}")
    print(f"Total Vectors in FAISS Store:      {faiss_index.ntotal}")
    print("=" * 80)

    # -------------------------------------------------------------
    # STAGE 4: Display Final Category Registry
    # -------------------------------------------------------------
    print("\n[FINAL REGISTRY STATE]:")
    category_manager.display_registry(registry)
    print("Pipeline run completed successfully!")


if __name__ == "__main__":
    run_pipeline()
