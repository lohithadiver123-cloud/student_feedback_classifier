"""
category_manager.py
Manages category IDs, metadata, JSON persistence, and complaint counts.
Python manages category IDs and counts (the LLM does not).
"""

import json
import os


REGISTRY_FILE = "category_registry.json"


def load_registry(filepath: str = REGISTRY_FILE) -> dict:
    """
    Loads category registry from a JSON file.
    If the file does not exist, returns an empty dictionary.
    """
    if not os.path.exists(filepath):
        return {}

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Warning] Could not load registry from {filepath}: {e}. Starting fresh.")
        return {}


def save_registry(registry: dict, filepath: str = REGISTRY_FILE) -> None:
    """
    Saves the category registry to a JSON file.
    """
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[Error] Failed to save category registry to {filepath}: {e}")


def get_next_category_id(registry: dict) -> str:
    """
    Generates a unique category ID like C1, C2, C3.
    Extracts the highest existing number and increments it by 1.
    """
    highest = 0
    for cat_id in registry.keys():
        if cat_id.startswith("C") and cat_id[1:].isdigit():
            val = int(cat_id[1:])
            if val > highest:
                highest = val
    return f"C{highest + 1}"


def find_category_by_room_and_issue(registry: dict, room_number: str, category_name: str) -> str | None:
    """
    Checks if an existing category in the registry has the same room number
    and a matching category name / topic.
    Returns the category_id if found, or None.
    """
    clean_room = str(room_number).strip().lower()
    clean_name = str(category_name).strip().lower()

    for cat_id, cat_info in registry.items():
        existing_room = str(cat_info.get("room_number", "")).strip().lower()
        existing_name = str(cat_info.get("category_name", "")).strip().lower()

        # Both the room and the category topic must match!
        if clean_room == existing_room and (clean_name in existing_name or existing_name in clean_name):
            return cat_id
    return None


def add_category(
    registry: dict,
    category_name: str,
    room_number: str,
    description: str,
    example_complaint: str,
    filepath: str = REGISTRY_FILE
) -> str:
    """
    Creates and records a new category in the registry:
    - Assigns next unique category ID (e.g. C1, C2)
    - Sets complaint_count to 1
    - Adds the initial example complaint
    - Saves the updated registry to JSON
    Returns the newly assigned category_id.
    """
    cat_id = get_next_category_id(registry)

    # Format the category name nicely with the room if applicable
    formatted_name = category_name.strip()
    if room_number and room_number != "General" and room_number not in formatted_name:
        formatted_name = f"{formatted_name} (Room {room_number})"

    registry[cat_id] = {
        "category_id": cat_id,
        "category_name": formatted_name,
        "room_number": str(room_number).strip(),
        "description": description.strip(),
        "complaint_count": 1,
        "example_complaints": [example_complaint.strip()] if example_complaint else []
    }

    save_registry(registry, filepath)
    return cat_id


def increment_category_count(
    registry: dict,
    category_id: str,
    new_example: str = None,
    filepath: str = REGISTRY_FILE
) -> None:
    """
    Increments the complaint count for an existing category.
    Optionally appends a new example complaint (keeps up to 5 examples).
    Saves the updated registry to JSON.
    """
    if category_id not in registry:
        print(f"[Warning] Category ID {category_id} not found in registry.")
        return

    registry[category_id]["complaint_count"] += 1

    if new_example:
        examples = registry[category_id].get("example_complaints", [])
        if new_example not in examples:
            if len(examples) < 5:
                examples.append(new_example.strip())
            registry[category_id]["example_complaints"] = examples

    save_registry(registry, filepath)


def display_registry(registry: dict) -> None:
    """
    Prints a formatted, beginner-friendly view of the category registry.
    """
    print("\n" + "=" * 80)
    print("                      CATEGORY REGISTRY SUMMARY")
    print("=" * 80)
    if not registry:
        print("Registry is currently empty.")
        print("=" * 80 + "\n")
        return

    for cat_id, info in registry.items():
        print(f"[{cat_id}] {info.get('category_name')}")
        print(f"    Room:        {info.get('room_number', 'N/A')}")
        print(f"    Total Count: {info.get('complaint_count', 0)}")
        print(f"    Description: {info.get('description', '')}")
        examples = info.get("example_complaints", [])
        if examples:
            print("    Examples:")
            for ex in examples[:3]:
                print(f"      - {ex}")
        print("-" * 80)
    print("=" * 80 + "\n")
