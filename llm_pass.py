"""
llm_pass.py
Handles LLM-powered category discovery and classification.
Supports Groq (Auto-detects active models), xAI Grok, and OpenAI.
Incorporate strict room-separation rules and precise category naming.
"""

import json
import os
import re
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROK_API_KEY = os.getenv("GROK_API_KEY", "").strip()
XAI_API_KEY = os.getenv("XAI_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
CUSTOM_MODEL = os.getenv("LLM_MODEL", "").strip() or os.getenv("OPENAI_MODEL", "").strip()

_CLIENT = None
_PROVIDER = None
_MODEL = None


def get_llm_client():
    """
    Initializes and returns (client, provider_name, model_name).
    Dynamically verifies available models to prevent 404 model-not-found errors.
    """
    global _CLIENT, _PROVIDER, _MODEL
    if _CLIENT is not None:
        return _CLIENT, _PROVIDER, _MODEL

    # Check for Groq API Key
    groq_key = GROQ_API_KEY
    if not groq_key and GROK_API_KEY.startswith("gsk_"):
        groq_key = GROK_API_KEY

    if groq_key:
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)

            # Query available models on this Groq account
            try:
                available_models = [
                    m.id for m in client.models.list().data
                    if "whisper" not in m.id and "guard" not in m.id
                ]
            except Exception:
                available_models = []

            # Determine best active model
            selected_model = None
            if CUSTOM_MODEL and CUSTOM_MODEL in available_models:
                selected_model = CUSTOM_MODEL
            else:
                preferred = [
                    "openai/gpt-oss-120b",
                    "openai/gpt-oss-20b",
                    "llama-3.3-70b-versatile",
                    "llama-3.1-8b-instant",
                    "qwen/qwen3.8-27b",
                    "mixtral-8x7b-32768"
                ]
                for p in preferred:
                    if p in available_models:
                        selected_model = p
                        break
                if not selected_model and available_models:
                    selected_model = available_models[0]
                elif not selected_model:
                    selected_model = CUSTOM_MODEL or "openai/gpt-oss-120b"

            _CLIENT = client
            _PROVIDER = "Groq"
            _MODEL = selected_model
            print(f"[LLM Provider] Connected to Groq API. Active Model: {_MODEL}")
            return _CLIENT, _PROVIDER, _MODEL

        except Exception as e:
            print(f"[LLM Warning] Could not initialize Groq client: {e}")

    # Check for xAI Grok API Key
    xai_key = XAI_API_KEY or (GROK_API_KEY if not GROK_API_KEY.startswith("gsk_") else "")
    if xai_key:
        try:
            from openai import OpenAI
            _CLIENT = OpenAI(api_key=xai_key, base_url="https://api.x.ai/v1")
            _PROVIDER = "xAI Grok"
            _MODEL = CUSTOM_MODEL or "grok-beta"
            print(f"[LLM Provider] Connected to xAI Grok API. Model: {_MODEL}")
            return _CLIENT, _PROVIDER, _MODEL
        except Exception as e:
            print(f"[LLM Warning] Could not initialize xAI Grok client: {e}")

    # Check for OpenAI API Key
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            _CLIENT = OpenAI(api_key=OPENAI_API_KEY)
            _PROVIDER = "OpenAI"
            _MODEL = CUSTOM_MODEL or "gpt-4o-mini"
            print(f"[LLM Provider] Connected to OpenAI API. Model: {_MODEL}")
            return _CLIENT, _PROVIDER, _MODEL
        except Exception as e:
            print(f"[LLM Warning] Could not initialize OpenAI client: {e}")

    return None, None, None


# =====================================================================
# Precise Local Fallback (Runs if internet/API fails)
# =====================================================================

ISSUE_KEYWORDS = [
    # Power & Charging
    ("charging", "Insufficient Charging Ports"),
    ("port", "Charging Port & Socket Issue"),
    ("socket", "Electrical Socket & Plug Issue"),
    ("plug", "Electrical Plug & Power Issue"),
    ("power", "Power Supply & Electrical Issue"),
    ("switch", "Switchboard & Electrical Issue"),
    # Appliances
    ("fan", "Ceiling Fan Malfunction"),
    ("tube light", "Lighting & Tube Light Issue"),
    ("light", "Lighting Issue"),
    ("ac", "Air Conditioner & Cooling Issue"),
    ("air conditioner", "Air Conditioner & Cooling Issue"),
    ("cooler", "Water Cooler Maintenance"),
    # Classroom / Seminar
    ("digital board", "Digital Board & Smart Class Issue"),
    ("projector", "Projector & Display Issue"),
    ("board", "Blackboard / Digital Board Issue"),
    ("mic", "Microphone & Sound System Issue"),
    ("speaker", "Audio Speaker Malfunction"),
    # Plumbing
    ("tap", "Plumbing & Water Tap Leakage"),
    ("leak", "Plumbing Leakage"),
    ("flush", "Washroom Flush Malfunction"),
    ("hot water", "Washroom Hot Water Supply"),
    ("water", "Water Supply Issue"),
    # Furniture & Carpentry
    ("door", "Carpentry & Door Lock Jam"),
    ("lock", "Carpentry & Door Lock Jam"),
    ("window", "Window Glass & Latch Issue"),
    ("chair", "Hostel Furniture & Chair Maintenance"),
    ("table", "Study Table Maintenance"),
    ("bed", "Hostel Bed & Cot Repair"),
    # Campus & Common Areas
    ("wi-fi", "Hostel Internet & Wi-Fi Connectivity"),
    ("wifi", "Hostel Internet & Wi-Fi Connectivity"),
    ("internet", "Hostel Internet & Connectivity"),
    ("mess", "Hostel Mess & Dining Food Quality"),
    ("food", "Hostel Mess & Dining Food Quality"),
    ("gym", "Hostel Gym Equipment Maintenance"),
    ("treadmill", "Hostel Gym Equipment Maintenance"),
]


def _rule_based_topic_and_description(complaint_text: str, room_number: str) -> tuple[str, str]:
    text_lower = complaint_text.lower()
    selected_topic = None

    for keyword, topic_name in ISSUE_KEYWORDS:
        if keyword in text_lower:
            selected_topic = topic_name
            break

    if not selected_topic:
        # Extract meaningful words from complaint text rather than generic fallback
        words = [w.capitalize() for w in text_lower.split() if len(w) > 3 and w not in ("room", "need", "have", "with", "from", "this", "that")]
        selected_topic = " ".join(words[:3]) + " Issue" if words else "Maintenance Issue"

    if room_number and room_number != "General":
        category_name = f"{selected_topic} - Room {room_number}"
        description = f"Complaints regarding {selected_topic.lower()} specifically located in Room {room_number}."
    else:
        category_name = f"{selected_topic} (Common Facilities)"
        description = f"Complaints concerning campus or hostel wide facility: {selected_topic.lower()}."

    return category_name, description


# =====================================================================
# LLM Category Discovery (Batches of up to 10)
# =====================================================================

def discover_categories_batch(batch_complaints: list[dict], existing_registry: dict) -> list[dict]:
    """
    Processes a batch of student complaints (up to 10 complaints).
    Uses LLM to identify specific, meaningful categories and reuses existing ones when appropriate.
    Strict Rule: Complaints in different rooms must have separate categories!
    """
    client, provider, model = get_llm_client()

    if client:
        try:
            existing_summary = [
                {
                    "category_id": cid,
                    "category_name": cinfo.get("category_name"),
                    "room_number": cinfo.get("room_number"),
                    "description": cinfo.get("description")
                }
                for cid, cinfo in existing_registry.items()
            ]

            prompt = f"""You are classifying student complaints into precise, actionable issue categories.

CLASSIFICATION RULES (follow these in order):
1. Classify each complaint independently. Ignore greetings, sign-offs, urgency labels, repeated-report wording, typos, and emotional language. Use the actual issue, not the email template or course name.
2. Assign exactly one primary issue to each complaint. Name the concrete problem and affected service/equipment/process (for example, "Colab Session Disconnects", "Assignments Not Graded", "Missing Assignment Feedback", "JupyterHub Errors", "Broken Corridor Lighting").
3. LOCATION AND ISSUE ARE SEPARATE:
   - The location value "General" means no specific room was identified. It is NOT an issue category and does NOT mean two complaints are related.
   - Two complaints in "General" may be merged only when they describe the same underlying problem and would require the same kind of corrective action.
   - Same room, department, course, urgency, or wording such as "please help" is never sufficient reason to merge complaints.
4. MATCH an existing category only when BOTH conditions hold:
   a. Its room/location is the same as the complaint's room/location; AND
   b. Its specific underlying problem and needed corrective action are the same.
   Compare the category name, description, and examples. If any of those indicate a different problem, do not match it.
5. Keep distinct issues separate even when they appear in the same complaint batch. Examples that MUST be separate include:
   - Colab/cluster access vs. assignments not graded.
   - Assignments not graded vs. grades given without feedback.
   - JupyterHub failures vs. lab computer setup problems.
   - Missing lecture slides vs. short lab sessions.
   - Project mentor not replying vs. access to datasets/research papers.
   NEVER create or reuse an umbrella category such as "General Academic & Campus Grievance", "General Complaints", "Miscellaneous", or "General Maintenance".
6. If no existing category meets the strict match criteria, choose CREATE_NEW and give it a short, specific issue name. Do not broaden the name to fit unrelated complaints.
7. For a vague complaint with no identifiable issue (for example, "poor facilities" or "too many problems"), do not attach it to a specific issue category. Use the specific name "Unclear Complaint - Needs Review" so it can be reviewed separately.
8. Preserve input order and return exactly one result for every input complaint. Copy each complaint_id and room_number exactly from that input item.
9. For a specific room, format a new name as "<Specific Issue> - <Room>" (for example, "Broken Fan - Room 27"). For General, do not add a fake room suffix.

Existing Categories:
{json.dumps(existing_summary, indent=2)}

Batch of Complaints to Process:
{json.dumps(batch_complaints, indent=2)}

Return strictly valid JSON only: an array in the same order as the input, with exactly one object per complaint. Do not wrap it in Markdown:
[
  {{
    "complaint_id": "string",
    "room_number": "string",
    "action": "MATCH" or "CREATE_NEW",
    "matched_category_id": "C1" or null,
    "proposed_name": "Specific Issue Title - Room X",
    "proposed_description": "Detailed explanation of the issue"
  }}
]"""

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a precise complaint classification assistant. Always respond in valid JSON array."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )

            raw_content = response.choices[0].message.content.strip()
            if raw_content.startswith("```"):
                raw_content = re.sub(r"^```(?:json)?\s*", "", raw_content)
                raw_content = re.sub(r"\s*```$", "", raw_content)

            parsed = json.loads(raw_content)
            print(f"[LLM API] Successfully processed batch using {provider} ({model}).")
            return parsed

        except Exception as e:
            print(f"[LLM Warning] {provider or 'API'} call failed ({e}). Falling back to local category discovery logic.")

    # -----------------------------------------------------------------
    # Local Rule-based Discovery Fallback
    # -----------------------------------------------------------------
    results = []
    local_batch_cats = {}

    for item in batch_complaints:
        cid = item.get("complaint_id")
        text = item.get("complaint_text")
        room = item.get("room_number", "General")
        prop_name, prop_desc = _rule_based_topic_and_description(text, room)

        matched_id = None
        for cat_id, cat_info in existing_registry.items():
            if str(cat_info.get("room_number")).strip().lower() == str(room).strip().lower():
                existing_name = cat_info.get("category_name", "").lower()
                topic_core = prop_name.split(" - ")[0].split(" (")[0].lower()
                if topic_core in existing_name or existing_name in topic_core:
                    matched_id = cat_id
                    break

        if not matched_id:
            batch_key = f"{room}_{prop_name.lower()}"
            if batch_key in local_batch_cats:
                matched_id = local_batch_cats[batch_key]

        if matched_id:
            results.append({
                "complaint_id": cid,
                "complaint_text": text,
                "room_number": room,
                "action": "MATCH",
                "matched_category_id": matched_id,
                "proposed_name": prop_name,
                "proposed_description": prop_desc
            })
        else:
            results.append({
                "complaint_id": cid,
                "complaint_text": text,
                "room_number": room,
                "action": "CREATE_NEW",
                "matched_category_id": None,
                "proposed_name": prop_name,
                "proposed_description": prop_desc
            })

    return results


# =====================================================================
# LLM Single Complaint Classification (For new incoming complaints)
# =====================================================================

def classify_or_propose_category(complaint_text: str, room_number: str, existing_registry: dict) -> dict:
    """
    Called when FAISS similarity is below threshold or room numbers do not match.
    The LLM checks if an existing category FOR THIS EXACT ROOM matches,
    or proposes a brand new specific category.
    """
    client, provider, model = get_llm_client()

    if client:
        try:
            candidates = [
                {
                    "category_id": cid,
                    "category_name": cinfo.get("category_name"),
                    "room_number": cinfo.get("room_number"),
                    "description": cinfo.get("description")
                }
                for cid, cinfo in existing_registry.items()
            ]

            prompt = f"""You are classifying one student complaint into a precise, actionable issue category.

Student Complaint: "{complaint_text}"
Location / Room Number: "{room_number}"

Existing Categories in Registry:
{json.dumps(candidates, indent=2)}

Rules:
1. Ignore greetings, signatures, urgency labels, repeated-report wording, typos, and emotional language. Classify the actual problem.
2. Name one concrete issue and its affected service, equipment, or process. Do not use umbrella labels such as "General Academic & Campus Grievance", "General Complaints", "Miscellaneous", or "General Maintenance".
3. "General" is only a location meaning that no specific room was identified. It is not evidence that complaints concern the same issue. Same location, department, course, urgency, or wording is never enough to reuse a category.
4. Choose MATCH only when an existing category has the exact same normalized room/location AND the same specific underlying problem with the same kind of corrective action. Compare its name, description, and examples. A broad category that contains unrelated issues is not a valid match.
5. Examples that must remain separate: Colab/cluster access vs. ungraded assignments; ungraded assignments vs. grades without feedback; JupyterHub errors vs. lab computer setup; missing lecture slides vs. short lab sessions.
6. If no existing category passes both match conditions, choose CREATE_NEW with a short, specific name. For a specific room, use "<Specific Issue> - <Room>". For General, do not invent a room suffix.
7. If the complaint is too vague to identify an issue, use the separate name "Unclear Complaint - Needs Review"; never place it in an unrelated existing category.

Return strictly valid JSON only:
{{
  "action": "MATCH" or "CREATE_NEW",
  "category_id": "C..." (if MATCH, otherwise null),
  "proposed_name": "Specific Descriptive Title - Room {room_number}",
  "proposed_description": "Detailed description of the issue in this room"
}}"""

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a precise complaint classification assistant. Always respond strictly in valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )

            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)

            parsed = json.loads(raw)
            return parsed

        except Exception as e:
            print(f"[LLM Warning] {provider or 'API'} call failed ({e}). Using local fallback.")

    # Fallback logic
    prop_name, prop_desc = _rule_based_topic_and_description(complaint_text, room_number)
    for cat_id, cat_info in existing_registry.items():
        if str(cat_info.get("room_number")).strip().lower() == str(room_number).strip().lower():
            existing_name = cat_info.get("category_name", "").lower()
            topic_core = prop_name.split(" - ")[0].split(" (")[0].lower()
            if topic_core in existing_name or existing_name in topic_core:
                return {
                    "action": "MATCH",
                    "category_id": cat_id,
                    "proposed_name": cat_info.get("category_name"),
                    "proposed_description": cat_info.get("description")
                }

    return {
        "action": "CREATE_NEW",
        "category_id": None,
        "proposed_name": prop_name,
        "proposed_description": prop_desc
    }
