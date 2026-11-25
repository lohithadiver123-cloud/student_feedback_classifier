"""
data_clean.py
Production-grade text cleaning, boilerplate stripping, and location extraction
for student complaints, emails, and feedback.
"""

import re
import pandas as pd


# Common email greetings, prefixes, and noise phrases to strip
NOISE_PREFIXES = [
    r"^(?:respected|dear)\s+(?:sir(?:/madam)?|madam|hod|all|faculty|prof(?:\.|essor)?\s*[\w\.\s]*?)\s*[:,!-]*\s*",
    r"^(?:hello|hi|hey)\s+(?:sir|all|hod|there|everyone|guys)\s*[:,!-]*\s*",
    r"^draft\s+for\s+hod\s*[:,!-]*\s*",
    r"^additional\s+comments\s*[:,!-]*\s*",
    r"^feedback\s*[:,!-]*\s*",
    r"^any\s+suggestions(?:\s+for\s+(?:the\s+)?department)?\s*[\?:,!-]*\s*",
    r"^what\s+problems\s+did\s+you\s+face\s*[\?:,!-]*\s*",
    r"^ok\s+so\s*[:,!-]*\s*",
    r"^concern\s*[:,!-]*\s*",
    r"^request\s*[:,!-]*\s*",
]

# Common email signatures, footers, and noise suffixes to strip
NOISE_SUFFIXES = [
    r"(?:sent\s+from\s+my\s+(?:iphone|android)|get\s+outlook\s+for\s+android).*",
    r"confidential:.*",
    r"contact:\s*\d+.*",
    r"mail\s+me\s+at\s+[\w\.-]+@[\w\.-]+.*",
    r"see\s+https?://\S+.*",
    r"\b(?:regards|sincerely|thanks|thank\s+you|yours\s+faithfully)\b.*",
    r"\b(?:lmk|pls\s+react\s+if\s+same|pls\s+reply|thoughts\?|sir\s+please\s+help)\b.*",
    r"\b(?:koi\s+sunta\s+hi\s+nahi|bilkul\s+bekaar\s+hai|ekdum\s+kharab\s+hai|kuch\s+kaam\s+nahi\s+kar\s+raha|yaar)\b.*",
]


def clean_text(text: str) -> str:
    """
    Cleans noisy student emails, portal tickets, and WhatsApp messages:
    - Strips HTML tags (<p>, <br>, etc.)
    - Removes priority labels ([High], [Critical], etc.)
    - Removes email greetings ("Dear Sir", "Respected HOD", etc.)
    - Removes email signatures and disclaimers ("Sent from my iPhone", etc.)
    - Normalizes typos and excessive punctuation
    """
    if pd.isna(text) or text is None:
        return ""

    t = str(text)

    # 1. Remove HTML tags
    t = re.sub(r"<[^>]+>", " ", t)

    # 2. Remove bracketed priority tags: [High], [Low], [Medium], [Critical]
    t = re.sub(r"\[\s*(?:high|low|medium|critical|urgent|medimu|loww)\s*\]", " ", t, flags=re.I)

    # 3. Remove leading greetings & prefixes (multiple passes for chained greetings)
    for _ in range(3):
        for pattern in NOISE_PREFIXES:
            t = re.sub(pattern, " ", t, flags=re.I)

    # 4. Remove footers, signatures, and disclaimers
    for pattern in NOISE_SUFFIXES:
        t = re.sub(pattern, " ", t, flags=re.I)

    # 5. Clean emojis, unreadable symbols, and multiple spaces
    t = re.sub(r"[^\w\s\.,\?!-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    # 6. If the text is purely punctuation or test string
    if len(t) < 3 or re.match(r"^[\.\?!,-]+$", t) or t.lower() in ("test", "test test", "hi", "hello"):
        return ""

    return t


def extract_room_number(text: str, default_room: str = "") -> str:
    """
    Extracts or normalizes room/location:
    - If a valid default_room is provided, use it.
    - Matches specific venues: 'Seminar Hall 2', 'AI Lab 1', 'Lab 2', 'Data Science Lab'.
    - Matches room numbers: 'Room 204', 'Room 107', 'Room 108', 'tutorial room 3'.
    - If no specific location is found, marks as 'General'.
    """
    if default_room and not pd.isna(default_room):
        clean_def = str(default_room).strip()
        if clean_def and clean_def.lower() not in ("nan", "none", "", "general"):
            return clean_def

    # Matches venues: "Seminar Hall 2", "AI Lab 1", "GPU Lab", "Data Science Lab", "Research Lab"
    match_venue = re.search(
        r"\b((?:seminar\s+hall|ai\s+lab|gpu\s+lab|data\s+science\s+lab|aiml\s+research\s+lab|tutorial\s+room)\s*#?\s*\d*)\b",
        text,
        re.IGNORECASE
    )
    if match_venue:
        venue = match_venue.group(1).strip()
        if len(venue) > 3:
            return venue.title()

    # Matches "room 204", "classroom A-204", "room B-110", "room 107"
    match_room = re.search(r"\b(?:room|classroom|cabin|block)\s*#?\s*([A-Za-z]?-?[0-9]+[a-zA-Z]?)\b", text, re.IGNORECASE)
    if match_room:
        return f"Room {match_room.group(1).upper()}"

    return "General"


def detect_columns(df: pd.DataFrame) -> tuple[str, str | None, str | None]:
    """
    Auto-detects the column names for text, room, and ID so any user dataset works.
    Returns: (text_col, room_col, id_col)
    """
    col_map = {c.lower().strip().replace(" ", "_"): c for c in df.columns}

    # Possible candidate names for the complaint text
    text_candidates = [
        "complaint_text", "complaint", "text", "review", "feedback",
        "issue", "description", "comment", "content", "student_review",
        "query", "message", "problem"
    ]
    text_col = None
    for cand in text_candidates:
        if cand in col_map:
            text_col = col_map[cand]
            break

    if not text_col:
        for c in df.columns:
            if df[c].dtype == object:
                text_col = c
                break

    if not text_col:
        raise ValueError(f"Could not find a text column in CSV. Found columns: {list(df.columns)}")

    # Possible candidate names for room number
    room_candidates = ["room_number", "room", "room_no", "roomno", "room_id", "location", "hostel_room"]
    room_col = None
    for cand in room_candidates:
        if cand in col_map:
            room_col = col_map[cand]
            break

    # Possible candidate names for ID
    id_candidates = ["complaint_id", "id", "ticket_id", "review_id", "cid", "index"]
    id_col = None
    for cand in id_candidates:
        if cand in col_map:
            id_col = col_map[cand]
            break

    return text_col, room_col, id_col


def load_and_clean_complaints(csv_path: str) -> list[dict]:
    """
    Reads complaints from any CSV file, strips noise/boilerplate, and extracts locations.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[Error] Failed to read CSV file at '{csv_path}': {e}")
        return []

    text_col, room_col, id_col = detect_columns(df)
    print(f"[Dataset] Loaded '{csv_path}' ({len(df)} rows). Auto-detected text column: '{text_col}'")

    clean_records = []
    for idx, row in df.iterrows():
        raw_text = row[text_col]
        cleaned = clean_text(raw_text)

        # Skip empty or uninformative complaints
        if not cleaned:
            continue

        raw_room = row[room_col] if room_col else ""
        room = extract_room_number(raw_text, default_room=raw_room)

        complaint_id = (
            str(row[id_col]).strip()
            if id_col and not pd.isna(row[id_col])
            else f"CMP_{idx+1:03d}"
        )

        clean_records.append({
            "complaint_id": complaint_id,
            "room_number": room,
            "complaint_text": str(raw_text).strip(),
            "clean_text": cleaned
        })

    return clean_records
