"""
app.py
Executive HOD Grievance & Academic Issue Intelligence Dashboard
Designed for Department Heads (HOD), Deans, and Academic Coordinators.

Run Command:
  python -m streamlit run app.py
"""

import json
import os
import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

import category_manager
import data_clean
import embeddings
import llm_pass
import vector_store

load_dotenv()
REGISTRY_JSON = "category_registry.json"
FAISS_INDEX_FILE = "faiss_index.bin"
VECTOR_METADATA_FILE = "vector_metadata.json"
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.80"))

# Page Setup
st.set_page_config(
    page_title="HOD Grievance & Academic Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        border-top: 4px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        font-weight: 600;
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# Academic Taxonomy & Pillar Classification
# =====================================================================

PILLAR_MAP = {
    "Computing & Lab Infrastructure": {
        "icon": "💻",
        "keywords": ["gpu", "cuda", "vram", "workstation", "pip", "package", "ram", "hang", "restart", "pc", "computer", "boot", "socket", "power", "plug", "extension", "timing", "close", "server", "jupyter"],
        "dept": "IT Systems & Hardware Lab Administrator"
    },
    "Examinations & Grading": {
        "icon": "📝",
        "keywords": ["quiz", "marks", "grade", "grading", "internal", "test", "mid-sem", "viva", "record", "coding style", "rubric", "revaluation", "result", "attendance portal", "absent"],
        "dept": "Department Examination & Evaluation Committee"
    },
    "Curriculum & Teaching Quality": {
        "icon": "📚",
        "keywords": ["theory", "hands-on", "coding example", "derivation", "lecture", "slide", "skipped", "question bank", "syllabus", "elective", "llm", "prompt", "workshop", "course"],
        "dept": "Academic Curriculum & Faculty Coordinator"
    },
    "Projects & Research Mentorship": {
        "icon": "👥",
        "keywords": ["guide", "mentor", "travelling", "meet", "co-guide", "review", "slot", "postpone", "project", "research", "credits", "deadline"],
        "dept": "Project Review & Mentorship Committee"
    },
    "Classroom & Campus Facilities": {
        "icon": "🏢",
        "keywords": ["projector", "dim", "flicker", "mic", "microphone", "speaker", "audio", "sound", "water", "dispenser", "cooler", "chair", "bench", "seat", "whiteboard", "fan", "ac", "air conditioner", "light", "washroom", "staircase", "pothole", "bus", "canteen", "lift"],
        "dept": "Campus Maintenance & Estate Operations"
    }
}


def classify_pillar_and_severity(name: str, desc: str, count: int) -> tuple[str, str, str, int]:
    """
    Maps category into (Pillar, Severity, Responsible_Dept, Weight)
    """
    combined = f"{name} {desc}".lower()

    # 1. Determine Academic Pillar
    assigned_pillar = "Classroom & Campus Facilities"
    assigned_dept = "Campus Maintenance & Estate Operations"

    for pillar, info in PILLAR_MAP.items():
        if any(kw in combined for kw in info["keywords"]):
            assigned_pillar = pillar
            assigned_dept = info["dept"]
            break

    # 2. Determine Severity
    critical_terms = [
        "slippery", "hazard", "pothole", "gpu server", "server access",
        "restart", "attendance portal", "fire", "pipe", "medical",
        "absent even though", "booking system", "conflict: overlapping", "staircase",
        "blocking my credits", "unable to log in", "500 error"
    ]
    high_terms = [
        "wi-fi", "wifi", "internet", "projector", "dim", "software",
        "boot", "air conditioner", "cooling", "drinking water",
        "power socket", "outlets", "bandwidth", "ram", "cuda", "gpu",
        "marks", "quiz", "revaluation", "rubric", "guide", "mentor", "pip"
    ]
    medium_terms = [
        "fan", "noise", "rattling", "soap", "speaker", "sound",
        "lights", "corridor", "chair", "bench", "whiteboard", "heat", "printer", "timing"
    ]

    severity = "Low"
    weight = 1

    if any(term in combined for term in critical_terms):
        severity = "Critical"
        weight = 4
    elif any(term in combined for term in high_terms):
        severity = "High"
        weight = 3
    elif any(term in combined for term in medium_terms):
        severity = "Medium"
        weight = 2

    # High frequency escalation
    if count >= 10 and weight < 3:
        severity = "High"
        weight = 3
    elif count >= 25:
        severity = "Critical"
        weight = 4

    return assigned_pillar, severity, assigned_dept, weight


@st.cache_data(ttl=5)
def load_dashboard_data():
    """Transforms raw categories into structured, executive-level analytics DataFrame."""
    registry = category_manager.load_registry(REGISTRY_JSON)
    if not registry:
        return pd.DataFrame(), {}

    rows = []
    for cat_id, info in registry.items():
        name = info.get("category_name", "Unknown")
        room = info.get("room_number", "General")
        desc = info.get("description", "")
        count = info.get("complaint_count", 0)
        examples = info.get("example_complaints", [])

        # Clean display name from raw boilerplate
        clean_name = re.sub(r"^(?:\[.*?\]|<p>|dear\s+hod,?\s*|hello\s+sir,?\s*|draft\s+for\s+hod:?\s*)", "", name, flags=re.I).strip()
        clean_name = re.sub(r"\s+\(Common Facilities\)$", "", clean_name).strip()

        pillar, severity, dept, weight = classify_pillar_and_severity(clean_name, desc, count)
        impact = count * weight

        rows.append({
            "Category ID": cat_id,
            "Grievance Category": clean_name,
            "Academic Pillar": pillar,
            "Location / Room": room,
            "Severity": severity,
            "Severity Weight": weight,
            "Reported Complaints": count,
            "HOD Impact Score": impact,
            "Action Authority": dept,
            "Executive Summary": desc,
            "Sample Student Quotes": " • ".join(examples[:2])
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["HOD Impact Score", "Reported Complaints"], ascending=[False, False])
    return df, registry


# =====================================================================
# Main Application View
# =====================================================================

def main():
    st.title("🎓 Department Grievance & Academic Intelligence")
    st.markdown("**AIML Department Executive Command Center** • *Categorized for Immediate HOD Resolution*")
    st.markdown("---")

    df, registry = load_dashboard_data()

    if df.empty:
        st.warning("⚠️ No complaints data found in registry! Please run `python main.py` first.")
        return

    # -------------------------------------------------------------
    # Sidebar Filters
    # -------------------------------------------------------------
    st.sidebar.header("🔍 Executive Filters")

    # Pillar filter
    pillar_options = ["All Pillars"] + list(PILLAR_MAP.keys())
    selected_pillar = st.sidebar.selectbox("Filter by Domain / Pillar:", pillar_options, index=0)

    # Severity filter
    severity_options = ["All Severities", "Critical", "High", "Medium", "Low"]
    selected_severity = st.sidebar.selectbox("Filter by Urgency Level:", severity_options, index=0)

    # Location filter
    location_list = ["All Locations"] + sorted(list(df["Location / Room"].unique()))
    selected_location = st.sidebar.selectbox("Filter by Venue / Lab:", location_list, index=0)

    # Search filter
    search_kw = st.sidebar.text_input("Search Grievance Keyword:", placeholder="e.g. GPU, Marks, Projector, Guide").strip()

    st.sidebar.markdown("---")
    st.sidebar.info("💡 **HOD Impact Score** = `Complaint Frequency` × `Urgency Weight`. Higher score = Fix First.")

    # Apply Filters
    filtered_df = df.copy()
    if selected_pillar != "All Pillars":
        filtered_df = filtered_df[filtered_df["Academic Pillar"] == selected_pillar]
    if selected_severity != "All Severities":
        filtered_df = filtered_df[filtered_df["Severity"] == selected_severity]
    if selected_location != "All Locations":
        filtered_df = filtered_df[filtered_df["Location / Room"] == selected_location]
    if search_kw:
        filtered_df = filtered_df[
            filtered_df["Grievance Category"].str.contains(search_kw, case=False, na=False) |
            filtered_df["Executive Summary"].str.contains(search_kw, case=False, na=False) |
            filtered_df["Sample Student Quotes"].str.contains(search_kw, case=False, na=False)
        ]

    # -------------------------------------------------------------
    # Top KPI Summary Cards
    # -------------------------------------------------------------
    total_complaints = df["Reported Complaints"].sum()
    total_categories = len(df)
    critical_issues = len(df[df["Severity"] == "Critical"])
    high_issues = len(df[df["Severity"] == "High"])

    top_pillar = df.groupby("Academic Pillar")["Reported Complaints"].sum().idxmax()
    top_pillar_count = df.groupby("Academic Pillar")["Reported Complaints"].sum().max()

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric(label="Total Student Grievances Logged", value=f"{total_complaints:,}")
    with kpi2:
        st.metric(label="Actionable Problem Categories", value=int(total_categories))
    with kpi3:
        st.metric(
            label="Critical Showstoppers",
            value=f"{critical_issues} Issues",
            delta=f"{critical_issues + high_issues} Total High/Critical",
            delta_color="inverse"
        )
    with kpi4:
        st.metric(
            label="Highest Complaint Area",
            value=top_pillar.split("&")[0].strip(),
            delta=f"{top_pillar_count} complaints"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # Academic Pillar Quick-Overview Cards
    # -------------------------------------------------------------
    st.subheader("🏛️ Department Grievance Distribution Across 5 Academic Pillars")
    pillar_cols = st.columns(5)

    for i, (p_name, p_info) in enumerate(PILLAR_MAP.items()):
        p_df = df[df["Academic Pillar"] == p_name]
        p_count = p_df["Reported Complaints"].sum()
        p_issues = len(p_df)
        with pillar_cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:24px;">{p_info['icon']}</div>
                <div style="font-weight:700; font-size:14px; margin-top:5px;">{p_name}</div>
                <div style="font-size:22px; font-weight:800; color:#1f77b4; margin-top:5px;">{p_count} <span style="font-size:12px; font-weight:400; color:#666;">complaints</span></div>
                <div style="font-size:12px; color:#888;">{p_issues} distinct categories</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # Dashboard Tabs
    # -------------------------------------------------------------
    tab_rank, tab_table, tab_test = st.tabs([
        "🔥 Priority Action Queue (What HOD Must Fix First)",
        "📋 Full Department Action Matrix & CSV Export",
        "🧪 Live Real-time Grievance Classifier"
    ])

    # -------------------------------------------------------------
    # TAB 1: Priority Queue & Visual Analytics
    # -------------------------------------------------------------
    with tab_rank:
        st.subheader("🚨 Top 10 Highest Impact Issues for HOD Intervention")
        top10 = df.head(10).sort_values(by="HOD Impact Score", ascending=True)

        color_map = {
            "Critical": "#d62728",
            "High": "#ff7f0e",
            "Medium": "#1f77b4",
            "Low": "#2ca02c"
        }

        col_chart1, col_chart2 = st.columns([3, 2])
        with col_chart1:
            fig_bar = px.bar(
                top10,
                x="HOD Impact Score",
                y="Grievance Category",
                color="Severity",
                color_discrete_map=color_map,
                orientation="h",
                text="Reported Complaints",
                hover_data=["Academic Pillar", "Location / Room", "Action Authority"],
                labels={"HOD Impact Score": "Impact Score (Volume × Urgency)", "Grievance Category": "Identified Problem"}
            )
            fig_bar.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_chart2:
            st.markdown("**Severity Distribution**")
            sev_df = df["Severity"].value_counts().reset_index()
            sev_df.columns = ["Severity", "Count"]

            fig_donut = px.pie(
                sev_df,
                values="Count",
                names="Severity",
                color="Severity",
                color_discrete_map=color_map,
                hole=0.45
            )
            fig_donut.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_donut, use_container_width=True)

        # Hotspots by Venue / Lab
        st.subheader("📍 Infrastructure Hotspots (Where are complaints concentrated?)")
        venue_df = df[df["Location / Room"] != "General"].groupby("Location / Room")["Reported Complaints"].sum().reset_index()
        venue_df = venue_df.sort_values(by="Reported Complaints", ascending=False).head(10)

        if not venue_df.empty:
            fig_loc = px.bar(
                venue_df,
                x="Location / Room",
                y="Reported Complaints",
                color="Reported Complaints",
                color_continuous_scale="Blues",
                labels={"Reported Complaints": "Complaints", "Location / Room": "Lab / Room"}
            )
            fig_loc.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_loc, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 2: Full Grievance Registry & Export
    # -------------------------------------------------------------
    with tab_table:
        st.subheader(f"📋 Action Matrix ({len(filtered_df)} Categories Displayed)")

        csv_export = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Official HOD Action Report (CSV)",
            data=csv_export,
            file_name="AIML_HOD_Grievance_Action_Report.csv",
            mime="text/csv",
            help="Exports filtered problem list with action authority to send to committees and faculty."
        )

        show_cols = [
            "Category ID", "Grievance Category", "Academic Pillar",
            "Location / Room", "Severity", "Reported Complaints",
            "HOD Impact Score", "Action Authority", "Executive Summary", "Sample Student Quotes"
        ]

        st.dataframe(
            filtered_df[show_cols],
            use_container_width=True,
            column_config={
                "Reported Complaints": st.column_config.NumberColumn("Volume", format="%d"),
                "HOD Impact Score": st.column_config.ProgressColumn(
                    "Impact",
                    min_value=0,
                    max_value=int(df["HOD Impact Score"].max() or 10),
                    format="%d"
                ),
                "Severity": st.column_config.TextColumn("Urgency"),
                "Executive Summary": st.column_config.TextColumn("Brief Description", width="large")
            },
            hide_index=True
        )

    # -------------------------------------------------------------
    # TAB 3: Live Classifier
    # -------------------------------------------------------------
    with tab_test:
        st.subheader("🧪 Live AI Grievance Classifier")
        st.markdown("Test how new incoming student feedback is parsed, stripped of boilerplate, and mapped to executive categories.")

        test_input = st.text_area(
            "Enter Raw Student Complaint / Email / Ticket:",
            placeholder="e.g. Respected HOD, The GPU server runs out of memory whenever we fine-tune BERT models in AI Lab 1. Please help. Regards, Student"
        )

        if st.button("Classify Grievance", type="primary"):
            if not test_input.strip():
                st.warning("Please enter some text!")
            else:
                with st.spinner("Analyzing semantics..."):
                    cleaned = data_clean.clean_text(test_input)
                    room = data_clean.extract_room_number(test_input)

                    faiss_index = vector_store.create_or_load_index(FAISS_INDEX_FILE)
                    vector_metadata = vector_store.load_vector_metadata(VECTOR_METADATA_FILE)
                    query_emb = embeddings.generate_single_embedding(cleaned)

                    search_res = vector_store.search_nearest_complaint(faiss_index, vector_metadata, query_emb)
                    is_match = False
                    sim = 0.0

                    if search_res:
                        sim = search_res["similarity"]
                        near_cat = search_res["category_id"]
                        near_room = search_res["room_number"]
                        is_room_match = (str(near_room).strip().lower() == str(room).strip().lower())

                        if sim >= SIMILARITY_THRESHOLD and is_room_match:
                            is_match = True
                            assigned_id = near_cat
                            cat_name = registry.get(assigned_id, {}).get("category_name", "Unknown")
                            method = f"⚡ FAISS Immediate Match (Cosine Sim {sim:.4f} >= {SIMILARITY_THRESHOLD})"

                    if not is_match:
                        llm_res = llm_pass.classify_or_propose_category(cleaned, room, registry)
                        if llm_res.get("action") == "MATCH" and llm_res.get("category_id") in registry:
                            assigned_id = llm_res["category_id"]
                            cat_name = registry[assigned_id]["category_name"]
                            method = f"🧠 LLM Match (Existing Category for {room})"
                        else:
                            cat_name = llm_res.get("proposed_name", f"Issue in {room}")
                            assigned_id = "NEW_CAT"
                            method = f"🧠 LLM New Category Discovered for {room}"

                    st.success("Analysis Complete!")
                    res1, res2 = st.columns(2)
                    with res1:
                        st.markdown(f"**Cleaned Core Grievance:** `{cleaned}`")
                        st.markdown(f"**Identified Location:** `{room}`")
                    with res2:
                        st.markdown(f"**Assigned Category:** `{cat_name}`")
                        st.caption(f"Method: {method}")


if __name__ == "__main__":
    main()
