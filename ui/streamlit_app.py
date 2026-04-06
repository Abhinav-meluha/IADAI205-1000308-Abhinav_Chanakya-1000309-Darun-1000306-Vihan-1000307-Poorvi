import sys
import os
import hashlib
import json
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import google.generativeai as genai
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from src.data_processing import load_dataset, preprocess_dataset
from src.recommender_engine import AIDestinationRecommender
from src.itinerary_generator import generate_itinerary
from src.location_optimizer import choose_best_region
import src.pdf_generator as pdf_generator
from src.gamma_exporter import export_itinerary_pdf_via_gamma, GammaAPIError
from src.feedback_system import (
    load_feedback,
    average_rating,
    most_liked_destinations,
    interest_trends,
)

try:
    from src.canva_mcp_exporter import export_itinerary_pdf_via_canva, CanvaMCPError
    CANVA_EXPORT_AVAILABLE = True
    CANVA_IMPORT_ERROR = ""
except Exception as canva_import_exc:  # noqa: BLE001
    export_itinerary_pdf_via_canva = None

    class CanvaMCPError(RuntimeError):
        pass

    CANVA_EXPORT_AVAILABLE = False
    CANVA_IMPORT_ERROR = str(canva_import_exc)

generate_pdf = pdf_generator.generate_pdf

if hasattr(pdf_generator, "generate_ai_itinerary_pdf"):
    generate_ai_itinerary_pdf = pdf_generator.generate_ai_itinerary_pdf
else:
    def generate_ai_itinerary_pdf(ai_itinerary_text, trip_details=None, file_path="travel_itinerary_ai.pdf"):
        itinerary_lines = []
        for line in str(ai_itinerary_text or "").splitlines():
            clean = line.strip()
            if clean:
                itinerary_lines.append({"destination": clean[:180]})
        if not itinerary_lines:
            itinerary_lines = [{"destination": "No AI itinerary available"}]
        trip_details = trip_details or {}
        return generate_pdf(
            itinerary_lines,
            trip_details.get("start_date", "N/A"),
            trip_details.get("days", len(itinerary_lines)),
        )

load_dotenv()


GEMINI_MODEL_OPTIONS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

INTEREST_OPTIONS = [
    "culture",
    "adventure",
    "nature",
    "beaches",
    "nightlife",
    "cuisine",
    "wellness",
    "urban",
    "seclusion",
]

TRAVEL_SYSTEM_PROMPT = (
    "You are a premium travel-planning specialist for GlobeTrek.\n"
    "Output must be formal, concise, and professional.\n"
    "Strict rules:\n"
    "- Never output placeholders like 'Not specified', 'N/A', 'null', 'unknown', or empty labels.\n"
    "- If a field is missing, omit that field entirely.\n"
    "- If a section has weak or missing data, remove that section.\n"
    "- Never include a 'General' time slot or section.\n"
    "- Avoid repetitive content.\n"
    "- Safety guidance belongs only in the 'Important Notes' section.\n"
    "- Budget must not be repeated in every activity line.\n"
    "Required output order:\n"
    "1) Trip Summary\n"
    "2) Overview (max 4-5 lines)\n"
    "3) Day-wise Itinerary\n"
    "4) Budget Summary\n"
    "5) Important Notes\n"
    "6) Packing List\n"
    "Day-wise format:\n"
    "Day X - Title\n"
    "Date\n"
    "Morning\n"
    "- Place:\n"
    "- Activity:\n"
    "- Duration:\n"
    "- Transport:\n"
    "- Food Tip:\n"
    "Afternoon\n"
    "- Place:\n"
    "- Activity:\n"
    "- Duration:\n"
    "- Transport:\n"
    "- Food Tip:\n"
    "Evening\n"
    "- Place:\n"
    "- Activity:\n"
    "- Duration:\n"
    "- Transport:\n"
    "- Food Tip:\n"
    "Night (optional)\n"
    "- Place:\n"
    "- Activity:\n"
    "- Duration:\n"
    "- Transport:\n"
    "- Food Tip:\n"
)

MISSING_TEXT_TOKENS = {
    "",
    "not specified",
    "n/a",
    "na",
    "null",
    "none",
    "unknown",
    "unspecified",
    "tbd",
    "-",
    "--",
    "---",
}


def _is_missing_text(value):
    text = str(value or "").strip().lower()
    return text in MISSING_TEXT_TOKENS


def sanitize_itinerary_text(raw_text):
    lines = str(raw_text or "").splitlines()
    clean_lines = []
    seen = set()

    for raw in lines:
        line = str(raw or "").strip()
        if not line:
            continue

        normalized = re.sub(r"\s+", " ", line).strip().lower()

        if any(token in normalized for token in ["not specified", " n/a", "null", "none", "unknown"]):
            continue
        if normalized in {"general", "general:", "- general", "section: general"}:
            continue
        if normalized in seen:
            continue

        seen.add(normalized)
        clean_lines.append(line)

    output = []
    last_compact = ""
    for line in clean_lines:
        compact = re.sub(r"[^a-z0-9]+", "", line.lower())
        if compact and compact == last_compact:
            continue
        output.append(line)
        last_compact = compact

    return "\n".join(output).strip()


def load_gemini_api_key():
    secret_key = ""
    try:
        secret_key = (
            st.secrets.get("GEMINI_API_KEY", "")
            or st.secrets.get("GOOGLE_API_KEY", "")
        )
        if not secret_key and "gemini" in st.secrets:
            gemini_block = st.secrets["gemini"]
            if "api_key" in gemini_block:
                secret_key = gemini_block["api_key"]
            elif "GEMINI_API_KEY" in gemini_block:
                secret_key = gemini_block["GEMINI_API_KEY"]
    except Exception:
        secret_key = ""

    env_key = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
    return secret_key or env_key


def load_gamma_api_key():
    secret_key = ""
    try:
        secret_key = st.secrets.get("GAMMA_API_KEY", "")
        if not secret_key and "gamma" in st.secrets:
            gamma_block = st.secrets["gamma"]
            if "api_key" in gamma_block:
                secret_key = gamma_block["api_key"]
            elif "GAMMA_API_KEY" in gamma_block:
                secret_key = gamma_block["GAMMA_API_KEY"]
    except Exception:
        secret_key = ""

    env_key = os.getenv("GAMMA_API_KEY", "")
    return secret_key or env_key


def load_canva_mcp_token():
    secret_key = ""
    try:
        secret_key = st.secrets.get("CANVA_MCP_BEARER_TOKEN", "")
        if not secret_key and "canva" in st.secrets:
            canva_block = st.secrets["canva"]
            if "mcp_bearer_token" in canva_block:
                secret_key = canva_block["mcp_bearer_token"]
            elif "CANVA_MCP_BEARER_TOKEN" in canva_block:
                secret_key = canva_block["CANVA_MCP_BEARER_TOKEN"]
    except Exception:
        secret_key = ""

    env_key = os.getenv("CANVA_MCP_BEARER_TOKEN", "")
    return secret_key or env_key


def get_trip_context_for_ai():
    if "trip_details" not in st.session_state:
        return ""

    details = st.session_state["trip_details"]
    context_lines = ["Trip Planner context:"]

    high_level_fields = [
        ("Start date", details.get("start_date", "")),
        ("Duration (days)", details.get("days", "")),
        ("Climate preference", details.get("climate", "")),
        ("Budget", details.get("budget", "")),
        ("Age group", details.get("age_group", "")),
        ("Accessibility needed", details.get("accessibility", "")),
        ("Interests", details.get("interests", "")),
        ("Free-text request", details.get("user_query", "")),
        ("Recommended country", details.get("country", "")),
    ]

    for label, value in high_level_fields:
        if not _is_missing_text(value):
            context_lines.append(f"- {label}: {value}")

    destination_lines = []
    for d in details.get("destinations", []):
        line_parts = []
        required_fields = [
            ("site", "site"),
            ("city", "city"),
            ("country", "country"),
            ("type", "type"),
        ]
        for key, label in required_fields:
            value = d.get(key, "")
            if not _is_missing_text(value):
                line_parts.append(f"{label}: {value}")

        optional_fields = [
            ("best_season", "best season"),
            ("avg_cost_usd", "avg cost usd"),
            ("budget_level", "budget level"),
            ("avg_rating", "avg rating"),
            ("unesco_site", "unesco"),
            ("climate_classification", "climate"),
            ("region", "region"),
            ("state", "state"),
        ]
        for key, label in optional_fields:
            value = d.get(key, None)
            if value is not None and not _is_missing_text(value):
                line_parts.append(f"{label}: {value}")

        if line_parts:
            destination_lines.append("- " + " | ".join(line_parts))

    if destination_lines:
        context_lines.append("- Suggested destinations:")
        context_lines.extend(destination_lines[:15])

    return "\n".join(context_lines)


def validate_gemini_key(api_key, model_name):
    genai.configure(api_key=api_key)
    models_to_try = [model_name] + [m for m in GEMINI_MODEL_OPTIONS if m != model_name]
    last_error = None
    for model in models_to_try:
        try:
            test = genai.GenerativeModel(model)
            test.generate_content("Reply with OK")
            return True, model, ""
        except Exception as e:
            last_error = e
            continue
    return False, model_name, str(last_error) if last_error else "Unknown validation error"


def build_detailed_itinerary_prompt():
    return (
        "Create a premium, structured itinerary from Trip Planner context.\n"
        "Hard constraints:\n"
        "- Never output 'Not specified', 'N/A', null, unknown, or empty labels.\n"
        "- If a field is unavailable, remove that field.\n"
        "- If a section is weak or empty, remove that section.\n"
        "- Do not use a 'General' block.\n"
        "- No duplicated bullets, repeated budget notes, or repeated safety notes.\n"
        "Required structure:\n"
        "1) Trip Summary\n"
        "2) Overview (max 4-5 lines)\n"
        "3) Day-wise Itinerary\n"
        "4) Budget Summary\n"
        "5) Important Notes\n"
        "6) Packing List\n"
        "Day-wise format for each day:\n"
        "Day X - Title\n"
        "Date\n"
        "Morning\n"
        "- Place:\n"
        "- Activity:\n"
        "- Duration:\n"
        "- Transport:\n"
        "- Food Tip:\n"
        "Afternoon\n"
        "- Place:\n"
        "- Activity:\n"
        "- Duration:\n"
        "- Transport:\n"
        "- Food Tip:\n"
        "Evening\n"
        "- Place:\n"
        "- Activity:\n"
        "- Duration:\n"
        "- Transport:\n"
        "- Food Tip:\n"
        "Night (optional)\n"
        "- Place:\n"
        "- Activity:\n"
        "- Duration:\n"
        "- Transport:\n"
        "- Food Tip:\n"
        "Use concise, professional language suitable for a luxury travel report."
    )


def ask_gemini_with_trip_context(user_prompt, history_messages=None):
    contextual_prompt = user_prompt
    trip_context = get_trip_context_for_ai()
    if trip_context:
        contextual_prompt = (
            f"{trip_context}\n\n"
            f"User request:\n{user_prompt}"
        )

    history = []
    if history_messages:
        for m in history_messages:
            role = "user" if m["role"] == "user" else "model"
            history.append({"role": role, "parts": [m["content"]]})

    try:
        genai.configure(api_key=st.session_state.gemini_key)
        model_name = st.session_state.get("gemini_active_model") or st.session_state.gemini_model
        model = genai.GenerativeModel(
            model_name,
            system_instruction=TRAVEL_SYSTEM_PROMPT,
        )
        if history:
            chat_session = model.start_chat(history=history)
            response = chat_session.send_message(contextual_prompt)
        else:
            response = model.generate_content(contextual_prompt)
        return response.text.strip() if response.text else "No response. Please try again."
    except Exception as e:
        return f"Error: {e}"


def build_local_designed_pdf_bytes(ai_itinerary_text, trip_details):
    pdf_path = generate_ai_itinerary_pdf(
        ai_itinerary_text,
        trip_details=trip_details,
        file_path="travel_itinerary_ai.pdf",
    )
    with open(pdf_path, "rb") as f:
        return f.read()


def build_canva_ready_text_with_gemini(ai_itinerary_text, trip_details):
    if not st.session_state.get("gemini_api_valid"):
        return ai_itinerary_text

    try:
        genai.configure(api_key=st.session_state.gemini_key)
        model_name = st.session_state.get("gemini_active_model") or st.session_state.gemini_model
        model = genai.GenerativeModel(model_name)
        destinations = []
        for d in trip_details.get("destinations", []):
            if isinstance(d, dict) and d.get("site"):
                destinations.append(str(d["site"]))

        prompt = (
            "Convert this travel itinerary into clean Canva-design-ready content.\n"
            "Rules:\n"
            "- Keep all schedule details faithful to the original.\n"
            "- Remove markdown symbols like **, ###, and code fences.\n"
            "- Output in structured plain text sections:\n"
            "  Cover Title\n"
            "  Trip Snapshot\n"
            "  Day-wise blocks with headings\n"
            "  Each day should contain Flight/Transfer/Hotel/Activity/Meal blocks where available.\n"
            "- Keep text concise and visually scannable for PDF layout.\n\n"
            f"Country: {trip_details.get('country', 'Not specified')}\n"
            f"Start date: {trip_details.get('start_date', 'Not specified')}\n"
            f"Duration: {trip_details.get('days', 'N/A')} days\n"
            f"Destinations: {', '.join(destinations[:10]) if destinations else 'Not specified'}\n\n"
            "Original Itinerary:\n"
            f"{ai_itinerary_text}"
        )
        response = model.generate_content(prompt)
        text = response.text.strip() if response and response.text else ""
        if text and not text.lower().startswith("error:"):
            return text
    except Exception:
        return ai_itinerary_text

    return ai_itinerary_text

# ----------------------------------------------------
# Page Config
# ----------------------------------------------------

st.set_page_config(
    page_title="GlobeTrek AI",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 GlobeTrek AI Travel Planner")


# ----------------------------------------------------
# Load Dataset
# ----------------------------------------------------

df = load_dataset()
df = preprocess_dataset(df)

# ----------------------------------------------------
# Session state for Gemini chat (Tab 3)
# ----------------------------------------------------

if "gemini_messages" not in st.session_state:
    st.session_state.gemini_messages = []
if "gemini_api_valid" not in st.session_state:
    st.session_state.gemini_api_valid = False
if "gemini_key" not in st.session_state:
    st.session_state.gemini_key = load_gemini_api_key()
if "gemini_model" not in st.session_state:
    st.session_state.gemini_model = "gemini-2.5-flash"
if "gemini_active_model" not in st.session_state:
    st.session_state.gemini_active_model = st.session_state.gemini_model
if "gemini_key_input" not in st.session_state:
    st.session_state.gemini_key_input = ""
if "gamma_api_key" not in st.session_state:
    st.session_state.gamma_api_key = load_gamma_api_key()
if "canva_mcp_token" not in st.session_state:
    st.session_state.canva_mcp_token = load_canva_mcp_token()
if "gamma_pdf_bytes" not in st.session_state:
    st.session_state.gamma_pdf_bytes = None
if "gamma_pdf_error" not in st.session_state:
    st.session_state.gamma_pdf_error = ""
if "gamma_pdf_source_hash" not in st.session_state:
    st.session_state.gamma_pdf_source_hash = ""
if "gamma_pdf_notice" not in st.session_state:
    st.session_state.gamma_pdf_notice = ""

# ----------------------------------------------------
# Navigation Tabs
# ----------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "Trip Planner",
    "Explore Destinations",
    "AI Travel Assistant",
    "Feedback Analytics"
])

# ====================================================
# TAB 1 — TRIP PLANNER
# ====================================================

with tab1:

    st.header("Plan Your Trip")

    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input("Start Date")

    with col2:
        days = st.slider("Trip Duration (days)", 1, 5, 3)

    col3, col4 = st.columns(2)

    with col3:
        climate = st.selectbox("Preferred Climate", ["Any", "Tropical", "Temperate", "Cold"])

    with col4:
        budget = st.selectbox("Budget Level", ["Low", "Mid-range", "Luxury"])

    col5, col6 = st.columns(2)

    with col5:
        age_group = st.selectbox("Traveler Age Group", ["Teen", "Adult", "Family", "Senior"])

    with col6:
        accessibility = st.checkbox("Wheelchair Accessible Locations")

    st.subheader("Travel Interests")

    interests = st.multiselect("Select interests", INTEREST_OPTIONS)

    user_query = st.text_input(
        "Describe your trip",
        placeholder="Example: cultural trip in Greece with good food"
    )

    combined_query = f"""
    Trip request: {user_query}
    Climate: {climate}
    Budget: {budget}
    Age group: {age_group}
    Accessibility: {accessibility}
    Interests: {', '.join(interests)}
    """

    if st.button("Generate Trip Plan"):

        recommender = AIDestinationRecommender(df)
        results = recommender.recommend(combined_query)
        results, country = choose_best_region(results)

        st.session_state["results"] = results
        st.session_state["country"] = country
        st.session_state["query"] = combined_query
        st.session_state["interests"] = interests

        destination_cols = [
            col for col in [
                "Site Name", "city", "state", "region", "country", "Type",
                "Best Season", "avg_cost_usd", "budget_level", "Avg Rating",
                "UNESCO Site", "climate_classification",
            ]
            if col in results.columns
        ]
        destination_records = (
            results[destination_cols]
            .head(10)
            .rename(columns={
                "Site Name": "site",
                "Type": "type",
                "Best Season": "best_season",
                "Avg Rating": "avg_rating",
                "UNESCO Site": "unesco_site",
            })
            .to_dict(orient="records")
        )
        st.session_state["trip_details"] = {
            "start_date": str(start_date),
            "days": days,
            "climate": climate,
            "budget": budget,
            "age_group": age_group,
            "accessibility": accessibility,
            "interests": ", ".join(interests) if interests else "",
            "user_query": user_query if user_query else "",
            "country": country,
            "destinations": destination_records,
        }
        itinerary = generate_itinerary(results, str(start_date), days)
        st.session_state["itinerary"] = itinerary
        st.session_state["ai_detailed_itinerary"] = ""
        st.session_state.gamma_pdf_bytes = None
        st.session_state.gamma_pdf_error = ""
        st.session_state.gamma_pdf_source_hash = ""
        st.session_state.gamma_pdf_notice = ""

    if "results" in st.session_state:

        trip_details = st.session_state.get("trip_details", {})

        st.subheader("AI Generated Itinerary")
        if not st.session_state.gemini_api_valid:
            st.info("Validate your Gemini key in the AI Travel Assistant tab, then return here.")
        else:
            if st.button("Generate Detailed AI Itinerary", key="tab1_ai_itinerary_btn"):
                detailed_prompt = build_detailed_itinerary_prompt()
                history_before = list(st.session_state.gemini_messages)
                ai_reply = ask_gemini_with_trip_context(
                    detailed_prompt,
                    history_messages=history_before
                )
                cleaned_reply = sanitize_itinerary_text(ai_reply)
                st.session_state["ai_detailed_itinerary"] = cleaned_reply
                st.session_state.gemini_messages.append({"role": "user", "content": detailed_prompt})
                st.session_state.gemini_messages.append({"role": "assistant", "content": cleaned_reply or ai_reply})

            if st.session_state.get("ai_detailed_itinerary"):
                st.markdown(st.session_state["ai_detailed_itinerary"])

        ai_itinerary_text = sanitize_itinerary_text(st.session_state.get("ai_detailed_itinerary", "").strip())
        if ai_itinerary_text and ai_itinerary_text != st.session_state.get("ai_detailed_itinerary", ""):
            st.session_state["ai_detailed_itinerary"] = ai_itinerary_text

        if ai_itinerary_text:
            with st.expander("AI Itinerary Export (Auto Designed)", expanded=True):
                gamma_api_key = st.session_state.gamma_api_key or load_gamma_api_key()
                canva_token = st.session_state.canva_mcp_token or load_canva_mcp_token()
                st.session_state.gamma_api_key = gamma_api_key
                st.session_state.canva_mcp_token = canva_token

                source_payload = {
                    "itinerary": ai_itinerary_text,
                    "trip_details": trip_details,
                }
                source_hash = hashlib.sha256(
                    json.dumps(source_payload, sort_keys=True, default=str).encode("utf-8")
                ).hexdigest()

                if source_hash != st.session_state.gamma_pdf_source_hash:
                    st.session_state.gamma_pdf_bytes = None
                    st.session_state.gamma_pdf_error = ""
                    st.session_state.gamma_pdf_source_hash = source_hash
                    st.session_state.gamma_pdf_notice = ""

                if st.session_state.gamma_pdf_error:
                    st.error(st.session_state.gamma_pdf_error)
                if st.session_state.gamma_pdf_notice:
                    st.info(st.session_state.gamma_pdf_notice)

                if st.session_state.gamma_pdf_bytes:
                    st.download_button(
                        label="Download AI Designed PDF",
                        data=st.session_state.gamma_pdf_bytes,
                        file_name="GlobeTrek_AI_Designed_Itinerary.pdf",
                        mime="application/pdf",
                        key="download_gamma_ai_pdf",
                    )
                else:
                    if st.button("Generate Designed AI PDF", key="generate_gamma_ai_pdf_btn", type="primary"):
                        st.session_state.gamma_pdf_error = ""
                        st.session_state.gamma_pdf_notice = ""
                        st.session_state.gamma_pdf_bytes = None
                        with st.spinner("Generating designed itinerary PDF..."):
                            canva_error = ""
                            gamma_error = ""

                            canva_ready_text = build_canva_ready_text_with_gemini(
                                ai_itinerary_text, trip_details,
                            )

                            if canva_token and CANVA_EXPORT_AVAILABLE:
                                try:
                                    canva_result = export_itinerary_pdf_via_canva(
                                        itinerary_text=canva_ready_text,
                                        trip_details=trip_details,
                                        canva_token=canva_token,
                                    )
                                    st.session_state.gamma_pdf_bytes = canva_result.pdf_bytes
                                    st.session_state.gamma_pdf_notice = "Designed with Canva AI + Gemini collaboration."
                                    st.rerun()
                                except Exception as e:
                                    canva_error = str(e)
                            elif canva_token and not CANVA_EXPORT_AVAILABLE:
                                canva_error = f"Canva exporter dependency missing. Details: {CANVA_IMPORT_ERROR}"
                            else:
                                canva_error = "Canva token is not configured."

                            if gamma_api_key:
                                try:
                                    gamma_result = export_itinerary_pdf_via_gamma(
                                        ai_itinerary_text, trip_details,
                                        gamma_api_key=gamma_api_key,
                                    )
                                    st.session_state.gamma_pdf_bytes = gamma_result.pdf_bytes
                                    st.session_state.gamma_pdf_notice = "Canva unavailable. Designed with Gamma AI backgrounds."
                                    st.rerun()
                                except Exception as e:
                                    gamma_error = str(e)
                            else:
                                gamma_error = "Gamma API key is not configured."

                            try:
                                st.session_state.gamma_pdf_bytes = build_local_designed_pdf_bytes(
                                    ai_itinerary_text, trip_details,
                                )
                                st.session_state.gamma_pdf_notice = ""
                                st.rerun()
                            except Exception as fallback_error:
                                st.session_state.gamma_pdf_error = (
                                    f"All export providers failed. "
                                    f"Canva: {canva_error or 'unknown'} | "
                                    f"Gamma: {gamma_error or 'unknown'} | "
                                    f"Local: {fallback_error}"
                                )
        else:
            st.info("Generate Detailed AI Itinerary first, then export as a designed PDF.")


# ====================================================
# TAB 2 - EXPLORE DESTINATIONS
# ====================================================

with tab2:

    st.header("Explore Destinations")

    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        country_options = ["All"] + sorted(df["country"].dropna().astype(str).unique().tolist())
        selected_country = st.selectbox("Country", country_options)

    with col_f2:
        type_options = ["All"] + sorted(df["Type"].dropna().astype(str).unique().tolist())
        selected_type = st.selectbox("Category", type_options)

    with col_f3:
        climate_options = ["All"]
        if "climate_classification" in df.columns:
            climate_options += sorted(df["climate_classification"].dropna().astype(str).unique().tolist())
        selected_climate = st.selectbox("Climate", climate_options)

    search_text = st.text_input("Search by place or city", placeholder="e.g. Paris, beach, temple")

    explore_df = df.copy()

    if selected_country != "All":
        explore_df = explore_df[explore_df["country"].astype(str) == selected_country]

    if selected_type != "All":
        explore_df = explore_df[explore_df["Type"].astype(str) == selected_type]

    if selected_climate != "All" and "climate_classification" in explore_df.columns:
        explore_df = explore_df[
            explore_df["climate_classification"].astype(str) == selected_climate
        ]

    if search_text.strip():
        query = search_text.strip().lower()
        explore_df = explore_df[
            explore_df["Site Name"].astype(str).str.lower().str.contains(query)
            | explore_df["city"].astype(str).str.lower().str.contains(query)
            | explore_df["country"].astype(str).str.lower().str.contains(query)
            | explore_df["Interests"].astype(str).str.lower().str.contains(query)
        ]

    st.caption(f"Showing {len(explore_df)} destinations")

    display_columns = [
        c for c in [
            "Site Name", "city", "country", "Type",
            "Best Season", "Interests", "Avg Rating", "avg_cost_usd",
        ]
        if c in explore_df.columns
    ]

    st.subheader("Recommended Places")
    for _, row in explore_df[display_columns].head(30).iterrows():
        place = row.get("Site Name", "Unknown Place")
        city = row.get("city", "N/A")
        country = row.get("country", "N/A")
        st.write(f"- {place} ({city}, {country})")


# ====================================================
# TAB 3 - AI TRAVEL ASSISTANT
# ====================================================

with tab3:

    st.header("AI Travel Assistant")

    with st.expander(
        "Gemini API Key" + (" Connected" if st.session_state.gemini_api_valid else " Not connected"),
        expanded=not st.session_state.gemini_api_valid
    ):
        st.markdown(
            "Get your free key at "
            "[aistudio.google.com](https://aistudio.google.com/app/apikey)",
        )

        key_input = st.text_input(
            "Paste your Gemini API key",
            type="password",
            placeholder="AIza...",
            key="gemini_key_input"
        )
        if st.session_state.gemini_key and not key_input:
            st.caption("Using API key from Streamlit secrets/environment. Click Validate Key to connect.")

        col_a, col_b = st.columns([1, 2])

        with col_b:
            st.selectbox("Gemini Model", GEMINI_MODEL_OPTIONS, key="gemini_model")

        with col_a:
            if st.button("Validate Key", type="primary"):
                candidate_key = key_input.strip() or st.session_state.gemini_key
                if candidate_key:
                    ok, working_model, error_message = validate_gemini_key(
                        candidate_key,
                        st.session_state.gemini_model
                    )
                    if ok:
                        st.session_state.gemini_api_valid = True
                        st.session_state.gemini_key = candidate_key
                        st.session_state.gemini_active_model = working_model
                        st.success(f"API key validated using {working_model}. You can now chat.")
                        if working_model != st.session_state.gemini_model:
                            st.info(f"Selected model was unavailable. Using {working_model} for this session.")
                        st.rerun()
                    else:
                        st.session_state.gemini_api_valid = False
                        if "API_KEY_INVALID" in error_message or "invalid" in error_message.lower():
                            st.error("Invalid API key. Please create a fresh key in Google AI Studio and try again.")
                        elif "PERMISSION_DENIED" in error_message or "permission" in error_message.lower():
                            st.error("API key was accepted but lacks permission for the selected model/project.")
                        else:
                            st.error(f"Could not validate key. Details: {error_message}")
                else:
                    st.warning("No API key found. Add it in Streamlit secrets or paste it above.")

    if st.session_state.gemini_messages:
        if st.button("Clear conversation"):
            st.session_state.gemini_messages = []
            st.rerun()

    for msg in st.session_state.gemini_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if not st.session_state.gemini_api_valid:
        st.info("Enter and validate your Gemini API key above to start chatting.")
    else:
        col_left, col_right = st.columns([3, 2])
        with col_right:
            if "trip_details" in st.session_state:
                if st.button("Build detailed itinerary from Trip Planner"):
                    auto_prompt = build_detailed_itinerary_prompt()
                    history_before = list(st.session_state.gemini_messages)
                    st.session_state.gemini_messages.append(
                        {"role": "user", "content": auto_prompt}
                    )
                    reply = ask_gemini_with_trip_context(
                        auto_prompt,
                        history_messages=history_before
                    )
                    reply = sanitize_itinerary_text(reply) or reply
                    st.session_state.gemini_messages.append(
                        {"role": "assistant", "content": reply}
                    )
                    st.rerun()
            else:
                st.caption("Generate a Trip Plan in Tab 1 to enable one-click detailed itinerary.")

        user_input = st.chat_input("Ask me anything about travel...")

        if user_input:
            history_before = list(st.session_state.gemini_messages)
            st.session_state.gemini_messages.append(
                {"role": "user", "content": user_input}
            )
            reply = ask_gemini_with_trip_context(
                user_input,
                history_messages=history_before
            )
            st.session_state.gemini_messages.append(
                {"role": "assistant", "content": reply}
            )
            st.rerun()


# ====================================================
# TAB 4 - FEEDBACK ANALYTICS
# ====================================================

with tab4:

    st.header("Feedback Analytics")

    df_feedback = load_feedback()

    if df_feedback is None or df_feedback.empty:
        st.info("No feedback data available yet.")
    else:
        avg = average_rating(df_feedback)
        st.metric("Average Rating", f"{avg:.2f}")

        st.subheader("Most Liked Destinations")
        top_destinations = most_liked_destinations(df_feedback)
        if top_destinations.empty:
            st.info("No destination preference data yet.")
        else:
            st.bar_chart(top_destinations)

        st.subheader("User Interest Trends")
        trends = interest_trends(df_feedback)
        if trends.empty:
            st.info("No interest trend data yet.")
        else:
            st.bar_chart(trends)

        with st.expander("View raw feedback data"):
            st.dataframe(df_feedback)
