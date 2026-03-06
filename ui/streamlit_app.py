import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd

from src.data_processing import load_dataset, preprocess_dataset
from src.video_generator import generate_travel_video
from moviepy.editor import ImageClip, concatenate_videoclips
from src.recommender_engine import AIDestinationRecommender
from src.itinerary_generator import generate_itinerary
from src.location_optimizer import choose_best_region
from src.chatbot import travel_chatbot, detect_trip_request
from src.pdf_generator import generate_pdf
from src.feedback_system import (
    save_feedback,
    load_feedback,
    average_rating,
    most_liked_destinations,
    interest_trends
)

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
# Navigation Tabs
# ----------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "🗺 Trip Planner",
    "🔎 Explore Destinations",
    "💬 AI Travel Assistant",
    "📊 Feedback Analytics"
])

# ====================================================
# TAB 1 — TRIP PLANNER
# ====================================================

with tab1:

    st.header("Plan Your Trip")

    # --------------------------------
    # Trip Basics
    # --------------------------------

    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input("Start Date")

    with col2:
        days = st.slider(
            "Trip Duration (days)",
            1,
            5,
            3
        )

    # --------------------------------
    # Traveler Preferences
    # --------------------------------

    col3, col4 = st.columns(2)

    with col3:
        climate = st.selectbox(
            "Preferred Climate",
            ["Any", "Tropical", "Temperate", "Cold"]
        )

    with col4:
        budget = st.selectbox(
            "Budget Level",
            ["Low", "Mid-range", "Luxury"]
        )

    col5, col6 = st.columns(2)

    with col5:
        age_group = st.selectbox(
            "Traveler Age Group",
            ["Teen", "Adult", "Family", "Senior"]
        )

    with col6:
        accessibility = st.checkbox(
            "Wheelchair Accessible Locations"
        )

    # --------------------------------
    # Travel Interests
    # --------------------------------

    st.subheader("Travel Interests")

    interests = st.multiselect(
        "Select interests",
        [
            "culture",
            "adventure",
            "nature",
            "beaches",
            "nightlife",
            "cuisine",
            "wellness",
            "urban",
            "seclusion"
        ]
    )

    # --------------------------------
    # Trip Description
    # --------------------------------

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

    # --------------------------------
    # Generate Trip Button
    # --------------------------------

    if st.button("Generate Trip Plan"):

        recommender = AIDestinationRecommender(df)

        results = recommender.recommend(combined_query)

        results, country = choose_best_region(results)

        # Save for feedback system
        st.session_state["results"] = results
        st.session_state["country"] = country
        st.session_state["query"] = combined_query
        st.session_state["interests"] = interests

        st.success(f"Recommended Country: {country}")

        st.subheader("Suggested Destinations")

        st.dataframe(results[
            ["Site Name", "city", "country"]
        ])

        itinerary = generate_itinerary(
            results,
            str(start_date),
            days
        )

        # Save itinerary
        st.session_state["itinerary"] = itinerary

        st.subheader("Your Travel Itinerary")

        for day in itinerary:

            st.write(
                f"📅 {day['date']} → {day['destination']}"
            )

        # --------------------------------
        # Generate PDF
        # --------------------------------

        pdf_path = generate_pdf(
            itinerary,
            start_date,
            days
        )

        with open(pdf_path, "rb") as file:

            st.download_button(
                label="📄 Download Travel Plan (PDF)",
                data=file,
                file_name="GlobeTrek_Travel_Plan.pdf",
                mime="application/pdf"
            )

        # --------------------------------
        # Feedback System
        # --------------------------------

if "results" in st.session_state:

    st.subheader("Rate This Travel Plan")

    rating = st.slider(
        "How useful was this recommendation?",
        1,
        5,
        3,
        key="rating_slider"
    )

    if st.button("Submit Feedback"):

        destinations = st.session_state["results"]["Site Name"].tolist()

        save_feedback(
            st.session_state["query"],
            st.session_state["country"],
            destinations,
            rating,
            st.session_state["interests"]
        )

        st.success("Thank you! Your feedback has been recorded.")

        # --------------------------------
        # Travel Video Generator
        # --------------------------------

st.subheader("Travel Recap Video")

if st.button("🎥 Generate Travel Video"):

    if "itinerary" in st.session_state:

        # DEBUG: show itinerary data
        st.write(st.session_state["itinerary"])

        video_path = generate_travel_video(
            st.session_state["itinerary"]
        )

        if video_path:

            st.success("Travel video created!")

            st.video(video_path)

        else:

            st.warning("Could not generate video.")

    else:

        st.warning("Please generate a trip first.")
       
 
# ====================================================
# TAB 2 — DESTINATION EXPLORER
# ====================================================

with tab2:

    st.header("Explore Destinations")

    search_query = st.text_input(
        "Search destinations",
        placeholder="Example: beaches in Spain"
    )

    if st.button("Find Destinations"):

        recommender = AIDestinationRecommender(df)

        results = recommender.recommend(search_query)

        results, country = choose_best_region(results)

        st.success(f"Top Region: {country}")

        st.dataframe(results[[
            "Site Name",
            "city",
            "country",
            "Type"
        ]])


# ====================================================
# TAB 3 — AI TRAVEL ASSISTANT
# ====================================================

with tab3:

   user_input = st.chat_input("Ask about travel plans...")

if user_input:

    st.chat_message("user").write(user_input)

    days_detected = detect_trip_request(user_input)

    if days_detected:

        st.chat_message("assistant").write(
            f"I'll plan a {days_detected}-day trip for you!"
        )

        recommender = AIDestinationRecommender(df)

        results = recommender.recommend(user_input)

        results, country = choose_best_region(results)

        itinerary = generate_itinerary(
            results,
            str(pd.Timestamp.today().date()),
            days_detected
        )

        for day in itinerary:

            st.write(
                f"{day['date']} → {day['destination']}"
            )

    else:

        response = travel_chatbot(user_input)

        st.chat_message("assistant").write(response)

# ====================================================
# TAB 4 — FEEDBACK ANALYTICS
# ====================================================

with tab4:

    st.header("User Feedback Analytics")

    df_feedback = load_feedback()

    if df_feedback is None or df_feedback.empty:

        st.info("No feedback data available yet.")

    else:

        # Average Rating
        avg_rating = average_rating(df_feedback)

        st.metric("Average User Rating", round(avg_rating, 2))

        # Most Liked Destinations
        st.subheader("Most Liked Destinations")

        liked_destinations = most_liked_destinations(df_feedback)

        if liked_destinations is not None:
            st.bar_chart(liked_destinations)

        # Interest Trends
        st.subheader("User Interest Trends")

        interest_data = interest_trends(df_feedback)

        if interest_data is not None:
            st.bar_chart(interest_data)

        # Debug (optional)
        st.dataframe(df_feedback)