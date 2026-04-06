GlobeTrek AI

Smart Cultural Tourism Recommendation Site

GlobeTrek AI is an AI travel app.
It helps users:

find good places to visit
create travel plans (itinerary)
chat with an AI travel assistant
make a travel video automatically

The system uses AI technologies like:

machine learning
vector search
natural language processing
recommendation systems
multimedia tools

The app is built using:

Python
Streamlit
transformer models
FAISS
Google Gemini AI
Key Features
AI Destination Recommendation

The system suggests places based on what the user wants.
It uses semantic search and vector embeddings.

Smart Itinerary Generation

The app creates a day-by-day travel plan automatically
based on user interests.

AI Travel Assistant

A chatbot using Google Gemini API.
It helps users:

plan trips
answer travel questions
Travel Video Recap Generator

Creates a slideshow video using images of selected places.

PDF Travel Plan Export

Generates a PDF file of the travel plan.

Feedback Learning System

Collects user data like:

ratings
liked destinations
interests

It gives analytics like:

average rating
popular places
user trends
Cloud Deployment

The app runs on Streamlit Cloud.

System Architecture

The system works step by step:

User Input
Query Processing
Embedding Generation
Vector Similarity Search
Destination Recommendation
Region Optimization
Itinerary Generation
Travel Video + PDF Export
User Feedback Learning

Different modules handle each step.
Mathematical & AI Concepts
Vector Embeddings

Text is converted into vectors using transformer models.

Model used:

SentenceTransformer("all-MiniLM-L6-v2")

Example:
Input:

"historic temples in japan"

Output:

[0.23, -0.41, 0.18, 0.72, ...]

These vectors represent meaning, not just keywords.

Cosine Similarity

Used to compare vectors.

Formula:

Similarity(A,B)=(A·B)/(|A|×|B|)

Where:

A = user query vector
B = destination vector

Meaning:

1.0 → perfect match
0.7+ → strong match
0 → no match
FAISS (Vector Search Optimization)

Used for fast searching.

Process:

convert destinations to vectors
store in FAISS index
quickly find similar results
Backend Modules
Data Processing

File: src/dataprocessing.py

Libraries:

pandas
numpy

Work:

load dataset
clean data
combine text

Example:

df2["combinedfeatures"]
Embedding Model

File: src/embeddingmodel.py

Library:

sentence-transformers

Work:

convert text to vectors
Recommendation Engine

File: src/recommenderengine.py

Libraries:

scikit-learn
faiss

Steps:

convert query to vector
compare with destination vectors
rank results

Example:

cosinesimilarity(userembedding, destinationembeddings)
Location Optimizer

File: src/locationoptimizer.py

Goal:

keep recommendations in the same region

Example:

countrycounts = results["country"].value_counts()
bestcountry = countrycounts.idxmax()

Itinerary Generator

File: src/itinerarygenerator.py

Work:

remove duplicates
group by city
divide into days

Example:

Day 1 - Parthenon  
Day 2 - Acropolis Museum  
Day 3 - Ancient Agora  
Video Generator

File: src/videogenerator.py

Libraries:

OpenCV
MoviePy
ImageIO

Work:

load images
convert to frames
create video

Output:

travelvideo.mp4
Chatbot

File: src/chatbot.py

Uses:

Google Gemini API

Work:

answer questions
suggest places
create plans
PDF Generator

File: src/pdfgenerator.py

Library:

fpdf

Example:

pdf.cell(200,10,txt="Travel Itinerary",ln=True)

Output:

GlobeTrekTravelPlan.pdf
Feedback System

File: src/feedbacksystem.py

Stores:

user query
country
recommendations
rating
interests
time

File:

data/feedbacklog.csv
Analytics Dashboard

Shows:

average rating
top destinations
user interest trends
Frontend Application

File: ui/streamlitapp.py

Framework:

Streamlit

Features:

trip planner
destination explorer
chatbot
analytics
video generator

Example:

st.slider("Trip Duration",1,10,3)
Project Structure
GlobeTrekAI
│
├── data
│   ├── mastertourismdataset.csv
│   └── feedbacklog.csv
│
├── assets
│   └── destinations
│
├── src
│   ├── dataprocessing.py
│   ├── embeddingmodel.py
│   ├── recommenderengine.py
│   ├── locationoptimizer.py
│   ├── itinerarygenerator.py
│   ├── chatbot.py
│   ├── videogenerator.py
│   ├── pdfgenerator.py
│   └── feedbacksystem.py
│
├── ui
│   └── streamlitapp.py
│
├── requirements.txt
└── README.md
Technologies Used
Programming:
Python
AI/ML:
Sentence Transformers
FAISS
Scikit-Learn
Web:
Streamlit
AI API:
Google Gemini API
Multimedia:
OpenCV
MoviePy
ImageIO
Data:
Pandas
NumPy

Deployment
The app is deployed using Streamlit Cloud.

Steps:

Push code to GitHub
Connect to Streamlit Cloud
Deploy app

Dependencies are taken from:

requirements.txt
