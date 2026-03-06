from src.video_generator import generate_travel_video

# Fake itinerary for testing
itinerary = [
    {"date": "2026-03-06", "destination": "Parthenon"},
    {"date": "2026-03-07", "destination": "Acropolis"},
    {"date": "2026-03-08", "destination": "Ancient Agora"}
]

video = generate_travel_video(itinerary)

print("Video path:", video)