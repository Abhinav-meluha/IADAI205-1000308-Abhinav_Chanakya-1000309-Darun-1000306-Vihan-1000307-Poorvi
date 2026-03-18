from google import genai
import re

# Create Gemini client (NEW way)
client = genai.Client(api_key="AIzaSyCkHqAXCbpBNDDzVuvyl71WvZPAxmzusL0")


# -----------------------------
# Detect trip request
# -----------------------------
def detect_trip_request(text):
    match = re.search(r"(\d+)\s*day", text.lower())
    if match:
        return int(match.group(1))
    return None


# -----------------------------
# Detect greetings
# -----------------------------
def detect_greeting(text):
    greetings = ["hi", "hello", "hey", "good morning", "good evening"]
    text = text.lower()
    return any(greet in text for greet in greetings)


# -----------------------------
# Main travel chatbot
# -----------------------------
def travel_chatbot(user_input):

    # Greeting response
    if detect_greeting(user_input):
        return (
            "Hello! 👋 I'm your AI Travel Assistant.\n\n"
            "I can help you:\n"
            "• Plan travel itineraries\n"
            "• Discover destinations\n"
            "• Learn about tourist sites\n"
            "• Get travel recommendations\n\n"
            "Try asking something like:\n"
            "“Plan a 5 day trip to Bali.”"
        )

    # Detect number of days (optional feature)
    days = detect_trip_request(user_input)

    # Smart prompt
    prompt = f"""
    You are an AI travel assistant for a tourism platform.

    Give clear, structured, and useful travel responses.

    If the user asks for a trip plan:
    - Include day-wise itinerary
    - Mention key attractions
    - Keep it realistic and helpful

    User request:
    {user_input}
    """

    # Generate response (NEW API)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )

    return response.text


# -----------------------------
# Run chatbot loop
# -----------------------------
if __name__ == "__main__":
    print("🌍 Globtrek AI Travel Chatbot Started (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit"]:
            print("Bot: Safe travels! ✈️🌴")
            break

        reply = travel_chatbot(user_input)
        print("\nBot:", reply, "\n")
