# ===== Imports =====
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled, function_tool
from dotenv import load_dotenv
import os
from openai import AsyncOpenAI
import asyncio
import streamlit as st

# ===== Load environment variables =====
load_dotenv()
set_tracing_disabled(True)

API_KEY = os.getenv("GEMINI_API_KEY")  # DO NOT TOUCH 🔐

# ===== Configure Gemini 2.5 Flash client =====
external_client = AsyncOpenAI(
    api_key=API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client
)

# ===== Tools =====

@function_tool
def get_flights(destination: str) -> str:
    # Mock data
    return f"Flights to {destination} from your city: Flight A, Flight B, Flight C."

@function_tool
def suggest_hotels(destination: str) -> str:
    # Mock data
    return f"Recommended hotels in {destination}: Hotel X, Hotel Y, Hotel Z."

# ===== Agents =====

destination_agent = Agent(
    name="DestinationAgent",
    instructions="Suggest a travel destination based on the user's mood or interests.",
    model=model
)

booking_agent = Agent(
    name="BookingAgent",
    instructions="Provide flight and hotel options based on destination.",
    model=model,
    tools=[get_flights, suggest_hotels]
)

explore_agent = Agent(
    name="ExploreAgent",
    instructions="Suggest attractions and food places for a given destination.",
    model=model
)

# ===== Runner =====

class TravelDesignerRunner:
    def __init__(self):
        self.runner = Runner()

    async def run(self, user_input: str) -> str:
        destination_response = await self.runner.run(destination_agent, user_input)
        destination_text = self.extract_final_text(destination_response)

        destination = self.extract_destination(destination_text)

        flights_response = await self.runner.run(booking_agent, f"Flights to {destination}")
        flights_text = self.extract_final_text(flights_response)

        hotels_response = await self.runner.run(booking_agent, f"Hotels in {destination}")
        hotels_text = self.extract_final_text(hotels_response)

        explore_response = await self.runner.run(explore_agent, f"Attractions and food in {destination}")
        explore_text = self.extract_final_text(explore_response)

        return (
            f"### 🌍 Suggested Destination:\n{destination_text}\n\n"
            f"### ✈️ Flight Options:\n{flights_text}\n\n"
            f"### 🏨 Hotel Suggestions:\n{hotels_text}\n\n"
            f"### 🍽 Attractions & Food:\n{explore_text}"
        )

    def extract_destination(self, text: str) -> str:
        known_destinations = ["Paris", "Tokyo", "New York"]
        for dest in known_destinations:
            if dest.lower() in text.lower():
                return dest
        return "Paris"

    def extract_final_text(self, result) -> str:
        # Get string from result object
        raw_text = result.value if hasattr(result, "value") else str(result)

        # Remove unwanted debug/metadata lines
        cleaned_lines = []
        for line in raw_text.splitlines():
            if any(keyword in line for keyword in [
                "new item(s)",
                "raw response(s)",
                "input guardrail result(s)",
                "output guardrail result(s)",
                "RunResult:",
                "Last agent:",
                "Final output (str):"
            ]):
                continue
            cleaned_lines.append(line.strip())
        cleaned_text = "\n".join(cleaned_lines).strip()

        return cleaned_text

# ===== Streamlit UI =====

def main():
    st.title("🧳 AI Travel Designer Agent")

    user_input = st.text_input("🧠 Tell me your mood or travel interests (e.g., I want a relaxing beach vacation)")

    if user_input and st.button("🚀 Plan My Trip"):
        with st.spinner("Planning your travel experience..."):
            runner = TravelDesignerRunner()
            result = asyncio.run(runner.run(user_input.strip()))
        st.markdown(result)

if __name__ == "__main__":
    main()
