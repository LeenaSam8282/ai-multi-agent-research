import streamlit as st
import os
import requests
from dotenv import load_dotenv

from langchain_community.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_classic import hub

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

@tool
def calculator(expression: str) -> dict:
    """Evaluate math expression."""
    try:
        result = eval(expression)
        return {
            "message": f"Result = {result}",
            "total_bill": "NA",
            "status": "Completed"
        }
    except:
        return {
            "message": "Calculation Error",
            "total_bill": "NA",
            "status": "Failed"
        }

@tool
def weather(city: str) -> dict:
    """Fetch weather for city."""
    api = os.getenv("WEATHER_API_KEY")
    r = requests.get(
        f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api}"
    ).json()

    if "main" not in r:
        return {
            "message": "City not found",
            "total_bill": "NA",
            "status": "Failed"
        }

    temp = round(r["main"]["temp"] - 273.15, 2)
    desc = r["weather"][0]["description"]

    return {
        "message": f"Weather in {city}: {temp}°C, {desc}",
        "total_bill": "NA",
        "status": "Completed"
    }

prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm=llm,
    tools=[calculator, weather],
    prompt=prompt
)

agent_exec = AgentExecutor(
    agent=agent,
    tools=[calculator, weather],
    verbose=True
)

st.title("Agent with Weather + Calculator (Structured Output)")

user_input = st.text_input("Enter Input (e.g., 28*92 or Hyderabad weather):")

if st.button("Submit"):
    result = agent_exec.invoke({"input": user_input})
    st.json(result)  # important: structured output