from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Backend is running 👍"}

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any, List
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_classic import hub

load_dotenv()

app = FastAPI()

class Query(BaseModel):
    query: str


# ------------------ LLM ------------------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


# ------------------ TOOLS ------------------
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
web = DuckDuckGoSearchRun()
arxiv = ArxivQueryRun()

tools = [wiki, web, arxiv]


# ------------------ PROMPT ------------------
prompt = hub.pull("hwchase17/react")


# ------------------ RESEARCH AGENT ------------------
research_agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

# IMPORTANT FIX → allow Gemini to return normal text
research_exec = AgentExecutor(
    agent=research_agent,
    tools=tools,
    verbose=False,
    handle_parsing_errors=True
)


# ------------------ SUMMARIZER ------------------
def summarizer_agent(content: str) -> str:
    prompt = f"""
Act as a professional research summarization agent.

Using the content below, generate a structured academic summary with the following sections:

1. Topic Overview
2. Background Context
3. Key Mechanisms / Core Concepts
4. Use-cases & Applications
5. Advantages
6. Challenges / Limitations
7. Future Scope
8. Conclusion

Content to summarize:
\"\"\"{content}\"\"\"

Ensure:
- correctness
- detail
- clarity
- technical depth
- no fluff
- no repetition
- no hallucination
"""
    response = llm.invoke(prompt)
    return response.content


# ------------------ EMAIL AGENT ------------------
def email_agent(content: str) -> str:
    first_line = content.split(".")[0][:60]
    subject = f"Subject: Overview on {first_line}"

    return f"""
{subject}

Dear Sir/Madam,

{content}

Regards,
Multi-Agent Research System
"""


# ------------------ ARXIV SMART TRIM ------------------
def trim_arxiv_output(raw_text: str) -> str:
    lines = raw_text.split("\n")
    papers = []
    for ln in lines:
        if ln.strip():
            papers.append(ln.strip())
        if len(papers) >= 5:
            break
    return "\n".join(papers)


# ------------------ MAIN PIPELINE (NO INTENTS) ------------------
@app.post("/run")
async def run_pipeline(payload: Query) -> Dict[str, Any]:

    research_block = None
    summary_block = None
    email_block = None
    raw_output = None

    # ---------- STEP 1: RESEARCH ----------
    try:
        result = research_exec.invoke({"input": payload.query})
        raw_output = result.get("output", payload.query)

        arxiv_trim = trim_arxiv_output(raw_output) if "arxiv" in raw_output.lower() else None

        research_block = {
            "wiki_web": raw_output,
            "arxiv": arxiv_trim
        }

    except Exception as e:
        raw_output = payload.query
        research_block = {"wiki_web": payload.query, "arxiv": None}


    # ---------- STEP 2: SUMMARY ----------
    summary_block = summarizer_agent(raw_output)


    # ---------- STEP 3: EMAIL ----------
    email_block = email_agent(summary_block)


    # ---------- FINAL RETURN (ALWAYS dict) ----------
    return {
        "research": research_block or None,
        "summary": summary_block or None,
        "email": email_block or None
    }