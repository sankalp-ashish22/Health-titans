from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from enum import Enum
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from dotenv import load_dotenv
import os
from langchain_fireworks import ChatFireworks
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.prompts import (
    ChatPromptTemplate,
)
from langchain.tools.render import render_text_description

# Load environment variables
load_dotenv()

# Retrieve API keys from environment variables
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")

class MenstrualPhase(str, Enum):
    menstrual = "Menstrual Phase (Day 1 to Day 7)"
    proliferative = "Proliferative Phase (Day 8 to Day 11)"
    ovulation = "Ovulation Phase (Day 12 to 17)"
    luteal = "Luteal Phase (Day 18 to Day 28)"

class PeriodFlowType(str, Enum):
    heavy = "Heavy"
    moderate = "Moderate"
    low = "Low"

class UserQuery(BaseModel):
    phase: MenstrualPhase
    day: int = Field(None, ge=1, le=28)
    abdominal_pain: bool = False
    period_flow: bool = False
    period_flow_type: PeriodFlowType = None
    additional_query: str = None

app = FastAPI()

def periodcarerecommender(input_text):
    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
    wiki = WikipediaQueryRun(api_wrapper=api_wrapper)
    search = TavilySearchAPIWrapper(tavily_api_key=TAVILY_API_KEY)
    tavily = TavilySearchResults(api_wrapper=search)
    pubmed = PubmedQueryRun()
    tools = [pubmed, wiki, tavily]
    
    llm = ChatFireworks(model="accounts/fireworks/models/mixtral-8x7b-instruct", api_key=FIREWORKS_API_KEY, max_tokens=300)
    rendered_tools = render_text_description(tools)

    prompt_template = f"""
        Your name is Maitri. You are a medical practitioner and specialize in questions
        regarding female menstrual health, periods, symptoms related to it, its solutions,
        diseases related to it and myths related to it. Answer the question as detailed as possible 
        from the given sources, make sure to provide all the details, don't provide the wrong answer to 
        things you do not know and you should not entertain any questions that are not related to female menstruation,
        periods, symptoms related to it, its solutions, diseases related to it and myths related to it.

        Make sure to use only the wiki pubmed tool for information and no other sources strictly.
        Do not provide article links but you can tell the sources wherever needed. Give a care routine during the phases of the menstrual cycle. Strictly summarize your answers in 150 tokens.

        Here are the names and descriptions for each tool:

        {rendered_tools}
    """
   
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_template),
            ("user", "{input}")
        ]
    )

    chain = prompt | llm 
    answer = chain.invoke({"input": input_text})
    return answer.content

@app.get("/phases")
async def get_phases():
    phases = [phase.value for phase in MenstrualPhase]
    return {"phases": phases}

@app.get("/period_flow_types")
async def get_period_flow_types():
    period_flow_types = [flow_type.value for flow_type in PeriodFlowType]
    return {"period_flow_types": period_flow_types}

@app.post("/get_suggestions")
async def get_suggestions(user_query: UserQuery):
    if user_query.phase == MenstrualPhase.menstrual:
        if user_query.day is None:
            raise HTTPException(status_code=400, detail="Day is required for the Menstrual Phase")
        period_flow_type = user_query.period_flow_type.value if user_query.period_flow else 'None'
        if user_query.day > 7 or user_query.day<0:
            return HTTPException(status_code=422, detail="Day should be between 0 and 7 for this phase")
        user_question = f"I'm in the Menstrual Phase, Day {user_query.day}. Abdominal pain: {'Yes' if user_query.abdominal_pain else 'No'}, Period flow: {period_flow_type}. Suggest me how should I take care of myself"
        if user_query.additional_query:
            user_question += f" {user_query.additional_query}"
    else:
        user_question = f"I am in the {user_query.phase}. Suggest me how should I take care of myself based on the {user_query.phase}"
        if user_query.additional_query:
            user_question += f" {user_query.additional_query}"
    
    suggestion = periodcarerecommender(user_question)
    return {"suggestion": suggestion}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)

