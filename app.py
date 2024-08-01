
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain.prompts import ChatPromptTemplate
from langchain.tools.render import render_text_description
from dotenv import load_dotenv
import os
import requests
from langchain_community.tools.tavily_search import TavilySearchResults

# Load environment variables
load_dotenv()

# Define input model
class PeriodCareInput(BaseModel):
    user_phase: str
    user_day: int = None
    abdominal_pain: bool
    period_flow: bool
    period_flow_type: str = None
    additional_issues: str = None

# Initialize FastAPI app
app = FastAPI()

# Initialize API wrappers and models
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)
search = TavilySearchAPIWrapper(tavily_api_key="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")
tavily = TavilySearchResults(api_wrapper=search)
pubmed = PubmedQueryRun()
tools = [pubmed, wiki, tavily]

gemini_api_key = os.environ['gemini_api_key']
rendered_tools = render_text_description(tools)

prompt_template = f""" Your name is Titans. You are a medical practitioner and specialize on questions
        regarding female menstrual health, periods, symptoms related to it, its solutions, 
        diseases related to it, and myths related to it. Answer the question as detailed as possible 
        from the given sources, make sure to provide all the details, don't provide the wrong answer to 
        things you do not know and you should not entertain any questions that are not related to female menstruation 
        , periods, symptoms related to it, its solutions, diseases related to it, and myths related to it.\n\n 
        Make sure to use only the wiki pubmed tool for information and no other sources strictly.
        Do not provide articles link but you can tell the sources wherever needed. Give a care routine during the phases of the menstrual cycle. Strictly Summarize your answers in 150 tokens.
        Here are the names and descriptions for each tool:

{rendered_tools}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompt_template),
        ("user", "{input}")
    ]
)

# Function to generate response using Gemini API
def generate_response_gemini(prompt):
    headers = {
        "Authorization": f"Bearer {gemini_api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gemini-pro",
        "prompt": prompt,
        "max_tokens": 300,
        "temperature": 0.1,
    }
    response = requests.post("https://api.gemini.com/v1/completions", headers=headers, json=data)
    if response.status_code == 200:
        return response.json().get("choices", [])[0].get("text", "")
    else:
        raise HTTPException(status_code=response.status_code, detail=response.text)

# Define the recommendation function
def period_care_recommender(input_data: PeriodCareInput):
    if input_data.user_phase == "Menstrual Phase (Day 1 to Day 7)":
        user_question = f"I'm in the Menstrual Phase, Day {input_data.user_day}. Abdominal pain: {'Yes' if input_data.abdominal_pain else 'No'}, Period flow: {input_data.period_flow_type if input_data.period_flow else 'None'}. Suggest me how should I take care of myself"
    else:
        user_question = f'I am in the {input_data.user_phase}. Suggest me how should I take care of myself based on the {input_data.user_phase}'

    if input_data.additional_issues:
        user_question += f" {input_data.additional_issues}"

    chain = prompt | (lambda input: generate_response_gemini(input))
    answer = chain.invoke({"input": user_question})
    print(answer)
    return answer

# Define the endpoint
@app.post("/recommendation")
async def get_recommendation(input_data: PeriodCareInput):
    try:
        result = period_care_recommender(input_data)
        return {"recommendation": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Main function (unnecessary in FastAPI as it uses ASGI server)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
