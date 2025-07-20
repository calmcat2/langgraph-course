from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()

@tool
def double(num: int):
    """
    Description: this function doubles the input value

    Input: integer that needs to be doubled
    Output: integer result
    """
    return num*2

tools = [double, TavilySearch()]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
llm_with_tools = llm.bind_tools(tools=tools)
