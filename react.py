from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_core.tools import tool

load_dotenv()

web_search = TavilySearch(max_results=3)

@tool
def square_root(num: float) -> float:
    """the function returns square root of a number"""
    print("======Square root tool is called======")
    return num**0.5


tools = [web_search, square_root]

llm_with_tools = ChatGoogleGenerativeAI(model="gemini-2.5-pro").bind_tools(tools)
