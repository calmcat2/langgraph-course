from dotenv import load_dotenv
from langchain_core.output_parsers.string import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import hub

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
prompt = hub.pull("rlm/rag-prompt")
answer_chain = prompt | llm | StrOutputParser()
