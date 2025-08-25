from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

class AnswerSchema(BaseModel):
    decision: Literal['web-search', 'retrieve'] = Field(
        description="decision on choice of 'web-search' or 'retrieve'"
    )

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash").with_structured_output(AnswerSchema)
system = """You are asked about a question and need to decide whether to search in the vectorstore or online.
The vectorstore has information about best practices for Tavily Crawl, Extract and Search.
If the question is not relavent to the content in the vectorstore then return 'web-search'.
If you decide to use the vectorestore then return 'retrieve'"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "question: {question}",
        ),
    ]
)
entry_point_chain = prompt | llm 