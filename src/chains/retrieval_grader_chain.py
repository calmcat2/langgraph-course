from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class GradeAnswer(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: bool = Field(
        description="Documents are relevant to the question, true or false"
    )


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash").with_structured_output(GradeAnswer)
system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'true' or 'false' to indicate whether the document is relevant to the question."""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Retrieved Content: \n\n {documents}\n\n User's Question: {question}",
        ),
    ]
)
retrieval_grader_chain = prompt | llm
