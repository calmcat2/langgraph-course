from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class GradeHallucinations(BaseModel):
    """Binary scores for halluciation check on generated answers."""

    binary_score: bool = Field(
        description="Documents are based on facts or references, 'true' or 'false'"
    )


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash").with_structured_output(GradeHallucinations)
system = """You are a grader assessing whether an LLM answer is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'true' or 'false'. 'false' means that the answer is grounded in / supported by the set of facts."""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Generated Content: \n\n {answer}\n\n Context: {documents}",
        ),
    ]
)
halluciation_grader_chain = prompt | llm
