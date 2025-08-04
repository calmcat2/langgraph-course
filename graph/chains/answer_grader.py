from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class GradeAnswer(BaseModel):
    """Binary score to assess answer quality."""

    binary_score: bool = Field(
        description="The answer is useful to the user, true or false"
    )


llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro").with_structured_output(GradeAnswer)
system = """You are a grader assessing whether an LLM generation provides a good answer to the question. \n 
     Give a binary score 'true' or 'false'. True means that the answer quality is good for the question."""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "User Question: \n\n {question} \n\n LLM Generation: {generation}",
        ),
    ]
)
answer_grader = prompt | llm
