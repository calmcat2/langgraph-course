from typing import TypedDict


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to perform a web search
        documents: list of documents
    """

    question: str
    answer: str
    web_search: bool
    documents: list[str]
    halluciation_score: bool 
    answer_score: bool
