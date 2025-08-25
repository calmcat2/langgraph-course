from langchain_tavily import TavilySearch
from models.state import GraphState
from typing import Dict, List
from langchain_core.documents import Document


def websearch(state: GraphState) -> Dict[str, List[Document] | bool]:
    """
    Performs a web search using the Tavily API. Based on the state, it either
    appends the results to existing documents or overrides them.

    Args:
        state: The current graph state.

    Returns:
        A dictionary with the updated list of documents and a reset web_search flag.
    """
    print("---Performing Web Search---")
    question = state["question"]
    documents = state.get("documents", [])
    is_complementary_search = state.get("web_search", False)
    search_res = TavilySearch(max_results=3).invoke(question)
    websearch_doc = [
        Document(page_content=res["content"], metadata={"source": res.get("url")})
        for res in search_res["results"]
    ]
    if is_complementary_search:
        print(" - Appending web search results to existing documents")
        documents.append(websearch_doc)
    else:
        print(" - Overriding documents with web search results ")
        documents = websearch_doc
    return {"documents": documents, "web_search": False}
