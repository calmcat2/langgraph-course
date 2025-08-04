from langchain_tavily import TavilySearch
from graph.state import GraphState
from typing import Dict
from langchain_core.documents import Document


def websearch(state: GraphState) -> Dict:
    websearch_tool = TavilySearch(max=3)
    question = state["question"]
    documents = state.get("documents",[])
    search_res = websearch_tool.invoke(question)
    contents = "\n".join([res["content"] for res in search_res["results"]])
    new_doc = Document(page_content=contents)
    if documents is not None:
        documents.append(new_doc)
    else:
        documents = [new_doc]
    return {"documents": documents, "question": question}
