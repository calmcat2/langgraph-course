from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState
from typing import Dict


def doc_grader(state: GraphState) -> Dict:
    print("---GRADING---")
    question = state["question"]
    documents = state["documents"]

    filtered_doc = []
    web_search = False
    for doc in documents:
        print("Grading document...")
        grade = retrieval_grader.invoke({"question": question, "documents": [doc]})
        if grade.binary_score:
            print("- Pass")
            filtered_doc.append(doc)
        else:
            web_search = True
            print("- Fail")
            print("Will use web search tool to complement the results.")
    return {"question": question, "documents": filtered_doc, "web_search": web_search}
