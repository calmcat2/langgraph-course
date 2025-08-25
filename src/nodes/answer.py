from chains.answer_chain import answer_chain
from models.state import GraphState


def answer(state: GraphState) -> str:
    print("---Generating an answer---")
    documents = state["documents"]
    question = state["question"]

    input = {"question": question, "context": documents}
    answer = answer_chain.invoke(input)
    return {"question": question, "documents": documents, "answer": answer}
