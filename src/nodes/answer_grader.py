from chains.answer_grader_chain import answer_grader_chain
from chains.hallucination_grader_chain import halluciation_grader_chain
from models.state import GraphState

def answer_grader(state: GraphState) -> GraphState:
    print("---Grading the answer---")
    question = state["question"]
    answer = state["answer"]
    documents = state["documents"]
    halluciation_score = halluciation_grader_chain.invoke(
        {"answer": answer, "documents": documents}
    )
    if not halluciation_score.binary_score:
        answer_score = answer_grader_chain.invoke(
            {"answer": answer, "question": question}
        )
        if answer_score.binary_score:
            return {"answer_score": True, "halluciation_score": False}
        else:
            return {"answer_score": False, "halluciation_score": False}
    else:
        return {"halluciation_score": True}
    