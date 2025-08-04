from graph.chains.generation import generation_chain
from graph.state import GraphState


def generation(state: GraphState) -> str:
    print("---GENERATE---")
    documents = state["documents"]
    question = state["question"]

    input = {"question": question, "context": documents}
    generation = generation_chain.invoke(input)
    return {"question": question, "documents": documents, "generation": generation}
