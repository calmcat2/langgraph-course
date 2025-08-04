from graph.state import GraphState
from ingestion import vectorstore


def retrieve(state: GraphState):
    print("---RETRIEVING---")
    retriever = vectorstore.as_retriever()
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


if __name__ == "__main__":
    query = "What is tavily crawl best practice"
    print(retriever({"question": query}))
