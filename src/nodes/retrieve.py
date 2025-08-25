from models.state import GraphState
from scripts.ingestion import vectorstore


def retrieve(state: GraphState):
    print("---Retrieving answer from the knowledge base---")
    retriever = vectorstore.as_retriever()
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


if __name__ == "__main__":
    query = "What is tavily crawl best practice"
    print(retrieve({"question": query}))
