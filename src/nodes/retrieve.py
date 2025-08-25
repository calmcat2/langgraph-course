from models.state import GraphState
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def retrieve(state: GraphState):
    print("---Retrieving answer from the knowledge base---")
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vectorstore = Chroma(
    collection_name="Langgraph_RAG",
    embedding_function=embeddings,
    persist_directory="./data/chroma_langgraph_db",
)
    retriever = vectorstore.as_retriever()
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


if __name__ == "__main__":
    query = "What is tavily crawl best practice"
    print(retrieve({"question": query}))
