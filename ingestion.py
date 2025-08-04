from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import asyncio
import hashlib

load_dotenv()

URLS = [
    "https://docs.tavily.com/documentation/best-practices/best-practices-crawl#7-semantic-search-or-rag-integration",
    "https://docs.tavily.com/documentation/best-practices/best-practices-extract",
    "https://docs.tavily.com/documentation/best-practices/best-practices-search",
]

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vectorstore = Chroma(
    collection_name="Langgraph_RAG",
    embedding_function=embeddings,
    persist_directory="./chroma_langgraph_db",
)


async def ingestion():
    print("Starting document ingestion process.")

    try:
        loader = WebBaseLoader(URLS)
        docs = [doc async for doc in loader.alazy_load()]
        print(f"Successfully loaded {len(docs)} from {len(URLS)}.")

        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250, chunk_overlap=20
        )
        split_docs = splitter.split_documents(docs)
        print(f"Split documents into {len(split_docs)} chunks.")

        uuids = [
            hashlib.sha256(
                f"{doc.metadata.get('source','')}-{doc.page_content}".encode("utf-8")
            ).hexdigest()
            for doc in split_docs
        ]
        vectorstore.add_documents(documents=split_docs, ids=uuids)
        print(f"Successfully loaded {len(split_docs)} into Chromadb")

    except Exception as e:
        print(f"An error occured: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(ingestion())
