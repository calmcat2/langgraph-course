from graph.graph import app

if __name__ == "__main__":
    print("Start RAG...")
    # query = "Difference between Tavily crawl and extract."
    query = "who's Billy Elish?"
    output = app.invoke({"question": query})
    print("---Final Answer---")
    print(output["generation"])
