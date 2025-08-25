from nodes.retrieve import retrieve
from nodes.web_search import websearch
from nodes.document_grader import doc_grader
from nodes.answer import answer
from nodes.answer_grader import answer_grader
from chains.entry_point_chain import entry_point_chain
from langgraph.graph import StateGraph, END
from models.state import GraphState

RETRIEVE = "Retrieve"
GRADE_DOCUMENT = "Grade_document"
WEB_SEARCH = "Web_research"
ANSWER = "Generate_Answer"
ANSWER_GRADER = "Answer_grader"


def entry_point(state: GraphState) -> str:
    """Let LLM decide if the question is related to the content in vectorstore.
    If yes, then retrieve the answer in the vectorestore. Else, do web search directly."""
    print("---Select first step based on the question---")
    question = state["question"]
    output = entry_point_chain.invoke({"question": question})
    if output.decision == 'web-search':
        print(" -DECISION: Starting web search ")
        return WEB_SEARCH
    elif output.decision == 'retrieve':
        print(" -DECISION: Starting retrieving the vectore store ")
        return RETRIEVE
    return WEB_SEARCH


def after_grade_document(state: GraphState) -> str:
    """After grade_docuemnt node is run, based on the state, 
    determine if further web search is needed or can generate answer directly."""
    print("---Deciding whether web search is needed---")
    if state["web_search"]:
        print(" -DECISION: Starting web search")
        return WEB_SEARCH
    else:
        print(" -DECISION: Generating answers without further web search")
        return ANSWER


def after_answer_grader(state: GraphState) -> str:
    """After generation node is run, based on the state, 
    if the answer has halluciation then re-generate answer. 
    If the answer does not answer the question, then start a web search."""
    print("---Deciding the answer---")

    halluciation_score = state["halluciation_score"]
    answer_score = state.get("answer_score", False)
    if not halluciation_score:
        print(" -Generated content is based on facts.")
        if answer_score:
            print(" -DECISION: The generated answer is good.")
            return END
        else:
            print(" -DECISION: The generated answer is not good. Starting web research again...")
            return WEB_SEARCH
    else:
        print(" -DECISION: The generated answer is halluciated. Generating a new answer...")
        return ANSWER


graph = StateGraph(GraphState)

graph.add_node(RETRIEVE, retrieve)
graph.add_node(GRADE_DOCUMENT, doc_grader)
graph.add_node(WEB_SEARCH, websearch)
graph.add_node(ANSWER, answer)
graph.add_node(ANSWER_GRADER, answer_grader)

graph.set_conditional_entry_point(entry_point, {RETRIEVE: RETRIEVE, WEB_SEARCH: WEB_SEARCH})
graph.add_edge(RETRIEVE, GRADE_DOCUMENT)
graph.add_conditional_edges(
    GRADE_DOCUMENT, after_grade_document, {WEB_SEARCH: WEB_SEARCH, ANSWER: ANSWER}
)
graph.add_edge(WEB_SEARCH, ANSWER)
graph.add_edge(ANSWER, ANSWER_GRADER)
graph.add_conditional_edges(
    ANSWER_GRADER, after_answer_grader, {WEB_SEARCH: WEB_SEARCH, ANSWER: ANSWER, END: END}
)

app = graph.compile()
app.get_graph().draw_mermaid_png(output_file_path="./docs/graph.png")


if __name__ == "__main__":
    print("Start RAG...")
    #query = "Difference between Tavily crawl and extract."
    query = "who's Billy Elish?"
    output = app.invoke({"question": query})
    print("---Final Answer---")
    print(output["answer"])
