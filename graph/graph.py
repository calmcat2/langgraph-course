from graph.nodes.retrieve import retrieve
from graph.nodes.web_search import websearch
from graph.nodes.document_grade import doc_grader
from graph.nodes.generation import generation
from graph.chains.hallucination_grader import halluciation_grader
from graph.chains.answer_grader import answer_grader
from graph.chains.entry_point import entry_point_chain
from langgraph.graph import StateGraph, END
from graph.state import GraphState

RETRIEVE = "Retrieve"
GRADE_DOCUMENT = "Grade_document"
WEB_SEARCH = "Web_research"
GENERATE = "Generate"

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
    # Fallback to web search if the decision is not clear
    return WEB_SEARCH


def after_grade_document(state: GraphState) -> str:
    """After grade_docuemnt node is run, based on the state, 
    determine if further web search is needed or can generate answer directly."""
    print("---Decide whether web search is needed---")
    if state["web_search"]:
        print(" -DECISION: Starting web search")
        return WEB_SEARCH
    else:
        print(" -DECISION: Generating answers without further web search")
        return GENERATE


def after_generation(state: GraphState) -> str:
    """After generation node is run, based on the state, 
    if the answer has halluciation then re-generate answer. 
    If the answer does not answer the question, then start a web search."""
    print("---Evaluating generated content---")
    question = state["question"]
    generation = state["generation"]
    documents = state["documents"]
    halluciation_score = halluciation_grader.invoke(
        {"generation": generation, "documents": documents}
    )
    if halluciation_score.binary_score:
        print(" -Generated content is based on facts.")
        answer_score = answer_grader.invoke(
            {"generation": generation, "question": question}
        )
        if answer_score.binary_score:
            print(" -DECISION: The generated answer is good.")
            return END
        else:
            print(" -DECISION: The generated answer is not good. Starting web research again...")
            return WEB_SEARCH
    else:
        print(" -DECISION: The generated answer is halluciated. Generating a new answer...")
        return GENERATE


graph = StateGraph(GraphState)

graph.add_node(RETRIEVE, retrieve)
graph.add_node(GRADE_DOCUMENT, doc_grader)
graph.add_node(WEB_SEARCH, websearch)
graph.add_node(GENERATE, generation)

graph.set_conditional_entry_point(entry_point, {RETRIEVE: RETRIEVE, WEB_SEARCH: WEB_SEARCH})
graph.add_edge(RETRIEVE, GRADE_DOCUMENT)
graph.add_conditional_edges(
    GRADE_DOCUMENT, after_grade_document, {WEB_SEARCH: WEB_SEARCH, GENERATE: GENERATE}
)
graph.add_edge(WEB_SEARCH, GENERATE)
graph.add_conditional_edges(
    GENERATE, after_generation, {WEB_SEARCH: WEB_SEARCH, GENERATE: GENERATE, END: END}
)

app = graph.compile()
app.get_graph().draw_mermaid_png(output_file_path="graph.png")
