from langgraph.graph import END, StateGraph
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage
from nodes import tool_node, reasoning_node, output

load_dotenv()
REASONING = "Reasoning_agent"
TOOL = "Tool_node"
OUTPUT = "Output_node"


class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


def should_continue(state: GraphState):
    if state["messages"][-1].tool_calls:
        print("======Tool is called======")
        return TOOL
    else:
        return OUTPUT


graph = StateGraph(GraphState)

graph.add_node(REASONING, reasoning_node)
graph.set_entry_point(REASONING)
graph.add_node(TOOL, tool_node)
graph.add_node(OUTPUT, output)

graph.add_conditional_edges(REASONING, should_continue, {TOOL: TOOL, OUTPUT: OUTPUT})
graph.add_edge(TOOL, REASONING)
graph.add_edge(OUTPUT, END)

app = graph.compile()
app.get_graph().draw_mermaid_png(output_file_path="graph.png")

if __name__ == "__main__":
    query = "Who are you?"
    print("start ReAct agent.")
    res = app.invoke({"messages": [HumanMessage(content=query)]})
    print("======Final Answer======")
    print(res["messages"][-1])

