from langgraph.graph import END, MessageGraph
from tool_executor import execute_tools
from chains import first_responder, revisor
from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from typing import List

load_dotenv()
FIRST_RESPONDER = "first_responder"
REVISOR = "revisor"
TOOL_EXECUTOR = "tool_executor"
MAX_REVISION = 3


def should_continue(state: List[AnyMessage]):
    tool_call_count = sum(
        1 for msg in state if isinstance(msg, AIMessage) and msg.tool_calls
    )
    if tool_call_count < MAX_REVISION:
        return TOOL_EXECUTOR
    else:
        return END


graph = MessageGraph()

graph.add_node(FIRST_RESPONDER, first_responder)
graph.set_entry_point(FIRST_RESPONDER)

graph.add_node(TOOL_EXECUTOR, execute_tools)
graph.add_edge(FIRST_RESPONDER, TOOL_EXECUTOR)

graph.add_node(REVISOR, revisor)
graph.add_edge(TOOL_EXECUTOR, REVISOR)

graph.add_conditional_edges(
    REVISOR, should_continue, {TOOL_EXECUTOR: TOOL_EXECUTOR, END: END}
)

app = graph.compile()
app.get_graph().draw_mermaid_png(output_file_path="graph.png")

if __name__ == "__main__":
    query = "How to take care of a cat with CKD."
    print("start Reflexion agent.")
    res = app.invoke([HumanMessage(content=query)])
    print(res[-1].tool_calls[0]["args"]["answer"])
