from langgraph.prebuilt import ToolNode
from react import tools, llm_with_tools

tool_node = ToolNode(tools=tools)

SYSYEM_MESSAGE="""
You are a helpful assistant that can use tools to answer questions.
"""

def reasoning_node(state):
    """Running a reasoning node."""
    print("======Reasoning agent is called======")
    messages = state["messages"]
    response = llm_with_tools.invoke(
        [{"role": "system", "content": SYSYEM_MESSAGE}, *messages]
    )
    return {"messages": [response]}

def output(state):
    all_msgs = state["messages"]
    if last := all_msgs[-1].content:
        return {"messages": [last]}
    else:
        return {"messages": [all_msgs[-2].content]}
