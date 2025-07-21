import logging
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import MessageGraph, END
from chains import generate_chain, reflect_chain
from dotenv import load_dotenv
from typing import List

load_dotenv()


GENERATE = "generate"
REFLECT = "reflect"


def generation_node(state: List[AnyMessage]) -> AnyMessage:
    print("------------------GENERATE node called------------------")
    response = generate_chain.invoke({"messages": state})
    # print(response.content)
    return [AIMessage(content=response.content)]


def reflection_node(state: List[AnyMessage]) -> List[HumanMessage]:
    print("------------------REFLECT node called------------------")
    response = reflect_chain.invoke({"messages": state})
    # print(response.content)
    return [HumanMessage(content=response.content)]


def should_continue(state: List[AnyMessage]):
    if len(state) > 4:
        return END
    return REFLECT


def create_graph():
    flow = MessageGraph()
    flow.add_node(GENERATE, generation_node)
    flow.add_node(REFLECT, reflection_node)
    flow.set_entry_point(GENERATE)
    flow.add_conditional_edges(
        GENERATE, should_continue, {END: END, REFLECT: REFLECT}
    )
    flow.add_edge(REFLECT, GENERATE)
    app = flow.compile()
    app.get_graph().draw_mermaid_png(output_file_path="graph.png")
    return app


def main():
    app = create_graph()
    print("start reflection agent.")
    query = (
        "Make this tweet better: '@LangChainAI — newly Tool Calling feature is "
        "seriously underrated. After a long wait, it's here- making the "
        "implementation of agents across different models with function "
        "calling - super easy. Made a video covering their newest blog post'"
    )
    final_response = app.invoke(HumanMessage(content=query))
    print("------------------FINAL RESPONSE------------------")
    print(final_response[-1].content)


if __name__ == "__main__":
    main()
