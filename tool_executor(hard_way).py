import json
from concurrent.futures import ThreadPoolExecutor
from typing import List

from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage, AIMessage

from chains import parser
from schemas import AnswerQuestion, Reflection

load_dotenv()

search = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(api_wrapper=search, max_results=5)


def execute_tools(state: List[BaseMessage]) -> List[ToolMessage]:
    """
    Executes tools based on the tool calls in the last message.

    This function is designed to handle the specific 'AnswerQuestion' tool call,
    extract the search queries from it, run them using the Tavily search tool,
    and then aggregate the results into a single ToolMessage corresponding to the
    original 'AnswerQuestion' call.
    """
    tool_invocation: AIMessage = state[-1]

    parsed_tool_calls = parser.invoke(tool_invocation)
    tool_messages = []
    for parsed_call in parsed_tool_calls:
        # The 'AnswerQuestion' tool call contains the search queries.
        # We execute these queries and return the results as a single observation.
        search_queries = parsed_call.get("args", {}).get("search_queries", [])

        with ThreadPoolExecutor() as executor:
            outputs = list(executor.map(tavily_tool.invoke, search_queries))

        tool_messages.append(
            ToolMessage(content=json.dumps(outputs), tool_call_id=parsed_call["id"])
        )
    return tool_messages


if __name__ == "__main__":
    print("Tool Executor Enter")

    human_message = HumanMessage(
        content="Write about AI-Powered SOC / autonomous soc  problem domain,"
        " list startups that do that and raised capital."
    )

    answer = AnswerQuestion(
        answer="",
        reflection=Reflection(missing="", superfluous=""),
        search_queries=[
            "AI-powered SOC startups funding",
            "AI SOC problem domain specifics",
            "Technologies used by AI-powered SOC startups",
        ],
    )

    raw_res = execute_tools(
        state=[
            human_message,
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": AnswerQuestion.__name__,
                        "args": answer.dict(),
                        "id": "call_KpYHichFFEmLitHFvFhKy1Ra",
                    }
                ],
            ),
        ]
    )
    print(raw_res)
