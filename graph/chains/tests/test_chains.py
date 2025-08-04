from graph.chains.retrieval_grader import GradeAnswer, retrieval_grader
from graph.nodes.retrieve import retrieve
from graph.chains.generation import generation_chain
from graph.chains.hallucination_grader import halluciation_grader
from graph.chains.answer_grader import answer_grader
from pprint import pprint


# def test_retrieval_grader_answer_yes():
#     question = "Tavily crawl best practice"
#     output = retrieve({"question": question})
#     grade_res: GradeAnswer = retrieval_grader.invoke(output)

#     assert grade_res.binary_score == "yes"


# def test_retrieval_grader_answer_no():
#     question = "Who is Billy Elish"
#     output = retrieve({"question": question})
#     grade_res: GradeAnswer = retrieval_grader.invoke(output)

#     assert grade_res.binary_score == "no"


# def test_generation_chain():
#     question = "Tavily crawl best practice"
#     output = retrieve({"question": question})
#     res = generation_chain.invoke(
#         {"question": output["question"], "context": output["documents"]}
#     )
#     pprint(res)

def test_hallucination_grader_good():
    question = "Tavily crawl best practice"
    output = retrieve({"question": question})
    res = generation_chain.invoke(
        {"question": output["question"], "context": output["documents"]}
    )
    output = halluciation_grader.invoke({"generation": res, "documents": output["documents"]})
    assert output.binary_score == True

def test_hallucination_grader_bad():
    question = "Tavily crawl best practice"
    output = retrieve({"question": question})
    res = generation_chain.invoke(
        {"question": output["question"], "context": output["documents"]}
    )
    output = halluciation_grader.invoke({"generation": "Tavily crawl is the best tool in the world.", "documents": output["documents"]})
    assert output.binary_score == False

def test_answer_grader_good():
    question = "Tavily crawl best practice"
    output = retrieve({"question": question})
    res = generation_chain.invoke(
        {"question": output["question"], "context": output["documents"]}
    )
    output = answer_grader.invoke({"generation": res, "question": question})
    assert output.binary_score == True

def test_answer_grader_bad():
    question = "Tavily crawl best practice"
    output = retrieve({"question": question})
    res = generation_chain.invoke(
        {"question": output["question"], "context": output["documents"]}
    )
    output = answer_grader.invoke({"generation": "Pizza is ready.", "question": question})
    assert output.binary_score == False