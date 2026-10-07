from admissions_agent.agent.state import AgentState
from admissions_agent.config import settings
from admissions_agent.guardrails.nlu_guard import check_input_nlu
from admissions_agent.guardrails.output_guard import check_output
from admissions_agent.guardrails.regex_guard import check_input_regex
from admissions_agent.rag.context_builder import build_context
from admissions_agent.rag.generator import generate_answer
from admissions_agent.rag.query_rewriter import rewrite_query
from admissions_agent.rag.retriever import retrieve

REFUSALS = {
    "regex": "Sorry, I can't process that message. Please remove any personal information or instructions.",
    "nlu": "Sorry, I can only help with admission-related questions.",
    "output": "Sorry, I couldn't find a reliable answer to that in my admission documents.",
}


def regex_guard_node(state: AgentState) -> dict:
    return {"guardrail_results": [check_input_regex(state["question"])]}


def nlu_guard_node(state: AgentState) -> dict:
    return {"guardrail_results": [check_input_nlu(state["question"])]}


def rewrite_query_node(state: AgentState) -> dict:
    return {"search_query": rewrite_query(state["question"])}


def retrieve_node(state: AgentState) -> dict:
    return {"chunks": retrieve(state["search_query"], settings.top_k)}


def build_context_node(state: AgentState) -> dict:
    return {"context": build_context(state["chunks"])}


def generate_node(state: AgentState) -> dict:
    return {"answer": generate_answer(state["question"], state["context"])}


def output_guard_node(state: AgentState) -> dict:
    result = check_output(state["question"], state["context"], state["answer"])
    return {"guardrail_results": [result]}


def refuse_node(state: AgentState) -> dict:
    failed_guard = state["guardrail_results"][-1].guard
    return {"answer": REFUSALS[failed_guard]}


def route_after_guard(state: AgentState) -> str:
    return "continue" if state["guardrail_results"][-1].allowed else "refuse"
