import operator
from typing import Annotated, TypedDict

from admissions_agent.guardrails.result import GuardrailResult
from admissions_agent.rag.models import RetrievedChunk


class AgentState(TypedDict, total=False):
    question: str
    search_query: str
    chunks: list[RetrievedChunk]
    context: str
    answer: str
    # Each guardrail node appends its result, so the list is the full guardrail history.
    guardrail_results: Annotated[list[GuardrailResult], operator.add]
