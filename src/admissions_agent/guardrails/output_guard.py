from pydantic import BaseModel, Field

from admissions_agent.guardrails.result import GuardrailResult
from admissions_agent.llm import get_guardrail_model

GUARD_NAME = "output"

JUDGE_PROMPT = """You are evaluating an answer from a university admissions assistant.

Check two things:
1. grounded: Is every fact n the answer (dates, fees, scores, requirements) supported by the context? Saying a value is not available when the context lacks it counts as grounded.
2. safe: Does the answer address what was asked, for the right university, programme and applicant group?
Context:
{context}

Question: {question}

Answer: {answer}"""


class Verdict(BaseModel):
    grounded: bool = Field(description="True if the answer is fully supported by the context")
    safe: bool = Field(description="True if the answer contains no unsafe advice")
    reason: str = Field(description="One short sentence explaining the verdict")


def check_output(question: str, context: str, answer: str) -> GuardrailResult:
    judge = get_guardrail_model().with_structured_output(Verdict)
    verdict = judge.invoke(JUDGE_PROMPT.format(context=context, question=question, answer=answer))
    return GuardrailResult(GUARD_NAME, verdict.grounded and verdict.safe, verdict.reason)
