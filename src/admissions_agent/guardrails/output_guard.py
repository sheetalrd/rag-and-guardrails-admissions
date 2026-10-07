from pydantic import BaseModel, Field

from admissions_agent.guardrails.result import GuardrailResult
from admissions_agent.llm import get_guardrail_model

GUARD_NAME = "output"

JUDGE_PROMPT = """You are evaluating an answer from a university admissions assistant.

Check two things:

1. grounded: Does the answer state any specific fact (a date, fee, score, required document, eligibility rule) that is missing from the context or contradicts it? If yes, grounded is false. Otherwise grounded is true.
   These never make an answer ungrounded:
   - saying that information is not available, not listed or incomplete, even if the context contains related details
   - giving only part of what the context contains
   - advising the applicant to check the official website or contact the admissions office

2. safe: Does the answer address the question that was asked, without mixing up the university, programme or applicant group? If the question names no programme, a general answer is safe.

Context:
{context}

Question: {question}

Answer: {answer}"""


class Verdict(BaseModel):
    reason: str = Field(description="One short sentence explaining the verdict")
    grounded: bool = Field(description="False only if the answer states a specific fact that the context does not support")
    safe: bool = Field(description="True if the answer addresses the question for the right university, programme and applicant group")


def check_output(question: str, context: str, answer: str) -> GuardrailResult:
    judge = get_guardrail_model().with_structured_output(Verdict)
    verdict = judge.invoke(JUDGE_PROMPT.format(context=context, question=question, answer=answer))
    failed = [name for name, ok in (("grounded", verdict.grounded), ("safe", verdict.safe)) if not ok]
    reason = f"failed {', '.join(failed)}: {verdict.reason}" if failed else verdict.reason
    return GuardrailResult(GUARD_NAME, not failed, reason)
