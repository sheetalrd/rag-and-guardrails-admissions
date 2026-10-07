from admissions_agent.llm import get_agent_model

ANSWER_PROMPT = """You are a helpful university admissions assistant.
Answer the question using only the context below and mention the source numbers you used, like [1].
Every fact, number, quantity and step in your answer must appear in the context.
Do not add information, even if it seems obvious.
If the context covers only part of the question, answer that part and say what the documents do not cover.
If the context does not contain the answer, say you don't know.

Context:
{context}

Question: {question}"""


def generate_answer(question: str, context: str) -> str:
    response = get_agent_model().invoke(ANSWER_PROMPT.format(context=context, question=question))
    return response.text.strip()
