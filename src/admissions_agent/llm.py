import os
from functools import lru_cache

from admissions_agent.config import settings


def build_chat_model(openai_model: str, ollama_model: str):
    if settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=ollama_model, temperature=0)

    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is missing. Copy .env.example to .env and add your key.")

        return ChatOpenAI(model=openai_model)

    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r} (use 'openai' or 'ollama')")


@lru_cache
def get_agent_model():
    """The model that reforms queries and writes answers."""
    return build_chat_model(settings.openai_agent_model, settings.ollama_agent_model)


@lru_cache
def get_guardrail_model():
    """The smaller model that judges answers in the output guardrail."""
    return build_chat_model(settings.openai_guardrail_model, settings.ollama_guardrail_model)
