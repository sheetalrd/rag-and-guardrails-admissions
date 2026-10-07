import os
from dataclasses import dataclass
from pathlib import Path

import truststore
from dotenv import load_dotenv

# Trust the certificates installed in the operating system, so downloads and API calls
# also work on office networks that inspect HTTPS traffic.
truststore.inject_into_ssl()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai").lower()
    openai_agent_model: str = os.getenv("OPENAI_AGENT_MODEL", "gpt-5.1")
    openai_guardrail_model: str = os.getenv("OPENAI_GUARDRAIL_MODEL", "gpt-5.4-nano")
    ollama_agent_model: str = os.getenv("OLLAMA_AGENT_MODEL", "gemma3:4b")
    ollama_guardrail_model: str = os.getenv("OLLAMA_GUARDRAIL_MODEL", "gemma3:4b")

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "200"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "40"))
    top_k: int = int(os.getenv("TOP_K", "4"))

    pdf_dir: Path = PROJECT_ROOT / "data" / "pdfs"
    training_csv: Path = PROJECT_ROOT / "data" / "guardrail_training.csv"
    chroma_dir: Path = PROJECT_ROOT / "chroma_db"
    collection_name: str = "UniversityAdmissionsAssistant"


settings = Settings()
