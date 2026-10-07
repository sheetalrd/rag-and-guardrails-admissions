"""Search pipeline: question -> reformed query -> top-K chunks -> context."""

import sys
from pathlib import Path

# Make the admissions_agent package in src/ importable without installing the project.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from admissions_agent.config import settings
from admissions_agent.rag.context_builder import build_context
from admissions_agent.rag.query_rewriter import rewrite_query
from admissions_agent.rag.retriever import retrieve

console = Console()


def chunk_table(chunks) -> Table:
    table = Table(show_lines=True)
    table.add_column("Rank")
    table.add_column("Source")
    table.add_column("Page")
    table.add_column("Distance")
    table.add_column("Chunk text")
    for rank, chunk in enumerate(chunks, start=1):
        table.add_row(str(rank), chunk.source, str(chunk.page), f"{chunk.distance:.3f}", chunk.text)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", default="how much protein should i eat to build muscle?")
    parser.add_argument("--top-k", type=int, default=settings.top_k)
    args = parser.parse_args()

    console.rule("Step 1: User question")
    console.print(f"  {args.question}")

    console.rule("Step 2: Query reformation")
    search_query = rewrite_query(args.question)
    console.print(f"  {search_query}")

    console.rule(f"Step 3: Retrieve top-{args.top_k} chunks from ChromaDB")
    chunks = retrieve(search_query, args.top_k)
    console.print(chunk_table(chunks))

    console.rule("Step 4: Build the context sent to the LLM")
    console.print(Panel(build_context(chunks), title="Context", border_style="green"))


if __name__ == "__main__":
    main()
