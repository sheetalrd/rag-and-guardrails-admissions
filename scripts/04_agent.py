"""The full LangGraph agent: guardrails + RAG, with a trace of every node."""

import sys
from pathlib import Path

# Make the admissions_agent package in src/ importable without installing the project.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse

from rich.console import Console
from rich.panel import Panel

from admissions_agent.agent.graph import build_graph

console = Console()


def describe(update: dict) -> str:
    if "guardrail_results" in update:
        result = update["guardrail_results"][-1]
        status = "[green]ALLOW[/green]" if result.allowed else "[red]BLOCK[/red]"
        return f"{status} {result.reason}"
    if "search_query" in update:
        return f"search query: {update['search_query']}"
    if "chunks" in update:
        hits = [f"{chunk.source} p{chunk.page} ({chunk.distance:.3f})" for chunk in update["chunks"]]
        return "retrieved: " + ", ".join(hits)
    if "context" in update:
        return f"context built ({len(update['context'].split())} words)"
    return "answer written"


def ask(graph, question: str):
    answer = ""
    for step in graph.stream({"question": question}, stream_mode="updates"):
        for node, update in step.items():
            console.print(f"  [bold cyan]{node:<14}[/bold cyan] {describe(update)}")
            answer = update.get("answer", answer)
    console.print(Panel(answer, title="Answer", border_style="green"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", help="ask one question and exit")
    args = parser.parse_args()

    graph = build_graph()

    if args.question:
        ask(graph, args.question)
        return

    console.print("Admissions agent ready. Type a question, or 'quit' to stop.")
    while True:
        question = console.input("\n[bold]You:[/bold] ").strip()
        if question.lower() in {"quit", "exit"}:
            break
        if question:
            ask(graph, question)


if __name__ == "__main__":
    main()
