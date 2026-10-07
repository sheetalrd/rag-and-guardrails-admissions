"""Ingestion pipeline: PDFs -> pages -> chunks -> embeddings -> ChromaDB."""

import sys
from pathlib import Path

# Make the admissions_agent package in src/ importable without installing the project.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from collections import Counter

from rich.console import Console
from rich.panel import Panel

from admissions_agent.config import settings
from admissions_agent.rag.chunker import chunk_pages
from admissions_agent.rag.ingestion import store_chunks
from admissions_agent.rag.loader import load_pdfs

console = Console()


def main():
    console.rule("Step 1: Load PDFs")
    pages = load_pdfs(settings.pdf_dir)
    if not pages:
        console.print(f"[red]No PDF text found in {settings.pdf_dir}. Add your PDFs there and run again.[/red]")
        return
    for source, count in Counter(page.source for page in pages).items():
        console.print(f"  {source}: {count} pages")

    console.rule("Step 2: Split pages into chunks")
    chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
    console.print(
        f"  {len(pages)} pages -> {len(chunks)} chunks "
        f"(chunk size {settings.chunk_size} words, overlap {settings.chunk_overlap} words)"
    )
    for chunk in chunks[:2]:
        console.print(Panel(chunk.text, title=f"Sample chunk: {chunk.id}", border_style="cyan"))

    console.rule("Step 3: Embed chunks and store them in ChromaDB")
    stored = store_chunks(chunks)
    console.print(f"  [green]Stored {stored} chunks in collection '{settings.collection_name}'[/green]")
    console.print(f"  Database folder: {settings.chroma_dir}")


if __name__ == "__main__":
    main()
