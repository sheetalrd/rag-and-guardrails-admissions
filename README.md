# University Admissions Assistant: RAG and Guardrails

A question-answering assistant for prospective students of the University of Amsterdam (UvA) and TU Delft. It answers questions about programmes, eligibility, fees and application dates from a small set of PDF documents, and uses three guardrails to stay on topic and to avoid inventing facts.

This repository is my submission for the RAG and guardrails assignment. I adapted the lesson project to my own use case, built a labelled test set for the search step, implemented the retrieval metrics by hand, and wrote up the results in [`report_sheetal-deshpande.md`](report_sheetal-deshpande.md).

## How it works

Every question passes through these steps, in order:

| Step | What it does |
|---|---|
| 1. Regex guard | Blocks prompt-injection phrases, harmful requests, personal data (email, phone, card numbers) and inputs over 500 characters. |
| 2. NLU guard | A TF-IDF and logistic regression classifier labels the message as `admission`, `off_topic`, `prompt_injection` or `harmful`. Only `admission` continues. |
| 3. Query rewriting | An LLM turns the question into a search query. |
| 4. Retrieval | ChromaDB returns the `TOP_K` chunks closest to the query. |
| 5. Answer generation | An LLM answers using only the retrieved chunks. |
| 6. Output guard | An LLM judge checks that the answer is supported by the chunks and is about the right university, programme and applicant group. |

If a guard blocks, the assistant replies with a refusal instead of an answer.

## Assignment deliverables

| Deliverable | Location |
|---|---|
| Documents for my use case | `data/pdfs/` |
| Guardrail training data | `data/guardrail_training.csv` |
| Test set: 26 questions with graded relevant chunks | `data/eval_set.json` |
| Chunk `id` returned by the retriever | `src/admissions_agent/rag/retriever.py`, `src/admissions_agent/rag/models.py` |
| Metrics in plain Python: Precision@K, Recall@K, F1@K, MRR, MAP, NDCG@K | `src/admissions_agent/evaluation/metrics.py` |
| Metric tests with hand-calculated examples | `tests/test_evaluation.py` |
| Evaluation script | `scripts/05_evaluate.py` |
| Report | `report_sheetal-deshpande.md` |

## Project layout

```
data/
  pdfs/                      the two source PDFs
  guardrail_training.csv     labelled messages for the NLU guard
  eval_set.json              test questions and relevant chunk ids
src/admissions_agent/
  config.py                  settings, read from .env
  llm.py                     chat models (OpenAI or Ollama)
  rag/                       PDF loading, chunking, vector store, retriever, query rewriter
  guardrails/                regex guard, NLU guard, output guard
  evaluation/metrics.py      retrieval metrics
scripts/                     numbered lesson scripts
tests/                       unit tests
reports/                     my report
chroma_db/                   the search index (created when you index the PDFs)
```

## Setup

Requires Python 3.13.

1. Install the dependencies. <!-- TODO: put the install command your project uses here, for example: python -m pip install -r requirements.txt -->
2. Create the settings file and add your API key:

   ```
   copy .env.example .env
   ```

   Then open `.env` and set `OPENAI_API_KEY`.

Settings you can change in `.env`:

| Setting | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` or `ollama` |
| `OPENAI_AGENT_MODEL` | `gpt-5.1` | model that rewrites queries and writes answers |
| `OPENAI_GUARDRAIL_MODEL` | `gpt-5.4-nano` | model that judges answers in the output guard |
| `OLLAMA_AGENT_MODEL`, `OLLAMA_GUARDRAIL_MODEL` | `gemma3:4b` | local models, used when the provider is `ollama` |
| `CHUNK_SIZE` | `200` | words per chunk |
| `CHUNK_OVERLAP` | `40` | words shared between neighbouring chunks |
| `TOP_K` | `4` | chunks passed to the answer step |

Changing `CHUNK_SIZE` or `CHUNK_OVERLAP` changes the chunk ids, so `data/eval_set.json` would have to be labelled again.

## Running it

Run every command from the project root.

| Command | What it does |
|---|---|
| <!-- TODO: indexing script name --> `python scripts/01_....py` | Loads the PDFs, chunks them and builds the ChromaDB index. Run this first. |
| `python scripts/02_search.py "your question"` | Rewrites the question and shows the retrieved chunks. |
| `python scripts/03_guardrails.py` | Tries the guardrails. |
| <!-- TODO: agent script name --> `python scripts/04_....py` | Starts the full assistant. |
| `python scripts/05_evaluate.py` | Evaluates the search with query rewriting. |
| `python scripts/05_evaluate.py --no-rewrite` | Evaluates the search with the question as typed. |
| `python -m pytest tests/test_evaluation.py` | Runs the metric tests. These need no LLM and no database. |

## Evaluation

`data/eval_set.json` holds 26 questions with 69 labelled chunks. Grade 2 means the chunk fully answers the question, and grade 1 means it is partly relevant. The set mixes easy questions, paraphrases, comparisons, a negation, one question in Dutch and one that needs both PDFs.

Results with query rewriting, averaged over the 26 questions:

| K  | Precision | Recall | F1    | MRR   | MAP   | NDCG  |
|----|-----------|--------|-------|-------|-------|-------|
| 1  | 0.500     | 0.214  | 0.278 | 0.500 | 0.214 | 0.404 |
| 3  | 0.346     | 0.489  | 0.367 | 0.635 | 0.357 | 0.464 |
| 5  | 0.277     | 0.600  | 0.346 | 0.660 | 0.403 | 0.505 |
| 10 | 0.173     | 0.719  | 0.262 | 0.665 | 0.436 | 0.560 |

What I found:

- **Precision falls and recall rises as K grows.** A question has 2.65 relevant chunks on average, so most extra slots hold irrelevant chunks.
- **Query rewriting made no clear difference.** It helped some questions and hurt others, and the averages stayed within the noise of one or two questions.
- **The search often finds the right page but the wrong chunk.** Only the first chunk of a page contains the programme name, so later chunks on the same page match a query about that programme poorly.

The full analysis, including the guardrail examples, is in the report.

## Known limitations

- The NLU guard matches word patterns and not meaning, so unusual phrasing of a normal question can be blocked.
- The guardrail training data includes questions about Imperial College, but the documents only cover UvA and TU Delft. Those questions pass the guards and the assistant then has nothing to answer from.
- The output guard can block a correct answer that says information is missing from the sources.

## About the data

The two PDFs are datasets compiled from public university web pages for this exercise. They are not official prospectuses. Fees, deadlines and requirements change every year, so check the universities' own websites before relying on any value.

## Author

Sheetal Deshpande