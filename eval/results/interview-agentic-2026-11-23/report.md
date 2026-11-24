# Guided interview Agentic RAG evaluation

November 23, 2026

## Setup

The paired run used 32 questions from `pub17_2025_rag_eval.json` with seed `542`.
The sample contains 9 boundary, 8 factual, 8 scenario, and 7 multi-context questions,
all about tax year 2025. Both modes used the live `rag_materials_v3` corpus, Qwen3
embeddings in Ollama, HNSW and BM25 retrieval, and listwise reranking. The answer
model was `gpt-4o-mini`. The judge was `gpt-4.1-mini-2025-04-14`. Four workers ran
pairs with alternating mode order.

The previous interview Q&A from `4d2d3f7` classifies the input, runs one search,
and generates an answer. The current version uses the same classification and
can check the retrieved evidence and request up to two targeted follow-up searches.
Both modes received the same question and neutral interview prompt. The answering
model had access to retrieved passages. The correctness judge also received the
reference answer and its evidence.

## Results

| Metric | Single-pass RAG | Agentic RAG |
|---|---:|---:|
| Mean answer correctness score | 85.94% | 85.94% |
| Mean faithfulness score | 94.27% | 100.00% |
| Answers returned | 32/32 | 32/32 |
| Questions with follow-up retrieval | 0/32 | 1/32 |
| Mean retrieval calls per question | 1.00 | 1.03 |
| Mean model calls per question | 3.00 | 3.06 |
| Median answer latency | 7.15 s | 7.42 s |
| P95 answer latency | 10.31 s | 12.65 s |

Both modes scored 27 questions as correct, 1 as partial, and 4 as incorrect.
Each question had the same correctness verdict in both modes. Faithfulness
improved on 3 questions and tied on 29. All 3 improvements occurred after a single
retrieval, including 2 cases where both modes received the same ranked passages.
On `pub17_2025_eval_115`, the agent requested a second search for a Schedule B
boundary question. Both modes scored as correct, and the agent cited passages
already present in its first set of results.

All 32 pairs received both automatic scores. Correctness awards 1 for correct,
0.5 for partial, and 0 for incorrect. Faithfulness averages each answer's share
of factual claims the judge found supported by the retrieved passages. Latency
excludes judging, and model calls include classification, reranking, and answer
generation. The live corpus contains 0 of the 153 original gold chunk IDs, so
there is no comparable ID-based Recall@5 score for this corpus.

This run measures a small stratified sample with one model judgment per metric.
The initial queries differed in 7 pairs despite the shared classifier settings,
and the complete ranked first result matched in 21 pairs. The evidence support
scores describe the answers in this run. Separate measurements would be needed
to estimate the effect of supplemental retrieval on harder questions.

## Regression verification

The full Python test suite passed with 266 tests and 10 subtests. The tests cover
follow-up retrieval, evidence deduplication, search limits, invalid citations,
and the interview response when attachments are sent with a tax question.

```powershell
python -m pytest -q -p no:cacheprovider
```

## Run artifacts

- [Per-question answers, searches, and judgments](pairs.jsonl)
- [Summary and run settings](summary.json)
- [Reproduction command](../../README.md#guided-interview-agentic-rag-comparison)
