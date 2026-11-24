For the source-blind relevance evaluation of the saved default interview top five, see
[the passage relevance report](results/passage-relevance-2026-11-17/report.md).
It evaluates 104 questions and 520 passages using grade 0 (unrelated), 1 (topical)
and 2 (directly relevant). It reuses saved passages, makes no retrieval/database
changes, and reports passage Precision@5 separately from question-level hit@5.
Automatic labels and evidence-quote review flags are provisional, not human truth
or answer-correctness scores. The single completed round refuses reruns:

```powershell
python -m eval.run_passage_relevance --env-file ../EE542-Finalproject/.env
```

For the paired 104-query index/reference-block cleanup experiment, see
[corpus-hygiene.md](../docs/corpus-hygiene.md).

For controlled 104-question intake candidate-pool comparisons, see
[candidate_experiment.md](candidate_experiment.md).

## Guided interview Agentic RAG comparison

The [November 23, 2026 comparison](results/interview-agentic-2026-11-23/report.md)
covers 32 paired questions with live retrieval.

Run the previous single-pass and current adaptive interview Q&A on the same questions:

```powershell
python -m eval.run_agentic_comparison --limit 32 --workers 4 `
  --judge-model gpt-4.1-mini-2025-04-14 --out eval/results/interview-agentic-repeat
```

Omit `--limit` to run all 200 cases. Limited runs sample across question types with
the recorded seed. Both variants use the same interview context, classification
prompt, answer model, and configured live HNSW/BM25 retriever and reranker. Single-pass
uses one retrieval and the previous answer prompt from `4d2d3f7`. Agentic Q&A can
review the evidence and retrieve up to three times. Pair order alternates. The runner
records individual queries, returned chunk IDs, answers, judge verdicts, model calls,
and latency before judging. `--judge-model` selects the model for both judges.

Correctness and faithfulness use the rubrics below. Paired means include only cases
with both scores for that metric, and the summary reports scored counts. Abstentions
are graded for correctness and have no faithfulness score. The active corpus uses
different chunk IDs from the benchmark's original Publication 17 index, so this run
uses the reference answers for correctness without scoring ID-based recall. Use a
fresh output directory for each run.

## Historical free-form chat baseline

Run the free-form chat agent against the same evaluation dataset:

```powershell
.\.venv\Scripts\python.exe -m eval.run_baseline --workers 2
```

The runner uses the configured OpenAI model for answers and automatic judging.
`EVAL_MODEL` can select a different judge before starting a new run. The judge
temperature is zero; the agent keeps its existing configuration. Only the
question is passed to the answering agent. Every case starts a fresh conversation.
The database is accessed through the existing read-only retrieval path.

Outputs are saved in `eval/results/baseline/`:

| File | Contents |
|---|---|
| `manifest.json` | Dataset and source hashes, models, prompts, concurrency, and metric definitions |
| `responses.jsonl` | Checkpointed answers, conversation messages, actual retrieval calls, and latency |
| `judgments.jsonl` | Automatic correctness verdicts and statement-level faithfulness judgments |
| `results.jsonl` | Per-question answers, references, retrieval metrics, scores, and diagnostics |
| `summary.json` | Macro metrics, scoring coverage, latency statistics, and breakdowns by type and year |
| `errors.jsonl` | Judge errors, if any |

Metrics:

- Recall@5 and MRR@5 use annotated chunk IDs, separately for HNSW, BM25, and
  RRF. The primary aggregate uses each question's first actual retrieval call.
  No retrieval call scores zero on an answerable question. Per-call metrics and
  cumulative recall are retained for diagnosing repeated retrievals; cumulative
  recall is not called Recall@5 because it can cover more than five chunks.
- Answer correctness uses an automatic reference-based rubric: correct = 1,
  partially correct = 0.5, incorrect = 0. Standard answers and their evidence
  are available only to this judge, never to the answering agent.
- Faithfulness is the proportion of assessable response claims supported by
  the passages actually returned to the agent, across all tool calls. This is
  a custom LLM judge implementation of statement support, not the RAGAS library.
  Answers with no assessable factual claims receive null, not an automatic 1;
  the summary reports the number of scored answers.
- End-to-end latency measures the agent invocation, including tool calls and
  generation, and excludes judging. Mean, median, and p95 are reported under
  the recorded concurrency.

LLM judging is approximate and should be checked against human samples. Chunk
relevance labels may be incomplete, so ID-based retrieval scores can miss other
valid evidence. Publication year and question year are retained separately:
the dataset's 2026 questions are not used to change the configured 2025 corpus.
The supplied passages can explicitly discuss a later year's rules.

Rerun the same command to resume. Completed agent responses and judgments are
reused. A changed dataset, pipeline, prompts, model, or concurrency requires a
different output directory with `--out`. Use `--limit 2` for an initial check
without changing the complete dataset fingerprint.
