from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
import hashlib
import json
from pathlib import Path
from random import Random
from statistics import mean, median
from time import perf_counter

from numpy import percentile

from eval.run_baseline import (
    CORRECTNESS_PROMPT, FAITHFULNESS_PROMPT, CorrectnessJudgment, FaithfulnessJudgment, faithfulness_score,
)
from src.config import ROOT, llm_settings
from src.llm.client import chat_json, set_call_guard
from src.llm.interview_qa import MAX_SEARCHES, NO_EVIDENCE, UNAVAILABLE, respond_to_question
from src.rag.retriever import retrieve


GUIDE = {"question": "What tax question do you have?", "fact": "tax_question", "input_type": "list", "options": []}
MODES = {"single_pass": 1, "agentic": MAX_SEARCHES}
VERDICTS = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}
model_calls = ContextVar("model_calls", default=None)


def count_model_call() -> None:
    calls = model_calls.get()
    if calls is not None:
        model_calls.set(calls + 1)


def select_cases(samples: list[dict], limit: int | None, seed: int) -> list[dict]:
    if limit is None or limit >= len(samples):
        return samples
    groups = {}
    for sample in samples:
        groups.setdefault(sample["question_type"], []).append(sample)
    random = Random(seed)
    buckets = [groups[key] for key in sorted(groups)]
    for bucket in buckets:
        random.shuffle(bucket)
    selected = []
    while len(selected) < limit:
        for bucket in buckets:
            if bucket:
                selected.append(bucket.pop())
                if len(selected) == limit:
                    break
    return selected


def run_variant(sample: dict, max_searches: int, source_year: int) -> tuple[dict, list[dict]]:
    searches = []
    evidence = {}

    def search(query):
        call = {"query": query}
        searches.append(call)
        try:
            result = retrieve(query)
        except Exception as exc:
            call["error"] = type(exc).__name__
            raise
        matches = [m for m in result["matches"][:5] if m.get("citation") and m.get("text")]
        call.update(chunk_ids=[m["chunk_id"] for m in matches], method=result["retrieval_method"],
                    rerank_fallback=result.get("rerank_fallback", False))
        for match in matches:
            key = (match["citation"], match["text"])
            evidence[key] = {"citation": match["citation"], "text": match["text"]}
        return result

    token = model_calls.set(0)
    started = perf_counter()
    try:
        response = respond_to_question(chat_json, search, GUIDE, sample["question"], source_year,
                                       max_searches=max_searches)
        answer = response["answer"] if response else ""
        citations = response["citations"] if response else []
        status = "unavailable" if answer == UNAVAILABLE else "answered" if citations else "abstained"
        row = {"status": status, "answer": answer, "citations": [c["citation"] for c in citations]}
    except Exception as exc:
        row = {"status": "unavailable", "error": type(exc).__name__}
    finally:
        elapsed, calls = perf_counter() - started, model_calls.get()
        model_calls.reset(token)
    row.update(elapsed_seconds=elapsed, searches=searches, model_calls=calls, retrieved_passages=len(evidence))
    return row, list(evidence.values())


def score(sample: dict, row: dict, evidence: list[dict], judge_model: str) -> None:
    if row["status"] == "unavailable":
        return
    answer = row["answer"] if row["answer"] != NO_EVIDENCE else ""
    common = {"question": sample["question"], "response": answer}
    rubrics = (
        ("correctness", CORRECTNESS_PROMPT, CorrectnessJudgment,
         {"reference": sample["standard_answer"], "reference_evidence": sample["relevant_documents"]}),
        ("faithfulness", FAITHFULNESS_PROMPT, FaithfulnessJudgment, {"retrieved_contexts": evidence}),
    )
    for metric, prompt, schema, context in rubrics:
        if metric == "faithfulness" and row["status"] == "abstained":
            continue
        try:
            system = prompt + "\nUse this JSON schema: " + json.dumps(schema.model_json_schema())
            reply = chat_json(system, json.dumps(common | context, ensure_ascii=False), model=judge_model, timeout=60)
            result = schema.model_validate(reply).model_dump()
        except Exception as exc:
            row[f"{metric}_error"] = type(exc).__name__
            continue
        row[f"{metric}_judgment"] = result
        row[metric] = VERDICTS[result["verdict"]] if metric == "correctness" else faithfulness_score(result)


def evaluate_pair(index: int, sample: dict, source_year: int, judge_model: str) -> dict:
    order = list(MODES) if index % 2 == 0 else list(reversed(MODES))
    responses = {mode: run_variant(sample, MODES[mode], source_year) for mode in order}
    for mode, (row, evidence) in responses.items():
        score(sample, row, evidence, judge_model)
    return {"id": sample["id"], "question": sample["question"], "question_type": sample["question_type"],
            "tax_year": sample["tax_year"], **{mode: responses[mode][0] for mode in MODES}}


def summarize(pairs: list[dict]) -> dict:
    summary = {"cases": len(pairs), "modes": {}, "paired": {}}
    for mode in MODES:
        rows = [pair[mode] for pair in pairs]
        completed = [row for row in rows if row["status"] != "unavailable"]
        latency = [row["elapsed_seconds"] for row in completed]
        summary["modes"][mode] = {
            "answered": sum(row["status"] == "answered" for row in rows),
            "abstained": sum(row["status"] == "abstained" for row in rows),
            "unavailable": len(rows) - len(completed),
            "mean_searches": mean(len(row["searches"]) for row in rows),
            "mean_model_calls": mean(row["model_calls"] for row in rows),
            "follow_up_cases": sum(len(row["searches"]) > 1 for row in rows),
            "retrieval_errors": sum("error" in call for row in rows for call in row["searches"]),
            "rerank_fallbacks": sum(call.get("rerank_fallback", False) for row in rows for call in row["searches"]),
            "latency_seconds": {"mean": mean(latency), "median": median(latency), "p95": float(percentile(latency, 95))}
            if latency else None,
        }
    for metric in ("correctness", "faithfulness"):
        comparable = [(p["single_pass"][metric], p["agentic"][metric]) for p in pairs
                      if all(p[mode].get(metric) is not None for mode in MODES)]
        summary["paired"][metric] = {
            "cases": len(comparable),
            "single_pass_scored": sum(p["single_pass"].get(metric) is not None for p in pairs),
            "agentic_scored": sum(p["agentic"].get(metric) is not None for p in pairs),
            "single_pass_mean": mean(a for a, _ in comparable) if comparable else None,
            "agentic_mean": mean(b for _, b in comparable) if comparable else None,
            "agentic_wins": sum(b > a for a, b in comparable),
            "ties": sum(a == b for a, b in comparable),
            "agentic_losses": sum(b < a for a, b in comparable),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "eval/pub17_2025_rag_eval.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=542)
    parser.add_argument("--judge-model")
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers and limit must be positive")
    dataset = args.dataset.read_bytes()
    samples = select_cases(json.loads(dataset), args.limit, args.seed)
    if not samples or len({sample["id"] for sample in samples}) != len(samples):
        parser.error("the selected dataset must have at least one case and unique IDs")
    settings = json.loads((ROOT / "config/settings.json").read_text(encoding="utf-8"))
    answer_model = llm_settings().extract_model
    judge_model = args.judge_model or answer_model
    args.out.mkdir(parents=True, exist_ok=True)
    pairs = []
    set_call_guard(count_model_call)
    try:
        with (args.out / "pairs.jsonl").open("x", encoding="utf-8") as output, ThreadPoolExecutor(args.workers) as pool:
            pending = [pool.submit(evaluate_pair, index, sample, settings["tax_year"], judge_model)
                       for index, sample in enumerate(samples)]
            for future in pending:
                pair = future.result()
                pairs.append(pair)
                output.write(json.dumps(pair, ensure_ascii=False) + "\n")
                output.flush()
                print(f"{len(pairs)}/{len(samples)} {pair['id']}", flush=True)
    finally:
        set_call_guard(None)
    summary = summarize(pairs)
    summary["run"] = {"dataset_sha256": hashlib.sha256(dataset).hexdigest(), "seed": args.seed,
                      "workers": args.workers, "answer_model": answer_model, "judge_model": judge_model,
                      "corpus_version": settings["corpus_version"], "corpus_tax_year": settings["tax_year"],
                      "rerank_method": settings["rerank_method"], "max_searches": MODES}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if summary["paired"]["correctness"]["cases"] != len(samples) or any(
        f"{metric}_error" in pair[mode] for pair in pairs for mode in MODES for metric in ("correctness", "faithfulness")
    ):
        raise SystemExit("The comparison has unavailable responses or failed judgments. See pairs.jsonl.")


if __name__ == "__main__":
    main()
