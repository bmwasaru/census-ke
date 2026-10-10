"""Evaluate retrieval, fixed-evidence generation, or the complete census pipeline."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time

BASE_DIR = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = BASE_DIR / "evaluation/questions.json"
FIXTURES_PATH = BASE_DIR / "evaluation/evidence.json"
NUMBER = re.compile(r"(?<![\w.,+-])[+-]?\d+(?:,\d{3})*(?:\.\d+)?(?![\w]|,\d|\.\d)")
CITATION = re.compile(r"\[S\d+\]")
ABSTENTION = re.compile(
    r"not (?:provide|contain|available|included|specified|given)|"
    r"(?:insufficient|not enough) (?:information|evidence)|"
    r"(?:cannot|could not|can't|unable to) (?:answer|determine|find)|"
    r"does(?:n't| not) (?:provide|contain|include|specify)|"
    r"no (?:information|evidence|data)|missing from|not present",
    re.IGNORECASE,
)


def numbers(text):
    """Compare exact numeric tokens, accepting thousands separators, not substrings."""
    return {
        format(Decimal(m.group().replace(",", "")).normalize(), "f")
        for m in NUMBER.finditer(CITATION.sub("", str(text)))
    }


def source_matches(result, expected_sources):
    return any(
        result.get("volume") == s["volume"] and result.get("page") in s["pages"]
        for s in expected_sources
    )


def reciprocal_rank(results, expected_sources):
    return next(
        (
            1 / rank
            for rank, r in enumerate(results, 1)
            if source_matches(r, expected_sources)
        ),
        0,
    )


def hit_at_k(results, expected_sources, k):
    return any(source_matches(r, expected_sources) for r in results[:k])


def validate_questions(questions, fixtures):
    if not isinstance(questions, list) or not questions:
        raise ValueError("Questions must be a nonempty list")
    seen = set()
    for q in questions:
        if q["id"] in seen:
            raise ValueError(f"Duplicate question id: {q['id']}")
        seen.add(q["id"])
        if not isinstance(q["answerable"], bool) or not q["question"].strip():
            raise ValueError(f"Invalid question: {q['id']}")
        if q.get("evidence_id") not in fixtures:
            raise ValueError(f"Missing evidence fixture: {q['id']}")
        if q["answerable"] and not q.get("expected_values"):
            raise ValueError(
                f"Answerable numeric case needs expected values: {q['id']}"
            )
        if "retrieval" in q["suites"] and q["answerable"] and not q["expected_sources"]:
            raise ValueError(f"Missing retrieval sources: {q['id']}")
        for s in q["expected_sources"]:
            if not s["pages"] or any(
                not isinstance(p, int) or p < 1 for p in s["pages"]
            ):
                raise ValueError(f"Pages must be one-based PDF pages: {q['id']}")


def score_answer(question, response, llm):
    """Diagnostic checks, not a semantic correctness verdict or LLM-as-judge."""
    answer = response["answer"]
    evidence = response.get("evidence", [])
    evidence_text, _, _ = llm._prepare_evidence(evidence, max_chunks=len(evidence) or 1)
    actual = numbers(answer)
    expected = {n for v in question.get("expected_values", []) for n in numbers(v)}
    forbidden = {n for v in question.get("forbidden_values", []) for n in numbers(v)}
    allowed = numbers(evidence_text) | {
        n for v in question.get("allowed_derived_values", []) for n in numbers(v)
    }
    citations = set(CITATION.findall(answer))
    supplied = {f"[{s['id']}]" for s in response.get("sources", [])}
    abstained = bool(ABSTENTION.search(answer))
    missing = sorted(expected - actual)
    forbidden_found = sorted(forbidden & actual)
    unsupported = sorted(actual - allowed)
    failures = []
    if not answer.strip():
        failures.append("empty_answer")
    if question["answerable"]:
        if missing:
            failures.append("missing_expected_values")
        if forbidden_found:
            failures.append("distractor_values_in_answer")
        if not citations:
            failures.append("missing_citation")
        if abstained:
            failures.append("unexpected_abstention")
    else:
        if not abstained:
            failures.append("missing_abstention")
        # A disclaimer followed by a guessed count is still a failure. Years in
        # the question/evidence are permitted; no counts are expected here.
        years = {
            n
            for n in numbers(question["question"] + " " + evidence_text)
            if Decimal(n) == Decimal(n).to_integral_value()
            and 1900 <= Decimal(n) <= 2100
        }
        unsupported = sorted(set(unsupported) - years)
        if actual - years:
            failures.append("numeric_claim_on_unanswerable_question")
    if citations - supplied:
        failures.append("invalid_citations")
    if unsupported:
        failures.append("unsupported_numbers")
    return {
        "automated_checks_passed": not failures,
        "failures": failures,
        "missing_expected_values": missing,
        "forbidden_values_found": forbidden_found,
        "unsupported_numbers": unsupported,
        "citations": sorted(citations),
        "invalid_citations": sorted(citations - supplied),
        "abstention_detected": abstained,
        "semantic_review": "required",
        "review_notes": "Check value-to-column/entity mapping, claim support, and readability; token checks cannot prove these.",
    }


def summarize(records):
    retrieval = list(
        {r["id"]: r["retrieval"] for r in records if r.get("retrieval")}.values()
    )
    generated = [r for r in records if "generation" in r or "error" in r]
    scored = [r for r in generated if "generation" in r]
    times = [r["generation_seconds"] for r in scored]
    unanswerable = [r for r in generated if not r["answerable"]]
    answerable = [r for r in generated if r["answerable"]]

    def rate(rows):
        return (
            (
                sum(
                    r.get("generation", {}).get("automated_checks_passed", False)
                    for r in rows
                )
                / len(rows)
            )
            if rows
            else None
        )

    def check_rate(rows, failures):
        return (
            sum(
                "generation" in r and not (set(r["generation"]["failures"]) & failures)
                for r in rows
            )
            / len(rows)
            if rows
            else None
        )

    return {
        "questions": len(records),
        "retrieval_questions": len(retrieval),
        **{
            f"hit@{k}": (
                sum(r[f"hit@{k}"] for r in retrieval) / len(retrieval)
                if retrieval
                else None
            )
            for k in (1, 3, 5)
        },
        "mrr": (
            statistics.mean(r["reciprocal_rank"] for r in retrieval)
            if retrieval
            else None
        ),
        "generation_questions": len(generated),
        "generation_errors": sum("error" in r for r in generated),
        "automated_check_pass_rate": rate(generated),
        "answerable_check_pass_rate": rate(answerable),
        "unanswerable_check_pass_rate": rate(unanswerable),
        "expected_value_presence_rate": check_rate(
            answerable, {"missing_expected_values"}
        ),
        "citation_presence_rate": check_rate(answerable, {"missing_citation"}),
        "citation_validity_rate": check_rate(generated, {"invalid_citations"}),
        "failure_counts": dict(
            Counter(failure for r in scored for failure in r["generation"]["failures"])
        ),
        "categories": {
            category: {"attempts": len(rows), "automated_check_pass_rate": rate(rows)}
            for category in sorted({r.get("category", "unknown") for r in generated})
            for rows in [
                [r for r in generated if r.get("category", "unknown") == category]
            ]
        },
        "median_generation_seconds": statistics.median(times) if times else None,
        "semantic_review": "required" if generated else "not_applicable",
    }


def evaluate(
    questions, fixtures, mode, engine=None, llm=None, top_k=5, max_chunks=5, repeat=1
):
    records = []
    for q in questions:
        suite = "generation" if mode == "generation" else "retrieval"
        if suite not in q["suites"]:
            continue
        results = fixtures[q["evidence_id"]]["results"]
        retrieval = None
        retrieval_seconds = None
        if mode != "generation":
            start = time.perf_counter()
            results = engine.search(q["question"], top_k=top_k, volume=q.get("volume"))
            retrieval_seconds = time.perf_counter() - start
            if q["answerable"]:
                retrieval = {
                    f"hit@{k}": hit_at_k(results, q["expected_sources"], k)
                    for k in (1, 3, 5)
                }
                retrieval["reciprocal_rank"] = reciprocal_rank(
                    results, q["expected_sources"]
                )
                retrieval["expected_values_present"] = all(
                    numbers(v) <= numbers(json.dumps(results))
                    for v in q["expected_values"]
                )
        for attempt in range(1, (repeat if mode != "retrieval" else 1) + 1):
            record = {
                "id": q["id"],
                "question": q["question"],
                "category": q["category"],
                "answerable": q["answerable"],
                "attempt": attempt,
                "expected_values": q["expected_values"],
                "expected_sources": q["expected_sources"],
                "retrieval": retrieval,
                "retrieval_seconds": retrieval_seconds,
                "input_evidence": results,
                "fixture_id": q["evidence_id"] if mode == "generation" else None,
            }
            if mode != "retrieval":
                start = time.perf_counter()
                try:
                    response = llm.answer(q["question"], results, max_chunks=max_chunks)
                    record.update(
                        answer=response["answer"],
                        sources=response["sources"],
                        supplied_evidence=response["evidence"],
                        generation=score_answer(q, response, llm),
                    )
                except Exception as exc:
                    # Keep failed requests in the denominator, preserve other results.
                    record["error"] = {"type": type(exc).__name__, "message": str(exc)}
                record["generation_seconds"] = time.perf_counter() - start
            records.append(record)
            print(
                f"{q['id']} [{attempt}]: "
                + (
                    str(record.get("generation", {}).get("failures", []))
                    if mode != "retrieval"
                    else str(retrieval)
                ),
                flush=True,
            )
    return records


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def rescore_report(report, questions):
    """Regrade saved answers without contacting Ollama or running retrieval."""
    from llm import CensusLLM

    by_id = {q["id"]: q for q in questions}
    for run in report["runs"]:
        llm = CensusLLM(model=run["model"])
        for record in run["records"]:
            if "answer" in record:
                record["generation"] = score_answer(
                    by_id[record["id"]],
                    {
                        "answer": record["answer"],
                        "sources": record["sources"],
                        "evidence": record["supplied_evidence"],
                    },
                    llm,
                )
        run["summary"] = summarize(run["records"])
    report["rescored_at"] = datetime.now(timezone.utc).isoformat()
    report["scorer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["retrieval", "generation", "end-to-end"], default="retrieval"
    )
    parser.add_argument("--models", nargs="+", help="Ollama model tags to compare")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Embedding device; CPU avoids platform GPU issues",
    )
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--fixtures", type=Path, default=FIXTURES_PATH)
    parser.add_argument(
        "--output", type=Path, default=BASE_DIR / "evaluation/results.json"
    )
    parser.add_argument("--ids", nargs="+", help="Only run specified question IDs")
    parser.add_argument("--top-k", type=positive_int, default=5)
    parser.add_argument("--max-chunks", type=positive_int, default=5)
    parser.add_argument("--repeat", type=positive_int, default=1)
    parser.add_argument(
        "--rescore", type=Path, help="Regrade an existing report offline"
    )
    parser.add_argument(
        "--fail-on-checks",
        action="store_true",
        help="Exit 1 on generation failures or errors",
    )
    args = parser.parse_args(argv)
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    questions = json.loads(args.questions.read_text())
    fixtures = json.loads(args.fixtures.read_text())
    validate_questions(questions, fixtures)
    if args.rescore:
        report = json.loads(args.rescore.read_text())
        if (
            report["questions_sha256"]
            != hashlib.sha256(args.questions.read_bytes()).hexdigest()
        ):
            parser.error(
                "Saved report uses different questions; rescore with its original dataset"
            )
        rescore_report(report, questions)
        return save_report(report, args.output, args.fail_on_checks)
    if args.ids:
        unknown = set(args.ids) - {q["id"] for q in questions}
        if unknown:
            parser.error(f"Unknown IDs: {sorted(unknown)}")
        questions = [q for q in questions if q["id"] in args.ids]
    suite = "generation" if args.mode == "generation" else "retrieval"
    if not any(suite in q["suites"] for q in questions):
        parser.error("No questions eligible for the selected mode")
    engine = None
    if args.mode != "generation":
        from search_engine import CensusSearchEngine

        engine = CensusSearchEngine(
            BASE_DIR / "index", BASE_DIR / "models/bge-small-en-v1.5"
        )
        if not engine.is_ready():
            parser.error("Census index missing; run python ingest.py")
        from sentence_transformers import SentenceTransformer

        engine._model = SentenceTransformer(
            str(engine.model_path), device=args.device, local_files_only=True
        )
    from llm import CensusLLM

    models = args.models or [os.getenv("OLLAMA_MODEL", "qwen3:1.7b")]
    runs = []
    for model in ([None] if args.mode == "retrieval" else models):
        llm = CensusLLM(model=model) if model else None
        records = evaluate(
            questions,
            fixtures,
            args.mode,
            engine,
            llm,
            args.top_k,
            args.max_chunks,
            args.repeat,
        )
        runs.append(
            {
                "model": model,
                "configuration": (
                    {
                        k: getattr(llm, k)
                        for k in (
                            "base_url",
                            "num_ctx",
                            "num_predict",
                            "temperature",
                            "keep_alive",
                            "timeout",
                        )
                    }
                    if llm
                    else None
                ),
                "summary": summarize(records),
                "records": records,
            }
        )
    report = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "top_k": args.top_k,
        "max_chunks": args.max_chunks,
        "repeat": args.repeat,
        "questions_sha256": hashlib.sha256(args.questions.read_bytes()).hexdigest(),
        "fixtures_sha256": hashlib.sha256(args.fixtures.read_bytes()).hexdigest(),
        "embedding_device": args.device if engine else None,
        "python_version": sys.version.split()[0],
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "index_sha256": (
            {
                name: hashlib.sha256((engine.index_dir / name).read_bytes()).hexdigest()
                for name in ("chunks.json", "census.faiss")
            }
            if engine
            else None
        ),
        "prompt_sha256": hashlib.sha256((BASE_DIR / "llm.py").read_bytes()).hexdigest(),
        "index_metadata": engine.metadata if engine else None,
        "limitations": "Automated numeric/citation/abstention checks are diagnostics, not proof of semantic correctness. Timings include model loading and warm-cache effects. Fixed fixtures are curated, not raw extraction.",
        "runs": runs,
    }
    return save_report(report, args.output, args.fail_on_checks)


def save_report(report, output, fail_on_checks):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    for run in report["runs"]:
        print(run["model"] or "retrieval", json.dumps(run["summary"], indent=2))
    print(f"Report: {output}")
    return int(
        fail_on_checks
        and any(
            "error" in r
            or r.get("generation", {}).get("automated_checks_passed") is False
            for run in report["runs"]
            for r in run["records"]
        )
    )


if __name__ == "__main__":
    sys.exit(main())
