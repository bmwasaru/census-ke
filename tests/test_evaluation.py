import json
from pathlib import Path

import pytest

from evaluation.evaluate import (
    evaluate,
    hit_at_k,
    numbers,
    reciprocal_rank,
    score_answer,
    source_matches,
    summarize,
    validate_questions,
)
from llm import CensusLLM, LLMServiceError

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = json.loads((ROOT / "evaluation/evidence.json").read_text())
QUESTIONS = json.loads((ROOT / "evaluation/questions.json").read_text())


def question(id):
    return next(q for q in QUESTIONS if q["id"] == id)


def response(q, answer):
    llm = CensusLLM()
    _, sources, evidence = llm._prepare_evidence(FIXTURES[q["evidence_id"]]["results"])
    return {"answer": answer, "sources": sources, "evidence": evidence}


def grade(id, answer):
    q = question(id)
    return score_answer(q, response(q, answer), CensusLLM())


def test_dataset_labels_and_unique_ids():
    validate_questions(QUESTIONS, FIXTURES)
    assert question("mombasa_population")["expected_sources"][0]["pages"] == [17]
    assert "retrieval" not in question("crop_missing_wheat")["suites"]


def test_duplicate_question_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        validate_questions([QUESTIONS[0], QUESTIONS[0]], FIXTURES)


def test_numeric_tokens_are_exact_and_ignore_citations():
    assert numbers("1,208,333 people. [S1] Average 3.1.") == {"1208333", "3.1"}
    assert "5681" not in numbers("15,681 or 56810")
    assert numbers("0.0 and 1.00") == {"0", "1"}


def test_source_matching_requires_volume_and_pdf_page():
    expected = [{"volume": "I", "pages": [17]}]
    results = [{"volume": "IV", "page": 17}, {"volume": "I", "page": 17}]
    assert not source_matches(results[0], expected)
    assert not hit_at_k(results, expected, 1)
    assert hit_at_k(results, expected, 3)
    assert reciprocal_rank(results, expected) == 0.5
    assert reciprocal_rank([], expected) == 0


def test_correct_number_and_citation_pass():
    result = grade("crop_maize_column", "5,681 households grow maize. [S1]")
    assert result["automated_checks_passed"]
    assert result["semantic_review"] == "required"


def test_wrong_column_fails_even_when_correct_value_also_appears():
    result = grade(
        "crop_maize_column", "12,497 grow maize. The exact count is 5,681. [S1]"
    )
    assert "distractor_values_in_answer" in result["failures"]


def test_invented_count_and_source_are_flagged():
    result = grade("crop_maize_column", "6,681 households. [S9]")
    assert set(result["failures"]) >= {
        "unsupported_numbers",
        "invalid_citations",
        "missing_expected_values",
    }


def test_missing_citation_fails():
    assert (
        "missing_citation"
        in grade("crop_maize_column", "5,681 households.")["failures"]
    )


def test_missing_wheat_requires_abstention_without_count():
    assert grade(
        "crop_missing_wheat",
        "The evidence does not provide enough information to answer this question.",
    )["automated_checks_passed"]
    result = grade(
        "crop_missing_wheat",
        "The evidence does not provide wheat data, but 12,497 grow wheat. [S1]",
    )
    assert "numeric_claim_on_unanswerable_question" in result["failures"]


def test_unanswerable_year_can_be_repeated_without_being_a_count():
    assert grade(
        "future_population", "The evidence does not provide the population for 2024."
    )["automated_checks_passed"]


def test_decimal_wrong_value_fails():
    result = grade("mombasa_household_size", "Average household size was 3.0. [S1]")
    assert "missing_expected_values" in result["failures"]
    assert "unsupported_numbers" in result["failures"]


def test_numeric_checks_cannot_prove_entity_mapping():
    # Both values are present with valid citations, but the entities are swapped.
    # The report must never describe this as verified semantic correctness.
    result = grade(
        "population_comparison", "Nairobi: 1,208,333. Mombasa: 4,397,073. [S1] [S2]"
    )
    assert result["automated_checks_passed"]
    assert result["semantic_review"] == "required"


def test_generation_does_not_call_retrieval_and_records_repeats(monkeypatch):
    llm = CensusLLM()
    q = question("crop_maize_column")
    monkeypatch.setattr(
        llm, "answer", lambda *args, **kwargs: response(q, "5,681 households. [S1]")
    )
    records = evaluate([q], FIXTURES, "generation", llm=llm, repeat=2)
    assert len(records) == 2
    assert records[0]["retrieval"] is None
    assert records[0]["supplied_evidence"]
    assert summarize(records)["automated_check_pass_rate"] == 1


def test_request_errors_count_as_failures_and_do_not_stop_run(monkeypatch):
    llm = CensusLLM()

    def fail(*args, **kwargs):
        raise LLMServiceError("Unavailable")

    monkeypatch.setattr(llm, "answer", fail)
    records = evaluate(
        [question("crop_maize_column"), question("crop_missing_wheat")],
        FIXTURES,
        "generation",
        llm=llm,
    )
    summary = summarize(records)
    assert summary["generation_errors"] == 2
    assert summary["automated_check_pass_rate"] == 0
    assert summary["unanswerable_check_pass_rate"] == 0


def test_unanswerable_questions_are_retrieved_but_excluded_from_hit_metrics():
    class Engine:
        def search(self, query, **kwargs):
            return []

    records = evaluate(
        [question("future_population")], FIXTURES, "retrieval", engine=Engine()
    )
    assert len(records) == 1
    assert summarize(records)["hit@1"] is None


def test_end_to_end_records_missing_retrieval_evidence(monkeypatch):
    class Engine:
        def search(self, query, **kwargs):
            return []

    records = evaluate(
        [question("mombasa_population")],
        FIXTURES,
        "end-to-end",
        engine=Engine(),
        llm=CensusLLM(),
    )
    assert records[0]["retrieval"]["hit@5"] is False
    assert "missing_expected_values" in records[0]["generation"]["failures"]


def test_signed_and_malformed_numbers():
    assert numbers("-5,681 and +2.9") == {"-5681", "2.9"}
    assert numbers("12,34") == set()
    assert (
        "missing_expected_values"
        in grade("crop_maize_column", "-5,681 households. [S1]")["failures"]
    )


def test_repeated_generation_does_not_duplicate_retrieval_denominator():
    r = {
        "id": "same",
        "answerable": True,
        "retrieval": {
            "hit@1": True,
            "hit@3": True,
            "hit@5": True,
            "reciprocal_rank": 1,
        },
    }
    assert summarize([r, r])["retrieval_questions"] == 1


def test_cli_saves_errors_and_returns_failure_status(tmp_path, monkeypatch):
    from evaluation.evaluate import main

    def fail(*args, **kwargs):
        raise LLMServiceError("Unavailable")

    monkeypatch.setattr(CensusLLM, "answer", fail)
    output = tmp_path / "results.json"
    status = main(
        [
            "--mode",
            "generation",
            "--models",
            "test-model",
            "--ids",
            "crop_maize_column",
            "--output",
            str(output),
            "--fail-on-checks",
        ]
    )
    assert status == 1
    report = json.loads(output.read_text())
    assert report["runs"][0]["summary"]["generation_errors"] == 1
    assert report["runs"][0]["records"][0]["error"]["message"] == "Unavailable"
    assert report["questions_sha256"]


def test_rescoring_uses_saved_answers_without_generation(monkeypatch):
    from evaluation.evaluate import rescore_report

    def fail(*args, **kwargs):
        raise AssertionError("Rescoring must not generate an answer")

    monkeypatch.setattr(CensusLLM, "answer", fail)
    q = question("crop_maize_column")
    saved = response(q, "12,497 households grow maize. [S1]")
    report = {
        "runs": [
            {
                "model": "test-model",
                "records": [
                    {
                        "id": q["id"],
                        "answerable": True,
                        "answer": saved["answer"],
                        "sources": saved["sources"],
                        "supplied_evidence": saved["evidence"],
                        "generation_seconds": 1,
                    }
                ],
            }
        ]
    }
    rescore_report(report, [q])
    assert report["runs"][0]["summary"]["automated_check_pass_rate"] == 0
    assert report["rescored_at"]
