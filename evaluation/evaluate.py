import json
from pathlib import Path

from search_engine import CensusSearchEngine

BASE_DIR = Path(__file__).resolve().parent.parent

QUESTIONS_PATH = BASE_DIR / "evaluation" / "questions.json"

INDEX_DIR = BASE_DIR / "index"

MODEL_DIR = BASE_DIR / "models" / "bge-small-en-v1.5"


def source_matches(
    result,
    expected_sources,
):
    for source in expected_sources:

        if result.get("volume") != source["volume"]:
            continue

        if result.get("page") in source["pages"]:
            return True

    return False


def reciprocal_rank(
    results,
    expected_sources,
):
    for rank, result in enumerate(
        results,
        start=1,
    ):

        if source_matches(
            result,
            expected_sources,
        ):
            return 1 / rank

    return 0


def hit_at_k(
    results,
    expected_sources,
    k,
):
    return any(
        source_matches(
            result,
            expected_sources,
        )
        for result in results[:k]
    )


def main():

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))

    engine = CensusSearchEngine(
        index_dir=INDEX_DIR,
        model_path=MODEL_DIR,
    )

    evaluated = []

    hit1 = 0
    hit3 = 0
    hit5 = 0

    reciprocal_ranks = []

    for question in questions:

        # Retrieval cannot be evaluated against
        # an expected page for intentionally
        # unanswerable questions.
        if not question["answerable"]:
            continue

        results = engine.search(
            question["question"],
            top_k=5,
        )

        expected = question["expected_sources"]

        h1 = hit_at_k(
            results,
            expected,
            1,
        )

        h3 = hit_at_k(
            results,
            expected,
            3,
        )

        h5 = hit_at_k(
            results,
            expected,
            5,
        )

        rr = reciprocal_rank(
            results,
            expected,
        )

        hit1 += int(h1)
        hit3 += int(h3)
        hit5 += int(h5)

        reciprocal_ranks.append(rr)

        evaluated.append(
            {
                "id": question["id"],
                "question": question["question"],
                "hit@1": h1,
                "hit@3": h3,
                "hit@5": h5,
                "reciprocal_rank": round(rr, 3),
                "returned": [
                    {
                        "rank": result["rank"],
                        "volume": result["volume"],
                        "page": result["page"],
                        "score": result["score"],
                    }
                    for result in results
                ],
            }
        )

    total = len(evaluated)

    if total == 0:
        print("No answerable questions " "to evaluate.")
        return

    mrr = sum(reciprocal_ranks) / total

    print()
    print("Retrieval Evaluation")

    print("====================")

    print(f"Questions: {total}")

    print(f"Hit@1: " f"{hit1 / total:.1%}")

    print(f"Hit@3: " f"{hit3 / total:.1%}")

    print(f"Hit@5: " f"{hit5 / total:.1%}")

    print(f"MRR: " f"{mrr:.3f}")

    output = BASE_DIR / "evaluation" / "results.json"

    output.write_text(
        json.dumps(
            evaluated,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(f"Detailed results: {output}")


if __name__ == "__main__":
    main()
