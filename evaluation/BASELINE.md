# Observed evaluation baseline

31 starter questions; one generation attempt per model/question. This is a small benchmark concentrated in Volume I, with synthetic crop regressions clearly separated. These reports preserve failures, not just successful answers.

| Run | Cases | Automatic checks passed | Expected values present (answerable cases) | Median generation time |
| --- | ---: | ---: | ---: | ---: |
| Fixed evidence: qwen3:1.7b | 31 | 51.6% | 100.0% | 0.68s |
| Fixed evidence: qwen3:0.6b | 31 | 54.8% | 91.3% | 0.33s |
| Full pipeline: Qwen3 1.7B | 26 | 3.8% | 9.5% | 7.12s |

Retrieval found the labelled PDF source in the top five for 2 of 21 answerable questions (Hit@5: 9.5%; MRR: 0.024). Hit@1 and Hit@3 were both zero. Other retrieved pages can contain relevant numbers; page-hit metrics use the verified labelled sources and do not prove row/column correctness.

The 1.7B fixed-evidence run included every expected number, but missed citations in 12 answers. The 0.6B run omitted required numbers in two answers and triggered five distractor-number flags, including the maize regression. Some distractor flags are legitimate extra context rather than wrong claims; inspect the saved answer before judging them. Both models supplied unsupported causal explanations for population growth.

Automatic pass rates are **not semantic accuracy**. Each generation answer still needs review for county/column/year mapping, citation support, comparison direction, and unsupported prose. The smaller model having a higher aggregate automatic pass rate does not establish that it is a better model. Timings include loading, caching, and different answer lengths and are not controlled speed benchmarks.

Start improvements with retrieval/table extraction, then source citation compliance and unsupported-question handling. Rerun the same cases and retain a separate report to compare changes. Expand ground truth into other volumes before making deployment decisions.

Reports: [retrieval](results.json), [fixed-evidence model comparison](model-results.json), [full pipeline](pipeline-results.json). Commands, schema, and limitations: [evaluation guide](README.md).
