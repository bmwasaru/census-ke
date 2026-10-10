# Census evaluation

Run from the repository root using the application virtual environment:

```sh
.venv/bin/python -m evaluation.evaluate
.venv/bin/python -m evaluation.evaluate --mode generation --models qwen3:1.7b qwen3:0.6b --output evaluation/model-results.json
.venv/bin/python -m evaluation.evaluate --mode end-to-end --models qwen3:1.7b --output evaluation/pipeline-results.json
```

`retrieval` searches the real local index. `generation` supplies curated fixed evidence directly to each model, isolating model behavior from retrieval. `end-to-end` feeds actual search results to the model. Generation runs require local Ollama and installed model tags; the runner never pulls models or changes the application's model configuration. Embeddings load locally on CPU by default (`--device` can override this). One OpenMP thread and disabled tokenizer parallelism avoid the embedding crash observed on the local macOS environment.

## Questions and evidence

`questions.json` contains 31 starter cases: 21 answerable census questions, five unsupported questions eligible for retrieval and generation, and five generation-only regression cases. Categories include county population, sex columns, household count and size, population scope, paraphrases, Swahili, comparison, volume filtering, future years, unsupported causes and policy, missing crop columns, wrong-county evidence, and empty evidence.

`evidence.json` stores normalized, manually verified rows from Volume I, Table 2.2 (PDF page 17, printed page 7) and Table 2.3 (PDF page 19, printed page 9). Page labels in retrieval expectations always use **one-based PDF page numbers**. Table 2.3 excludes special populations; its population figures must not be used as total-population labels. The crop fixture comes from the existing unit-test regression example and is **synthetic**, not an official agriculture benchmark. It is excluded from retrieval and end-to-end runs. Empty evidence tests the application's early abstention path rather than model inference.

For each case specify:

- A unique `id`, `question`, `category`, and `suites` (`retrieval`, `generation`).
- `answerable`, relative to the supplied fixed evidence in generation mode.
- `expected_sources`: volume and acceptable PDF pages, required for answerable retrieval cases.
- `expected_values`: exact required numeric values; no substring matching.
- `forbidden_values`: known distractor counts that should not appear in this concise answer. This conservative check can flag a legitimate explanatory contrast; reviewers decide.
- `evidence_id`: a key in the evidence fixtures.
- Optional `volume`: restrict retrieval to one volume.
- Optional `allowed_derived_values`: explicitly verified calculations permitted beyond evidence numbers.

When adding ground truth, inspect the original PDF table and its footnotes, then record fixture provenance. Do not copy a model answer into ground truth. Add verified sub-county, age, education, disability, employment, housing, and official agriculture questions next; the current starter set is concentrated in Volume I and does not establish quality across all census volumes.

## Reports and interpretation

Reports record model/options, UTC timestamp, question/fixture/prompt-code SHA-256 hashes, index metadata, raw input evidence, the evidence actually supplied after deduplication, model answers, source labels, latency, errors, and per-question checks. `results.json` is the retrieval baseline, `model-results.json` compares fixed-evidence answers, and `pipeline-results.json` evaluates the real pipeline. These files contain observed results, not target scores.

Retrieval summaries use answerable cases only for Hit@1/3/5 and MRR. Unanswerable questions still retrieve evidence and remain in the report; page-hit metrics are not meaningful for them. A page hit alone does not establish that the correct table row or column was extracted. `expected_values_present` helps diagnose this, but a matching number elsewhere in a chunk can also be misleading.

Generation checks flag missing expected values, known distractors, unsupported numeric tokens, missing/unknown citations, unexpected abstention, and numeric answers to unanswerable questions. A disclaimer followed by a guessed count fails. Source labels are checked against labels actually supplied, not labels inferred from the number of retrieval hits. Exact numeric tokens accept comma formatting and decimals, excluding `[S1]` citation digits. Counts written in words, semantic column/entity errors, citation-to-claim support, and subtly unsupported prose require review.

**An automatic pass is not verified correctness.** Every generation result is marked `semantic_review: required`. Read the answer alongside its supplied evidence, checking the county, column, year, population scope, direction of comparison, citation support, abstention, and readability. In particular, an answer can swap two correct population numbers and still pass token checks. A unit test explicitly demonstrates this limitation. Do not use the automatic pass rate as a deployment accuracy score.

Errors count as failed generation attempts. Timing includes cold model loads, warm-cache effects, and output length, so medians from different sequential models are descriptive rather than a controlled performance benchmark.

## Focused runs and CI

```sh
.venv/bin/python -m evaluation.evaluate --mode generation --models qwen3:1.7b --ids crop_maize_column crop_missing_wheat --repeat 3 --output /tmp/crop-eval.json
.venv/bin/python -m evaluation.evaluate --mode generation --models qwen3:1.7b --fail-on-checks --output /tmp/check-eval.json
.venv/bin/python -m pytest tests/test_evaluation.py -q
```

`--repeat` repeats generation attempts; retrieval is performed once per question/model and reused within those attempts. `--top-k` controls search candidates; `--max-chunks` controls evidence passed to the model. `--questions` and `--fixtures` select alternative datasets. `--fail-on-checks` returns status 1 when a generation check fails or a request errors; retrieval misses alone do not trigger this flag. By default the runner completes and saves failures for inspection.

Saved answers can be regraded offline after changing the scoring code:

```sh
.venv/bin/python -m evaluation.evaluate --rescore evaluation/model-results.json --output evaluation/model-results.json
```

Rescoring checks that the question dataset hash matches, preserves the original answers, timings, and generation metadata, and adds a new scorer hash and timestamp. It does not invoke search or Ollama. Summary fields also break down expected-value presence, citation presence/validity, failure types, and question categories. Presence of an expected value is not proof that the model attributed it correctly.
