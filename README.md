# Kenya Census Semantic Search

A local-first semantic search and retrieval-augmented generation (RAG) application for exploring the **2019 Kenya Population and Housing Census (KPHC)** using natural-language questions.

The application combines semantic retrieval, structured table extraction, FAISS vector search, and a small local language model to help users find and understand census information without manually navigating large PDF reports.

Example questions include:

* What was the population of Mombasa County?
* How many households in Mombasa grow maize?

The application is designed around one core principle:

> **The census documents remain the source of truth.**

The language model does not independently answer census questions. It explains evidence retrieved directly from indexed census documents.

---

## Features

The project currently supports:

* Natural-language semantic search
* Local embedding generation
* FAISS vector search
* Search across multiple census volumes
* Filtering by census volume
* Structured extraction of census tables
* Row-level indexing of census tables
* Local RAG using Qwen3
* Plain-language explanations of census statistics
* Evidence-backed answers
* Source citations such as `[S1]` and `[S2]`
* Links to the exact PDF page supporting a result
* Expandable original evidence
* Local/offline embedding model operation
* Local LLM inference through Ollama
* REST API endpoints
* Automated tests with `pytest`
* Retrieval and LLM evaluation tooling

---

# Architecture

The application uses a retrieval-augmented generation pipeline.

```text
User Question
      │
      ▼
BAAI/bge-small-en-v1.5
      │
      ▼
Query Embedding
      │
      ▼
FAISS Vector Search
      │
      ▼
Relevant Census Evidence
      │
      ├── Text chunks
      │
      └── Structured table rows
      │
      ▼
Qwen3 1.7B
      │
      ▼
Plain-language answer
      │
      ├── Source citations
      ├── Original evidence
      └── Exact PDF page links
```

The language model only receives the evidence returned by semantic search. It is instructed not to introduce unsupported information.

---

# Models Used

## 1. Embedding Model

The semantic search system uses:

```text
BAAI/bge-small-en-v1.5
```

The embedding model converts both census content and user questions into numerical vector representations.

These vectors are compared using FAISS.

### Why BGE Small?

`BAAI/bge-small-en-v1.5` was selected because it provides a useful balance between:

* retrieval quality
* model size
* CPU performance
* memory requirements
* local deployment
* low-cost server deployment

This makes it practical for deployment on relatively small DigitalOcean servers.

The embedding model is downloaded once and stored locally:

```text
models/
└── bge-small-en-v1.5/
```

The application then loads it with:

```python
SentenceTransformer(
    str(model_path),
    local_files_only=True,
)
```

The query is encoded with the BGE retrieval instruction:

```text
Represent this sentence for searching relevant passages:
```

---

## 2. Language Model

The explanation layer uses:

```text
Qwen3 1.7B
```

through **Ollama**.

Default model configuration:

```env
OLLAMA_MODEL=qwen3:1.7b
OLLAMA_NUM_CTX=4096
OLLAMA_NUM_PREDICT=450
OLLAMA_TEMPERATURE=0.1
OLLAMA_KEEP_ALIVE=5m
```

Qwen is not treated as the source of census knowledge.

Its role is to:

* interpret retrieved census evidence
* explain statistics clearly
* summarize relevant values
* make simple comparisons when evidence supports them
* preserve census numbers
* cite evidence using labels such as `[S1]`

The model is instructed to abstain when retrieved evidence is insufficient.

For example:

```text
The retrieved census information does not provide enough information to answer this question.
```

The model is also instructed to return plain text rather than Markdown-heavy output.

The application performs an additional cleanup step to remove formatting such as:

```text
#
##
**
__
```

if the model generates them despite the prompt.

---

# Census Data

The application is designed around the **2019 Kenya Population and Housing Census**.

The initial dataset includes the major census volumes used by the project, including:

```text
Volume I
Population by County and Sub-County

Volume II
Distribution of Population by Administrative Units

Volume III
Distribution of Population by Age and Sex

Volume IV
Distribution of Population by Socio-Economic Characteristics
```

PDF files are stored under:

```text
data/
```

Example:

```text
data/
├── 2019-kphc-volume-1.pdf
├── 2019-kphc-volume-2.pdf
├── 2019-kphc-volume-3.pdf
└── 2019-kphc-volume-4.pdf
```

Additional census documents can be added to the ingestion process.

---

# Structured Census Table Extraction

One of the main challenges when working with census PDFs is that tables may be extracted as unreadable flattened text.

For example:

```text
MOMBASA 12,497 5,681 - - - 2,849 - - 2,232...
```

This format is difficult for:

* semantic search
* users
* language models
* numerical verification

The ingestion pipeline therefore detects tables using PyMuPDF and converts them into structured row-level records.

For example:

```json
{
  "type": "table",
  "volume": "IV",
  "page": 215,
  "table_number": "2.22",
  "table_title": "Distribution of Households Growing Other Crops",
  "row": {
    "County/Sub County": "MOMBASA",
    "Households Farming": "12,497",
    "Maize": "5,681",
    "Beans": "2,849",
    "Green grams": "2,232"
  }
}
```

The same row is converted into semantic-search text:

```text
Table 2.22: Distribution of Households Growing Other Crops.
County/Sub County: MOMBASA.
Households Farming: 12,497.
Maize: 5,681.
Beans: 2,849.
Green grams: 2,232.
```

This representation is embedded with BGE.

The original structured row is retained for display and LLM evidence.

This allows a query such as:

```text
How many households in Mombasa grow maize?
```

to retrieve a precise row instead of an entire multi-page table.

---

# Search Index

The ingestion process creates three files:

```text
index/
├── census.faiss
├── chunks.json
└── metadata.json
```

### `census.faiss`

Contains the vector index used for similarity search.

### `chunks.json`

Contains the metadata associated with each vector.

Records may be either:

```text
type: text
```

or:

```text
type: table
```

### `metadata.json`

Stores information about the generated index.

Example:

```json
{
  "year": 2019,
  "model": "BAAI/bge-small-en-v1.5",
  "documents": 4,
  "chunks": 18442,
  "text_chunks": 10891,
  "table_rows": 7551,
  "embedding_dimensions": 384
}
```

---

# Exact Page Citations

Every indexed record retains:

```text
source
page
```

The web application exposes indexed PDFs through:

```text
/census-pdf/<filename>
```

A search result can therefore link directly to a supporting page:

```text
/census-pdf/2019-kphc-volume-4.pdf#page=215
```

The interface displays links such as:

```text
Open census page ↗
```

and:

```text
View original page ↗
```

This allows users to verify answers against the original census report.

Exact-page navigation depends on the browser's PDF viewer. Most modern desktop browsers support the `#page=` fragment.

---

# Project Structure

The project structure is:

```text
kenya-census-semantic-search/
│
├── app.py
├── ingest.py
├── search_engine.py
├── llm.py
├── census_sources.py
│
├── download_model.py
├── download_census.py
│
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── .env
├── .env.example
├── .gitignore
├── README.md
│
├── data/
│   ├── 2019-kphc-volume-1.pdf
│   ├── 2019-kphc-volume-2.pdf
│   ├── 2019-kphc-volume-3.pdf
│   └── 2019-kphc-volume-4.pdf
│
├── models/
│   └── bge-small-en-v1.5/
│
├── index/
│   ├── census.faiss
│   ├── chunks.json
│   └── metadata.json
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
├── evaluation/
│   ├── __init__.py
│   ├── evaluate.py
│   ├── answer_checks.py
│   ├── questions.json
│   └── results.json
│
└── tests/
    ├── conftest.py
    ├── test_app.py
    ├── test_app_ask.py
    ├── test_ingest.py
    ├── test_llm.py
    └── test_search_engine.py
```

---

# Installation

Python 3.11+ is recommended.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

macOS/Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip

python -m pip install -r requirements.txt
```

For development and testing:

```bash
python -m pip install -r requirements-dev.txt
```

---

# Environment Configuration

Create:

```text
.env
```

Example:

```env
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1

EMBEDDING_MODEL_PATH=models/bge-small-en-v1.5

OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3:1.7b
OLLAMA_TIMEOUT=120
OLLAMA_NUM_CTX=4096
OLLAMA_NUM_PREDICT=450
OLLAMA_TEMPERATURE=0.1
OLLAMA_KEEP_ALIVE=5m

HOST=127.0.0.1
PORT=5000
FLASK_DEBUG=1
```

---

# Download the Embedding Model

The embedding model only needs to be downloaded once.

Run:

```bash
python download_model.py
```

The application itself then operates with:

```env
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

and:

```python
local_files_only=True
```

This helps avoid unexpected internet requests during search and ingestion.

---

# Install Qwen with Ollama

Install Ollama, then download the model:

```bash
ollama pull qwen3:1.7b
```

Verify:

```bash
ollama list
```

You should see:

```text
qwen3:1.7b
```

You can test it directly:

```bash
ollama run qwen3:1.7b
```

---

# Download Census Files

If using the project downloader:

```bash
python download_census.py
```

Alternatively, place the census PDFs manually inside:

```text
data/
```

---

# Build the Search Index

Run:

```bash
python ingest.py
```

The process:

1. discovers census PDFs
2. extracts ordinary text
3. detects census tables
4. converts table rows into structured records
5. removes detected tables from normal prose extraction
6. creates BGE embeddings
7. normalizes the embedding vectors
8. creates a FAISS inner-product index
9. saves the vectors and metadata

The output should include a breakdown such as:

```text
Extraction complete.

Total records: 18,442
Text chunks: 10,891
Table rows: 7,551
```

Whenever the extraction format changes, rebuild the index:

```bash
rm -f index/census.faiss
rm -f index/chunks.json
rm -f index/metadata.json

python ingest.py
```

---

# Running the Application

Make sure Ollama is available:

```bash
ollama list
```

Start Flask:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

---

# API

## Ask a Census Question

```http
POST /api/ask
```

Example request:

```json
{
  "query": "How many households in Mombasa grow maize?",
  "volume": "IV",
  "top_k": 6
}
```

Example response:

```json
{
  "query": "How many households in Mombasa grow maize?",
  "volume": "IV",
  "answer": "In Mombasa, 5,681 households reported growing maize. [S1]",
  "sources": [
    {
      "id": "S1",
      "volume": "IV",
      "page": 215,
      "table_number": "2.22",
      "page_url": "/census-pdf/2019-kphc-volume-4.pdf#page=215"
    }
  ],
  "evidence": []
}
```

---

## Raw Semantic Search

```http
POST /api/search
```

Example:

```json
{
  "query": "maize farming in Mombasa",
  "top_k": 5
}
```

This endpoint returns retrieval results without asking Qwen to generate an explanation.

It is particularly useful for debugging and evaluation.

---

## Health Check

```http
GET /health
```

Reports:

* index status
* embedding-model status
* indexed volumes
* index metadata
* Ollama availability
* Qwen model availability

---

# Testing

Run tests using the same Python environment as the application:

```bash
python -m pytest -v
```

Using:

```bash
python -m pytest
```

instead of only:

```bash
pytest
```

helps avoid situations where `pytest` points to a different Python environment.

Run coverage:

```bash
python -m pytest \
    --cov=app \
    --cov=ingest \
    --cov=search_engine \
    --cov=llm \
    --cov-report=term-missing
```

The automated test suite covers:

* text cleaning
* chunking
* table extraction
* duplicate table headers
* structured row generation
* exclusion of flattened tables from prose
* FAISS index loading
* semantic-search result structure
* volume filtering
* table-result preservation
* LLM evidence preparation
* source IDs
* Qwen request configuration
* Markdown cleanup
* LLM failures
* Flask search APIs
* exact-page URLs
* indexed PDF delivery
* path-traversal protection

---

# Evaluation

Automated unit tests confirm that the application behaves as expected, but they do not demonstrate that search results are actually correct.

The project therefore uses a separate evaluation process for retrieval and generated answers.

---

## Evaluation Dataset

Evaluation questions are stored in:

```text
evaluation/questions.json
```

Questions should be manually verified against the original census documents.

The starter benchmark now contains 31 cases with verified population and household labels plus clearly marked synthetic regressions. It supports retrieval, fixed-evidence generation, and full-pipeline evaluation. See [the evaluation guide](evaluation/README.md) for the dataset schema, commands, report fields, and automatic-grading limitations.

A useful evaluation set should include a mixture of:

* direct fact lookup
* county questions
* sub-county questions
* demographic questions
* agriculture questions
* socio-economic questions
* table queries
* paraphrased queries
* comparison questions
* ambiguous questions
* intentionally unanswerable questions

A starting dataset of around 50–100 manually verified questions is recommended.

---

# Retrieval Evaluation

Semantic retrieval is evaluated separately from Qwen.

Important metrics include:

## Hit@1

Did the first result contain the expected evidence?

```text
Hit@1
```

## Hit@3

Was correct evidence within the first three results?

```text
Hit@3
```

## Hit@5

Was correct evidence within the first five results?

```text
Hit@5
```

## Mean Reciprocal Rank

MRR measures how high the first relevant result appears.

If the correct result is:

```text
rank 1 → reciprocal rank 1.0
rank 2 → reciprocal rank 0.5
rank 4 → reciprocal rank 0.25
```

The mean is calculated over the evaluation dataset.

Possible internal quality targets are:

```text
Hit@1 >= 80%
Hit@3 >= 90%
Hit@5 >= 95%
MRR >= 0.80
```

These are project quality targets, not universal standards.

Run retrieval evaluation with:

```bash
python -m evaluation.evaluate
```

Using module execution is recommended so imports resolve from the project root.

The default run saves a structured report with metadata, summaries, and per-question evidence to:

```text
evaluation/results.json
```

---

# LLM Evaluation

The Qwen answer should be evaluated separately from retrieval.

A retrieval result can be correct while the generated explanation is wrong.

Likewise, Qwen cannot produce the correct answer if the correct evidence was never retrieved.

The project therefore evaluates several dimensions.

---

## Numeric Faithfulness

Every important number in the generated answer should exist in the retrieved evidence or be a clearly supported transformation.

Example evidence:

```text
Mombasa population: 1,208,333
```

Good answer:

```text
Mombasa had 1,208,333 residents. [S1]
```

Potential hallucination:

```text
Mombasa had 1,308,333 residents. [S1]
```

The second answer fails numeric faithfulness.

---

## Citation Validity

Every generated source label should correspond to evidence actually supplied to Qwen.

If the available evidence is:

```text
[S1]
[S2]
```

the model should not generate:

```text
[S5]
```

---

## Evidence Faithfulness

Every factual statement should be supported by retrieved census evidence.

The model should not infer unsupported explanations such as:

```text
The population increased because of migration.
```

unless the supplied census evidence explicitly establishes that conclusion.

---

## Abstention Accuracy

The system should recognize when retrieved material cannot support an answer.

Example:

```text
What was Nairobi's population in 2024?
```

If the indexed dataset only contains the 2019 census, an appropriate response is an abstention rather than an invented value.

---

# Human Evaluation

Automated checks are useful, but human review remains important.

Answers can be scored from 1–5 for:

* factual accuracy
* evidence faithfulness
* clarity
* citation correctness
* usefulness

Possible internal targets:

```text
Accuracy: >= 4.5 / 5
Clarity: >= 4.0 / 5
Usefulness: >= 4.0 / 5
```

Critical numerical errors should ideally be:

```text
0
```

---

# Why Retrieval and Generation Are Evaluated Separately

RAG has two major failure points.

### Retrieval failure

The correct census passage is not returned.

```text
Question
   ↓
wrong evidence
   ↓
LLM cannot reliably answer
```

### Generation failure

Correct evidence is retrieved but the model misinterprets or changes it.

```text
Question
   ↓
correct evidence
   ↓
incorrect explanation
```

Separating these evaluations makes it easier to identify whether improvements should be made to:

* ingestion
* chunking
* table parsing
* embeddings
* search ranking
* prompting
* evidence formatting
* Qwen configuration

---

# Important Design Decisions

## Retrieval Is the Source of Truth

Qwen should never be treated as the census database.

The authoritative flow is:

```text
Census PDF
    ↓
Retrieved evidence
    ↓
Qwen explanation
```

not:

```text
Question
    ↓
Qwen memory
```

---

## Tables Are Indexed at Row Level

Large tables are not embedded as one giant chunk.

This improves precision for questions involving:

* counties
* sub-counties
* age
* sex
* agriculture
* employment
* household characteristics

---

## Exact Values Are Preserved

Census statistics should normally retain their original value.

For example:

```text
1,208,333
```

rather than only:

```text
1.2 million
```

A user-friendly answer may say:

```text
About 1.21 million people (1,208,333). [S1]
```

---

## Python Should Handle Important Calculations

For calculations involving:

* percentages
* totals
* growth rates
* ratios
* differences

the preferred future architecture is:

```text
retrieved values
      ↓
Python calculation
      ↓
Qwen explanation
```

rather than expecting the language model to perform all numerical calculations.

---

# Current Limitations

The project still has several important limitations.

### PDF table extraction

Census reports contain complex multi-row and multi-column tables.

PyMuPDF table detection may not perfectly interpret every table.

Some tables may require custom normalization.

### PDF page numbers

The indexed page number is the PDF page number detected during extraction.

This may occasionally differ from a page number printed inside the census report.

### Browser PDF navigation

`#page=N` works in most desktop browser PDF viewers but may behave differently on mobile devices or external PDF applications.

### Retrieval quality

Embedding similarity does not guarantee that the highest-ranked passage is factually the correct one.

This is why evaluation is required.

### LLM hallucination

Qwen can still make mistakes despite strict prompting.

Answers should therefore always remain linked to underlying evidence.

### Complex calculations

The current LLM is best suited to explanation rather than complex statistical analysis.

Complex computations should be handled deterministically in Python.

---

# Deployment

The architecture was selected with affordable CPU deployment in mind.

A practical starting point is a small DigitalOcean VPS running:

```text
Flask application
+
BGE-small
+
FAISS
+
Ollama
+
Qwen3 1.7B
```

A server with approximately:

```text
2 vCPU
4 GB RAM
```

can be used as an initial deployment target, although actual memory and response-time requirements should be tested using the final index and expected traffic.

For production use, Flask's development server should not be exposed directly.

A typical deployment would use:

```text
Nginx
   ↓
Gunicorn
   ↓
Flask
   ↓
FAISS + BGE
   ↓
Ollama / Qwen3
```

---

# Offline Operation

The embedding component can run entirely from local files.

Set:

```env
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

and load the embedding model with:

```python
SentenceTransformer(
    str(model_path),
    local_files_only=True,
)
```

Ollama also runs Qwen locally.

Once the required models and census documents are downloaded, the primary search and answer pipeline therefore does not require external AI APIs.

# Development Workflow

Typical first-time setup:

```bash
python -m venv .venv

source .venv/bin/activate

python -m pip install --upgrade pip

python -m pip install -r requirements-dev.txt

python download_model.py

python download_census.py

python ingest.py

ollama pull qwen3:1.7b

python -m pytest -v

python -m evaluation.evaluate

python app.py
```

Normal development startup after setup:

```bash
source .venv/bin/activate

ollama list

python app.py
```

---

# Rebuilding After Census Data Changes

If PDFs or extraction logic change, rebuild the semantic index:

```bash
rm -f index/census.faiss
rm -f index/chunks.json
rm -f index/metadata.json

python ingest.py
```

The index should also be rebuilt after changes to:

* chunking logic
* table extraction
* table normalization
* embedding model
* embedding instructions

Changes limited to:

* frontend styling
* Qwen prompting
* Flask presentation

normally do not require re-indexing.

