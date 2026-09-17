import os
from pathlib import Path
from unittest.mock import Mock

from dotenv import load_dotenv


load_dotenv()


from flask import (
    Flask,
    abort,
    jsonify,
    render_template,
    request,
    send_from_directory,
    url_for,
)

from llm import (
    CensusLLM,
    LLMServiceError,
)

from search_engine import (
    CensusSearchEngine,
)


BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
)


DATA_DIR = (
    BASE_DIR
    / "data"
)


INDEX_DIR = (
    BASE_DIR
    / "index"
)


# ---------------------------------------------------------
# LOCAL BGE PATH
# ---------------------------------------------------------

configured_model_path = (
    os.getenv(
        "EMBEDDING_MODEL_PATH"
    )
)


if configured_model_path:

    MODEL_DIR = Path(
        configured_model_path
    )

    if not MODEL_DIR.is_absolute():

        MODEL_DIR = (
            BASE_DIR
            / MODEL_DIR
        )

    MODEL_DIR = (
        MODEL_DIR.resolve()
    )

else:

    MODEL_DIR = (
        BASE_DIR
        / "models"
        / "bge-small-en-v1.5"
    )


# ---------------------------------------------------------
# APP
# ---------------------------------------------------------

app = Flask(
    __name__
)


engine = CensusSearchEngine(
    index_dir=
        INDEX_DIR,

    model_path=
        MODEL_DIR,
)


llm = CensusLLM()


# ---------------------------------------------------------
# PAGE URL HELPERS
# ---------------------------------------------------------

def source_page_url(
    source,
    page,
):
    """
    Build a URL that opens a local census PDF
    at the exact page.

    Example:

    /census-pdf/2019-kphc-volume-4.pdf#page=215
    """

    if not source:
        return None

    if page is None:
        return None

    try:
        page_number = int(
            page
        )

    except (
        TypeError,
        ValueError,
    ):
        return None

    base_url = url_for(
        "census_pdf",
        filename=source,
    )

    return (
        f"{base_url}"
        f"#page={page_number}"
    )


def add_page_urls(
    records,
):
    """
    Add page_url to retrieval results,
    Qwen sources and evidence records.
    """

    enriched = []

    for record in records or []:

        item = dict(
            record
        )

        page_url = source_page_url(
            item.get(
                "source"
            ),
            item.get(
                "page"
            ),
        )

        if page_url:
            item[
                "page_url"
            ] = page_url

        enriched.append(
            item
        )

    return enriched


# ---------------------------------------------------------
# REQUEST PARAMETERS
# ---------------------------------------------------------

def get_request_parameters():

    payload = (
        request.get_json(
            silent=True
        )
        or {}
    )

    query = (
        payload.get(
            "query"
        )
        or
        request.args.get(
            "q"
        )
        or ""
    ).strip()

    volume = (
        payload.get(
            "volume"
        )
        or
        request.args.get(
            "volume"
        )
        or ""
    ).strip() or None

    try:

        top_k = int(
            payload.get(
                "top_k"
            )
            or
            request.args.get(
                "top_k"
            )
            or 6
        )

    except (
        TypeError,
        ValueError,
    ):

        top_k = 6

    top_k = max(
        1,
        min(
            top_k,
            20,
        ),
    )

    return (
        query,
        volume,
        top_k,
    )


def validate_search_request(
    query,
    volume,
):

    if not query:

        return (
            jsonify(
                {
                    "error":
                        (
                            "Please provide "
                            "a search query."
                        )
                }
            ),
            400,
        )

    if not engine.is_ready():

        return (
            jsonify(
                {
                    "error":
                        (
                            "The census search "
                            "index has not been "
                            "created yet. Run "
                            "`python ingest.py`."
                        )
                }
            ),
            503,
        )

    if not engine.model_path.exists():

        return (
            jsonify(
                {
                    "error":
                        (
                            "The local BGE "
                            "embedding model is "
                            "missing."
                        )
                }
            ),
            503,
        )

    valid_volumes = {
        item[
            "volume"
        ]
        for item
        in engine.volumes
    }

    if (
        volume
        and
        volume
        not in valid_volumes
    ):

        return (
            jsonify(
                {
                    "error":
                        (
                            "Unknown or "
                            "unindexed volume: "
                            f"{volume}"
                        )
                }
            ),
            400,
        )

    return None


# ---------------------------------------------------------
# MAIN PAGE
# ---------------------------------------------------------

@app.get("/")
def home():

    return render_template(
        "index.html",

        indexed=
            engine.is_ready(),

        volumes=
            engine.volumes,
    )


# ---------------------------------------------------------
# SERVE LOCAL CENSUS PDFs
# ---------------------------------------------------------

@app.get(
    "/census-pdf/<path:filename>"
)
def census_pdf(
    filename,
):
    """
    Serve only PDFs that belong to the
    currently indexed census sources.
    """

    safe_name = (
        Path(filename).name
    )

    # Do not allow directory traversal or
    # nested paths.
    if safe_name != filename:
        abort(
            404
        )

    allowed_sources = set(
        engine.metadata.get(
            "sources",
            [],
        )
    )

    if (
        safe_name
        not in allowed_sources
    ):
        abort(
            404
        )

    pdf_path = (
        DATA_DIR
        / safe_name
    )

    if not pdf_path.exists():
        abort(
            404
        )

    return send_from_directory(
        DATA_DIR,
        safe_name,
        mimetype=
            "application/pdf",
        as_attachment=
            False,
        conditional=
            True,
    )


# ---------------------------------------------------------
# HEALTH
# ---------------------------------------------------------

@app.get("/health")
def health():

    return jsonify(
        {
            "ok":
                True,

            "search": {
                "indexed":
                    engine.is_ready(),

                "chunks":
                    engine.chunk_count,

                "model_available":
                    engine.model_path.exists(),

                "metadata":
                    engine.metadata,
            },

            "llm":
                llm.health(),

            "volumes":
                engine.volumes,
        }
    )


# ---------------------------------------------------------
# VOLUMES
# ---------------------------------------------------------

@app.get("/api/volumes")
def api_volumes():

    return jsonify(
        {
            "volumes":
                engine.volumes
        }
    )


# ---------------------------------------------------------
# SEMANTIC SEARCH
# ---------------------------------------------------------

@app.route(
    "/api/search",
    methods=[
        "GET",
        "POST",
    ],
)
def api_search():

    (
        query,
        volume,
        top_k,
    ) = get_request_parameters()

    error = validate_search_request(
        query,
        volume,
    )

    if error:
        return error

    results = engine.search(
        query=
            query,

        top_k=
            top_k,

        volume=
            volume,
    )

    results = add_page_urls(
        results
    )

    return jsonify(
        {
            "query":
                query,

            "volume":
                volume or "all",

            "results":
                results,
        }
    )


# ---------------------------------------------------------
# RAG ANSWER
# ---------------------------------------------------------

@app.route(
    "/api/ask",
    methods=[
        "GET",
        "POST",
    ],
)
def api_ask():

    (
        query,
        volume,
        top_k,
    ) = get_request_parameters()

    error = validate_search_request(
        query,
        volume,
    )

    if error:
        return error

    results = engine.search(
        query=
            query,

        top_k=
            top_k,

        volume=
            volume,
    )

    if not results:

        return jsonify(
            {
                "query":
                    query,

                "volume":
                    volume or "all",

                "answer":
                    (
                        "I could not find "
                        "enough relevant census "
                        "information to answer "
                        "this question."
                    ),

                "sources":
                    [],

                "evidence":
                    [],
            }
        )

    try:

        generated = llm.answer(
            question=
                query,

            search_results=
                results,

            max_chunks=
                5,
        )

    except LLMServiceError as exc:

        return (
            jsonify(
                {
                    "error":
                        str(
                            exc
                        ),

                    "results":
                        add_page_urls(
                            results
                        ),
                }
            ),
            503,
        )

    sources = add_page_urls(
        generated[
            "sources"
        ]
    )

    evidence = add_page_urls(
        generated[
            "evidence"
        ]
    )

    return jsonify(
        {
            "query":
                query,

            "volume":
                volume or "all",

            "answer":
                generated[
                    "answer"
                ],

            "sources":
                sources,

            "evidence":
                evidence,
        }
    )


# ---------------------------------------------------------
# DEV SERVER
# ---------------------------------------------------------

if __name__ == "__main__":

    app.run(
        host=os.getenv(
            "HOST",
            "127.0.0.1",
        ),

        port=int(
            os.getenv(
                "PORT",
                "5000",
            )
        ),

        debug=(
            os.getenv(
                "FLASK_DEBUG",
                "1",
            )
            == "1"
        ),
    )
