import argparse
import json
import re
from pathlib import Path

from dotenv import load_dotenv


# Load environment variables before importing
# SentenceTransformers / Hugging Face libraries.
load_dotenv()


import faiss
import fitz
import numpy as np

from sentence_transformers import SentenceTransformer

from census_sources import SOURCE_BY_FILENAME


BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

DEFAULT_INDEX_DIR = BASE_DIR / "index"

DEFAULT_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "bge-small-en-v1.5"
)


# ---------------------------------------------------------
# GENERAL TEXT CLEANING
# ---------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Clean text extracted from census PDF pages.
    """

    if not text:
        return ""

    text = text.replace(
        "\u00ad",
        "",
    )

    text = text.replace(
        "\u2011",
        "-",
    )

    # Join words split by hyphen + line break.
    text = re.sub(
        r"-\s*\n\s*(?=\w)",
        "",
        text,
    )

    lines = []

    for line in text.splitlines():
        line = re.sub(
            r"\s+",
            " ",
            line,
        ).strip()

        if line:
            lines.append(
                line
            )

    return "\n".join(
        lines
    )


def clean_cell(value) -> str:
    """
    Normalize text inside a PDF table cell.
    """

    if value is None:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value),
    ).strip()


# ---------------------------------------------------------
# TEXT CHUNKING
# ---------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = 170,
    overlap: int = 35,
):
    """
    Split normal prose into overlapping word chunks.
    """

    if not text:
        return []

    words = (
        text
        .replace(
            "\n",
            " \n ",
        )
        .split()
    )

    if not words:
        return []

    chunks = []

    start = 0

    while start < len(words):

        end = min(
            start + chunk_size,
            len(words),
        )

        chunk = " ".join(
            words[start:end]
        )

        chunk = re.sub(
            r"\s*\n\s*",
            "\n",
            chunk,
        ).strip()

        if chunk:
            chunks.append(
                chunk
            )

        if end >= len(words):
            break

        start = max(
            end - overlap,
            start + 1,
        )

    return chunks


# ---------------------------------------------------------
# CENSUS SOURCE METADATA
# ---------------------------------------------------------

def source_metadata(
    pdf_path: Path,
):
    """
    Get volume metadata from census_sources.py.
    """

    configured = (
        SOURCE_BY_FILENAME.get(
            pdf_path.name
        )
    )

    if configured:

        return {
            "volume":
                configured[
                    "volume"
                ],

            "title":
                configured[
                    "title"
                ],
        }

    return {
        "volume":
            "Other",

        "title":
            (
                pdf_path.stem
                .replace(
                    "-",
                    " ",
                )
                .replace(
                    "_",
                    " ",
                )
                .title()
            ),
    }


# ---------------------------------------------------------
# TABLE HELPERS
# ---------------------------------------------------------

def normalize_headers(
    headers,
):
    """
    Clean table headers and ensure duplicate
    names become unique.

    Example:
        ["Male", "Male"]

    becomes:
        ["Male", "Male 2"]
    """

    result = []
    seen = {}

    for index, value in enumerate(
        headers
    ):
        name = (
            clean_cell(value)
            or f"Column {index + 1}"
        )

        count = (
            seen.get(
                name,
                0,
            )
            + 1
        )

        seen[name] = count

        if count == 1:
            result.append(
                name
            )
        else:
            result.append(
                f"{name} {count}"
            )

    return result


def meaningful_header_names(
    names,
):
    """
    Determine whether PyMuPDF supplied useful
    table header names.
    """

    if not names:
        return False

    cleaned = [
        clean_cell(
            value
        )
        for value in names
    ]

    non_empty = sum(
        bool(value)
        for value in cleaned
    )

    return (
        non_empty
        >= max(
            1,
            len(cleaned) // 2,
        )
    )


def rows_from_table_data(
    table_data,
    header_names=None,
):
    """
    Convert table.extract() output into:

        columns = [...]
        rows = [
            {...},
            {...}
        ]
    """

    if not table_data:
        return [], []

    raw_rows = []

    for row in table_data:

        if not row:
            continue

        raw_rows.append(
            [
                clean_cell(
                    value
                )
                for value in row
            ]
        )

    if not raw_rows:
        return [], []

    if meaningful_header_names(
        header_names
    ):
        headers = normalize_headers(
            header_names
        )

        body = raw_rows

        # Sometimes extract() also includes
        # the same header as its first row.
        first_row = normalize_headers(
            raw_rows[0][
                :len(headers)
            ]
        )

        if first_row == headers:
            body = raw_rows[1:]

    else:
        headers = normalize_headers(
            raw_rows[0]
        )

        body = raw_rows[1:]

    rows = []

    for raw in body:

        values = (
            raw
            + [""] * max(
                0,
                len(headers) - len(raw),
            )
        )

        row = {}

        for index, header in enumerate(
            headers
        ):
            row[header] = (
                values[index]
                if index < len(values)
                else ""
            )

        # Ignore completely empty rows.
        if not any(
            value
            and value not in {
                "-",
                "—",
                "–",
            }
            for value in row.values()
        ):
            continue

        # Ignore repeated table headings
        # appearing halfway through a long table.
        normalized_values = [
            clean_cell(
                value
            ).lower()
            for value in row.values()
        ]

        normalized_headers = [
            clean_cell(
                header
            ).lower()
            for header in headers
        ]

        if (
            normalized_values[
                :len(
                    normalized_headers
                )
            ]
            == normalized_headers
        ):
            continue

        rows.append(
            row
        )

    return (
        headers,
        rows,
    )


def table_number_from_title(
    title: str,
):
    """
    Extract a table number such as 2.22.
    """

    match = re.search(
        r"\bTable\s+([A-Za-z0-9.\-]+)",
        title or "",
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    return match.group(1)


def find_table_title(
    page,
    table_bbox,
) -> str:
    """
    Find the closest text block immediately
    above a detected table.
    """

    if not table_bbox:
        return ""

    try:
        bbox = fitz.Rect(
            table_bbox
        )

        candidates = []

        for block in page.get_text(
            "blocks"
        ):
            block_rect = fitz.Rect(
                block[:4]
            )

            # Only inspect nearby blocks above
            # the table.
            if (
                block_rect.y1
                <= bbox.y0 + 3
                and
                bbox.y0 - block_rect.y1
                <= 140
            ):
                text = clean_text(
                    block[4]
                )

                if text:
                    candidates.append(
                        (
                            block_rect.y1,
                            text,
                        )
                    )

        if not candidates:
            return ""

        candidate_text = sorted(
            candidates,
            key=lambda item:
                item[0],
        )[-1][1]

        lines = (
            candidate_text
            .splitlines()
        )

        # Prefer an explicit "Table X..." line.
        for index, line in enumerate(
            lines
        ):
            if re.match(
                r"^Table\s+",
                line,
                flags=re.IGNORECASE,
            ):
                return " ".join(
                    lines[index:]
                )

        # Fallback to nearest preceding line.
        return lines[-1]

    except Exception:
        return ""


def table_row_to_search_text(
    table_title: str,
    row: dict,
) -> str:
    """
    Create clean semantic-search text from
    one structured table row.

    This is what BGE embeds.
    """

    parts = []

    if table_title:
        parts.append(
            table_title.rstrip(
                "."
            )
        )

    for key, value in row.items():

        value = clean_cell(
            value
        )

        if (
            not value
            or value in {
                "-",
                "—",
                "–",
            }
        ):
            continue

        parts.append(
            f"{key}: {value}"
        )

    if not parts:
        return ""

    return (
        ". ".join(
            parts
        ).strip()
        + "."
    )


def page_tables(
    page,
):
    """
    Safely run PyMuPDF table detection.
    """

    try:
        finder = (
            page.find_tables()
        )

        return list(
            getattr(
                finder,
                "tables",
                [],
            )
            or []
        )

    except Exception as exc:

        print(
            "Warning: could not "
            f"extract page table: {exc}"
        )

        return []


# ---------------------------------------------------------
# REMOVE TABLE CONTENT FROM NORMAL PROSE
# ---------------------------------------------------------

def rects_intersect(
    a,
    b,
) -> bool:
    """
    Return True if two PyMuPDF rectangles overlap.
    """

    return not (
        a.x1 <= b.x0
        or
        a.x0 >= b.x1
        or
        a.y1 <= b.y0
        or
        a.y0 >= b.y1
    )


def prose_outside_tables(
    page,
    table_bboxes,
):
    """
    Extract normal page text while excluding
    blocks belonging to tables.

    This prevents tables from being indexed
    twice:
        1. structured rows
        2. ugly flattened text
    """

    pieces = []

    table_rects = [
        fitz.Rect(
            bbox
        )
        for bbox in table_bboxes
    ]

    for block in page.get_text(
        "blocks"
    ):
        block_rect = fitz.Rect(
            block[:4]
        )

        if any(
            rects_intersect(
                block_rect,
                table_rect,
            )
            for table_rect
            in table_rects
        ):
            continue

        text = clean_text(
            block[4]
        )

        if text:
            pieces.append(
                text
            )

    return "\n".join(
        pieces
    )


# ---------------------------------------------------------
# PAGE / PDF EXTRACTION
# ---------------------------------------------------------

def extract_chunks(
    pdf_path: Path,
    chunk_size: int,
    overlap: int,
    start_id: int,
):
    """
    Extract both:

    - structured table rows
    - normal prose chunks
    """

    document = fitz.open(
        pdf_path
    )

    source = source_metadata(
        pdf_path
    )

    chunks = []

    chunk_id = start_id

    print()

    print(
        f"Extracting Volume "
        f"{source['volume']} — "
        f"{source['title']}"
    )

    print(
        f"Pages: {len(document)}"
    )

    for page_number, page in enumerate(
        document,
        start=1,
    ):

        tables = page_tables(
            page
        )

        table_bboxes = []

        # -------------------------------------------------
        # TABLES
        # -------------------------------------------------

        for table_index, table in enumerate(
            tables,
            start=1,
        ):

            bbox = getattr(
                table,
                "bbox",
                None,
            )

            if bbox:
                table_bboxes.append(
                    bbox
                )

            try:
                table_data = (
                    table.extract()
                )

            except Exception as exc:

                print(
                    "Warning: table extraction "
                    f"failed on page "
                    f"{page_number}: {exc}"
                )

                continue

            header = getattr(
                table,
                "header",
                None,
            )

            header_names = getattr(
                header,
                "names",
                None,
            )

            (
                columns,
                rows,
            ) = rows_from_table_data(
                table_data,
                header_names=header_names,
            )

            if not rows:
                continue

            table_title = (
                find_table_title(
                    page,
                    bbox,
                )
                if bbox
                else ""
            )

            if not table_title:
                table_title = (
                    f"Table on page "
                    f"{page_number}"
                )

            table_number = (
                table_number_from_title(
                    table_title
                )
            )

            for row in rows:

                search_text = (
                    table_row_to_search_text(
                        table_title,
                        row,
                    )
                )

                if not search_text:
                    continue

                chunks.append(
                    {
                        "chunk_id":
                            chunk_id,

                        "type":
                            "table",

                        "year":
                            2019,

                        "volume":
                            source[
                                "volume"
                            ],

                        "volume_title":
                            source[
                                "title"
                            ],

                        "page":
                            page_number,

                        "source":
                            pdf_path.name,

                        "table_index":
                            table_index,

                        "table_number":
                            table_number,

                        "table_title":
                            table_title,

                        "columns":
                            columns,

                        "row":
                            row,

                        # BGE representation.
                        "text":
                            search_text,
                    }
                )

                chunk_id += 1

        # -------------------------------------------------
        # NORMAL TEXT
        # -------------------------------------------------

        prose = clean_text(
            prose_outside_tables(
                page,
                table_bboxes,
            )
        )

        for text_chunk in chunk_text(
            prose,
            chunk_size,
            overlap,
        ):

            chunks.append(
                {
                    "chunk_id":
                        chunk_id,

                    "type":
                        "text",

                    "year":
                        2019,

                    "volume":
                        source[
                            "volume"
                        ],

                    "volume_title":
                        source[
                            "title"
                        ],

                    "page":
                        page_number,

                    "source":
                        pdf_path.name,

                    "text":
                        text_chunk,
                }
            )

            chunk_id += 1

    document.close()

    table_count = sum(
        chunk.get(
            "type"
        )
        == "table"
        for chunk in chunks
    )

    text_count = sum(
        chunk.get(
            "type"
        )
        == "text"
        for chunk in chunks
    )

    print(
        f"Created "
        f"{len(chunks):,} "
        f"searchable records."
    )

    print(
        f"  Text chunks: "
        f"{text_count:,}"
    )

    print(
        f"  Table rows: "
        f"{table_count:,}"
    )

    return chunks


# ---------------------------------------------------------
# FILE DISCOVERY
# ---------------------------------------------------------

def discover_pdfs(
    explicit_paths,
):
    """
    If file paths are supplied, index only those.

    Otherwise index every PDF in data/.
    """

    if explicit_paths:

        paths = [
            Path(
                path
            ).resolve()
            for path in explicit_paths
        ]

    else:

        paths = sorted(
            DATA_DIR.glob(
                "*.pdf"
            )
        )

    missing = [
        path
        for path in paths
        if not path.exists()
    ]

    if missing:

        missing_text = "\n".join(
            f"  - {path}"
            for path in missing
        )

        raise SystemExit(
            "PDF file(s) not found:\n"
            f"{missing_text}"
        )

    if not paths:

        raise SystemExit(
            "No census PDFs found.\n\n"
            "Run:\n\n"
            "    python download_census.py\n"
        )

    return paths


# ---------------------------------------------------------
# LOCAL EMBEDDING MODEL
# ---------------------------------------------------------

def validate_model_path(
    model_path: Path,
):
    """
    Ensure BGE exists locally.
    """

    if not model_path.exists():

        raise SystemExit(
            "\nLocal embedding model "
            "not found.\n\n"
            f"Expected:\n"
            f"    {model_path}\n\n"
            "Run:\n"
            "    python download_model.py\n"
        )

    if not model_path.is_dir():

        raise SystemExit(
            "Model path is not a "
            f"directory: {model_path}"
        )


def load_local_model(
    model_path: Path,
):
    """
    Load BGE without contacting Hugging Face.
    """

    validate_model_path(
        model_path
    )

    print()

    print(
        "Loading local "
        "embedding model:"
    )

    print(
        f"    {model_path}"
    )

    return SentenceTransformer(
        str(
            model_path
        ),
        local_files_only=True,
    )


# ---------------------------------------------------------
# EMBEDDINGS + FAISS
# ---------------------------------------------------------

def create_embeddings(
    model,
    chunks,
    batch_size: int,
):
    """
    Embed all text and structured-table
    representations.
    """

    documents = [
        chunk["text"]
        for chunk in chunks
    ]

    print()

    print(
        f"Embedding "
        f"{len(documents):,} "
        f"census records..."
    )

    embeddings = model.encode(
        documents,

        batch_size=
            batch_size,

        show_progress_bar=
            True,

        normalize_embeddings=
            True,

        convert_to_numpy=
            True,
    )

    return np.asarray(
        embeddings,
        dtype=np.float32,
    )


def create_faiss_index(
    embeddings,
):
    """
    Use normalized vectors + inner product,
    equivalent to cosine similarity.
    """

    dimensions = (
        embeddings.shape[1]
    )

    index = (
        faiss.IndexFlatIP(
            dimensions
        )
    )

    index.add(
        embeddings
    )

    return index


# ---------------------------------------------------------
# SAVE INDEX
# ---------------------------------------------------------

def save_index(
    index,
    chunks,
    pdf_paths,
    embeddings,
    index_dir: Path,
    model_path: Path,
    chunk_size: int,
    overlap: int,
):
    """
    Save FAISS index, searchable records,
    and metadata.
    """

    index_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    faiss_path = (
        index_dir
        / "census.faiss"
    )

    chunks_path = (
        index_dir
        / "chunks.json"
    )

    metadata_path = (
        index_dir
        / "metadata.json"
    )

    faiss.write_index(
        index,
        str(
            faiss_path
        ),
    )

    chunks_path.write_text(
        json.dumps(
            chunks,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    indexed_volumes = sorted(
        {
            (
                chunk[
                    "volume"
                ],
                chunk[
                    "volume_title"
                ],
            )
            for chunk in chunks
        },
        key=lambda item:
            item[0],
    )

    text_chunks = sum(
        chunk.get(
            "type"
        )
        == "text"
        for chunk in chunks
    )

    table_rows = sum(
        chunk.get(
            "type"
        )
        == "table"
        for chunk in chunks
    )

    metadata = {
        "year":
            2019,

        "model":
            "BAAI/bge-small-en-v1.5",

        "model_path":
            str(
                model_path
            ),

        "documents":
            len(
                pdf_paths
            ),

        "chunks":
            len(
                chunks
            ),

        "text_chunks":
            text_chunks,

        "table_rows":
            table_rows,

        "embedding_dimensions":
            int(
                embeddings.shape[1]
            ),

        "chunk_size_words":
            chunk_size,

        "overlap_words":
            overlap,

        "volumes": [
            {
                "volume":
                    volume,

                "title":
                    title,
            }
            for (
                volume,
                title,
            )
            in indexed_volumes
        ],

        "sources": [
            path.name
            for path in pdf_paths
        ],
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return (
        faiss_path,
        chunks_path,
        metadata_path,
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build the Kenya census semantic-search "
            "index using local BGE embeddings and "
            "structured census table extraction."
        )
    )

    parser.add_argument(
        "pdfs",
        nargs="*",
        help=(
            "Optional PDF paths. "
            "If omitted, all PDFs "
            "inside data/ are indexed."
        ),
    )

    parser.add_argument(
        "--index-dir",
        type=Path,
        default=
            DEFAULT_INDEX_DIR,
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        default=
            DEFAULT_MODEL_PATH,
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=170,
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=35,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    args = (
        parser.parse_args()
    )

    if args.chunk_size <= 0:
        raise SystemExit(
            "--chunk-size must "
            "be greater than zero."
        )

    if args.overlap < 0:
        raise SystemExit(
            "--overlap cannot "
            "be negative."
        )

    if (
        args.overlap
        >= args.chunk_size
    ):
        raise SystemExit(
            "--overlap must be "
            "smaller than "
            "--chunk-size."
        )

    if args.batch_size <= 0:
        raise SystemExit(
            "--batch-size must "
            "be greater than zero."
        )

    model_path = (
        args.model_path
        .expanduser()
        .resolve()
    )

    validate_model_path(
        model_path
    )

    pdf_paths = (
        discover_pdfs(
            args.pdfs
        )
    )

    print()
    print(
        "Kenya Census Semantic Index"
    )

    print(
        "============================"
    )

    print()

    print(
        f"PDFs found: "
        f"{len(pdf_paths)}"
    )

    for pdf_path in pdf_paths:
        print(
            f"  - "
            f"{pdf_path.name}"
        )

    all_chunks = []

    next_chunk_id = 0

    for pdf_path in pdf_paths:

        chunks = extract_chunks(
            pdf_path=
                pdf_path,

            chunk_size=
                args.chunk_size,

            overlap=
                args.overlap,

            start_id=
                next_chunk_id,
        )

        all_chunks.extend(
            chunks
        )

        next_chunk_id += (
            len(chunks)
        )

    if not all_chunks:

        raise SystemExit(
            "No searchable content "
            "could be extracted from "
            "the census PDFs."
        )

    text_count = sum(
        chunk.get(
            "type"
        )
        == "text"
        for chunk in all_chunks
    )

    table_count = sum(
        chunk.get(
            "type"
        )
        == "table"
        for chunk in all_chunks
    )

    print()
    print(
        "Extraction complete."
    )

    print(
        f"Total records: "
        f"{len(all_chunks):,}"
    )

    print(
        f"Text chunks: "
        f"{text_count:,}"
    )

    print(
        f"Table rows: "
        f"{table_count:,}"
    )

    model = load_local_model(
        model_path
    )

    embeddings = (
        create_embeddings(
            model=
                model,

            chunks=
                all_chunks,

            batch_size=
                args.batch_size,
        )
    )

    print()

    print(
        "Creating FAISS index..."
    )

    faiss_index = (
        create_faiss_index(
            embeddings
        )
    )

    (
        faiss_path,
        chunks_path,
        metadata_path,
    ) = save_index(

        index=
            faiss_index,

        chunks=
            all_chunks,

        pdf_paths=
            pdf_paths,

        embeddings=
            embeddings,

        index_dir=
            args.index_dir,

        model_path=
            model_path,

        chunk_size=
            args.chunk_size,

        overlap=
            args.overlap,
    )

    print()
    print(
        "Index complete."
    )

    print(
        "==============="
    )

    print(
        f"Documents: "
        f"{len(pdf_paths):,}"
    )

    print(
        f"Search records: "
        f"{len(all_chunks):,}"
    )

    print(
        f"Text chunks: "
        f"{text_count:,}"
    )

    print(
        f"Table rows: "
        f"{table_count:,}"
    )

    print(
        f"FAISS vectors: "
        f"{faiss_index.ntotal:,}"
    )

    print(
        f"Embedding dimensions: "
        f"{embeddings.shape[1]}"
    )

    print()

    print(
        "Files created:"
    )

    print(
        f"  {faiss_path}"
    )

    print(
        f"  {chunks_path}"
    )

    print(
        f"  {metadata_path}"
    )


if __name__ == "__main__":
    main()
