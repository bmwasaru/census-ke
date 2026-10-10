import json
from pathlib import Path
from threading import Lock

import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "bge-small-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for " "searching relevant passages: "


class CensusSearchEngine:
    def __init__(self, index_dir: Path, model_path: Path = DEFAULT_MODEL_PATH):
        self.index_dir = Path(index_dir)
        self.model_path = Path(model_path)
        self._model = None
        self._model_lock = Lock()
        self._chunks = []
        self._index = None
        self.metadata = {}
        self._loaded = False
        self._load_index()

    # -----------------------------------------------------
    # PROPERTIES
    # -----------------------------------------------------
    @property
    def chunk_count(self):
        return len(self._chunks)

    @property
    def volumes(self):
        return self.metadata.get("volumes", [])

    # -----------------------------------------------------
    # MODEL
    # -----------------------------------------------------
    def _get_model(self):
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    if not (self.model_path.exists()):
                        raise RuntimeError(
                            "Local embedding "
                            "model not found at "
                            f"{self.model_path}. "
                            "Run "
                            "`python download_model.py` "
                            "first."
                        )

                    self._model = SentenceTransformer(
                        str(self.model_path), local_files_only=True
                    )

        return self._model

    # -----------------------------------------------------
    # INDEX LOADING
    # -----------------------------------------------------
    def _load_index(self):
        chunks_path = self.index_dir / "chunks.json"
        faiss_path = self.index_dir / "census.faiss"
        metadata_path = self.index_dir / "metadata.json"

        if not chunks_path.exists() or not faiss_path.exists():
            self._loaded = False
            return

        self._chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        self._index = faiss.read_index(str(faiss_path))

        if metadata_path.exists():
            self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        if len(self._chunks) != self._index.ntotal:
            raise RuntimeError(
                "Search index is "
                "inconsistent: FAISS "
                "vector count does not "
                "match chunk metadata."
            )

        self._loaded = True

    def reload(self):
        self._load_index()

    def is_ready(self):
        return self._loaded and self._index is not None and bool(self._chunks)

    # -----------------------------------------------------
    # QUERY EMBEDDING
    # -----------------------------------------------------
    def _encode_query(self, query: str):
        model = self._get_model()
        prepared_query = f"{QUERY_INSTRUCTION}" f"{query.strip()}"
        embedding = model.encode(
            [prepared_query], normalize_embeddings=True, convert_to_numpy=True
        )

        return np.asarray(embedding, dtype=np.float32)

    # -----------------------------------------------------
    # RESULT FORMATTER
    # -----------------------------------------------------
    def _result_from_chunk(self, chunk, score, rank):
        result = {
            "rank": rank,
            "score": round(float(score), 4),
            "year": chunk.get("year", 2019),
            "type": chunk.get("type", "text"),
            "volume": chunk.get("volume"),
            "volume_title": chunk.get("volume_title"),
            "page": chunk.get("page"),
            "chunk_id": chunk.get("chunk_id"),
            "source": chunk.get("source"),
            "text": chunk.get("text", ""),
        }

        if result["type"] == "table":
            result["table_number"] = chunk.get("table_number")
            result["table_title"] = chunk.get("table_title")
            result["columns"] = chunk.get("columns", [])
            result["row"] = chunk.get("row", {})

        return result

    # -----------------------------------------------------
    # SEARCH
    # -----------------------------------------------------
    def search(self, query: str, top_k: int = 6, volume: str | None = None):
        if not self.is_ready():
            return []

        query_vector = self._encode_query(query)

        # When filtering by volume, retrieve
        # additional candidates first.
        candidate_k = min(
            len(self._chunks), (max(top_k * 20, 100) if volume else top_k)
        )
        scores, indices = self._index.search(query_vector, candidate_k)
        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            chunk = self._chunks[int(idx)]

            if volume and chunk.get("volume") != volume:
                continue

            result = self._result_from_chunk(
                chunk=chunk, score=score, rank=len(results) + 1
            )
            results.append(result)

            if len(results) >= top_k:
                break

        return results
