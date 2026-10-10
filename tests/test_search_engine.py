import json
from pathlib import Path

import faiss
import numpy as np
import pytest

from search_engine import CensusSearchEngine


@pytest.fixture
def indexed_engine(tmp_path):
    """
    Create a tiny real FAISS index without
    loading BGE.
    """

    index_dir = tmp_path / "index"
    index_dir.mkdir()
    chunks = [
        {
            "chunk_id": 0,
            "type": "text",
            "year": 2019,
            "volume": "I",
            "volume_title": ("Population by County " "and Sub-County"),
            "page": 10,
            "source": ("2019-kphc-volume-1.pdf"),
            "text": ("Mombasa had a population " "of 1,208,333."),
        },
        {
            "chunk_id": 1,
            "type": "table",
            "year": 2019,
            "volume": "IV",
            "volume_title": (
                "Distribution of Population " "by Socio-Economic Characteristics"
            ),
            "page": 215,
            "source": ("2019-kphc-volume-4.pdf"),
            "table_number": "2.22",
            "table_title": ("Table 2.22: Distribution " "of Households Growing Crops"),
            "columns": ["Area", "Households Farming", "Maize"],
            "row": {
                "Area": "MOMBASA",
                "Households Farming": "12,497",
                "Maize": "5,681",
            },
            "text": (
                "Table 2.22. "
                "Area: MOMBASA. "
                "Households Farming: 12,497. "
                "Maize: 5,681."
            ),
        },
    ]
    (index_dir / "chunks.json").write_text(json.dumps(chunks), encoding="utf-8")
    metadata = {
        "volumes": [
            {"volume": "I", "title": "Population"},
            {"volume": "IV", "title": ("Socio-Economic " "Characteristics")},
        ],
        "sources": ["2019-kphc-volume-1.pdf", "2019-kphc-volume-4.pdf"],
    }
    (index_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    index = faiss.IndexFlatIP(2)
    index.add(vectors)
    faiss.write_index(index, str(index_dir / "census.faiss"))
    model_path = tmp_path / "model"
    model_path.mkdir()

    return CensusSearchEngine(index_dir=index_dir, model_path=model_path)


def test_engine_loads_index(indexed_engine):
    assert indexed_engine.is_ready() is True
    assert indexed_engine.chunk_count == 2


def test_search_returns_table_structure(indexed_engine, monkeypatch):
    monkeypatch.setattr(
        indexed_engine,
        "_encode_query",
        lambda query: np.array([[0.0, 1.0]], dtype=np.float32),
    )
    results = indexed_engine.search("maize in Mombasa", top_k=1)
    assert len(results) == 1
    result = results[0]
    assert result["type"] == "table"
    assert result["page"] == 215
    assert result["table_number"] == "2.22"
    assert result["row"]["Maize"] == "5,681"


def test_search_returns_text_result(indexed_engine, monkeypatch):
    monkeypatch.setattr(
        indexed_engine,
        "_encode_query",
        lambda query: np.array([[1.0, 0.0]], dtype=np.float32),
    )
    results = indexed_engine.search("Mombasa population", top_k=1)
    assert results[0]["type"] == "text"
    assert results[0]["page"] == 10


def test_volume_filter(indexed_engine, monkeypatch):
    monkeypatch.setattr(
        indexed_engine,
        "_encode_query",
        lambda query: np.array([[0.0, 1.0]], dtype=np.float32),
    )
    results = indexed_engine.search("population", top_k=1, volume="I")
    assert len(results) == 1
    assert results[0]["volume"] == "I"


def test_volumes_property(indexed_engine):
    volumes = indexed_engine.volumes
    assert len(volumes) == 2
    assert volumes[0]["volume"] == "I"


def test_inconsistent_index_raises(tmp_path):
    index_dir = tmp_path / "bad-index"
    index_dir.mkdir()
    (index_dir / "chunks.json").write_text(
        json.dumps([{"chunk_id": 0}]), encoding="utf-8"
    )
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    index = faiss.IndexFlatIP(2)
    index.add(vectors)
    faiss.write_index(index, str(index_dir / "census.faiss"))

    with pytest.raises(RuntimeError):
        CensusSearchEngine(index_dir=index_dir, model_path=tmp_path / "model")
