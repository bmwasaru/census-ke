from unittest.mock import Mock

SAMPLE_RESULTS = [
    {
        "rank": 1,
        "score": 0.82,
        "year": 2019,
        "volume": "I",
        "volume_title": ("Population by County " "and Sub-County"),
        "page": 12,
        "chunk_id": 100,
        "source": "2019-kphc-volume-1.pdf",
        "text": ("Mombasa County had " "1,208,333 residents."),
    }
]


def configure_ready_engine(
    app_module,
    monkeypatch,
):
    """
    Configure the CensusSearchEngine as ready
    without requiring a real FAISS index or
    local embedding model.
    """

    monkeypatch.setattr(
        app_module.engine,
        "is_ready",
        lambda: True,
    )

    # `volumes` is a read-only property that reads
    # from engine.metadata, so patch metadata instead.
    monkeypatch.setattr(
        app_module.engine,
        "metadata",
        {
            "volumes": [
                {
                    "volume": "I",
                    "title": ("Population by County " "and Sub-County"),
                }
            ]
        },
    )

    fake_model_path = Mock()

    fake_model_path.exists.return_value = True

    monkeypatch.setattr(
        app_module.engine,
        "model_path",
        fake_model_path,
    )


def test_api_ask_success(
    client,
    app_module,
    monkeypatch,
):
    configure_ready_engine(
        app_module,
        monkeypatch,
    )

    monkeypatch.setattr(
        app_module.engine,
        "search",
        lambda **kwargs: SAMPLE_RESULTS,
    )

    monkeypatch.setattr(
        app_module.llm,
        "answer",
        lambda **kwargs: {
            "answer": ("Mombasa had 1,208,333 " "residents. [S1]"),
            "sources": [
                {
                    "id": "S1",
                    "volume": "I",
                    "volume_title": ("Population by County " "and Sub-County"),
                    "page": 12,
                    "source": ("2019-kphc-volume-1.pdf"),
                    "score": 0.82,
                }
            ],
            "evidence": SAMPLE_RESULTS,
        },
    )

    response = client.post(
        "/api/ask",
        json={"query": ("What was Mombasa's " "population?")},
    )

    assert response.status_code == 200

    data = response.get_json()

    assert "1,208,333" in data["answer"]

    assert data["sources"][0]["id"] == "S1"


def test_api_ask_requires_question(
    client,
):
    response = client.post(
        "/api/ask",
        json={},
    )

    assert response.status_code == 400

    data = response.get_json()

    assert "error" in data


def test_api_ask_returns_no_results(
    client,
    app_module,
    monkeypatch,
):
    configure_ready_engine(
        app_module,
        monkeypatch,
    )

    monkeypatch.setattr(
        app_module.engine,
        "search",
        lambda **kwargs: [],
    )

    response = client.post(
        "/api/ask",
        json={"query": ("Unknown census question")},
    )

    assert response.status_code == 200

    data = response.get_json()

    assert data["sources"] == []

    assert data["evidence"] == []


def test_api_ask_calls_retrieval_first(
    client,
    app_module,
    monkeypatch,
):
    configure_ready_engine(
        app_module,
        monkeypatch,
    )

    calls = []

    def fake_search(**kwargs):
        calls.append("search")

        return SAMPLE_RESULTS

    def fake_answer(**kwargs):
        calls.append("llm")

        return {
            "answer": ("Answer [S1]"),
            "sources": [],
            "evidence": SAMPLE_RESULTS,
        }

    monkeypatch.setattr(
        app_module.engine,
        "search",
        fake_search,
    )

    monkeypatch.setattr(
        app_module.llm,
        "answer",
        fake_answer,
    )

    response = client.post(
        "/api/ask",
        json={"query": ("Mombasa population")},
    )

    assert response.status_code == 200

    assert calls == [
        "search",
        "llm",
    ]


def test_api_ask_passes_volume_filter(
    client,
    app_module,
    monkeypatch,
):
    configure_ready_engine(
        app_module,
        monkeypatch,
    )

    captured = {}

    def fake_search(
        query,
        top_k,
        volume,
    ):
        captured["query"] = query
        captured["top_k"] = top_k
        captured["volume"] = volume

        return SAMPLE_RESULTS

    monkeypatch.setattr(
        app_module.engine,
        "search",
        fake_search,
    )

    monkeypatch.setattr(
        app_module.llm,
        "answer",
        lambda **kwargs: {
            "answer": "Answer [S1]",
            "sources": [],
            "evidence": SAMPLE_RESULTS,
        },
    )

    response = client.post(
        "/api/ask",
        json={
            "query": ("Mombasa population"),
            "volume": "I",
            "top_k": 5,
        },
    )

    assert response.status_code == 200

    assert captured["volume"] == "I"

    assert captured["top_k"] == 5


def test_api_ask_rejects_unknown_volume(
    client,
    app_module,
    monkeypatch,
):
    configure_ready_engine(
        app_module,
        monkeypatch,
    )

    response = client.post(
        "/api/ask",
        json={
            "query": ("Mombasa population"),
            "volume": "IX",
        },
    )

    assert response.status_code == 400

    data = response.get_json()

    assert "Unknown or unindexed volume" in data["error"]
