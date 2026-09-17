from pathlib import Path
from unittest.mock import Mock

SAMPLE_RESULTS = [
    {
        "rank": 1,
        "score": 0.91,
        "type": "table",
        "year": 2019,
        "volume": "IV",
        "volume_title": (
            "Distribution of Population " "by Socio-Economic Characteristics"
        ),
        "page": 215,
        "chunk_id": 500,
        "source": ("2019-kphc-volume-4.pdf"),
        "table_number": "2.22",
        "table_title": ("Table 2.22: Distribution " "of Households Growing Crops"),
        "row": {
            "Area": "MOMBASA",
            "Maize": "5,681",
        },
        "text": ("Area: MOMBASA. " "Maize: 5,681."),
    }
]


def configure_ready_engine(
    app_module,
    monkeypatch,
):

    monkeypatch.setattr(
        app_module.engine,
        "is_ready",
        lambda: True,
    )

    monkeypatch.setattr(
        app_module.engine,
        "metadata",
        {
            "volumes": [
                {
                    "volume": "IV",
                    "title": ("Socio-Economic " "Characteristics"),
                }
            ],
            "sources": ["2019-kphc-" "volume-4.pdf"],
        },
    )

    fake_model_path = Mock()

    fake_model_path.exists.return_value = True

    monkeypatch.setattr(
        app_module.engine,
        "model_path",
        fake_model_path,
    )


def test_api_search_adds_page_url(
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

    response = client.post(
        "/api/search",
        json={"query": "maize in Mombasa"},
    )

    assert response.status_code == 200

    data = response.get_json()

    result = data["results"][0]

    assert result["page_url"] == ("/census-pdf/" "2019-kphc-volume-4.pdf" "#page=215")


def test_api_ask_adds_source_page_url(
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
            "answer": ("5,681 households " "reported growing " "maize. [S1]"),
            "sources": [
                {
                    "id": "S1",
                    "volume": "IV",
                    "page": 215,
                    "source": ("2019-kphc-" "volume-4.pdf"),
                    "type": "table",
                    "table_number": "2.22",
                    "table_title": ("Table 2.22"),
                }
            ],
            "evidence": SAMPLE_RESULTS,
        },
    )

    response = client.post(
        "/api/ask",
        json={"query": ("How many Mombasa " "households grow maize?")},
    )

    assert response.status_code == 200

    data = response.get_json()

    assert data["sources"][0]["page_url"] == (
        "/census-pdf/" "2019-kphc-volume-4.pdf" "#page=215"
    )

    assert data["evidence"][0]["page_url"] == (
        "/census-pdf/" "2019-kphc-volume-4.pdf" "#page=215"
    )


def test_api_ask_requires_query(
    client,
):

    response = client.post(
        "/api/ask",
        json={},
    )

    assert response.status_code == 400


def test_unknown_volume_rejected(
    client,
    app_module,
    monkeypatch,
):

    configure_ready_engine(
        app_module,
        monkeypatch,
    )

    response = client.post(
        "/api/search",
        json={
            "query": "population",
            "volume": "IX",
        },
    )

    assert response.status_code == 400


def test_pdf_route_serves_indexed_file(
    client,
    app_module,
    monkeypatch,
    tmp_path,
):

    configure_ready_engine(
        app_module,
        monkeypatch,
    )

    data_dir = tmp_path / "data"

    data_dir.mkdir()

    pdf = data_dir / "2019-kphc-volume-4.pdf"

    # Enough for testing Flask file delivery;
    # it does not need to be a valid PDF here.
    pdf.write_bytes(b"%PDF-1.4\n" b"test\n" b"%%EOF")

    monkeypatch.setattr(
        app_module,
        "DATA_DIR",
        data_dir,
    )

    response = client.get(("/census-pdf/" "2019-kphc-volume-4.pdf"))

    assert response.status_code == 200

    assert response.mimetype == "application/pdf"


def test_pdf_route_rejects_unknown_file(
    client,
    app_module,
    monkeypatch,
):

    configure_ready_engine(
        app_module,
        monkeypatch,
    )

    response = client.get("/census-pdf/not-indexed.pdf")

    assert response.status_code == 404


def test_pdf_route_rejects_directory_traversal(
    client,
    app_module,
    monkeypatch,
):

    configure_ready_engine(
        app_module,
        monkeypatch,
    )

    response = client.get("/census-pdf/../secret.pdf")

    assert response.status_code == 404
