import requests
import pytest

from llm import (
    CensusLLM,
    LLMServiceError,
    clean_llm_text,
)


SAMPLE_RESULTS = [
    {
        "rank": 1,
        "score": 0.91,
        "year": 2019,
        "type": "table",
        "volume": "IV",
        "volume_title": (
            "Distribution of Population "
            "by Socio-Economic Characteristics"
        ),
        "page": 215,
        "chunk_id": 500,
        "source": (
            "2019-kphc-volume-4.pdf"
        ),
        "table_number": "2.22",
        "table_title": (
            "Table 2.22: Distribution "
            "of Households Growing Crops"
        ),
        "row": {
            "County/Sub County":
                "MOMBASA",
            "Households Farming":
                "12,497",
            "Maize":
                "5,681",
            "Beans":
                "2,849",
        },
        "text": (
            "Area: MOMBASA. "
            "Maize: 5,681."
        ),
    }
]


class FakeResponse:

    def __init__(
        self,
        data=None,
        status_code=200,
    ):
        self.data = (
            data
            if data is not None
            else {}
        )

        self.status_code = (
            status_code
        )

    def raise_for_status(
        self,
    ):

        if self.status_code >= 400:

            raise requests.HTTPError(
                f"HTTP "
                f"{self.status_code}"
            )

    def json(
        self,
    ):
        return self.data


def test_clean_llm_text_removes_heading():

    output = clean_llm_text(
        "## Mombasa Agriculture"
    )

    assert output == (
        "Mombasa Agriculture"
    )


def test_clean_llm_text_removes_bold():

    output = clean_llm_text(
        "**5,681 households** "
        "grow maize. [S1]"
    )

    assert output == (
        "5,681 households "
        "grow maize. [S1]"
    )


def test_clean_llm_text_keeps_citation():

    output = clean_llm_text(
        "# Result\n\n"
        "**Mombasa** has "
        "5,681 maize-growing "
        "households. [S1]"
    )

    assert "[S1]" in output

    assert "#" not in output

    assert "**" not in output


def test_clean_llm_text_converts_bullets():

    output = clean_llm_text(
        "- Maize: 5,681\n"
        "- Beans: 2,849"
    )

    assert (
        "• Maize: 5,681"
        in output
    )


def test_table_evidence_is_structured():

    llm = CensusLLM()

    output = (
        llm._table_evidence_text(
            SAMPLE_RESULTS[0]
        )
    )

    assert (
        "County/Sub County: "
        "MOMBASA"
        in output
    )

    assert (
        "Maize: 5,681"
        in output
    )

    assert (
        "Beans: 2,849"
        in output
    )


def test_prepare_evidence_assigns_source():

    llm = CensusLLM()

    (
        evidence,
        sources,
        selected,
    ) = llm._prepare_evidence(
        SAMPLE_RESULTS
    )

    assert "[S1]" in evidence

    assert (
        sources[0]["id"]
        == "S1"
    )

    assert (
        sources[0]["type"]
        == "table"
    )

    assert (
        sources[0][
            "table_number"
        ]
        == "2.22"
    )

    assert len(selected) == 1


def test_answer_cleans_markdown(
    monkeypatch,
):

    llm = CensusLLM()

    def fake_post(
        url,
        json,
        timeout,
    ):

        return FakeResponse(
            {
                "message": {
                    "content": (
                        "## Answer\n\n"
                        "**5,681 households** "
                        "reported growing "
                        "maize. [S1]"
                    )
                }
            }
        )

    monkeypatch.setattr(
        llm.session,
        "post",
        fake_post,
    )

    result = llm.answer(
        (
            "How many Mombasa "
            "households grow maize?"
        ),
        SAMPLE_RESULTS,
    )

    assert (
        result["answer"]
        ==
        (
            "Answer\n\n"
            "5,681 households "
            "reported growing "
            "maize. [S1]"
        )
    )


def test_answer_disables_qwen_thinking(
    monkeypatch,
):

    llm = CensusLLM()

    captured = {}

    def fake_post(
        url,
        json,
        timeout,
    ):

        captured["payload"] = json

        return FakeResponse(
            {
                "message": {
                    "content":
                        "5,681 households. [S1]"
                }
            }
        )

    monkeypatch.setattr(
        llm.session,
        "post",
        fake_post,
    )

    llm.answer(
        "How many?",
        SAMPLE_RESULTS,
    )

    assert (
        captured[
            "payload"
        ]["think"]
        is False
    )

    assert (
        captured[
            "payload"
        ]["stream"]
        is False
    )


def test_answer_uses_qwen_model(
    monkeypatch,
):

    llm = CensusLLM()

    captured = {}

    def fake_post(
        url,
        json,
        timeout,
    ):

        captured["payload"] = json

        return FakeResponse(
            {
                "message": {
                    "content":
                        "Answer [S1]"
                }
            }
        )

    monkeypatch.setattr(
        llm.session,
        "post",
        fake_post,
    )

    llm.answer(
        "Question",
        SAMPLE_RESULTS,
    )

    assert (
        captured[
            "payload"
        ]["model"]
        == "qwen3:1.7b"
    )


def test_connection_error(
    monkeypatch,
):

    llm = CensusLLM()

    def fail(
        *args,
        **kwargs,
    ):

        raise (
            requests.ConnectionError(
                "connection refused"
            )
        )

    monkeypatch.setattr(
        llm.session,
        "post",
        fail,
    )

    with pytest.raises(
        LLMServiceError
    ):

        llm.answer(
            "Question",
            SAMPLE_RESULTS,
        )


def test_empty_answer_raises(
    monkeypatch,
):

    llm = CensusLLM()

    def fake_post(
        *args,
        **kwargs,
    ):

        return FakeResponse(
            {
                "message": {
                    "content": ""
                }
            }
        )

    monkeypatch.setattr(
        llm.session,
        "post",
        fake_post,
    )

    with pytest.raises(
        LLMServiceError
    ):

        llm.answer(
            "Question",
            SAMPLE_RESULTS,
        )


def test_health_model_available(
    monkeypatch,
):

    llm = CensusLLM()

    def fake_get(
        url,
        timeout,
    ):

        return FakeResponse(
            {
                "models": [
                    {
                        "name":
                            "qwen3:1.7b",

                        "model":
                            "qwen3:1.7b",
                    }
                ]
            }
        )

    monkeypatch.setattr(
        llm.session,
        "get",
        fake_get,
    )

    status = (
        llm.health()
    )

    assert (
        status["reachable"]
        is True
    )

    assert (
        status[
            "model_available"
        ]
        is True
    )
