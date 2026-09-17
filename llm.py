import os
import re
from typing import Any

import requests

SYSTEM_PROMPT = """
You are the Kenya Census Guide.

Your job is to help members of the public understand official Kenya Population and Housing Census information.

You will receive census evidence retrieved from official census documents.

Rules:

1. Answer ONLY from the census evidence provided.

2. Never use facts from your own knowledge.

3. Never invent a number, percentage, statistic, trend, cause, comparison, county fact, or citation.

4. Preserve important census numbers exactly.

5. If you round a large number for readability, also include the exact number immediately afterwards.

Example:
about 1.21 million people (1,208,333)

6. Explain statistics in clear everyday language.

7. Cite factual claims using ONLY the evidence labels supplied to you, such as [S1], [S2], and [S3].

8. Never create a citation label that was not supplied.

9. If different pieces of evidence appear to conflict, clearly say so.

10. If the evidence does not provide enough information to answer the question, say:

"The retrieved census information does not provide enough information to answer this question."

11. Do not make political, legal, economic, social, or policy conclusions that are not explicitly supported by the evidence.

12. Do not perform complicated calculations from incomplete tables.

13. Simple comparisons are acceptable only when all required values are clearly present in the supplied evidence.

14. Keep answers concise unless the user asks for more detail.

15. Explain technical census terminology when useful.

16. Output plain text only.

17. Do NOT use Markdown headings.

18. Do NOT use # symbols.

19. Do NOT use **bold** formatting.

20. Do NOT use __underline__ formatting.

21. Do NOT use Markdown tables.

22. Do NOT use backticks.

23. Prefer short paragraphs.

24. Use simple bullets only when a list genuinely improves clarity.

Your goal is accuracy, clarity, accessibility, and traceability to the census evidence.
""".strip()


class LLMServiceError(RuntimeError):
    """
    Raised when Ollama cannot provide
    a usable answer.
    """


def clean_llm_text(
    text: str,
) -> str:
    """
    Remove Markdown artifacts that may still
    appear even when the model was asked for
    plain-text output.

    Source citations such as [S1] remain intact.
    """

    text = (text or "").strip()

    # Remove fenced code markers.
    text = re.sub(
        r"```(?:\w+)?\s*",
        "",
        text,
    )

    text = text.replace(
        "```",
        "",
    )

    # ## Heading -> Heading
    text = re.sub(
        r"(?m)^\s*#{1,6}\s*",
        "",
        text,
    )

    # **bold** -> bold
    text = re.sub(
        r"\*\*(.*?)\*\*",
        r"\1",
        text,
    )

    # __bold__ -> bold
    text = re.sub(
        r"__(.*?)__",
        r"\1",
        text,
    )

    # `text` -> text
    text = re.sub(
        r"`([^`]+)`",
        r"\1",
        text,
    )

    # Markdown bullets -> clean bullet.
    text = re.sub(
        r"(?m)^\s*[-*+]\s+",
        "• ",
        text,
    )

    # Remove Markdown blockquote marker.
    text = re.sub(
        r"(?m)^\s*>\s?",
        "",
        text,
    )

    # Remove horizontal rules.
    text = re.sub(
        r"(?m)^\s*[-*_]{3,}\s*$",
        "",
        text,
    )

    # Prevent huge vertical gaps.
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


class CensusLLM:

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
    ):

        self.base_url = (
            base_url
            or os.getenv(
                "OLLAMA_BASE_URL",
                "http://127.0.0.1:11434",
            )
        ).rstrip("/")

        self.model = model or os.getenv(
            "OLLAMA_MODEL",
            "qwen3:1.7b",
        )

        self.timeout = int(
            timeout
            or os.getenv(
                "OLLAMA_TIMEOUT",
                "120",
            )
        )

        self.num_ctx = int(
            os.getenv(
                "OLLAMA_NUM_CTX",
                "4096",
            )
        )

        self.num_predict = int(
            os.getenv(
                "OLLAMA_NUM_PREDICT",
                "450",
            )
        )

        self.temperature = float(
            os.getenv(
                "OLLAMA_TEMPERATURE",
                "0.1",
            )
        )

        self.keep_alive = os.getenv(
            "OLLAMA_KEEP_ALIVE",
            "5m",
        )

        self.session = requests.Session()

    # -----------------------------------------------------
    # HEALTH CHECK
    # -----------------------------------------------------

    def health(
        self,
    ) -> dict[str, Any]:

        try:

            response = self.session.get(
                f"{self.base_url}" "/api/tags",
                timeout=3,
            )

            response.raise_for_status()

            data = response.json()

            models = data.get(
                "models",
                [],
            )

            installed = set()

            for model in models:

                name = model.get("name")

                model_name = model.get("model")

                if name:
                    installed.add(name)

                if model_name:
                    installed.add(model_name)

            return {
                "reachable": True,
                "model": self.model,
                "model_available": self.model in installed,
            }

        except (
            requests.RequestException,
            ValueError,
        ):

            return {
                "reachable": False,
                "model": self.model,
                "model_available": False,
            }

    # -----------------------------------------------------
    # TABLE EVIDENCE
    # -----------------------------------------------------

    def _table_evidence_text(
        self,
        result: dict,
    ) -> str:
        """
        Convert structured census table row
        into clean LLM context.
        """

        row = result.get("row") or {}

        lines = []

        table_title = result.get("table_title")

        if table_title:

            lines.append(f"Table: " f"{table_title}")

        for key, value in row.items():

            value = "" if value is None else str(value).strip()

            if not value or value in {
                "-",
                "—",
                "–",
            }:
                continue

            lines.append(f"{key}: {value}")

        return "\n".join(lines) or result.get(
            "text",
            "",
        )

    # -----------------------------------------------------
    # EVIDENCE PREPARATION
    # -----------------------------------------------------

    def _prepare_evidence(
        self,
        results: list[dict],
        max_chunks: int = 5,
    ):
        """
        Assign [S1], [S2] source labels and
        prepare Qwen context.
        """

        selected = []

        seen = set()

        for result in results:

            key = (
                result.get("source"),
                result.get("page"),
                result.get("chunk_id"),
            )

            if key in seen:
                continue

            seen.add(key)

            selected.append(result)

            if len(selected) >= max_chunks:
                break

        evidence_blocks = []

        sources = []

        for number, result in enumerate(
            selected,
            start=1,
        ):

            source_id = f"S{number}"

            evidence_type = result.get(
                "type",
                "text",
            )

            if evidence_type == "table":

                evidence_body = self._table_evidence_text(result)

            else:

                evidence_body = result.get(
                    "text",
                    "",
                ).strip()

            block = (
                f"[{source_id}]\n"
                f"Census: "
                f"2019 Kenya Population "
                f"and Housing Census\n"
                f"Volume: "
                f"{result.get('volume', 'Unknown')}\n"
                f"Volume title: "
                f"{result.get('volume_title', '')}\n"
                f"Page: "
                f"{result.get('page', '?')}\n"
                f"Evidence type: "
                f"{evidence_type}\n"
                f"Evidence:\n"
                f"{evidence_body}"
            )

            evidence_blocks.append(block)

            sources.append(
                {
                    "id": source_id,
                    "volume": result.get("volume"),
                    "volume_title": result.get("volume_title"),
                    "page": result.get("page"),
                    "source": result.get("source"),
                    "type": evidence_type,
                    "table_number": result.get("table_number"),
                    "table_title": result.get("table_title"),
                    "score": result.get("score"),
                }
            )

        return (
            "\n\n---\n\n".join(evidence_blocks),
            sources,
            selected,
        )

    # -----------------------------------------------------
    # PROMPT
    # -----------------------------------------------------

    def _build_user_prompt(
        self,
        question: str,
        evidence: str,
    ) -> str:

        return f"""
USER QUESTION

{question}


CENSUS EVIDENCE

{evidence}


TASK

Answer the question using only the census evidence above.

Explain the answer so that someone without a statistics background can understand it.

Preserve important figures accurately.

Use evidence labels such as [S1] and [S2] when citing factual claims.

Return clean plain text, not Markdown.

Do not mention information that is not present in the evidence.
""".strip()

    # -----------------------------------------------------
    # ANSWER
    # -----------------------------------------------------

    def answer(
        self,
        question: str,
        search_results: list[dict],
        max_chunks: int = 5,
    ) -> dict[str, Any]:

        if not search_results:

            return {
                "answer": (
                    "I could not find enough "
                    "relevant census information "
                    "to answer this question."
                ),
                "sources": [],
                "evidence": [],
            }

        (
            evidence_text,
            sources,
            selected_results,
        ) = self._prepare_evidence(
            search_results,
            max_chunks=max_chunks,
        )

        user_prompt = self._build_user_prompt(
            question,
            evidence_text,
        )

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "stream": False,
            # Qwen3 thinking is unnecessary
            # for this lightweight explanation task.
            "think": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
            },
        }

        try:

            response = self.session.post(
                f"{self.base_url}" "/api/chat",
                json=payload,
                timeout=self.timeout,
            )

            response.raise_for_status()

        except requests.ConnectionError as exc:

            raise LLMServiceError(
                "Could not connect to Ollama. " "Make sure Ollama is running."
            ) from exc

        except requests.Timeout as exc:

            raise LLMServiceError(
                "The local language model " "took too long to respond."
            ) from exc

        except requests.RequestException as exc:

            raise LLMServiceError(f"Ollama request failed: " f"{exc}") from exc

        try:

            data = response.json()

            answer = data.get(
                "message",
                {},
            ).get(
                "content",
                "",
            )

        except ValueError as exc:

            raise LLMServiceError("Ollama returned an " "invalid response.") from exc

        answer = clean_llm_text(answer)

        if not answer:

            raise LLMServiceError("The language model " "returned an empty answer.")

        return {
            "answer": answer,
            "sources": sources,
            "evidence": selected_results,
        }
