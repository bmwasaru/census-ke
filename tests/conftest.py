import os
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# Tests should never contact Hugging Face.
os.environ.setdefault(
    "HF_HUB_OFFLINE",
    "1",
)

os.environ.setdefault(
    "TRANSFORMERS_OFFLINE",
    "1",
)

os.environ.setdefault(
    "FLASK_DEBUG",
    "0",
)

os.environ.setdefault(
    "OLLAMA_BASE_URL",
    "http://127.0.0.1:11434",
)

os.environ.setdefault(
    "OLLAMA_MODEL",
    "qwen3:1.7b",
)


@pytest.fixture
def app_module():
    """
    Import the Flask application module.
    """

    import app

    app.app.config.update(
        TESTING=True,
    )

    return app


@pytest.fixture
def flask_app(
    app_module,
):
    return app_module.app


@pytest.fixture
def client(
    flask_app,
):
    with flask_app.test_client() as test_client:
        yield test_client
