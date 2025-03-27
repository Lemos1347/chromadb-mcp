"""ChromaDB integration through the Model Context Protocol."""

import os

from dotenv import load_dotenv

__version__ = "0.1.0"


def check_vars() -> None:
    chroma_path = os.environ.get("CHROMA_PERSIST_DIRECTORY", None)
    google_api_key = os.environ.get("GOOGLE_API_KEY", None)

    if chroma_path is None or google_api_key is None:
        raise RuntimeError("Env vars not set")


load_dotenv()
check_vars()
