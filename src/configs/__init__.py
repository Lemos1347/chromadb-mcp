import os
from typing import Optional

from .chromadb import ChromaDBClient

__all__ = ["get_chroma_client"]

# Get the persist directory from environment variable, with a default fallback

# Global variable to hold the single instance
_chroma_db_client: Optional[ChromaDBClient] = None


def get_chroma_client(
    embedding_function, collection_name="langchain"
) -> ChromaDBClient:
    """
    Returns the single ChromaDBClient instance for the application.

    Args:
        embedding_function: The embedding function to use with the collection.
        collection_name: The name of the collection to use.

    Returns:
        The ChromaDBClient instance.
    """
    global _chroma_db_client

    CHROMA_PERSIST_DIRECTORY = os.environ.get("CHROMA_PERSIST_DIRECTORY", "./chroma_db")

    if _chroma_db_client is None:
        _chroma_db_client = ChromaDBClient(
            persist_directory=CHROMA_PERSIST_DIRECTORY,
            embedding_function=embedding_function,
            collection_name=collection_name,
        )

    return _chroma_db_client
