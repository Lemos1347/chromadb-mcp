from abc import abstractmethod
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel


class RetrieveChunksOutputDTO(BaseModel):
    chunks: list[str]
    metadatas: list[dict[str, Any]]


@runtime_checkable
class VectorStore(Protocol):
    """
    Abstract interface for vector store implementations.
    Any class that implements this protocol can be used as a vector store.
    """

    @abstractmethod
    def get_chunks(
        self, inputs: list[str], k: int = 2
    ) -> None | RetrieveChunksOutputDTO:
        """
        Retrieve chunks from the vector store based on input queries.

        Args:
            inputs: List of input texts to query against the vector store
            k: Number of results to return for each query

        Returns:
            A RetrieveChunksOutputDTO containing the chunks and metadata,
            or None if no chunks were found
        """
        pass
