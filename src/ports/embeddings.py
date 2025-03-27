from abc import abstractmethod
from typing import List, Protocol, runtime_checkable


@runtime_checkable
class Embeddings(Protocol):
    """
    Abstract interface for embeddings generation.
    Any class that implements this protocol can be used as an embeddings generator.
    """

    @abstractmethod
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.

        Args:
            documents: List of text documents to generate embeddings for

        Returns:
            A list of embeddings, where each embedding is a list of floats
        """
        pass

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embeddings for a single query text.

        Args:
            query: Text to generate embeddings for

        Returns:
            A single embedding as a list of floats
        """
        pass 