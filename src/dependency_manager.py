import os
from typing import Optional

from src.configs import get_chroma_client
from src.infra import ChromaDBRepository, GoogleAIEmbeddings
from src.ports import Embeddings, VectorStore


class DependencyContainer:
    """
    Singleton container for managing application dependencies.
    """

    __instance: Optional["DependencyContainer"] = None
    __vector_store: Optional[VectorStore] = None
    __embeddings: Optional[Embeddings] = None

    def __new__(cls) -> "DependencyContainer":
        if cls.__instance is None:
            cls.__instance = super(DependencyContainer, cls).__new__(cls)
        return cls.__instance

    @classmethod
    def get_vector_store(cls) -> VectorStore:
        """
        Get the singleton instance of VectorStore.

        Returns:
            The VectorStore implementation.
        """
        if cls.__instance is None:
            cls.__instance = DependencyContainer()

        if cls.__instance.__vector_store is None:
            cls.__instance.__initialize_vector_store()

        return cls.__instance.__vector_store

    @classmethod
    def get_embeddings(cls) -> Embeddings:
        """
        Get the singleton instance of Embeddings.

        Returns:
            The Embeddings implementation.
        """
        if cls.__instance is None:
            cls.__instance = DependencyContainer()

        if cls.__instance.__embeddings is None:
            cls.__instance.__initialize_embeddings()

        return cls.__instance.__embeddings

    def __initialize_vector_store(self) -> None:
        """
        Initialize the default VectorStore implementation.
        """
        # Create a ChromaDB client and collection
        client = get_chroma_client(self.get_embeddings())

        # Create and set the ChromaDBRepository
        self.__vector_store = ChromaDBRepository(conn=client.get_collection())

    def __initialize_embeddings(self) -> None:
        """
        Initialize the default Embeddings implementation.
        """
        # Get API key from environment variables
        api_key = os.environ.get("GOOGLE_API_KEY")

        # Create and set the GoogleAIEmbeddings
        self.__embeddings = GoogleAIEmbeddings(api_key=api_key)


dependency_container = DependencyContainer()
