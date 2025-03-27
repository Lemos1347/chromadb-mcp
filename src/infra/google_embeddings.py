from typing import List

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.ports.embeddings import Embeddings


class GoogleAIEmbeddings(Embeddings):
    """
    Implementation of the Embeddings interface using Google's Generative AI embeddings.
    """

    def __init__(self, api_key: str, model_name: str = "models/text-embedding-004"):
        """
        Initialize the Google AI embeddings with the specified model and API key.

        Args:
            model_name: The name of the Google Generative AI model to use
            api_key: Google API key, if not provided will look for GOOGLE_API_KEY env variable
        """
        self._embeddings = GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=api_key,
        )

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.

        Args:
            documents: List of text documents to generate embeddings for

        Returns:
            A list of embeddings, where each embedding is a list of floats
        """
        return self._embeddings.embed_documents(documents)

    def embed_query(self, query: str) -> List[float]:
        """
        Generate embeddings for a single query text.

        Args:
            query: Text to generate embeddings for

        Returns:
            A single embedding as a list of floats
        """
        return self._embeddings.embed_query(query)
        
    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Implement the __call__ method expected by ChromaDB.
        This allows the GoogleAIEmbeddings class to be used directly with ChromaDB.
        
        Args:
            input: List of text documents to generate embeddings for
            
        Returns:
            A list of embeddings, where each embedding is a list of floats
        """
        return self.embed_documents(input)

