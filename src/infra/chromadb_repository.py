import chromadb

from src.ports import RetrieveChunksOutputDTO
from src.ports.vector_store import VectorStore


class ChromaDBRepository(VectorStore):
    def __init__(self, conn: chromadb.Collection) -> None:
        self.__conn = conn

    def get_chunks(
        self, inputs: list[str], k: int = 2
    ) -> None | RetrieveChunksOutputDTO:
        results = self.__conn.query(query_texts=inputs, n_results=k)

        chunks = results.get("documents", None)
        metadatas = results.get("metadatas", None)

        if chunks is None or len(chunks[0]) <= 0:
            return None

        return RetrieveChunksOutputDTO(
            chunks=chunks[0],
            metadatas=metadatas[0],
        )
