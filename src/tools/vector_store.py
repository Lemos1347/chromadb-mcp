from src.ports import VectorStore


class Tools:
    def __init__(self, vector_store: VectorStore) -> None:
        self.__vector_store = vector_store

    def retrieve_similar_chunks(self, query: str, k: int = 2) -> str:
        results = self.__vector_store.get_chunks(inputs=[query], k=k)
        if results is None:
            return "No results were found"
        return results.model_dump_json()
