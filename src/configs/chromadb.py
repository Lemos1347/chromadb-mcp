import chromadb

_DEFAULT_COLLECTION_NAME = "langchain"


class ChromaDBClient:
    def __init__(
        self,
        persist_directory: str,
        embedding_function,
        collection_name: str = _DEFAULT_COLLECTION_NAME,
    ) -> None:
        self.__client = chromadb.PersistentClient(path=persist_directory)

        self.__collection = self.__client.get_collection(
            name=collection_name, embedding_function=embedding_function
        )

    def get_collection(self) -> chromadb.Collection:
        return self.__collection
