from langchain_chroma import Chroma

from integrations.vector_store.langchain_base import LangChainVectorStore


class ChromaStore(LangChainVectorStore):

    def __init__(
        self,
        collection_name,
        embedding,
    ) -> None:
        self.collection_name = collection_name
        self.embedding = embedding

        super().__init__()


    def _build_client(self) -> Chroma:

        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding,
        )