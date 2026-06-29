from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


from integrations.vector_store.langchain_base import (
    LangChainVectorStore
)



class QdrantVectorStoreProvider(
    LangChainVectorStore
):


    def __init__(
        self,
        url,
        collection_name,
        embedding,
        api_key=None,
    ) -> None:

        self.url = url
        self.api_key = api_key
        self.collection_name = collection_name
        self.embedding = embedding

        super().__init__()



    def _build_client(self) -> QdrantVectorStore:

        client = QdrantClient(
            url=self.url,
            api_key=self.api_key,
        )


        return QdrantVectorStore(
            client=client,
            collection_name=self.collection_name,
            embedding=self.embedding,
        )