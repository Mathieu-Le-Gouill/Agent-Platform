from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain.embeddings import Embeddings
from langchain_core.documents import Document
from uuid import UUID

class QdrantStore:
    vector_store: QdrantVectorStore

    def __init__(
        self,
        url: str,
        api_key: str,
        cloud_inference: bool,
        collection_name: str,
        embeddings: Embeddings,
    ) -> None:
        
        client = QdrantClient(url=url, api_key=api_key, cloud_inference=cloud_inference)
        collection_name = collection_name

        self.vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
        )
        

    async def add(
        self,
        documents: list[Document]
    ) -> None: 
        self.vector_store.add_documents(documents=documents)

    
    async def delete(
        self,
        ids: list[UUID]
    ) -> None:
        self.vector_store.delete(ids)


    async def retrieve(
        self,
        query: str
    ) -> list[Chunk]:
        results = self.client.query(collection_name=self.collection_name, query_text=query)
        ...