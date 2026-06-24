from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain.embeddings import Embeddings
from core.entities.chunk import Chunk
from core.entities.document import Document
from platform.adapters.loaders._langchain_bridge import from_langchain, to_langchain
from uuid import UUID


class QdrantStore: # implements VectorStorePort
    vector_store: QdrantVectorStore
    client: QdrantClient

    def __init__(
        self,
        url: str,
        api_key: str,
        cloud_inference: bool,
        collection_name: str,
        embeddings: Embeddings,
    ) -> None:
        
        self.client = QdrantClient(
            url=url,
            api_key=api_key,
            cloud_inference=cloud_inference
        )

        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=embeddings,
        )
        

    async def add(
        self,
        documents: list[Document]
    ) -> None: 
        
        lc_docs = [
            to_langchain(chunk, document)
            for document in documents
            for chunk in document.chunks
        ]

        await self.vector_store.aadd_documents(lc_docs)

    
    async def delete(
        self,
        document_ids: list[UUID],
    ) -> None:
        await self.vector_store.adelete(
            filter={
                "should": [
                    {
                        "key": "document_id",
                        "match": {
                            "value": str(document_id),
                        },
                    }
                    for document_id in document_ids
                ]
            }
        )


    async def search(
        self,
        query: str,
        k: int = 5,
    ) -> list[Chunk]:
        
        results = await self.vector_store.asimilarity_search(
            query,
            k=k,
        )

        return [
            from_langchain(doc)
            for doc in results
        ]
