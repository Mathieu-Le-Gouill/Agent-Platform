from typing import Protocol

class VectorStorePort(Protocol):
    async def store(self, vectorStore) -> str: 
        ...