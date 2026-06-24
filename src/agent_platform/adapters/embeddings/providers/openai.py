from openai import AsyncOpenAI
from core.value_objects.embeddable import Embeddable
from core.entities.embedding import Embedding

class OpenAIEmbeddings:
    client: AsyncOpenAI
    model: str


    def __init__(
        self,
        api_key: str,
        model: str,
    ) -> None:
        
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model


    async def encode(
        self,
        item: Embeddable,
    ) -> Embedding:
        
        response = await self.client.embeddings.create(input=item.content, model=self.model)

        emb = response.data[0].embedding 
        
        return Embedding.from_list(emb, self.model)
        