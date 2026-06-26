from openai import AsyncOpenAI
from models.protocols.text_unit import TextUnit
from models.embedding import Embedding

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
        item: TextUnit,
    ) -> Embedding:
        
        response = await self.client.embeddings.create(input=item.text, model=self.model)

        emb = response.data[0].embedding 
        
        return Embedding.from_list(emb, self.model)
        