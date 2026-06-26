from openai import AsyncOpenAI
from models.token import TokenUsage
from models.protocols.text_unit import TextUnit
from models.embedding import Embedding
from mappers.openai.utils import omit_none
from adapters.embeddings.response import EmbeddingResponse
from adapters.embeddings.config import EmbeddingConfig


class OpenAIEmbeddings:
    _client: AsyncOpenAI
    _model: str


    def __init__(
        self,
        api_key: str,
        model: str,
    ) -> None:
        
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model


    async def encode(
        self,
        items: list[TextUnit],
        config: EmbeddingConfig = EmbeddingConfig(),
    ) -> EmbeddingResponse:
        
        texts = [item.text for item in items]

        response = await self._client.embeddings.create(
            input=texts,
            model=self._model,
            dimensions=omit_none(config.dimensions),
        )

        embeddings = [
            Embedding.from_list(data.embedding, model=response.model, id=item.id)
            for item, data in zip(items, response.data)
        ]

        return EmbeddingResponse(
            embeddings=embeddings,
            model=response.model,
            usage=TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=0,
            ),
        )
        