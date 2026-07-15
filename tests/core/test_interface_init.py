from agent_platform.core.credentials import NoCredentials
from agent_platform.core.interfaces.embeddings.base import BaseEmbeddingProvider
from agent_platform.core.interfaces.embeddings.response import EmbeddingResponse
from agent_platform.core.interfaces.llm.base import BaseLLMProvider
from agent_platform.core.interfaces.speech.base import BaseSpeechToText
from agent_platform.core.interfaces.vector_store.base import BaseVectorStore
from agent_platform.core.schemas.embedding import Embedding


class TestProviderInit:

    def test_embedding_provider_init(self):
        creds = NoCredentials()

        class _Prov(BaseEmbeddingProvider):
            def embed_document(self, items, config=None):
                return EmbeddingResponse(embeddings=[], model="")

            async def aembed_document(self, items, config=None):
                return EmbeddingResponse(embeddings=[], model="")

            def embed_query(self, query, config=None):
                return EmbeddingResponse(embeddings=[], model="")

            async def aembed_query(self, query, config=None):
                return EmbeddingResponse(embeddings=[], model="")

        p = _Prov(credentials=creds)
        assert p._credentials is creds

    def test_llm_provider_init(self):
        creds = NoCredentials()

        class _Prov(BaseLLMProvider):
            def generate(self, prompt, config=None, tools=None):
                from agent_platform.core.interfaces.llm.response import LLMResponse
                return LLMResponse(message=None, model="", finish_reason=None)

            async def agenerate(self, prompt, config=None, tools=None):
                from agent_platform.core.interfaces.llm.response import LLMResponse
                return LLMResponse(message=None, model="", finish_reason=None)

            def stream(self, prompt, config=None):
                return iter([])

        p = _Prov(credentials=creds)
        assert p._credentials is creds

    def test_speech_provider_init(self):
        creds = NoCredentials()

        class _Prov(BaseSpeechToText):
            async def transcribe(self, audio, config=None):
                from agent_platform.core.schemas.conversation import Transcript
                return Transcript(utterances=[])

            def stream(self, audio, config=None):
                return iter([])

        p = _Prov(credentials=creds)
        assert p._credentials is creds

    def test_vector_store_init(self):
        creds = NoCredentials()

        class _Prov(BaseVectorStore):
            async def add(self, chunks):
                return []

            async def delete(self, ids):
                pass

            async def search(self, vector, k, filter=None, config=None):
                return []

            async def search_with_scores(self, vector, k, filter=None, config=None):
                return []

        p = _Prov(credentials=creds)
        assert p._credentials is creds


class TestEmbeddingResponse:
    def test_iter(self):
        emb = Embedding.from_list([0.1, 0.2])
        response = EmbeddingResponse(embeddings=[emb], model="test")
        items = list(iter(response))
        assert items == [emb]

    def test_len(self):
        emb = Embedding.from_list([0.1, 0.2])
        response = EmbeddingResponse(embeddings=[emb], model="test")
        assert len(response) == 1

    def test_len_empty(self):
        response = EmbeddingResponse(embeddings=[], model="test")
        assert len(response) == 0

    def test_iter_multiple(self):
        e1 = Embedding.from_list([0.1])
        e2 = Embedding.from_list([0.2])
        response = EmbeddingResponse(embeddings=[e1, e2], model="test")
        assert list(iter(response)) == [e1, e2]
