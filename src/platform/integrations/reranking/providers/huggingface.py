from langchain_community.document_compressors import CrossEncoderReranker

from langchain_community.cross_encoders import HuggingFaceCrossEncoder


from integrations.reranking.langchain_base import LangChainReranker


class HuggingFaceRerankerProvider(LangChainReranker):

    def __init__(
        self,
        model="BAAI/bge-reranker-v2-m3",
    ):
        self._model = model

        super().__init__()


    def _client(self) -> CrossEncoderReranker:

        encoder = HuggingFaceCrossEncoder(model_name=self._model)

        return CrossEncoderReranker(model=encoder)