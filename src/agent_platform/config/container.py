"""
class Container:
    def __init__(self, settings: Settings):
        # --- services ---
        self.stt         = WhisperCppClient(settings.whisper_url)
        self.translator  = NllbTranslator(settings.nllb_model)
        self.llm         = OpenAILLM(settings.llm_model)
        self.embedder    = SentenceTransformerEmbedder(settings.embedder_model)
        self.store       = QdrantStore(settings.qdrant_url)
        self.reranker    = BgeReranker(settings.reranker_model)

        # --- pipelines ---
        self.speech_translation = SpeechTranslationPipeline(self.stt, self.translator)
        self.rag_query          = RagQueryPipeline(self.embedder, self.store, self.reranker, self.llm)
        self.summarization      = SummarizationPipeline(self.llm)
"""