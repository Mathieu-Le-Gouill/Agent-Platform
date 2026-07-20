from agent_platform.core.interfaces.speech.config import SpeechConfig
from agent_platform.integrations.speech_to_text.deepgram.config import DeepgramConfig
from agent_platform.integrations.speech_to_text.whisperx.config import WhisperXConfig


class TestSpeechConfig:
    def test_default_model(self):
        cfg = SpeechConfig()
        assert cfg.model == ""

    def test_custom_model(self):
        cfg = SpeechConfig(model="nova-2-phonecall")
        assert cfg.model == "nova-2-phonecall"


class TestDeepgramConfig:
    def test_inherits_defaults(self):
        cfg = DeepgramConfig()
        assert isinstance(cfg, SpeechConfig)
        assert cfg.model == "nova-2"

    def test_custom_model(self):
        cfg = DeepgramConfig(model="whisper")
        assert cfg.model == "whisper"


class TestWhisperXConfig:
    def test_defaults(self):
        cfg = WhisperXConfig()
        assert cfg.model_size == "large-v3"
        assert cfg.device == "cpu"
        assert cfg.compute_type == "float32"
        assert cfg.batch_size == 16
        assert cfg.min_duration_ms == 5000

    def test_custom_values(self):
        cfg = WhisperXConfig(
            model_size="small",
            device="cuda",
            compute_type="float16",
            batch_size=32,
            min_duration_ms=1000,
        )
        assert cfg.model_size == "small"
        assert cfg.device == "cuda"
        assert cfg.compute_type == "float16"
        assert cfg.batch_size == 32
        assert cfg.min_duration_ms == 1000
