from agent_platform.core.interfaces.vad.config import VADConfig
from agent_platform.integrations.vad.pvcobra.config import PvcobraVadConfig
from agent_platform.integrations.vad.silero.config import SileroVadConfig
from agent_platform.integrations.vad.ten.config import TenVadConfig
from agent_platform.integrations.vad.webrtc.config import WebrtcVadConfig


class TestVADConfig:
    def test_defaults(self):
        cfg = VADConfig()
        assert cfg.sample_rate == 16000
        assert cfg.speech_pad_ms == 30
        assert cfg.min_silence_duration_ms == 100
        assert cfg.min_speech_duration_ms == 250

    def test_custom_values(self):
        cfg = VADConfig(
            sample_rate=8000,
            speech_pad_ms=50,
            min_silence_duration_ms=200,
            min_speech_duration_ms=500,
        )
        assert cfg.sample_rate == 8000
        assert cfg.speech_pad_ms == 50
        assert cfg.min_silence_duration_ms == 200
        assert cfg.min_speech_duration_ms == 500

    def test_accepts_all_positive_values(self):
        cfg = VADConfig(sample_rate=16000)
        assert cfg.sample_rate == 16000


class TestSileroVadConfig:
    def test_inherits_defaults(self):
        cfg = SileroVadConfig()
        assert isinstance(cfg, VADConfig)
        assert cfg.sample_rate == 16000
        assert cfg.threshold == 0.5

    def test_custom_threshold(self):
        cfg = SileroVadConfig(threshold=0.8)
        assert cfg.threshold == 0.8

    def test_max_speech_duration_s_defaults_to_infinite(self):
        cfg = SileroVadConfig()
        assert cfg.max_speech_duration_s == float("inf")

    def test_custom_max_speech_duration_s(self):
        cfg = SileroVadConfig(max_speech_duration_s=30.0)
        assert cfg.max_speech_duration_s == 30.0

    def test_max_samples_default_covers_several_seconds_at_16khz(self):
        cfg = SileroVadConfig()
        assert cfg.max_samples == 80_000
        assert cfg.max_samples / cfg.sample_rate >= 1.0

    def test_custom_max_samples(self):
        cfg = SileroVadConfig(max_samples=50)
        assert cfg.max_samples == 50


class TestWebrtcVadConfig:
    def test_inherits_defaults(self):
        cfg = WebrtcVadConfig()
        assert isinstance(cfg, VADConfig)
        assert cfg.sample_rate == 16000
        assert cfg.mode == 1

    def test_custom_mode(self):
        cfg = WebrtcVadConfig(mode=3)
        assert cfg.mode == 3

    def test_aggressiveness_field_no_longer_exists(self):
        assert "aggressiveness" not in WebrtcVadConfig.model_fields
        cfg = WebrtcVadConfig(mode=1, aggressiveness=3)
        assert not hasattr(cfg, "aggressiveness")


class TestPvcobraVadConfig:
    def test_inherits_defaults(self):
        cfg = PvcobraVadConfig()
        assert isinstance(cfg, VADConfig)
        assert cfg.sample_rate == 16000
        assert cfg.device is None
        assert cfg.library_path is None
        assert cfg.threshold == 0.5

    def test_custom_device(self):
        cfg = PvcobraVadConfig(device="/dev/snd")
        assert cfg.device == "/dev/snd"


class TestTenVadConfig:
    def test_inherits_defaults(self):
        cfg = TenVadConfig()
        assert isinstance(cfg, VADConfig)
        assert cfg.sample_rate == 16000
        assert cfg.hop_size == 256
        assert cfg.threshold == 0.5

    def test_custom_hop_size(self):
        cfg = TenVadConfig(hop_size=160)
        assert cfg.hop_size == 160

    def test_hop_size_rejects_values_outside_160_256(self):
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            TenVadConfig(hop_size=512)

        with pytest.raises(ValidationError):
            TenVadConfig(hop_size=64)
