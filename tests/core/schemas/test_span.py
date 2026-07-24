from agent_platform.core.schemas.span import SampleSpan, TimeSpan


class TestTimeSpan:
    def test_duration_positive(self):
        span = TimeSpan(start=0.0, end=10.0)
        assert span.duration == 10.0

    def test_duration_zero(self):
        span = TimeSpan(start=5.0, end=5.0)
        assert span.duration == 0.0

    def test_duration_negative(self):
        span = TimeSpan(start=10.0, end=0.0)
        assert span.duration == -10.0

    def test_duration_float(self):
        span = TimeSpan(start=1.5, end=3.7)
        assert span.duration == 2.2

    def test_default_unit(self):
        span = TimeSpan(start=0.0, end=1.0)
        assert span.unit == "ms"

    def test_custom_unit(self):
        span = TimeSpan(start=0.0, end=1.0, unit="s")
        assert span.unit == "s"


class TestSampleSpan:
    def test_sample_size(self):
        span = SampleSpan(start=0, end=100)
        assert span.sample_size == 100

    def test_sample_size_zero(self):
        span = SampleSpan(start=50, end=50)
        assert span.sample_size == 0

    def test_sample_size_single(self):
        span = SampleSpan(start=10, end=11)
        assert span.sample_size == 1

    def test_to_ms(self):
        span = SampleSpan(start=0, end=16000)
        ts = span.to_ms(16000)
        assert isinstance(ts, TimeSpan)
        assert ts.start == 0
        assert ts.end == 1
        assert ts.unit == "ms"

    def test_to_ms_non_uniform(self):
        span = SampleSpan(start=8000, end=48000)
        ts = span.to_ms(16000)
        assert ts.start == 0
        assert ts.end == 3

    def test_to_ms_division_truncation(self):
        span = SampleSpan(start=0, end=15999)
        ts = span.to_ms(16000)
        assert ts.start == 0
        assert ts.end == 0
