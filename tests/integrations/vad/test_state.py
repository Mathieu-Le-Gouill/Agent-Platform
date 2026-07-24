from agent_platform.core.interfaces.vad.state import VADState


def test_defaults():
    state = VADState()
    assert state.in_speech is False
    assert state.segment_start is None
    assert state.last_speech_end is None
    assert state.speech_ms == 0
    assert state.silence_ms == 0


def test_mutation():
    state = VADState()
    state.in_speech = True
    assert state.in_speech is True
    state.segment_start = 100
    assert state.segment_start == 100
    state.last_speech_end = 500
    assert state.last_speech_end == 500
    state.speech_ms = 300
    assert state.speech_ms == 300
    state.silence_ms = 200
    assert state.silence_ms == 200


def test_reset_from_used_state():
    state = VADState(
        in_speech=True,
        segment_start=100,
        last_speech_end=500,
        speech_ms=300,
        silence_ms=200,
    )
    state.in_speech = False
    state.segment_start = None
    state.speech_ms = 0
    state.silence_ms = 0
    assert state == VADState(last_speech_end=500)
