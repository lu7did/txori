from txori.audio import DefaultAudioSource
import pytest


def test_audio_invalid_duration():
    src = DefaultAudioSource(sample_rate=8000, channels=1)
    with pytest.raises(ValueError):
        src.record(-1.0)
