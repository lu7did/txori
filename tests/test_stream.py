import pytest
from txori.audio import StreamAudioSource


def test_stream_invalid_blocksize():
    src = StreamAudioSource(sample_rate=8000, channels=1, blocksize=0)
    with pytest.raises(ValueError):
        next(src.blocks())
