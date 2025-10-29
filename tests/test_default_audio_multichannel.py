import numpy as np
from txori.audio import DefaultAudioSource

def test_default_audio_multichannel(monkeypatch):
    frames = 32
    def _rec(nframes, samplerate, channels, dtype):  # noqa: D401
        assert channels == 1 or channels == 1  # ensure call
        # Simula 2 canales para probar promedio
        return (np.random.rand(nframes, 2).astype("float32") * 2 - 1)
    monkeypatch.setattr("txori.audio.sd.rec", _rec)
    monkeypatch.setattr("txori.audio.sd.wait", lambda: None)
    src = DefaultAudioSource(sample_rate=8000, channels=2)
    x = src.record(frames / 8000.0)
    assert x.ndim == 1 and x.shape[0] == frames
