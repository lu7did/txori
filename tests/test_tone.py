import numpy as np
from txori.audio import ToneAudioSource


def test_tone_record_shape():
    src = ToneAudioSource(sample_rate=8000, frequency=600.0)
    x = src.record(0.1)
    assert x.ndim == 1 and len(x) == 800
    # energía no nula
    assert np.abs(x).mean() > 0.1


def test_tone_blocks_iter(monkeypatch):
    src = ToneAudioSource(sample_rate=8000, frequency=600.0, blocksize=64)
    it = src.blocks()
    b1 = next(it)
    b2 = next(it)
    assert b1.shape == (64,) and b2.shape == (64,)
    # continuidad de fase aproximada: correlación positiva
    assert (b1[-8:] * b2[:8]).mean() > 0
