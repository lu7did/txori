"""Pruebas básicas del pipeline fuente->cpu->waterfall."""
from __future__ import annotations

import math
import os
import wave
from tempfile import TemporaryDirectory

import numpy as np

from txori.sources import FileSource
from txori.cpu import NoOpProcessor
from txori.waterfall import SpectrogramAnimator


def _make_wav(path: str, sr: int = 4000, secs: float = 0.2) -> None:
    n = int(sr * secs)
    t = np.arange(n, dtype=np.float32) / sr
    x = (0.5 * np.sin(2 * math.pi * 440 * t)).astype(np.float32)
    # a int16 PCM
    xi = np.clip((x * 32767.0), -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(xi.tobytes())


def test_filesource_and_noop():
    with TemporaryDirectory() as td:
        wav = os.path.join(td, "test.wav")
        _make_wav(wav, sr=3200, secs=0.1)
        src = FileSource(wav)
        assert src.sample_rate == 3200
        chunk = src.read(64)
        assert chunk.dtype == np.float32
        assert chunk.size <= 64
        proc = NoOpProcessor()
        out = proc.process(chunk)
        assert np.allclose(chunk, out)
        src.close()


def test_waterfall_compute_spec():
    with TemporaryDirectory() as td:
        wav = os.path.join(td, "test.wav")
        _make_wav(wav, sr=3200, secs=0.2)
        src = FileSource(wav)
        anim = SpectrogramAnimator(fs=src.sample_rate)
        # Push algunos datos y calcular spec
        anim._push(src.read(1024))
        Pxx, freqs, bins = anim.compute_spec()
        assert Pxx.ndim == 2
        assert freqs.ndim == 1 and bins.ndim == 1
        src.close()
