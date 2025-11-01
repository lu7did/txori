"""Additional tests for sources module to boost coverage."""
from __future__ import annotations

import os
import wave
from tempfile import TemporaryDirectory

import numpy as np

from txori.sources import FileSource


def _write_wav(path: str, sr: int, sampwidth: int, data: np.ndarray) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sr)
        wf.writeframes(data.tobytes())


def test_filesource_reads_8bit_and_16bit_pcm() -> None:
    with TemporaryDirectory() as td:
        sr = 8000
        # 8-bit unsigned
        x8 = (np.sin(2 * np.pi * 440 * np.arange(sr) / sr) * 0.5 + 0.5)
        d8 = np.clip((x8 * 255).astype(np.uint8), 0, 255)
        w8 = os.path.join(td, "a8.wav")
        _write_wav(w8, sr, 1, d8)
        s8 = FileSource(w8)
        y8 = s8.read(1000)
        assert y8.dtype == np.float32
        assert s8.sample_rate == sr
        assert np.isfinite(y8).all()
        s8.close()
        # 16-bit signed
        x16 = np.sin(2 * np.pi * 880 * np.arange(sr) / sr) * 0.8
        d16 = np.clip((x16 * 32767).astype(np.int16), -32768, 32767)
        w16 = os.path.join(td, "a16.wav")
        _write_wav(w16, sr, 2, d16)
        s16 = FileSource(w16)
        y16 = s16.read(2000)
        assert y16.dtype == np.float32
        assert s16.sample_rate == sr
        assert np.isfinite(y16).all()
        s16.close()
