"""Tests for ToneSource and stereo FileSource branch coverage."""
from __future__ import annotations

import os
import wave
from tempfile import TemporaryDirectory

import numpy as np

from txori.sources import FileSource, ToneSource


def _write_wav_stereo(path: str, sr: int, sampwidth: int, left: np.ndarray, right: np.ndarray) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sr)
        interleaved = np.empty(left.size + right.size, dtype=left.dtype)
        interleaved[0::2] = left
        interleaved[1::2] = right
        wf.writeframes(interleaved.tobytes())


def test_tone_source_generates_expected_length_and_range() -> None:
    ts = ToneSource(freq_hz=440.0, fs=4000)
    x = ts.read(1000)
    assert x.shape == (1000,)
    assert x.dtype == np.float32 and np.max(np.abs(x)) <= 1.0 + 1e-6


def test_filesource_stereo_mixes_to_mono() -> None:
    with TemporaryDirectory() as td:
        sr = 8000
        t = np.arange(sr) / sr
        l = (np.sin(2 * np.pi * 440 * t) * 0.9 + 0.0)
        r = (np.sin(2 * np.pi * 880 * t) * 0.9 + 0.0)
        d16_l = np.clip((l * 32767).astype(np.int16), -32768, 32767)
        d16_r = np.clip((r * 32767).astype(np.int16), -32768, 32767)
        wav = os.path.join(td, "stereo.wav")
        _write_wav_stereo(wav, sr, 2, d16_l, d16_r)
        src = FileSource(wav)
        y = src.read(2000)
        assert y.ndim == 1 and y.dtype == np.float32 and y.size > 0
        src.close()
