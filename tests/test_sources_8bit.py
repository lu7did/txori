"""Additional coverage for FileSource 8-bit mono path."""
from __future__ import annotations

import os
import wave
from tempfile import TemporaryDirectory

import numpy as np

from txori.sources import FileSource


def test_filesource_reads_8bit_mono() -> None:
    with TemporaryDirectory() as td:
        path = os.path.join(td, "m8.wav")
        sr = 4000
        n = sr // 4
        t = np.arange(n) / sr
        x = (np.sin(2 * np.pi * 300 * t) * 0.5 + 0.5)
        d8 = np.clip((x * 255.0).astype(np.uint8), 0, 255)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(1)
            wf.setframerate(sr)
            wf.writeframes(d8.tobytes())
        src = FileSource(path)
        y = src.read(512)
        assert y.ndim == 1 and y.dtype == np.float32 and y.size > 0
        src.close()
