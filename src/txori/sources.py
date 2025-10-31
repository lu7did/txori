"""Sources: entrada de audio para el pipeline."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional
import wave
import numpy as np


class Source(ABC):
    """Interfaz de fuente de audio."""

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Frecuencia de muestreo en Hz."""

    @abstractmethod
    def read(self, n: int) -> np.ndarray:
        """Lee hasta n muestras mono normalizadas en [-1, 1]."""

    @abstractmethod
    def close(self) -> None:
        """Libera recursos de la fuente."""


class FileSource(Source):
    """Fuente que lee muestras desde un archivo WAV."""

    def __init__(self, path: str) -> None:
        self._wf = wave.open(path, "rb")
        self._sr = int(self._wf.getframerate())
        self._sampwidth = self._wf.getsampwidth()
        self._channels = self._wf.getnchannels()
        if self._sampwidth not in (1, 2):
            self._wf.close()
            raise ValueError("Solo se soportan WAV PCM de 8 o 16 bits")

    @property
    def sample_rate(self) -> int:
        return self._sr

    def read(self, n: int) -> np.ndarray:
        frames = self._wf.readframes(max(0, int(n)))
        if not frames:
            return np.array([], dtype=np.float32)
        if self._sampwidth == 1:
            # PCM8 unsigned: 0..255 -> [-1, 1)
            data = np.frombuffer(frames, dtype=np.uint8)
            if self._channels > 1:
                data = data.reshape(-1, self._channels).mean(axis=1)
            out = ((data.astype(np.float32) - 128.0) / 128.0).astype(np.float32)
            return out
        elif self._sampwidth == 2:
            # PCM16 signed: -32768..32767 -> [-1, 1)
            data = np.frombuffer(frames, dtype=np.int16)
            if self._channels > 1:
                data = data.reshape(-1, self._channels).mean(axis=1)
            out = (data.astype(np.float32)) / 32768.0
            return out
        else:
            # No debería ocurrir por validación en __init__
            return np.array([], dtype=np.float32)

    def close(self) -> None:
        try:
            self._wf.close()
        except Exception:
            pass


class ToneSource(Source):
    """Fuente que sintetiza un tono senoidal continuo."""

    def __init__(self, freq_hz: float = 600.0, fs: int = 4000) -> None:
        self._sr = int(fs)
        self._freq = float(freq_hz)
        self._phase = 0.0
        self._dphi = 2.0 * np.pi * (self._freq / float(self._sr))

    @property
    def sample_rate(self) -> int:
        return self._sr

    def read(self, n: int) -> np.ndarray:
        n = max(0, int(n))
        if n == 0:
            return np.array([], dtype=np.float32)
        idx = np.arange(n, dtype=np.float64)
        phase = self._phase + idx * self._dphi
        x = np.sin(phase).astype(np.float32)
        self._phase = float((self._phase + n * self._dphi) % (2.0 * np.pi))
        return x

    def close(self) -> None:
        return
