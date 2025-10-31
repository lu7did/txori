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
        if self._sampwidth != 2:
            # Mantener simple: solo PCM16
            self._wf.close()
            raise ValueError("Solo se soportan WAV PCM de 16 bits")

    @property
    def sample_rate(self) -> int:
        return self._sr

    def read(self, n: int) -> np.ndarray:
        frames = self._wf.readframes(max(0, int(n)))
        if not frames:
            return np.array([], dtype=np.float32)
        data = np.frombuffer(frames, dtype=np.int16)
        if self._channels > 1:
            data = data.reshape(-1, self._channels).mean(axis=1).astype(np.int16)
        # Normaliza a [-1, 1]
        out = (data.astype(np.float32)) / 32768.0
        return out

    def close(self) -> None:
        try:
            self._wf.close()
        except Exception:
            pass
