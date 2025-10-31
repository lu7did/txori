"""CPU: módulos procesadores de señal."""
from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class Processor(ABC):
    """Interfaz de procesador de muestras."""

    @abstractmethod
    def process(self, x: np.ndarray) -> np.ndarray:
        """Procesa y devuelve las muestras."""


class NoOpProcessor(Processor):
    """Procesador por defecto: no modifica la señal."""

    def process(self, x: np.ndarray) -> np.ndarray:  # noqa: D401
        """Devuelve x sin modificaciones."""
        return x


class LpfProcessor(Processor):
    """Filtro pasabajos + remuestreo a 6 kHz (si fs_in > 6 kHz)."""

    def __init__(self, fs_in: int, cutoff_hz: float = 3000.0, target_fs: int = 6000) -> None:
        self.fs_in = int(fs_in)
        self.target_fs = int(target_fs) if self.fs_in > target_fs else self.fs_in
        # Diseño FIR sencillo ventana Hamming
        num_taps = 63
        n = np.arange(num_taps) - (num_taps - 1) / 2.0
        fc = min(cutoff_hz, 0.49 * (self.fs_in / 2.0))
        sinc = np.sinc(2.0 * fc / self.fs_in * n)
        window = np.hamming(num_taps)
        h = sinc * window
        h /= np.sum(h)
        self._h = h.astype(np.float32)
        self._xprev = np.zeros(num_taps - 1, dtype=np.float32)
        # Buffer para remuestreo fraccional y fase
        self._ybuf = np.zeros(0, dtype=np.float32)
        self._t = 0.0  # índice fraccional en ybuf
        self.fs_out = self.target_fs

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        xi = x.astype(np.float32, copy=False)
        inp = np.concatenate((self._xprev, xi))
        y = np.convolve(inp, self._h, mode="valid").astype(np.float32)
        self._xprev = inp[-(self._h.size - 1) :]
        if self.fs_out >= self.fs_in:
            # Sin remuestreo (igual o menor Fs de entrada)
            return y
        # Remuestreo lineal a fs_out
        self._ybuf = np.concatenate((self._ybuf, y))
        step = self.fs_in / float(self.fs_out)
        outs = []
        t = self._t
        # Necesitamos al menos 2 muestras para interpolar
        while t + 1.0 < self._ybuf.size:
            i = int(t)
            frac = t - i
            v = (1.0 - frac) * self._ybuf[i] + frac * self._ybuf[i + 1]
            outs.append(v)
            t += step
        # Descartar las muestras consumidas
        drop = int(t)
        if drop > 0:
            self._ybuf = self._ybuf[drop:]
            t -= drop
        self._t = t
        return np.asarray(outs, dtype=np.float32)
