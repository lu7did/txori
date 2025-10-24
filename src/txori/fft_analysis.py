"""Análisis espectral mediante FFT."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(slots=True)
class FFTAnalyzer:
    """Calcula energía por intervalos de frecuencia hasta fc con ancho ~bin_hz."""

    fs: float
    fc: float
    bin_hz: float

    def analyze(self, window: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        nfft_min = int(math.ceil(self.fs / max(self.bin_hz, 1e-9)))
        nfft = 1 << (nfft_min - 1).bit_length()  # potencia de 2 >= nfft_min
        # Aplica ventana de Hann para reducir leakage; zero-pad a nfft
        w = np.hanning(window.size)
        xw = (window * w).astype(np.float64)
        if nfft > xw.size:
            pad = np.zeros(nfft - xw.size, dtype=np.float64)
            xw = np.concatenate([xw, pad])
        X = np.fft.rfft(xw, n=nfft)
        freqs = np.fft.rfftfreq(nfft, d=1.0 / self.fs)
        mask = freqs <= self.fc
        freqs = freqs[mask]
        power = (X.real**2 + X.imag**2)[mask]
        # Agrupa por intervalo de bin_hz
        bin_index = np.floor(freqs / self.bin_hz).astype(int)
        n_bins = int(math.floor(self.fc / self.bin_hz)) + 1
        out = np.zeros(n_bins, dtype=np.float64)
        np.add.at(out, bin_index, power)
        return out
