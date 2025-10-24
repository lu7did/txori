# (c) Dr. Pedro E. Colla 2020-2025 (LU7DZ)
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
        # Elegir tamaño FFT potencia de 2 >= longitud de ventana
        nfft = 1 << (max(1, window.size) - 1).bit_length()
        # Ventana de Hann para reducir leakage
        w = np.hanning(window.size)
        xw = (window * w).astype(np.float64)
        if nfft > xw.size:
            xw = np.pad(xw, (0, nfft - xw.size))
        X = np.fft.rfft(xw, n=nfft)
        freqs = np.fft.rfftfreq(nfft, d=1.0 / self.fs)
        # Limitar a banda [0, fc)
        mask = freqs < (self.fc - 1e-9)
        freqs = freqs[mask]
        power = (X.real**2 + X.imag**2)[mask]
        # Mapear a bins uniformes de ancho bin_hz: indices 0..n_bins-1
        n_bins = int(math.floor(self.fc / max(self.bin_hz, 1e-9))) + 1
        out = np.zeros(n_bins, dtype=np.float64)
        idx = np.floor(freqs / max(self.bin_hz, 1e-9)).astype(int)
        idx = np.clip(idx, 0, n_bins - 1)
        np.add.at(out, idx, power)
        return out
