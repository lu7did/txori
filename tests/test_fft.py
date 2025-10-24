# (c) Dr. Pedro E. Colla 2020-2025 (LU7DZ)
from __future__ import annotations

import numpy as np

from txori.fft_analysis import FFTAnalyzer


def test_fft_bins_count_and_energy() -> None:
    fs = 48000.0
    fc = 3000.0
    bin_hz = 3.0
    n_bins = int(fc // bin_hz) + 1
    analyzer = FFTAnalyzer(fs=fs, fc=fc, bin_hz=bin_hz)
    # Señal: tono puro a 1000 Hz
    t = np.arange(100, dtype=float) / fs
    x = np.sin(2 * np.pi * 1000.0 * t)
    energy = analyzer.analyze(x)
    assert energy.shape[0] == n_bins
    assert energy.sum() > 0
