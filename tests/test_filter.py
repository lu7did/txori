# (c) Dr. Pedro E. Colla 2020-2025 (LU7DZ)
from __future__ import annotations

import numpy as np

from txori.filtering import OnePoleLowPass


def test_lowpass_monotonic_response() -> None:
    fs = 48000.0
    fc = 3000.0
    lp = OnePoleLowPass(fs=fs, fc=fc)
    x = np.zeros(10)
    x[0] = 1.0
    y = lp.process_window(x)
    # Respuesta al impulso: positiva y decreciente
    assert np.all(y >= 0)
    assert np.all(np.diff(y) <= 1e-9)
