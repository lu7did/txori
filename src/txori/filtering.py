"""Subsistema de límite de ancho de banda (filtro pasabajos)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


@dataclass(slots=True)
class OnePoleLowPass:
    """Filtro IIR de primer orden eficiente.

    Ecuación: y[n] = y[n-1] + a * (x[n] - y[n-1])
    donde a = (2*pi*fc) / (2*pi*fc + fs)
    """

    fs: float
    fc: float
    y_prev: float = 0.0
    _alpha: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not (0 < self.fc < self.fs / 2):
            raise ValueError("fc debe estar entre 0 y fs/2")
        self._alpha = (2 * math.pi * self.fc) / (2 * math.pi * self.fc + self.fs)

    def process_sample(self, x: float) -> float:
        y = self.y_prev + self._alpha * (x - self.y_prev)
        self.y_prev = y
        return y

    def process_window(self, window: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        out: npt.NDArray[np.float64] = np.empty_like(window)
        # Procesa desde la muestra más reciente hacia las antiguas
        for i in range(window.shape[0]):
            out[i] = self.process_sample(float(window[i]))
        return out
