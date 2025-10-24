# (c) Dr. Pedro E. Colla 2020-2025 (LU7DZ)
"""Subsistema de procesamiento de señales (placeholder)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


class IdentityProcessor:
    """Por ahora solo copia el arreglo de entrada."""

    def process(self, window: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return window.copy()


class AGCProcessor:
    """Control Automático de Ganancia (placeholder): copia sin cambios."""

    def process(self, window: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return window.copy()


class DSPProcessor:
    """Módulo DSP (placeholder): copia sin cambios."""

    def process(self, window: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return window.copy()
