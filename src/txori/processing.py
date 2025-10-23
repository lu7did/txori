"""Subsistema de procesamiento de señales (placeholder)."""

from __future__ import annotations

import numpy as np


class IdentityProcessor:
    """Por ahora solo copia el arreglo de entrada."""

    def process(self, window: np.ndarray) -> np.ndarray:
        return window.copy()
