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
