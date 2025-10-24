# (c) Dr. Pedro E. Colla 2020-2025 (LU7DZ)
"""Txori: biblioteca de procesamiento de señales y visualización.

Este paquete proporciona una canalización modular para:
- Captura de audio o señales sintéticas.
- Limitación de ancho de banda (filtro pasabajos).
- Procesamiento (placeholder inicial).
- Análisis espectral mediante FFT con resolución de ~3 Hz hasta 3 kHz.
- Visualización tipo espectrograma desplazable.
"""

from .config import SystemConfig
from .pipeline import Pipeline

__all__ = ["SystemConfig", "Pipeline"]
