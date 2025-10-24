# (c) Dr. Pedro E. Colla 2020-2025 (LU7DZ)
"""Visualización de espectro como imagen desplazable."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from PIL import Image

from .exceptions import VisualizationError


@dataclass(slots=True)
class SpectrogramRenderer:
    """Genera un espectrograma desplazable y promediado."""

    height: int
    width: int
    average_frames: int = 100
    update_interval: int = 5
    _accum: deque[npt.NDArray[np.float64]] = field(default_factory=deque, init=False)
    _image: npt.NDArray[np.uint8] = field(init=False)
    _norm_eps: float = 1e-12
    _dirty: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.height <= 0 or self.width <= 0:
            raise VisualizationError("Dimensiones inválidas de imagen")
        self._image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Navy oscuro de fondo
        self._image[:, :] = (0, 0, 80)

    @property
    def image(self) -> npt.NDArray[np.uint8]:
        return self._image

    def _energy_to_color(self, e: float, emax: float) -> tuple[int, int, int]:
        # Escala logarítmica en dB relativa al máximo
        e_safe = max(e, self._norm_eps)
        emax_safe = max(emax, self._norm_eps)
        db = 20.0 * math.log10(e_safe / emax_safe)
        # Mapear rango -80 dB .. 0 dB a 0..1
        t = (db + 80.0) / 80.0
        t = max(0.0, min(1.0, t))
        # azul -> cian -> verde -> amarillo -> rojo (gradiente simple)
        r = int(255 * max(0.0, 2 * t - 1.0))
        g = int(255 * min(1.0, 2 * t))
        b = int(255 * (1.0 - t))
        # Escala por bandas de 10 dB (relativa al máximo):
        # <= -50 dB: light blue; -50..-40: cyan; -40..-30: verde; -30..-20: amarillo; -20..-10: rojo; > -10: blanco
        if db <= -50.0:
            return (173, 216, 230)  # light blue
        if db <= -40.0:
            return (0, 255, 255)  # cyan
        if db <= -30.0:
            return (0, 255, 0)  # green
        if db <= -20.0:
            return (255, 255, 0)  # yellow
        if db <= -10.0:
            return (255, 0, 0)  # red
        return (255, 255, 255)  # white

    def push_spectrum(self, spectrum: npt.NDArray[np.float64]) -> None:
        if spectrum.ndim != 1:
            raise VisualizationError("El espectro debe ser 1-D")
        if spectrum.size != self.height:
            raise VisualizationError(
                "La altura de imagen debe coincidir con el espectro"
            )
        self._accum.append(spectrum.astype(np.float64))
        if len(self._accum) > self.average_frames:
            self._accum.popleft()
        # Dibuja cada update_interval arreglos
        if len(self._accum) % self.update_interval == 0:
            avg = np.mean(np.stack(list(self._accum), axis=0), axis=0)
            emax = float(np.max(avg) if avg.size else 1.0)
            # Desplaza todo a la izquierda y coloca nueva columna a la derecha (tiempo avanza → izquierda)
            self._image = np.roll(self._image, shift=-1, axis=1)
            column = np.zeros((self.height, 3), dtype=np.uint8)
            for y in range(self.height):
                column[y] = self._energy_to_color(float(avg[y]), emax)
            self._image[:, -1, :] = column
            self._dirty = True

    def to_pil(self) -> Image.Image:
        return Image.fromarray(self._image, mode="RGB")

    def save(self, path: str) -> None:
        self.to_pil().save(path)

    def consume_frame(self) -> npt.NDArray[np.uint8] | None:
        if self._dirty:
            self._dirty = False
            return self._image
        return None
