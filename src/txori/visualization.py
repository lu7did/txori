"""Visualización de espectro como imagen desplazable."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .exceptions import VisualizationError


@dataclass(slots=True)
class SpectrogramRenderer:
    """Genera un espectrograma desplazable y promediado."""

    height: int
    width: int
    average_frames: int = 100
    update_interval: int = 5
    _accum: deque[np.ndarray] = field(default_factory=deque, init=False)
    _image: np.ndarray = field(init=False)
    _norm_eps: float = 1e-12
    _dirty: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.height <= 0 or self.width <= 0:
            raise VisualizationError("Dimensiones inválidas de imagen")
        self._image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Navy oscuro de fondo
        self._image[:, :] = (0, 0, 80)

    @property
    def image(self) -> np.ndarray:
        return self._image

    def _energy_to_color(self, e: float, emax: float) -> tuple[int, int, int]:
        t = float(e / (emax + self._norm_eps))
        t = max(0.0, min(1.0, t))
        # azul -> azul claro -> casi blanco azulado
        base_blue = 80
        inc = int(175 * t)
        b = min(255, base_blue + inc)
        w = int(200 * t)
        r = min(255, w)
        g = min(255, w)
        return (r, g, b)

    def push_spectrum(self, spectrum: np.ndarray) -> None:
        if spectrum.ndim != 1:
            raise VisualizationError("El espectro debe ser 1-D")
        if spectrum.size != self.height:
            raise VisualizationError("La altura de imagen debe coincidir con el espectro")
        self._accum.append(spectrum.astype(np.float64))
        if len(self._accum) > self.average_frames:
            self._accum.popleft()
        # Dibuja cada update_interval arreglos
        if len(self._accum) % self.update_interval == 0:
            avg = np.mean(np.stack(list(self._accum), axis=0), axis=0)
            emax = float(np.max(avg) if avg.size else 1.0)
            # Desplaza todo a la izquierda y coloca nueva columna en x=0
            self._image = np.roll(self._image, shift=1, axis=1)
            column = np.zeros((self.height, 3), dtype=np.uint8)
            for y in range(self.height):
                column[y] = self._energy_to_color(float(avg[y]), emax)
            self._image[:, 0, :] = column
            self._dirty = True

    def to_pil(self) -> Image.Image:
        return Image.fromarray(self._image, mode="RGB")

    def save(self, path: str) -> None:
        self.to_pil().save(path)

    def consume_frame(self) -> np.ndarray | None:
        if self._dirty:
            self._dirty = False
            return self._image
        return None
