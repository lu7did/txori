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
        # Mapeo en bandas con umbrales fijos (no se usa t continuo)
        # Gradiente suave con umbrales dB definidos para transiciones
        stops = [
            (0, 0, 128),   # navy (fondo)
            (0, 0, 255),   # blue
            (135, 206, 235),  # sky blue
            (0, 255, 255),   # cyan
            (0, 255, 0),   # green
            (255, 255, 0),   # yellow
            (255, 0, 0),   # red
            (255, 255, 255),  # white
        ]
        # Umbrales en dB relativos al máximo para los stops anteriores
        # [-120, -90] blue, [-90, -70] sky, [-70, -50] cyan, [-50, -35] green, [-35, -20] yellow, [-20, -10] red, [-10, 0] white
        thr = [-120.0, -90.0, -70.0, -50.0, -35.0, -20.0, -10.0, 0.0]
        if db <= thr[0]:
            return stops[0]
        if db >= thr[-1]:
            return stops[-1]
        # Buscar segmento de forma robusta con bisect
        i = max(0, min(len(stops) - 2, bisect_right(thr, db) - 1))
        # Interpolación lineal dentro del segmento i..i+1
        span = max(1e-9, thr[i + 1] - thr[i])
        f = (db - thr[i]) / span
        r = int(round(stops[i][0] + f * (stops[i + 1][0] - stops[i][0])))
        g = int(round(stops[i][1] + f * (stops[i + 1][1] - stops[i][1])))
        b = int(round(stops[i][2] + f * (stops[i + 1][2] - stops[i][2])))
        return (r, g, b)

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
