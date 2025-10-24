# (c) Dr. Pedro E. Colla 2020-2025 (LU7DZ)
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

# Declaración para tipado sin depender de matplotlib en tiempo de chequeo
plt: Any = None


@dataclass
class LiveViewer:
    """Visor en vivo de espectrograma en ventana propia."""

    title: str = "Txori - Espectrograma"
    max_freq_hz: float = 3000.0
    bin_hz: float = 3.0
    seconds_per_col: float = 0.001
    title_text: str | None = None
    device_text: str | None = None
    _fig: Any | None = None
    _ax_spec: Any | None = None
    _im_spec: Any | None = None

    def _ensure_backend(self) -> None:  # pragma: no cover
        # Importa perezosamente matplotlib para no requerirlo en CI/tests
        global plt
        try:
            import importlib

            plt = importlib.import_module("matplotlib.pyplot")
        except Exception as e:  # pragma: no cover - dependiente del entorno
            raise RuntimeError(
                "La visualización en vivo requiere matplotlib. Instala con 'pip install matplotlib'."
            ) from e
        if self._fig is None:
            plt.ion()
            self._fig, self._ax_spec = plt.subplots(1, 1)
            self._fig.canvas.manager.set_window_title(self.title)
            # Textos externos, más alejados del gráfico
            if self.title_text:
                self._fig.text(0.01, 0.995, self.title_text, ha="left", va="top")
            if self.device_text:
                self._fig.text(0.99, 0.995, self.device_text, ha="right", va="top")

    def update(
        self, image: npt.NDArray[np.uint8], level: float | None = None
    ) -> None:  # pragma: no cover - depende de matplotlib
        self._ensure_backend()
        assert self._ax_spec is not None
        global plt
        # Espectrograma
        if self._im_spec is None:
            self._im_spec = self._ax_spec.imshow(image, origin="lower")
            self._ax_spec.set_xlabel("Tiempo (seg)")
            self._ax_spec.set_ylabel("Frecuencia (Hz)")
            # Eje Y a la derecha
            self._ax_spec.yaxis.tick_right()
            self._ax_spec.yaxis.set_label_position("right")
            # Configura ticks de frecuencia en Hz
            try:
                import numpy as _np

                y_freqs = _np.linspace(0.0, float(self.max_freq_hz), num=6)
                y_pos = (y_freqs / max(float(self.bin_hz), 1e-9)).astype(int)
                y_pos = _np.clip(y_pos, 0, image.shape[0] - 1)
                self._ax_spec.set_yticks(y_pos)
                self._ax_spec.set_yticklabels([f"{int(f)}" for f in y_freqs])
            except Exception:
                pass
        else:
            self._im_spec.set_data(image)
        # Escala de tiempo real: 0s a la derecha, máximo a la izquierda (recalcular por si cambia tamaño)
        try:
            import numpy as _np

            w = int(image.shape[1])
            max_span = w * float(self.seconds_per_col)
            xt_pos = _np.linspace(w - 1, 0, num=6)
            xt_lbl = [f"{t:.1f}" for t in _np.linspace(0.0, max_span, num=6)]
            self._ax_spec.set_xticks(xt_pos)
            self._ax_spec.set_xticklabels(xt_lbl)
        except Exception:
            pass
        assert self._fig is not None
        self._fig.canvas.draw_idle()
        plt.pause(0.001)


@dataclass
class TimeViewer:
    """Ventana de señal temporal sin procesar.

    Muestra la señal cruda (entrada) desplazándose de derecha a izquierda.
    """

    sample_rate: int
    span_seconds: float
    title: str = "Txori - Tiempo"
    _fig: Any | None = None
    _ax: Any | None = None
    _line: Any | None = None
    _buf: npt.NDArray[np.float64] | None = None
    _last_draw_t: float = 0.0

    def _ensure_backend(self) -> None:  # pragma: no cover
        global plt
        try:
            import importlib

            plt = importlib.import_module("matplotlib.pyplot")
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "La visualización temporal requiere matplotlib. Instala con 'pip install matplotlib'."
            ) from e
        if self._fig is None:
            import numpy as _np

            plt.ion()
            self._fig, self._ax = plt.subplots(1, 1)
            self._fig.canvas.manager.set_window_title(self.title)
            total = max(1, int(self.sample_rate * self.span_seconds))
            n_points = min(total, 5000)
            self._push_every = max(1, int(math.ceil(total / n_points)))
            self._buf = _np.zeros(n_points, dtype=_np.float64)
            x = _np.linspace(-self.span_seconds, 0.0, n_points)
            (self._line,) = self._ax.plot(x, self._buf, color="lime")
            self._ax.set_xlim(-self.span_seconds, 0.0)
            self._ax.set_ylim(-1.1, 1.1)
            self._ax.set_xlabel("Tiempo (s)")
            self._ax.set_ylabel("Amplitud (V)")
            self._ax.grid(True, alpha=0.2)

    def push_sample(
        self, sample: float
    ) -> None:  # pragma: no cover - depende de matplotlib
        self._ensure_backend()
        assert (
            self._buf is not None and self._line is not None and self._fig is not None
        )
        # Desplaza buffer y agrega nueva muestra cruda
        self._buf = np.roll(self._buf, -1)
        self._buf[-1] = float(sample)
        # Throttle de redibujado: ~40 FPS
        import time as _time

        now = _time.perf_counter()
        if now - self._last_draw_t >= 1.0 / 40.0:
            self._line.set_ydata(self._buf)
            self._fig.canvas.draw_idle()
            self._last_draw_t = now
            global plt
            plt.pause(0.001)
