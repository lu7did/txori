from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LiveViewer:
    """Visor en vivo con espectrograma y traza de intensidad superior."""

    title: str = "Txori - Espectrograma"
    max_freq_hz: float = 3000.0
    bin_hz: float = 3.0
    seconds_per_col: float = 0.001
    title_text: str | None = None
    device_text: str | None = None
    _fig: object | None = None
    _ax_spec: object | None = None
    _ax_top: object | None = None
    _im_spec: object | None = None
    _im_top: object | None = None
    _trace_img: np.ndarray | None = None
    _level_max: float = 1e-6

    def _ensure_backend(self) -> None:  # pragma: no cover
        # Importa perezosamente matplotlib para no requerirlo en CI/tests
        global plt
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception as e:  # pragma: no cover - dependiente del entorno
            raise RuntimeError(
                "La visualización en vivo requiere matplotlib. Instala con 'pip install matplotlib'."
            ) from e
        if self._fig is None:
            plt.ion()
            self._fig, (self._ax_top, self._ax_spec) = plt.subplots(
                2, 1, sharex=True, gridspec_kw={"height_ratios": [1, 8]}
            )
            self._fig.canvas.manager.set_window_title(self.title)  # type: ignore[attr-defined]
            # Ajusta fondo negro para traza superior
            self._ax_top.set_facecolor("black")
            self._ax_top.set_yticks([])
            self._ax_top.set_ylabel("Nivel", color="lime")
            for spine in self._ax_top.spines.values():
                spine.set_color("white")
            # Textos externos, más alejados del gráfico
            if self.title_text:
                self._fig.text(0.01, 0.995, self.title_text, ha="left", va="top")
            if self.device_text:
                self._fig.text(0.99, 0.995, self.device_text, ha="right", va="top")

    def update(
        self, image: np.ndarray, level: float | None = None
    ) -> None:  # pragma: no cover - depende de matplotlib
        self._ensure_backend()
        assert self._ax_spec is not None and self._ax_top is not None
        global plt  # type: ignore[name-defined]
        # Espectrograma (panel inferior)
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
        # Traza de intensidad (panel superior)
        trace_h = 32
        w = int(image.shape[1])
        if self._trace_img is None or self._trace_img.shape[1] != w:
            self._trace_img = np.zeros((trace_h, w, 3), dtype=np.uint8)
            self._ax_top.set_xlim(0, w)
            self._ax_top.set_ylim(0, trace_h)
            if self._im_top is None:
                self._im_top = self._ax_top.imshow(
                    self._trace_img, origin="lower", aspect="auto"
                )
            else:
                self._im_top.set_data(self._trace_img)
        # Desplaza y dibuja la muestra de nivel
        assert self._trace_img is not None
        self._trace_img = np.roll(self._trace_img, shift=-1, axis=1)
        self._trace_img[:, -1, :] = 0  # negro
        if level is not None:
            # Normalización dinámica con leve decaimiento (nivel instantáneo tras pasabajos)
            self._level_max = max(float(level), self._level_max * 0.995)
            denom = max(self._level_max, 1e-9)
            t = max(0.0, min(1.0, float(level) / denom))
            y = int(round(t * (trace_h - 1)))
            self._trace_img[y, -1] = (0, 255, 0)  # lime
        if self._im_top is not None:
            self._im_top.set_data(self._trace_img)
        # Escala de tiempo real: 0s a la derecha, máximo a la izquierda (recalcular por si cambia tamaño)
        try:
            import numpy as _np

            max_span = w * float(self.seconds_per_col)
            xt_pos = _np.linspace(w - 1, 0, num=6)
            xt_lbl = [f"{t:.1f}" for t in _np.linspace(0.0, max_span, num=6)]
            self._ax_spec.set_xticks(xt_pos)
            self._ax_spec.set_xticklabels(xt_lbl)
        except Exception:
            pass
        self._fig.canvas.draw_idle()  # type: ignore[union-attr]
        plt.pause(0.001)
