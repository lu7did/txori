from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class LiveViewer:
    title: str = "Txori - Espectrograma"
    max_freq_hz: float = 3000.0
    bin_hz: float = 3.0
    seconds_per_col: float = 0.001
    title_text: Optional[str] = None
    device_text: Optional[str] = None
    _fig: Optional[object] = None
    _ax: Optional[object] = None
    _im: Optional[object] = None

    def _ensure_backend(self) -> None:
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
            self._fig, self._ax = plt.subplots()
            self._fig.canvas.manager.set_window_title(self.title)  # type: ignore[attr-defined]
            # Textos externos
            if self.title_text:
                self._fig.text(0.01, 0.98, self.title_text, ha="left", va="top")
            if self.device_text:
                self._fig.text(0.99, 0.98, self.device_text, ha="right", va="top")

    def update(self, image: np.ndarray) -> None:
        self._ensure_backend()
        assert self._ax is not None
        global plt  # type: ignore[name-defined]
        if self._im is None:
            self._im = self._ax.imshow(image, origin="lower")
            self._ax.set_xlabel("Tiempo (seg)")
            self._ax.set_ylabel("Frecuencia (Hz)")
            # Eje Y a la derecha
            self._ax.yaxis.tick_right()
            self._ax.yaxis.set_label_position("right")
            # Configura ticks de frecuencia en Hz
            try:
                import numpy as _np
                y_freqs = _np.linspace(0.0, float(self.max_freq_hz), num=6)
                y_pos = (y_freqs / max(float(self.bin_hz), 1e-9)).astype(int)
                y_pos = _np.clip(y_pos, 0, image.shape[0] - 1)
                self._ax.set_yticks(y_pos)
                self._ax.set_yticklabels([f"{int(f)}" for f in y_freqs])
            except Exception:
                pass
            # Escala de tiempo real: 0s a la derecha, máximo a la izquierda
            try:
                import numpy as _np
                w = image.shape[1]
                max_span = w * float(self.seconds_per_col)
                xt_pos = _np.linspace(w - 1, 0, num=6)
                xt_lbl = [f"{t:.1f}" for t in _np.linspace(0.0, max_span, num=6)]
                self._ax.set_xticks(xt_pos)
                self._ax.set_xticklabels(xt_lbl)
            except Exception:
                pass
            self._fig.tight_layout()  # type: ignore[union-attr]
        else:
            self._im.set_data(image)
        self._fig.canvas.draw_idle()  # type: ignore[union-attr]
        plt.pause(0.001)
