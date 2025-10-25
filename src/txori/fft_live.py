"""Live FFT spectrogram for --dsp using librosa.

Computes STFT over a rolling buffer and updates the view at ~30 FPS.
Imports are lazy to avoid hard deps in CI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import time
from collections import deque

import numpy as np
import numpy.typing as npt

plt: Any = None
librosa = None  # type: ignore[assignment]
libdisp = None  # type: ignore[assignment]


@dataclass
class DSPLibrosaSpectrogram:
    sr: int
    span_seconds: float = 30.0
    fps: float = 30.0
    n_fft: int = 2048
    hop_length: int | None = None
    title: str = "Txori - Espectrograma (DSP)"
    y_max_hz: float = 3000.0
    ext_ax: Any | None = None

    _fig: Any | None = None
    _ax: Any | None = None
    _im: Any | None = None
    _cb: Any | None = None
    _buf: deque[float] = field(default_factory=deque, init=False)
    _last_draw: float = 0.0

    def _ensure_backend(self) -> None:  # pragma: no cover
        global plt, librosa, libdisp
        if plt is None:
            import importlib

            plt = importlib.import_module("matplotlib.pyplot")
        if librosa is None:
            import importlib

            librosa = importlib.import_module("librosa")
            libdisp = importlib.import_module("librosa.display")
        # Usar eje externo si fue provisto
        if self._ax is None and getattr(self, "ext_ax", None) is not None:
            self._ax = self.ext_ax
            self._fig = getattr(self.ext_ax, "figure", None)
            if self._ax is not None:
                self._ax.set_ylim(0.0, self.y_max_hz)
                self._ax.set_xlabel("Tiempo (s)")
                self._ax.set_ylabel("Frecuencia (Hz)")
            return
        if self._fig is None:
            plt.ion()
            self._fig, self._ax = plt.subplots(1, 1)
            try:
                self._fig.set_size_inches(6.0, 4.0, forward=True)
            except Exception:
                pass
            self._fig.canvas.manager.set_window_title(self.title)
            assert self._ax is not None
            self._ax.set_ylim(0.0, self.y_max_hz)
            self._ax.set_xlabel("Tiempo (s)")
            self._ax.set_ylabel("Frecuencia (Hz)")
            plt.show(block=False)
            self._fig.canvas.draw_idle()
            plt.pause(0.01)

    def show(self) -> None:  # pragma: no cover
        self._ensure_backend()

    @property
    def _buf_len(self) -> int:
        return int(max(1, self.sr * self.span_seconds))

    def push_sample(self, x: float) -> None:  # pragma: no cover
        self._ensure_backend()
        self._buf.append(float(x))
        while len(self._buf) > self._buf_len:
            self._buf.popleft()
        now = time.perf_counter()
        if now - self._last_draw >= 1.0 / max(self.fps, 1e-3):
            self._last_draw = now
            self._draw()

    def _draw(self) -> None:  # pragma: no cover
        assert self._ax is not None and self._fig is not None
        if len(self._buf) < max(self.n_fft, 64):
            return
        x = np.asarray(self._buf, dtype=np.float32)
        # Plantilla estilo test2.py: STFT sobre todo el buffer
        S = librosa.stft(x, n_fft=self.n_fft, hop_length=self.hop_length)
        A = np.abs(S)
        D_db = librosa.amplitude_to_db(A, ref=np.max)
        # Limitar frecuencia máxima visible
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        k = int(np.searchsorted(freqs, float(self.y_max_hz), side="right") - 1)
        if k >= 0:
            D_db = D_db[: k + 1, :]
        self._ax.clear()
        im = libdisp.specshow(D_db, sr=self.sr, x_axis="time", y_axis="hz", ax=self._ax)
        self._ax.set_ylim(0.0, self.y_max_hz)
        self._ax.set_xlabel("Tiempo (s)")
        self._ax.set_ylabel("Frecuencia (Hz)")
        # Mantener eje temporal constante de 0 a span_seconds
        try:
            self._ax.set_xlim(0.0, float(self.span_seconds))
        except Exception:
            pass
        if self._fig is not None:
            if self._cb is None:
                self._cb = self._fig.colorbar(im, ax=self._ax, format="%+2.0f dB")
            else:
                self._cb.update_normal(im)
            self._fig.canvas.draw_idle()
        else:
            # Redibujar canvas del eje externo si existe
            try:
                self._ax.figure.canvas.draw_idle()
            except Exception:
                pass
        try:
            import matplotlib.pyplot as _plt
            _plt.pause(0.001)
        except Exception:
            pass
