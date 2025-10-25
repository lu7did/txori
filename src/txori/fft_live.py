"""Live FFT spectrogram for --dsp using librosa.

Computes STFT over a rolling buffer and updates the view at ~30 FPS.
Imports are lazy to avoid hard deps in CI.
"""
# ruff: noqa: I001
# fmt: off
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import time
from typing import Any

import numpy as np

plt: Any = None
librosa: Any | None = None


@dataclass
class DSPLibrosaSpectrogram:
    """Espectrómetro en vivo con STFT (librosa) sobre ventana deslizante."""

    sr: int
    span_seconds: float = 30.0
    fps: float = 30.0
    n_fft: int = 2048
    hop_length: int | None = None
    title: str = "Txori - Espectrograma (DSP)"
    y_max_hz: float = 3000.0
    ext_ax: Any | None = None
    device_name: str | None = None
    decim_factor: int = 1
    user_text: str | None = None
    device_sr: int | None = None
    cw_mode: bool = False
    cw_center_hz: float = 600.0
    cw_bw_hz: float = 20.0
    cw_extra_centers: list[float] | None = None


    _fig: Any | None = None
    _ax: Any | None = None
    _im: Any | None = None
    _cb: Any | None = None
    _buf: deque[float] = field(default_factory=deque, init=False)
    _last_draw: float = 0.0
    _img_db: np.ndarray | None = None
    _ncols: int = 0
    _txt_info: Any | None = None
    _dev_samples: int = 0
    _ref_amp: float = 1e-9

    def _ensure_backend(self) -> None:  # pragma: no cover
        global plt, librosa
        if plt is None:
            import importlib

            plt = importlib.import_module("matplotlib.pyplot")
        if librosa is None:
            librosa = None
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
        # Contar muestras de dispositivo estimadas (decimación)
        try:
            self._dev_samples += int(max(1, self.decim_factor))
        except Exception:
            self._dev_samples += 1
        now = time.perf_counter()
        if now - self._last_draw >= 1.0 / max(self.fps, 1e-3):
            self._last_draw = now
            self._draw()

    def _draw(self) -> None:  # noqa: C901, pragma: no cover
        assert self._ax is not None
        if len(self._buf) < max(self.n_fft, 64):
            return
        x = np.asarray(self._buf, dtype=np.float32)
        # Construir nueva columna a partir de la última ventana n_fft
        xw = x[-self.n_fft:]
        if xw.size < 2:
            return
        # Ventana: usar Kaiser beta=14 en CW para >-90 dB lóbulos laterales
        win = (np.kaiser(xw.size, 14.0) if self.cw_mode else np.hanning(xw.size)).astype(np.float32)
        X = np.fft.rfft(xw * win, n=self.n_fft)
        A = np.abs(X)
        # En CW, confinar tonos a ±bw alrededor de los centros
        if self.cw_mode:
            freqs_all = np.fft.rfftfreq(self.n_fft, d=1.0 / float(self.sr))
            centers = [float(self.cw_center_hz)] + (list(self.cw_extra_centers) if self.cw_extra_centers else [])
            bin_w = float(self.sr) / float(self.n_fft)
            half_bins = max(1, int(np.ceil(float(self.cw_bw_hz) / max(bin_w, 1e-9))))
            mask = np.zeros_like(freqs_all, dtype=bool)
            for c in centers:
                ic = int(np.argmin(np.abs(freqs_all - float(c))))
                lo = max(0, ic - half_bins)
                hi = min(mask.size - 1, ic + half_bins)
                mask[lo : hi + 1] = True
            if A.size == mask.size:
                A = A * mask.astype(A.dtype) + (1e-12 * (~mask).astype(A.dtype))
        # Normalización con referencia dinámica de decaimiento
        amax = float(np.max(A)) if A.size else 0.0
        self._ref_amp = max(self._ref_amp * 0.995, amax, 1e-9)
        ref = self._ref_amp
        col_db = 20.0 * np.log10(np.maximum(A, 1e-12) / ref)
        # Limitar frecuencia máxima visible
        freqs = np.fft.rfftfreq(self.n_fft, d=1.0 / float(self.sr))
        k = int(np.searchsorted(freqs, float(self.y_max_hz), side="right") - 1)
        if k >= 0:
            col_db = col_db[: k + 1]
        # Inicializar imagen fija en tiempo
        eff_hop = int(self.hop_length or max(1, int(self.sr // max(1, int(self.fps)))))
        if self._img_db is None:
            self._ncols = max(1, int(round(float(self.span_seconds) * (float(self.sr) / float(eff_hop)))))
            self._img_db = np.full((col_db.size, self._ncols), -120.0, dtype=np.float32)
            self._im = self._ax.imshow(
                self._img_db,
                origin="lower",
                aspect="auto",
                extent=[0.0, float(self.span_seconds), 0.0, float(self.y_max_hz)],
                vmin=-120.0,
                vmax=0.0,
            )
            self._ax.set_ylim(0.0, self.y_max_hz)
            self._ax.set_xlabel("Tiempo (s)")
            self._ax.set_ylabel("Frecuencia (Hz)")
            if self._fig is not None:
                self._cb = self._fig.colorbar(self._im, ax=self._ax, format="%+2.0f dB")
        # Desplazar a la derecha y colocar nueva columna en la izquierda
        assert self._img_db is not None and self._im is not None
        self._img_db = np.roll(self._img_db, 1, axis=1)
        self._img_db[:, 0] = col_db[: self._img_db.shape[0]]
        self._im.set_data(self._img_db)
        # Mantener eje temporal constante
        try:
            self._ax.set_xlim(0.0, float(self.span_seconds))
        except Exception:
            pass
        # Texto informativo: texto usuario | dispositivo | fs | n_fft
        fs_show = int(self.device_sr) if self.device_sr is not None else int(self.sr)
        info = f"{(self.user_text or '').strip()} | {(self.device_name or '').strip()} | fs={fs_show} Hz | stft_in={int(self.n_fft)}"
        self._ax.set_title(info, fontsize=9)
        # Redibujar
        if self._fig is not None:
            self._fig.canvas.draw_idle()
        try:
            import matplotlib.pyplot as _plt
            _plt.pause(0.001)
        except Exception:
            pass
