"""Waterfall: espectrograma en tiempo real con animación derecha a izquierda."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import mlab
from matplotlib import mlab

from .sources import Source
from .cpu import Processor


class SpectrogramAnimator:
    """Administra el buffer y el dibujo del espectrograma en tiempo real."""

    def __init__(
        self,
        *,
        fs: int,
        nfft: int = 256,
        hop: int = 56,  # overlap = nfft - hop => nfft-56
        frames_per_update: int = 4,
        width_cols: int = 400,
        fft_window: str = "blackman",
        cmap: str = "ocean",
        pixels: int = 640,
    ) -> None:
        self.fs = int(fs)
        self.nfft = int(nfft)
        self.hop = int(hop)
        self.frames_per_update = int(frames_per_update)
        self.width_cols = int(width_cols)
        self._win = str(fft_window).lower()
        self._cmap = str(cmap) if cmap else "ocean"
        self.pixel_width = max(1, int(pixels))
        # Tamaño de buffer para cubrir width_cols columnas
        self._buf_len = self.nfft + (self.width_cols - 1) * self.hop
        self._buffer = np.zeros(self._buf_len, dtype=np.float32)

    def _window_fn(self, x: np.ndarray) -> np.ndarray:
        """Ventana seleccionable para mlab.specgram."""
        N = len(x)
        w = self._win
        if w in ("blackman", "blackmann"):
            return np.blackman(N)
        if w in ("hanning", "hann"):
            return np.hanning(N)
        if w == "hamming":
            return np.hamming(N)
        if w in ("rect", "rectangular", "boxcar", "none"):
            return np.ones(N, dtype=float)
        if w in ("blackmanharris", "blackman-harris", "bh4"):
            n = np.arange(N, dtype=float)
            a0, a1, a2, a3 = 0.35875, 0.48829, 0.14128, 0.01168
            return (
                a0
                - a1 * np.cos(2.0 * np.pi * n / (N - 1))
                + a2 * np.cos(4.0 * np.pi * n / (N - 1))
                - a3 * np.cos(6.0 * np.pi * n / (N - 1))
            )
        if w in ("flattop", "flat-top", "flat_top"):
            # 5-term flattop (Harris)
            n = np.arange(N, dtype=float)
            a0, a1, a2, a3, a4 = 0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368
            return (
                a0
                - a1 * np.cos(2.0 * np.pi * n / (N - 1))
                + a2 * np.cos(4.0 * np.pi * n / (N - 1))
                - a3 * np.cos(6.0 * np.pi * n / (N - 1))
                + a4 * np.cos(8.0 * np.pi * n / (N - 1))
            )
        # Fallback a Blackman
        return np.blackman(N)

    def _push(self, x: np.ndarray) -> None:
        if x.size == 0:
            return
        x = x.astype(np.float32, copy=False)
        if x.size >= self._buf_len:
            self._buffer = x[-self._buf_len :]
            return
        keep = self._buf_len - x.size
        self._buffer = np.concatenate((self._buffer[-keep:], x))

    def compute_spec(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calcula el espectrograma actual usando matplotlib.specgram."""
        fig, ax = plt.subplots()
        try:
            Pxx, freqs, bins = mlab.specgram(
                x=self._buffer,
                NFFT=self.nfft,
                Fs=self.fs,
                noverlap=self.nfft - self.hop,
                window=self._window_fn,
            )
        finally:
            plt.close(fig)
        return Pxx, freqs, bins

    def run(self, source: Source, cpu: Processor) -> None:
        """Inicia la animación en tiempo real consumiendo de la fuente y CPU."""
        fig, ax = plt.subplots()
        # Ajustar ancho en píxeles manteniendo alto
        try:
            dpi = float(fig.get_dpi())
            w, h = fig.get_size_inches()
            fig.set_size_inches(self.pixel_width / max(dpi, 1.0), h, forward=True)
        except Exception:
            pass
        manager = getattr(fig.canvas, "manager", None)
        if manager is not None and hasattr(manager, "set_window_title"):
            manager.set_window_title("Txori Waterfall")

        def _update(_frame: int):
            need = self.frames_per_update * self.hop
            x = source.read(need)
            x = cpu.process(x)
            self._push(x)
            ax.cla()
            Pxx, freqs, bins = mlab.specgram(
                x=self._buffer,
                NFFT=self.nfft,
                Fs=self.fs,
                noverlap=self.nfft - self.hop,
                window=self._window_fn,
            )
            Z = 10.0 * np.log10(Pxx + 1e-12)
            extent = (0.0, float(bins[-1]) if bins.size else 0.0, float(freqs[0]), float(freqs[-1]))
            ax.imshow(Z, origin="lower", aspect="auto", extent=extent, cmap=self._cmap)
            ax.set_title(f"Sample rate: {self.fs} Hz")
            ax.set_xlabel("Tiempo [s] (derecha→izquierda)")
            ax.set_ylabel("Frecuencia [Hz]")
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")

        interval_ms = int(1000 * (self.frames_per_update * self.hop) / float(self.fs))
        anim = FuncAnimation(
            fig,
            _update,
            interval=max(1, interval_ms),
            cache_frame_data=False,
        )
        plt.show()
        del anim  # mantener referencia hasta que show() regrese
