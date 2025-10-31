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
        fft_ema: float | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> None:
        self.fs = int(fs)
        self.nfft = int(nfft)
        self.hop = int(hop)
        self.frames_per_update = int(frames_per_update)
        self.width_cols = int(width_cols)
        self._win = str(fft_window).lower()
        self._cmap = str(cmap) if cmap else "ocean"
        self.pixel_width = max(1, int(pixels))
        self._ema_alpha = float(fft_ema) if fft_ema is not None else None
        self._ema_state: np.ndarray | None = None
        self._vmin = vmin
        self._vmax = vmax
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

    def run(self, source: Source, cpu: Processor, *, spkr: bool = False, time_plot: bool = False) -> None:
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

        # Configuración de gráfico de tiempo opcional
        time_fig = None
        time_line = None
        last_samples = None
        time_len = self.frames_per_update * self.hop
        if time_plot:
            time_fig, time_ax = plt.subplots()
            time_ax.set_title("Time plot (fuente)")
            time_ax.set_ylim([-1.0, 1.0])
            time_ax.set_xlim([0, max(1, time_len)])
            (time_line,) = time_ax.plot(np.zeros(max(1, time_len), dtype=np.float32))
            last_samples = np.zeros(max(1, time_len), dtype=np.float32)

        stream = None
        if spkr and sd is not None:
            try:
                stream = sd.OutputStream(samplerate=self.fs, channels=1, dtype="float32")
                stream.start()
            except Exception:
                stream = None

        def _update(_frame: int):
            nonlocal last_samples
            need = self.frames_per_update * self.hop
            x = source.read(need)
            if time_plot and last_samples is not None:
                n = len(last_samples)
                if x.size < n:
                    y = np.zeros(n, dtype=np.float32)
                    y[: x.size] = x
                else:
                    y = x[:n]
                last_samples = y
            if stream is not None and x.size:
                try:
                    stream.write(x.reshape(-1, 1))
                except Exception:
                    pass
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
            # Suavizado EMA (en potencia) si está habilitado
            if self._ema_alpha is not None and 0.0 < self._ema_alpha < 1.0:
                if self._ema_state is None:
                    self._ema_state = Pxx
                else:
                    self._ema_state = (
                        self._ema_alpha * self._ema_state + (1.0 - self._ema_alpha) * Pxx
                    )
                Pxx_plot = self._ema_state
            else:
                Pxx_plot = Pxx
            Z = 10.0 * np.log10(Pxx_plot + 1e-12)
            extent = (0.0, float(bins[-1]) if bins.size else 0.0, float(freqs[0]), float(freqs[-1]))
            ax.imshow(
                Z,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap=self._cmap,
                vmin=self._vmin,
                vmax=self._vmax,
            )
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
        if time_plot and time_fig is not None and time_line is not None:
            def _update_time(_frame: int):
                if last_samples is not None:
                    time_line.set_ydata(last_samples)
                return (time_line,)
            anim_time = FuncAnimation(
                time_fig,
                _update_time,
                interval=max(1, interval_ms),
                blit=True,
                cache_frame_data=False,
            )
            # Mantener referencia para evitar GC prematuro
            try:
                time_fig._txori_anim = anim_time  # type: ignore[attr-defined]
            except Exception:
                pass
        try:
            plt.show()
        finally:
            if stream is not None:
                try:
                    stream.stop(); stream.close()
                except Exception:
                    pass
        del anim  # mantener referencia hasta que show() regrese
