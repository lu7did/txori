"""Waterfall: espectrograma en tiempo real con animación derecha a izquierda."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import mlab
from matplotlib import mlab
import threading
import queue
import time

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

    def run(self, source: Source, cpu: Processor, *, spkr: bool = False, time_plot: bool = False, time_scale: float = 1.0) -> None:
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
        # Escala de tiempo: factor (0<time_scale<=1) para reducir ventana temporal del time plot
        time_len = int(self.frames_per_update * self.hop)
        if time_plot:
            try:
                scale = float(time_scale)
            except Exception:
                scale = 1.0
            if not (0.0 < scale <= 1.0):
                scale = 1.0
            time_len = max(1, int(time_len * scale))
            time_fig, time_ax = plt.subplots()
            time_ax.set_title("Time plot (fuente)")
            time_ax.set_ylim([-1.0, 1.0])
            time_ax.set_xlim([0, time_len])
            (time_line,) = time_ax.plot(np.zeros(time_len, dtype=np.float32))
            last_samples = np.zeros(time_len, dtype=np.float32)

        stream = None
        _to_out = None  # función de conversión y resampleo para salida de altavoz
        spkr_q = None
        spkr_run = False
        spkr_thr = None
        spkr_fs = int(self.fs)
        _spkr_buf = np.zeros(0, dtype=np.float32)
        _spkr_t = 0.0
        if spkr and sd is not None:
            def _make_out_stream(target_fs: int, cb):
                s = sd.OutputStream(samplerate=target_fs, channels=1, dtype="float32", callback=cb, blocksize=0)
                s.start()
                return s
            # callback de audio que consume de la cola
            spkr_q = queue.Queue(maxsize=16)
            _cb_buf = np.zeros(0, dtype=np.float32)
            def _cb(outdata, frames, time_info, status):  # noqa: D401
                nonlocal _cb_buf
                if status:
                    pass
                if _cb_buf.size < frames:
                    try:
                        while _cb_buf.size < frames:
                            _cb_buf = np.concatenate((_cb_buf, spkr_q.get_nowait()))
                    except queue.Empty:
                        pass
                if _cb_buf.size >= frames:
                    outdata[:, 0] = _cb_buf[:frames]
                    _cb_buf = _cb_buf[frames:]
                else:
                    outdata.fill(0.0)
            try:
                # Preferir Fs del procesado; si no es soportado, caer a 48000
                stream = _make_out_stream(spkr_fs, _cb)
            except Exception:
                try:
                    spkr_fs = 48000
                    stream = _make_out_stream(spkr_fs, _cb)
                except Exception:
                    stream = None
            if stream is not None:
                def _to_out(a: np.ndarray) -> np.ndarray:
                    nonlocal _spkr_buf, _spkr_t
                    x = a.astype(np.float32)
                    if spkr_fs == self.fs:
                        return x
                    _spkr_buf = np.concatenate((_spkr_buf, x))
                    step = self.fs / float(spkr_fs)
                    outs = []
                    t = _spkr_t
                    while t + 1.0 < _spkr_buf.size:
                        i = int(t)
                        frac = t - i
                        v = (1.0 - frac) * _spkr_buf[i] + frac * _spkr_buf[i + 1]
                        outs.append(v)
                        t += step
                    drop = int(t)
                    if drop > 0:
                        _spkr_buf = _spkr_buf[drop:]
                        t -= drop
                    _spkr_t = t
                    return np.asarray(outs, dtype=np.float32)

        # Productor en hilo separado para desacoplar lectura de la fuente del render
        prod_run = True
        def _produce():
            nonlocal last_samples
            chunk = max(1, self.hop)
            sr_in = int(getattr(source, "sample_rate", self.fs))
            while prod_run:
                t0 = time.monotonic()
                x = source.read(chunk)
                if x.size:
                    # time plot: últimas muestras de la fuente (independiente del waterfall)
                    if time_plot and last_samples is not None:
                        n = len(last_samples)
                        if x.size < n:
                            y = np.zeros(n, dtype=np.float32); y[: x.size] = x
                        else:
                            y = x[-n:]
                        last_samples = y
                    # procesar para waterfall y speaker (mismas muestras post-CPU)
                    x_proc = x
                    try:
                        x_proc = cpu.process(x)
                    except Exception:
                        x_proc = x
                    # waterfall: empujar procesado
                    try:
                        if x_proc.size:
                            self._push(x_proc)
                    except Exception:
                        pass
                    # speaker: cola asíncrona con mismas muestras que el waterfall
                    if stream is not None and spkr_q is not None and x_proc.size:
                        try:
                            y_sp = _to_out(x_proc) if _to_out is not None else x_proc.astype(np.float32)
                            spkr_q.put_nowait(y_sp.reshape(-1, 1))
                        except Exception:
                            pass
                # Ritmo en tiempo real
                dt = time.monotonic() - t0
                wait = max(0.0, (x.size / float(sr_in)) - dt) if x.size else (chunk / float(sr_in))
                time.sleep(min(0.1, wait))
        prod_thr = threading.Thread(target=_produce, name="txori-producer")
        prod_thr.start()

        def _update(_frame: int):
            # Solo render: el productor alimenta buffers
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
            # Detener hilos y liberar recursos limpiamente en Ctrl+C o cierre de ventana
            try:
                prod_run = False
                try:
                    prod_thr.join(timeout=0.5)
                except Exception:
                    pass
            except Exception:
                pass
            if stream is not None:
                try:
                    spkr_run = False
                    try:
                        spkr_thr.join(timeout=0.5)
                    except Exception:
                        pass
                except Exception:
                    pass
                try:
                    stream.stop(); stream.close()
                except Exception:
                    pass
        del anim  # mantener referencia hasta que show() regrese
