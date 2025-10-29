"""Cálculo y render del gráfico waterfall (espectrograma)."""
from __future__ import annotations

from dataclasses import dataclass
import time
from collections import deque

import numpy as np
from matplotlib import pyplot as plt

class TimePlotLive:
    """Visualizador simple de señal en tiempo."""

    def __init__(self) -> None:
        self.fig, self.ax = plt.subplots(figsize=(10, 3))
        self.line, = self.ax.plot(np.zeros(1, dtype=np.float32))
        self.ax.set_ylim([-1.0, 1.0])
        self.ax.set_xlim([0, 1])
        self.ax.invert_xaxis()  # Tiempo derecha→izquierda, igual que waterfall
        self.ax.set_title("Timeplot en vivo")
        self.ax.set_xlabel("Muestras")
        self.ax.set_ylabel("Amplitud")

    def update(self, block: np.ndarray) -> None:
        y = np.asarray(block, dtype=np.float32).reshape(-1)
        # Muestra más reciente a la derecha (x=0), más antigua a la izquierda (x=max)
        x = np.arange(y.size - 1, -1, -1, dtype=np.float32)
        self.line.set_data(x, y)
        self.ax.set_xlim([0, max(1, y.size)])

    def redraw(self) -> None:
        self.fig.canvas.draw_idle()


@dataclass(slots=True)
class WaterfallComputer:
    """Calcula el espectrograma en dBFS a partir de una señal mono."""

    nfft: int = 1024
    overlap: float = 0.5  # en [0, 1)

    def compute(self, signal: np.ndarray) -> np.ndarray:
        """Devuelve matriz (frames x (nfft/2+1)) con magnitudes en dB.

        Args:
            signal: Señal mono ``float32``.

        Raises:
            ValueError: Si la señal es demasiado corta o parámetros inválidos.
        """
        if not (0 <= self.overlap < 1):
            raise ValueError("overlap debe estar en [0, 1)")
        x = np.asarray(signal, dtype=np.float32)
        if x.ndim != 1:
            raise ValueError("La señal debe ser mono (1D)")
        step = int(self.nfft * (1 - self.overlap)) or 1
        if len(x) < self.nfft:
            raise ValueError("Señal demasiado corta para el nfft indicado")
        n_frames = 1 + (len(x) - self.nfft) // step
        win = np.hanning(self.nfft).astype(np.float32)
        spec = np.empty((n_frames, self.nfft // 2 + 1), dtype=np.float32)
        for i in range(n_frames):
            start = i * step
            frame = x[start : start + self.nfft]
            frame = frame * win
            fft = np.fft.rfft(frame, n=self.nfft)
            mag = np.abs(fft)
            spec[i] = 20.0 * np.log10(mag + 1e-12)
        return spec


@dataclass(slots=True)
class WaterfallRenderer:
    """Renderiza un espectrograma usando Matplotlib."""

    cmap: str = "viridis"

    def show(self, spec: np.ndarray, sample_rate: int, nfft: int, overlap: float) -> None:
        """Muestra el gráfico waterfall con F vertical y tiempo horizontal (R→L) en segundos."""
        plt.figure(figsize=(10, 6))
        data = spec.T  # filas=frecuencia, columnas=tiempo
        n_frames = data.shape[1]
        hop = int(nfft * (1 - overlap)) or 1
        t_max = float(max(0, n_frames - 1) * hop) / float(sample_rate)
        extent = (t_max, 0.0, 0.0, float(sample_rate) / 2.0)  # tiempo (s) derecha→izquierda
        plt.imshow(
            data,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap=self.cmap,
        )
        plt.colorbar(label="dBFS")
        plt.xlabel("Tiempo [s]")
        plt.ylabel("Frecuencia [Hz]")
        plt.title("Waterfall (Espectrograma)")
        plt.tight_layout()
        plt.show()


@dataclass(slots=True)
class WaterfallLive:
    """Render en vivo con buffer rodante y actualización continua."""

    nfft: int = 1024
    overlap: float = 0.5
    cmap: str = "viridis"
    max_frames: int = 400
    enable_timeplot: bool = False

    def run(self, blocks_iter, sample_rate: int) -> None:  # noqa: ANN001
        """Ejecuta render en vivo con ventana de tiempo fija (buffer rodante) hasta Ctrl+C.

        Args:
            blocks_iter: Iterador de bloques mono float32.
            sample_rate: Frecuencia de muestreo (Hz).
        """
        if not (0 <= self.overlap < 1):
            raise ValueError("overlap debe estar en [0, 1)")
        step = int(self.nfft * (1 - self.overlap)) or 1
        win = np.hanning(self.nfft).astype(np.float32)
        eps = 1e-12
        buf = np.empty(0, dtype=np.float32)

        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 6))
        bins = self.nfft // 2 + 1
        img_data = np.zeros((bins, self.max_frames), dtype=np.float32)
        # Ventana de tiempo constante basada en hop=step
        window_s = float((self.max_frames - 1) * step) / float(sample_rate)
        img = ax.imshow(
            img_data,
            aspect="auto",
            origin="lower",
            extent=(window_s, 0.0, 0.0, float(sample_rate) / 2.0),  # ventana de tiempo constante
            cmap=self.cmap,
        )
        plt.colorbar(img, label="dBFS")
        tplot = TimePlotLive() if self.enable_timeplot else None
        ax.set_xlabel("Tiempo [s]")
        ax.set_ylabel("Frecuencia [Hz]")
        ax.set_title("Waterfall en vivo")
        plt.tight_layout()

        try:
            for block in blocks_iter:
                if self.enable_timeplot and 'tplot' in locals() and tplot is not None:
                    try:
                        tplot.update(block)
                    except Exception:
                        pass
                buf = np.concatenate((buf, block.astype(np.float32, copy=False)))
                updated = False
                while len(buf) >= self.nfft:
                    frame = buf[: self.nfft] * win
                    mag = np.abs(np.fft.rfft(frame, n=self.nfft))
                    row = (20.0 * np.log10(mag + eps)).astype(np.float32, copy=False)
                    # Desplaza a la izquierda y coloca la fila nueva a la derecha
                    img_data[:, :-1] = img_data[:, 1:]
                    img_data[:, -1] = row
                    buf = buf[step:]
                    updated = True
                if updated:
                    img.set_data(img_data)
                    if self.enable_timeplot and 'tplot' in locals() and tplot is not None:
                        tplot.redraw()
                    plt.pause(0.001)
        except KeyboardInterrupt:
            pass
        finally:
            plt.ioff()
            plt.show()
