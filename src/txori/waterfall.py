"""Cálculo y render del gráfico waterfall (espectrograma)."""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque

import numpy as np
from matplotlib import pyplot as plt


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

    def show(self, spec: np.ndarray, sample_rate: int, nfft: int) -> None:
        """Muestra el gráfico waterfall interactivo."""
        plt.figure(figsize=(10, 6))
        extent = (0.0, float(sample_rate) / 2.0, float(spec.shape[0]), 0.0)
        plt.imshow(
            spec,
            aspect="auto",
            origin="upper",
            extent=extent,
            cmap=self.cmap,
        )
        plt.colorbar(label="dBFS")
        plt.xlabel("Frecuencia [Hz]")
        plt.ylabel("Tiempo [frames]")
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

    def run(self, blocks_iter, sample_rate: int) -> None:  # noqa: ANN001
        """Ejecuta el render en vivo hasta interrupción con Ctrl+C."""
        if not (0 <= self.overlap < 1):
            raise ValueError("overlap debe estar en [0, 1)")
        step = int(self.nfft * (1 - self.overlap)) or 1
        win = np.hanning(self.nfft).astype(np.float32)
        eps = 1e-12
        buf = np.empty(0, dtype=np.float32)
        rows: deque[np.ndarray] = deque(maxlen=self.max_frames)

        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 6))
        bins = self.nfft // 2 + 1
        img = ax.imshow(
            np.zeros((1, bins), dtype=np.float32),
            aspect="auto",
            origin="upper",
            extent=(0.0, float(sample_rate) / 2.0, 1.0, 0.0),
            cmap=self.cmap,
        )
        plt.colorbar(img, label="dBFS")
        ax.set_xlabel("Frecuencia [Hz]")
        ax.set_ylabel("Tiempo [frames]")
        ax.set_title("Waterfall en vivo")
        plt.tight_layout()

        try:
            for block in blocks_iter:
                buf = np.concatenate((buf, block.astype(np.float32, copy=False)))
                updated = False
                while len(buf) >= self.nfft:
                    frame = buf[: self.nfft] * win
                    mag = np.abs(np.fft.rfft(frame, n=self.nfft))
                    row = 20.0 * np.log10(mag + eps)
                    rows.append(row.astype(np.float32, copy=False))
                    buf = buf[step:]
                    updated = True
                if updated and rows:
                    data = np.vstack(rows)
                    img.set_data(data)
                    img.set_extent((0.0, float(sample_rate) / 2.0, float(len(rows)), 0.0))
                    plt.pause(0.001)
        except KeyboardInterrupt:
            pass
        finally:
            plt.ioff()
            plt.show()
