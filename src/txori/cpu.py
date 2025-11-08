"""CPU: módulos procesadores de señal."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np




class Processor(ABC):
    """Interfaz de procesador de muestras."""

    @abstractmethod
    def process(self, x: np.ndarray) -> np.ndarray:
        """Procesa y devuelve las muestras."""




class NoOpProcessor(Processor):
    """Procesador por defecto: no modifica la señal."""

    def process(self, x: np.ndarray) -> np.ndarray:  # noqa: D401
        """Devuelve x sin modificaciones."""
        return x




class LpfProcessor(Processor):
    """Filtro pasabajos + remuestreo a 2*fc cuando fs_in>4000; bypass si fs_in<=4000."""

    def __init__(self, fs_in: int, cutoff_hz: float = 2000.0) -> None:
        """Inicializa LPF y calcula Fs objetivo=2*fc si Fs_in>4000."""
        self.fs_in = int(fs_in)
        self.cutoff = float(cutoff_hz)
        self.fs_out = int(2 * self.cutoff) if self.fs_in > 4000 else self.fs_in
        self._bypass = self.fs_in <= 4000
        # Diseño FIR ventana Hamming para LPF
        num_taps = 63
        n = np.arange(num_taps) - (num_taps - 1) / 2.0
        fc = min(self.cutoff, 0.49 * (self.fs_in / 2.0))
        sinc = np.sinc(2.0 * fc / self.fs_in * n)
        window = np.hamming(num_taps)
        h = sinc * window
        h /= np.sum(h)
        self._h = h.astype(np.float32)
        self._xprev = np.zeros(num_taps - 1, dtype=np.float32)
        self._ybuf = np.zeros(0, dtype=np.float32)
        self._t = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        if self._bypass:
            return x
        xi = x.astype(np.float32, copy=False)
        inp = np.concatenate((self._xprev, xi))
        y = np.convolve(inp, self._h, mode="valid").astype(np.float32)
        self._xprev = inp[-(self._h.size - 1) :]
        # Remuestreo lineal a fs_out
        if self.fs_out >= self.fs_in:
            return y
        self._ybuf = np.concatenate((self._ybuf, y))
        step = self.fs_in / float(self.fs_out)
        outs = []
        t = self._t
        while t + 1.0 < self._ybuf.size:
            i = int(t)
            frac = t - i
            v = (1.0 - frac) * self._ybuf[i] + frac * self._ybuf[i + 1]
            outs.append(v)
            t += step
        drop = int(t)
        if drop > 0:
            self._ybuf = self._ybuf[drop:]
            t -= drop
        self._t = t
        return np.asarray(outs, dtype=np.float32)




class SkimmerProcessor(Processor):
    """Skimmer: si Fs<=8000 pasa directo; si Fs>8000 aplica LPF 4 kHz más agresivo y remuestrea a 8 kHz."""

    def __init__(self, fs_in: int) -> None:
        self.fs_in = int(fs_in)
        self.cutoff = 4000.0
        self.fs_out = 8000 if self.fs_in > 8000 else self.fs_in
        self._bypass = self.fs_in <= 8000
        # LPF más agresivo: más taps y ventana Blackman
        num_taps = 127
        n = np.arange(num_taps) - (num_taps - 1) / 2.0
        fc = min(self.cutoff, 0.49 * (self.fs_in / 2.0))
        sinc = np.sinc(2.0 * fc / self.fs_in * n)
        window = np.blackman(num_taps)
        h = sinc * window
        h /= np.sum(h)
        self._h = h.astype(np.float32)
        self._xprev = np.zeros(num_taps - 1, dtype=np.float32)
        self._ybuf = np.zeros(0, dtype=np.float32)
        self._t = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:  # noqa: D401
        """Devuelve x si Fs<=8000; si no, LPF agresivo + remuestreo a 8 kHz."""
        if x.size == 0:
            return x
        if self._bypass:
            return x
        xi = x.astype(np.float32, copy=False)
        inp = np.concatenate((self._xprev, xi))
        y = np.convolve(inp, self._h, mode="valid").astype(np.float32)
        self._xprev = inp[-(self._h.size - 1) :]
        # Remuestreo lineal a fs_out (8000 Hz)
        if self.fs_out >= self.fs_in:
            return y
        self._ybuf = np.concatenate((self._ybuf, y))
        step = self.fs_in / float(self.fs_out)
        outs = []
        t = self._t
        while t + 1.0 < self._ybuf.size:
            i = int(t)
            frac = t - i
            v = (1.0 - frac) * self._ybuf[i] + frac * self._ybuf[i + 1]
            outs.append(v)
            t += step
        drop = int(t)
        if drop > 0:
            self._ybuf = self._ybuf[drop:]
            t -= drop
        self._t = t
        return np.asarray(outs, dtype=np.float32)


class BandPassProcessor(Processor):
    """Filtro pasabanda (BPF) con centro f0 y ancho BW."""

    def __init__(self, fs: int, center_hz: float = 600.0, bw_hz: float = 200.0) -> None:
        """Disef1a un BPF FIR centrado en f0 con ancho BW."""
        self.fs = int(fs)
        self.f0 = float(center_hz)
        self.bw = float(bw_hz)
        num_taps = 63
        n = np.arange(num_taps) - (num_taps - 1) / 2.0
        # Diseño por diferencia de low-pass: h_bp = h_lp(fc_high) - h_lp(fc_low)
        def _lp(fc: float) -> np.ndarray:
            fc = np.clip(fc, 0.0, 0.49 * (self.fs / 2.0))
            sinc = np.sinc(2.0 * fc / self.fs * n)
            win = np.hamming(num_taps)
            h = sinc * win
            s = np.sum(h)
            return (h / s) if s != 0 else h
        fc_low = max(1.0, self.f0 - self.bw / 2.0)
        fc_high = min(self.fs / 2.0 - 1.0, self.f0 + self.bw / 2.0)
        hbp = _lp(fc_high) - _lp(fc_low)
        self._h = hbp.astype(np.float32)
        self._xprev = np.zeros(num_taps - 1, dtype=np.float32)

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        xi = x.astype(np.float32, copy=False)
        inp = np.concatenate((self._xprev, xi))
        y = np.convolve(inp, self._h, mode="valid").astype(np.float32)
        self._xprev = inp[-(self._h.size - 1) :]
        return y



class SkimmerWithBpf(Processor):
    """Skimmer + BPF: post-diezmado aplica BPF; expone helpers para pre/post BPF."""

    def __init__(self, fs_in: int) -> None:
        self._sk = SkimmerProcessor(fs_in=fs_in)
        fs_bpf = 8000 if fs_in > 8000 else fs_in
        self._bpf = BandPassProcessor(fs=fs_bpf, center_hz=600.0, bw_hz=200.0)

    def process(self, x: np.ndarray) -> np.ndarray:
        y_pre = self._sk.process(x)
        return self._bpf.process(y_pre)

    # Helpers detectados por waterfall para speaker/time plot
    def process_pre_bpf(self, x: np.ndarray) -> np.ndarray:
        return self._sk.process(x)

    def apply_post_bpf(self, y_pre: np.ndarray) -> np.ndarray:
        return self._bpf.process(y_pre)


class ChainProcessor(Processor):
    """Aplica una secuencia de procesadores en orden."""

    def __init__(self, procs: list[Processor]) -> None:
        """Encadena varios procesadores a ejecutar en secuencia."""
        self._procs = list(procs)

    def process(self, x: np.ndarray) -> np.ndarray:
        """Ejecutar cada procesador sobre x en orden y devolver resultado."""
        y = x
        for p in self._procs:
            y = p.process(y)
        return y


