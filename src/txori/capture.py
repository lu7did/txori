# (c) Dr. Pedro E. Colla 2020-2025 (LU7DZ)
"""Subsistema de captura de señales."""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from math import sin, tau
from typing import Any

import numpy as np
import numpy.typing as npt

from .config import SystemConfig
from .exceptions import AudioUnavailableError

try:
    sd: Any = importlib.import_module("sounddevice")
except Exception:  # pragma: no cover - dependiente del entorno
    sd = None


class BaseCapture(ABC):
    """Estrategia de captura de señal."""

    @abstractmethod
    def next_sample(self) -> float:
        """Obtiene la siguiente muestra (valor escalar)."""

    def label(self) -> str:
        return "Entrada"


class SyntheticSineCapture(BaseCapture):
    """Generador sintético de señal senoidal para pruebas y CI."""

    def __init__(self, freq_hz: float = 440.0, cfg: SystemConfig | None = None) -> None:
        self.cfg = cfg or SystemConfig()
        self._t = 0
        self._freq = float(freq_hz)
        self._omega = tau * self._freq / self.cfg.sample_rate

    def next_sample(self) -> float:
        v = sin(self._omega * self._t)
        self._t += 1
        return float(v)

    def label(self) -> str:
        return f"Sintético {int(self._freq)} Hz"


class SyntheticCWToneCapture(BaseCapture):
    """Generador CW: tono senoidal conmutado ON/OFF cada 57 ms.

    Por defecto 600 Hz; la duración del bloque ON y OFF es 57 ms cada uno.
    """

    def __init__(self, freq_hz: float = 600.0, cfg: SystemConfig | None = None) -> None:
        self.cfg = cfg or SystemConfig()
        self._t = 0
        self._freq = float(freq_hz)
        self._omega = tau * self._freq / self.cfg.sample_rate
        self._block_samples = max(1, int(round(0.057 * self.cfg.sample_rate)))
        self._remain = self._block_samples
        self._on = True

    def next_sample(self) -> float:
        val = sin(self._omega * self._t) if self._on else 0.0
        self._t += 1
        self._remain -= 1
        if self._remain <= 0:
            self._on = not self._on
            self._remain = self._block_samples
        return float(val)

    def label(self) -> str:
        return f"CW {int(self._freq)} Hz (57 ms on/off)"


class SyntheticCWPairCapture(BaseCapture):
    """Mezcla dos generadores CW: principal y QRN a 1000 Hz."""

    def __init__(self, freq_main: float = 600.0, freq_qrn: float = 1000.0, cfg: SystemConfig | None = None) -> None:
        self._g1 = SyntheticCWToneCapture(freq_main, cfg)
        self._g2 = SyntheticCWToneCapture(freq_qrn, cfg)

    def next_sample(self) -> float:
        v = float(self._g1.next_sample() + self._g2.next_sample())
        return float(max(-2.0, min(2.0, v)))

    def label(self) -> str:
        return "CW+QRN"


class SyntheticCWToneGroupCapture(BaseCapture):
    """Mezcla varios generadores CW (sin ruido)."""

    def __init__(self, freqs_hz: list[float], cfg: SystemConfig | None = None) -> None:
        self._gens = [SyntheticCWToneCapture(float(f), cfg) for f in freqs_hz]

    def next_sample(self) -> float:
        # Sumar portadoras a igual nivel que --cw/--qrn; aplicar limitador suave para evitar clipping
        v = float(sum(g.next_sample() for g in self._gens))
        # Recorte suave para evitar saturación excesiva, preservando niveles relativos
        v = float(max(-1.0, min(1.0, v)))
        return v

    def label(self) -> str:
        return "CW+QRM"

class AudioInputCapture(BaseCapture):
    """Captura desde el dispositivo de audio usando sounddevice."""

    def __init__(
        self, cfg: SystemConfig | None = None, device: int | str | None = None
    ) -> None:
        if sd is None:
            raise AudioUnavailableError("sounddevice no disponible en este entorno")
        self.cfg = cfg or SystemConfig()
        self._stream = sd.InputStream(
            samplerate=self.cfg.sample_rate,
            channels=1,
            dtype="float32",
            device=device,
            blocksize=1,
        )
        self._stream.start()
        # Resuelve nombre del dispositivo
        try:  # pragma: no cover - dependiente del entorno
            dev = getattr(self._stream, "device", device)
            info = (
                sd.query_devices(dev)
                if dev is not None
                else sd.query_devices(kind="input")
            )
            self._device_name = str(info.get("name", "dispositivo"))
        except Exception:  # pragma: no cover - robustez
            self._device_name = "dispositivo"

    def next_sample(self) -> float:  # pragma: no cover - requiere hardware
        assert self._stream is not None
        frames, _ = self._stream.read(1)
        return float(frames[0, 0])

    def label(self) -> str:  # pragma: no cover - depende de hardware
        return f"Audio: {getattr(self, '_device_name', 'dispositivo')}"


class SlidingWindow:
    """Ventana deslizante con la muestra más reciente en la posición 0.

    Cada nueva muestra desplaza las anteriores una posición a la derecha,
    descarta la más antigua y coloca la nueva en la posición 0.
    """

    def __init__(self, size: int) -> None:
        if size <= 0:
            raise ValueError("size debe ser > 0")
        self.size = size
        self._buf: npt.NDArray[np.float64] = np.zeros(size, dtype=np.float64)

    def push(self, sample: float) -> None:
        self._buf = np.roll(self._buf, 1)
        self._buf[0] = sample

    @property
    def array(self) -> npt.NDArray[np.float64]:
        return self._buf


class CaptureController:
    """Controla la captura acumulando en una ventana deslizante."""

    def __init__(self, strategy: BaseCapture, window_size: int) -> None:
        self.strategy = strategy
        self.window = SlidingWindow(window_size)

    def step(self) -> npt.NDArray[np.float64]:
        """Toma una muestra y retorna la ventana actualizada."""
        self.window.push(self.strategy.next_sample())
        return self.window.array
