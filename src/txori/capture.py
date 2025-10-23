"""Subsistema de captura de señales."""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import sin, tau

import numpy as np

from .config import SystemConfig
from .exceptions import AudioUnavailableError

try:
    import sounddevice as sd  # type: ignore
except Exception:  # pragma: no cover - dependiente del entorno
    sd = None  # type: ignore


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


class AudioInputCapture(BaseCapture):
    """Captura desde el dispositivo de audio usando sounddevice."""

    def __init__(self, cfg: SystemConfig | None = None, device: int | str | None = None) -> None:
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
            info = sd.query_devices(dev) if dev is not None else sd.query_devices(kind="input")
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
        self._buf: np.ndarray = np.zeros(size, dtype=np.float64)

    def push(self, sample: float) -> None:
        self._buf = np.roll(self._buf, 1)
        self._buf[0] = sample

    @property
    def array(self) -> np.ndarray:
        return self._buf


class CaptureController:
    """Controla la captura acumulando en una ventana deslizante."""

    def __init__(self, strategy: BaseCapture, window_size: int) -> None:
        self.strategy = strategy
        self.window = SlidingWindow(window_size)

    def step(self) -> np.ndarray:
        """Toma una muestra y retorna la ventana actualizada."""
        self.window.push(self.strategy.next_sample())
        return self.window.array
