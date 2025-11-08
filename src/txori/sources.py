"""Sources: entrada de audio para el pipeline."""
from __future__ import annotations

from abc import ABC, abstractmethod
import wave
import threading
import time
from collections import deque

import numpy as np

try:  # Backend de audio opcional para fuente en vivo
    import sounddevice as sd  # type: ignore
except Exception:  # pragma: no cover
    sd = None




class Source(ABC):
    """Interfaz de fuente de audio."""

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Frecuencia de muestreo en Hz."""

    @abstractmethod
    def read(self, n: int) -> np.ndarray:
        """Lee hasta n muestras mono normalizadas en [-1, 1]."""

    @abstractmethod
    def close(self) -> None:
        """Libera recursos de la fuente."""




class FileSource(Source):
    """Fuente que lee muestras desde un archivo WAV."""

    def __init__(self, path: str) -> None:
        """Abrir un WAV PCM y preparar atributos de formato (sr, bits, canales)."""
        self._wf = wave.open(path, "rb")
        self._sr = int(self._wf.getframerate())
        self._sampwidth = self._wf.getsampwidth()
        self._channels = self._wf.getnchannels()
        if self._sampwidth not in (1, 2):
            self._wf.close()
            raise ValueError("Solo se soportan WAV PCM de 8 o 16 bits")

    @property
    def sample_rate(self) -> int:
        return self._sr

    def read(self, n: int) -> np.ndarray:
        """Leer hasta n muestras mono normalizadas en [-1, 1]."""
        frames = self._wf.readframes(max(0, int(n)))
        if not frames:
            return np.array([], dtype=np.float32)
        if self._sampwidth == 1:
            # PCM8 unsigned: 0..255 -> [-1, 1)
            data = np.frombuffer(frames, dtype=np.uint8)
            if self._channels > 1:
                data = data.reshape(-1, self._channels).mean(axis=1)
            out = ((data.astype(np.float32) - 128.0) / 128.0).astype(np.float32)
            return out
        elif self._sampwidth == 2:
            # PCM16 signed: -32768..32767 -> [-1, 1)
            data = np.frombuffer(frames, dtype=np.int16)
            if self._channels > 1:
                data = data.reshape(-1, self._channels).mean(axis=1)
            out = (data.astype(np.float32)) / 32768.0
            return out
        else:
            # No debería ocurrir por validación en __init__
            return np.array([], dtype=np.float32)

    def close(self) -> None:
        """Cerrar el archivo WAV si este1 abierto."""
        try:
            self._wf.close()
        except Exception:  # nosec B110 - ignore close errors on teardown
            pass




class ToneSource(Source):
    """Fuente que sintetiza un tono senoidal continuo."""

    def __init__(self, freq_hz: float = 600.0, fs: int = 4000) -> None:
        """Configurar un tono senoidal continuo a frecuencia y Fs dados."""
        self._sr = int(fs)
        self._freq = float(freq_hz)
        self._phase = 0.0
        self._dphi = 2.0 * np.pi * (self._freq / float(self._sr))

    @property
    def sample_rate(self) -> int:
        return self._sr

    def read(self, n: int) -> np.ndarray:
        """Generar n muestras del seno y avanzar la fase interna."""
        n = max(0, int(n))
        if n == 0:
            return np.array([], dtype=np.float32)
        idx = np.arange(n, dtype=np.float64)
        phase = self._phase + idx * self._dphi
        x = np.sin(phase).astype(np.float32)
        self._phase = float((self._phase + n * self._dphi) % (2.0 * np.pi))
        return x

    def close(self) -> None:
        """No requiere cierre expledcito."""
        return


class LineSource(Source):
    """Fuente que captura audio en vivo del dispositivo de entrada predeterminado."""

    def __init__(self, blocksize: int = 1024, device: int | None = None) -> None:
        if sd is None:
            raise RuntimeError("sounddevice no disponible para --source line")
        try:
            dev_in = sd.default.device
            if isinstance(dev_in, (list, tuple)):
                dev_in = dev_in[0]
            # Si se provee un dispositivo explito, usarlo
            if device is not None:
                dev_in = int(device)
            info = sd.query_devices(dev_in)
        except Exception:
            info = {"default_samplerate": 48000}
            dev_in = device
        self._sr = int(info.get("default_samplerate", 48000) or 48000)
        self._buf = deque()  # type: ignore[var-annotated]
        self._lock = threading.Lock()
        self._closed = False

        def _cb(indata, frames, time_info, status) -> None:  # noqa: D401
            if status:
                pass
            try:
                mono = indata[:, 0].astype(np.float32, copy=True)
            except Exception:
                mono = np.zeros(frames, dtype=np.float32)
            with self._lock:
                self._buf.append(mono)

        try:
            self._stream = sd.InputStream(
                samplerate=self._sr,
                channels=1,
                dtype="float32",
                callback=_cb,
                blocksize=blocksize,
                device=dev_in if dev_in is not None else None,
            )
            self._stream.start()
        except Exception as e:  # pragma: no cover - error de backend
            raise RuntimeError(f"No se pudo iniciar captura de audio: {e}")

    @property
    def sample_rate(self) -> int:
        return self._sr

    def read(self, n: int) -> np.ndarray:
        if self._closed or n <= 0:
            return np.array([], dtype=np.float32)
        n = int(n)
        out = []
        remaining = n
        t0 = time.monotonic()
        # Espera activa breve para acumular muestras (no bloqueante prolongado)
        while remaining > 0:
            chunk = None
            with self._lock:
                if self._buf:
                    chunk = self._buf.popleft()
            if chunk is not None:
                if chunk.size > remaining:
                    out.append(chunk[:remaining])
                    # devolver la porción sobrante al frente
                    rest = chunk[remaining:]
                    if rest.size:
                        with self._lock:
                            self._buf.appendleft(rest)
                    remaining = 0
                else:
                    out.append(chunk)
                    remaining -= chunk.size
            else:
                # Sin datos disponibles; breve sleep para no consumir CPU
                if time.monotonic() - t0 > 0.25:  # timeout corto
                    break
                time.sleep(0.005)
        if not out:
            return np.array([], dtype=np.float32)
        return np.concatenate(out).astype(np.float32, copy=False)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:  # nosec B110 - ignorar errores de cierre
            pass
