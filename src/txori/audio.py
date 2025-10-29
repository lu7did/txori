"""Módulo de captura de audio desde la entrada predeterminada.

Se provee una clase de alto nivel que realiza la grabación en bloque para luego
ser procesada por el módulo de waterfall. La lógica de captura está aislada de la
visualización para favorecer testeo y mantenibilidad.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Iterator
import queue

import numpy as np
import sounddevice as sd


class AudioError(Exception):
    """Error relacionado con la captura de audio."""


@dataclass(slots=True)
class DefaultAudioSource:
    """Fuente de audio por defecto.

    Aísla la interacción con ``sounddevice`` para facilitar el testeo de la lógica
    de procesamiento sin requerir hardware.
    """

    sample_rate: int = 48_000
    channels: int = 1

    def record(self, duration_s: float) -> np.ndarray:
        """Graba ``duration_s`` segundos y devuelve un vector mono ``float32``.

        Args:
            duration_s: Duración en segundos (> 0).

        Returns:
            np.ndarray: Señal mono en el rango [-1, 1].

        Raises:
            AudioError: Si ocurre un problema al acceder al dispositivo.
            ValueError: Si la duración es inválida.
        """
        if duration_s <= 0:
            raise ValueError("duration_s debe ser > 0")
        try:
            frames: Final[int] = int(duration_s * self.sample_rate)
            rec = sd.rec(
                frames,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
            )
            sd.wait()
        except Exception as exc:  # noqa: BLE001 - superficie controlada
            raise AudioError("Fallo al capturar audio") from exc

        data = rec.astype(np.float32, copy=False)
        if self.channels > 1:
            data = np.mean(data, axis=1)
        else:
            data = data.reshape(-1)
        return data


@dataclass(slots=True)
class StreamAudioSource:
    """Fuente de audio continua basada en InputStream de sounddevice.

    Entrega bloques mono de tamaño fijo vía un iterador infinito hasta interrupción.
    """

    sample_rate: int = 48_000
    channels: int = 1
    blocksize: int = 1024

    def blocks(self) -> Iterator[np.ndarray]:
        """Itera bloques mono de tamaño fijo hasta Ctrl+C.

        Yields:
            np.ndarray: bloque mono float32 en [-1, 1] de longitud ``blocksize``.
        """
        if self.blocksize <= 0:
            raise ValueError("blocksize debe ser > 0")
        q: queue.Queue[np.ndarray] = queue.Queue(maxsize=50)

        def _cb(indata: np.ndarray, frames: int, time, status) -> None:  # noqa: ANN001
            if status:  # status puede reportar xruns; no elevamos excepción aquí
                pass
            # Intento no bloqueante compatible con colas stub de tests
            try:
                q.put_nowait(indata.copy())
            except AttributeError:
                try:
                    q.put(indata.copy(), block=False)
                except queue.Full:
                    # Descarta más antiguo y reintenta
                    try:
                        try:
                            q.get_nowait()
                        except AttributeError:
                            try:
                                q.get(block=False)
                            except TypeError:
                                q.get()
                        q.put(indata.copy(), block=False)
                    except Exception:
                        pass
            except queue.Full:
                # Cola llena: descartar el bloque más antiguo y reintentar; si vuelve a fallar, soltar este bloque
                try:
                    try:
                        q.get_nowait()
                    except AttributeError:
                        try:
                            q.get(block=False)
                        except TypeError:
                            q.get()
                    try:
                        q.put_nowait(indata.copy())
                    except AttributeError:
                        q.put(indata.copy(), block=False)
                except Exception:
                    pass

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                blocksize=self.blocksize,
                callback=_cb,
            ):
                while True:
                    data = q.get()
                    x = data.astype(np.float32, copy=False)
                    if self.channels > 1:
                        x = np.mean(x, axis=1)
                    else:
                        x = x.reshape(-1)
                    yield x
        except Exception as exc:  # noqa: BLE001
            raise AudioError("Fallo en el stream de audio") from exc


@dataclass(slots=True)
class ToneAudioSource:
    """Fuente de audio de tono senoidal de 600 Hz para pruebas y demo.

    Provee captura sintética tanto por bloques (stream) como en grabación fija.
    """

    sample_rate: int = 48_000
    frequency: float = 600.0
    blocksize: int = 1024
    _phase: float = field(default=0.0, init=False, repr=False)

    def record(self, duration_s: float) -> np.ndarray:
        """Genera una señal senoidal mono por la duración indicada."""
        if duration_s <= 0:
            raise ValueError("duration_s debe ser > 0")
        n = int(duration_s * self.sample_rate)
        t = np.arange(n, dtype=np.float32) / float(self.sample_rate)
        x = np.sin(2 * np.pi * self.frequency * t).astype(np.float32)
        return x

    def blocks(self) -> Iterator[np.ndarray]:
        """Itera bloques senoidales continuos preservando fase hasta Ctrl+C."""
        if self.blocksize <= 0:
            raise ValueError("blocksize debe ser > 0")
        try:
            while True:
                idx = np.arange(self.blocksize, dtype=np.float32)
                t = (self._phase + idx) / float(self.sample_rate)
                x = np.sin(2 * np.pi * self.frequency * t).astype(np.float32)
                self._phase += self.blocksize
                yield x
        except Exception as exc:  # noqa: BLE001
            raise AudioError("Fallo en generador de tono") from exc
