"""Módulo de captura de audio desde la entrada predeterminada.

Se provee una clase de alto nivel que realiza la grabación en bloque para luego
ser procesada por el módulo de waterfall. La lógica de captura está aislada de la
visualización para favorecer testeo y mantenibilidad.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

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
