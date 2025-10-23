"""Configuración del sistema."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SystemConfig:
    """Parámetros principales de la aplicación.

    Atributos:
        sample_rate: Frecuencia de muestreo en Hz.
        window_size: Muestras mantenidas en la ventana deslizante.
        cutoff_hz: Frecuencia de corte del filtro pasabajos.
        fft_bin_hz: Ancho máximo de cada intervalo de frecuencia en Hz.
        image_width: Ancho del espectrograma en píxeles.
        average_frames: Cantidad máxima de columnas a promediar (p. ej. 100).
        update_interval: Cada cuántos arreglos recibidos se dibuja una columna.
        use_audio: Si True intenta usar entrada de audio real; si False, señal sintética.
    """

    sample_rate: int = 48_000
    window_size: int = 100
    cutoff_hz: float = 3_000.0
    fft_bin_hz: float = 3.0
    image_width: int = 1_200
    average_frames: int = 100
    update_interval: int = 5
    use_audio: bool = False
