# (c) Dr. Pedro E. Colla 2020-2025 (LU7DZ)
"""Configuración del sistema."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SystemConfig:
    """Parámetros principales de la aplicación.

    Atributos:
        sample_rate: Frecuencia de muestreo en Hz (entrada, típicamente 48 kHz).
        window_size: Muestras mantenidas en la ventana deslizante (dominio de entrada).
        cutoff_hz: Frecuencia de corte del filtro pasabajos (por defecto 3 kHz, modificable).
        fft_bin_hz: Ancho máximo de cada intervalo de frecuencia en Hz.
        image_width: Ancho del espectrograma en píxeles (por defecto 1200).
        average_frames: Cantidad máxima de columnas a promediar (suavizado entre columnas).
        update_interval: Cada cuántos arreglos recibidos se dibuja una columna (renderer); típicamente 1.
        samples_per_col: Cantidad de muestras diezmadas que se promedian por columna (por defecto 15).
        use_audio: Si True intenta usar entrada de audio real; si False, señal sintética.
    """

    sample_rate: int = 48_000
    window_size: int = 100
    cutoff_hz: float = 3_000.0
    fft_bin_hz: float = 3.0
    image_width: int = 1_200
    average_frames: int = 100
    update_interval: int = 1
    samples_per_col: int = 15
    use_audio: bool = False
    test_tone_hz: float = 1000.0
    cw_mode: bool = False
    cw_tone_hz: float = 600.0
