# CHANGELOG

- 2025-10-31 - Build 012: Evita warning divide-by-zero en specgram usando epsilon antes de log10.

- 2025-10-31 - Build 011: Añade --fft-nfft y --fft-overlap (por defecto NFFT-56).

- 2025-10-31 - Build 010: Agrega --fft-window con múltiples opciones de ventana (default Blackman).

- 2025-10-31 - Build 009: Soporte para WAV PCM de 8 bits en FileSource.

- 2025-10-31 - Build 008: Título del espectrograma ahora indica sample rate (Fs).

- 2025-10-31 - Build 007: README documenta configuración de referencia y argumentos (versión 1.0).

- 2025-10-31 - Build 006: Fix de runtime en animación (window Blackman ahora función compatible).

- 2025-10-31 - Build 005: Arreglo de runtime (window Blackman en specgram) usando función adecuada.

- 2025-10-31 - Build 004: Evita error de runtime cuando el backend no expone window manager en Matplotlib.

- 2025-10-31 - Build 003: Corrige warnings de Matplotlib en animación (cache_frame_data=False) y conserva referencia anim.

- 2025-10-31 - Build 002: Implementa --source file, --in, --cpu none y espectrograma en tiempo real (Blackman, NFFT=256, overlap=NFFT-56) con animación derecha→izquierda.
- 2025-10-31 - Build 001: Resync funcional.
