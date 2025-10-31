# Historias

- 2025-10-31T16:22:43.267Z: Crear procesador de sonidos con --source file (WAV, respeta sample rate), --cpu none, y waterfall en tiempo real usando matplotlib.specgram con window Blackman, NFFT=256, overlap=NFFT-56, animación derecha a izquierda, deslizamiento en tiempo real.

- 2025-10-31T16:31:12Z: Corregir warnings de ejecución (FuncAnimation sin referencia y cache_frame_data).

- 2025-10-31T16:32:27Z: Arreglar error de runtime por fig.canvas.manager inexistente en ciertos backends.
