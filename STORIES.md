# Historias

- 2025-10-31T16:22:43.267Z: Crear procesador de sonidos con --source file (WAV, respeta sample rate), --cpu none, y waterfall en tiempo real usando matplotlib.specgram con window Blackman, NFFT=256, overlap=NFFT-56, animación derecha a izquierda, deslizamiento en tiempo real.

- 2025-10-31T16:31:12Z: Corregir warnings de ejecución (FuncAnimation sin referencia y cache_frame_data).

- 2025-10-31T16:32:27Z: Arreglar error de runtime por fig.canvas.manager inexistente en ciertos backends.

- 2025-10-31T16:33:46Z: Corregir ValueError en numpy.blackman pasando función window compatible con mlab.specgram.

- 2025-10-31T16:34:37Z: Ajustar window Blackman en specgram a callable que acepta vector para evitar ValueError.
- 2025-10-31T19:53:24.763Z: Documentar en README la configuración de referencia y todos los argumentos con defaults; subir build.

- 2025-10-31T20:01:21.262Z: Colocar como título en el espectrograma el sample rate que se le entrega.
- 2025-10-31T20:06:53Z: Agregar soporte para archivos .wav de 8 bits (normalización u8→[-1,1]).
