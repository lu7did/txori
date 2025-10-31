# Historias

- 2025-10-31T16:22:43.267Z: Crear procesador de sonidos con --source file (WAV, respeta sample rate), --cpu none, y waterfall en tiempo real usando matplotlib.specgram con window Blackman, NFFT=256, overlap=NFFT-56, animación derecha a izquierda, deslizamiento en tiempo real.

- 2025-10-31T16:31:12Z: Corregir warnings de ejecución (FuncAnimation sin referencia y cache_frame_data).

- 2025-10-31T16:32:27Z: Arreglar error de runtime por fig.canvas.manager inexistente en ciertos backends.

- 2025-10-31T16:33:46Z: Corregir ValueError en numpy.blackman pasando función window compatible con mlab.specgram.

- 2025-10-31T16:34:37Z: Ajustar window Blackman en specgram a callable que acepta vector para evitar ValueError.
- 2025-10-31T19:53:24.763Z: Documentar en README la configuración de referencia y todos los argumentos con defaults; subir build.

- 2025-10-31T20:01:21.262Z: Colocar como título en el espectrograma el sample rate que se le entrega.
- 2025-10-31T20:06:53Z: Agregar soporte para archivos .wav de 8 bits (normalización u8→[-1,1]).
- 2025-10-31T20:12:33.903Z: Agregar argumento --fft-window (Blackman, BlackmanHarris, FlatTop, Hamming, Hanning, Rectangular).
- 2025-10-31T20:19:27.564Z: Introducir argumentos --fft-nfft y --fft-overlap con defaults (256 y NFFT-56).
- 2025-10-31T20:21:35Z: Corregir RuntimeWarning divide by zero en log10 del espectrograma aplicando epsilon.

- 2025-10-31T20:23:18.805Z: Al presionar Control+C, terminar sin excepción y mostrar 'Programa terminado por el usuario'.

- 2025-10-31T20:24:58Z: Suprimir RuntimeWarning divide by zero reemplazando ax.specgram por mlab.specgram + epsilon.
- 2025-10-31T20:26:07.525Z: Usar colormap 'ocean' por defecto e implementar --fft-cmap para personalizarlo.
- 2025-10-31T20:29:48Z: Implementar argumento --fft-pixels (default 640), escalar width_cols proporcionalmente y reflejar en título.
- 2025-10-31T20:42:09.103Z: Agregar argumento --wide para fijar 4096 píxeles horizontales manteniendo los verticales.
- 2025-10-31T20:43:54Z: Corregir --wide para que fije 4096 px de ancho usando DPI del backend.
- 2025-10-31T20:44:49.432Z: Mover la escala de frecuencias al margen derecho del gráfico.
- 2025-10-31T20:47:19Z: Implementar --fft-ema y --vmin/--vmax para control de contraste.
- 2025-10-31T20:53:04.901Z: Documentar en README todos los argumentos y sus defaults.
- 2025-10-31T20:56:12.057Z: Agregar fuente de tono sintetizado (--source tone) con parámetros --tone-freq y --tone-fsr.
- 2025-10-31T20:58:31Z: Añadir clase ToneSource y corregir error de import al usar --source tone.
- 2025-10-31T21:05:17.132Z: Implementar --spkr para reproducir la salida de la fuente en el dispositivo de audio por defecto.
