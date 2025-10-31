Txori
Programa para procesamiento de sonidos de señales CW

Versión 1.0 build 016


Novedad (Build 002): Procesador de sonidos en tiempo real
- Fuente inicial: --source file con ruta --in a un .WAV (se respeta su sample rate).
- CPU inicial: --cpu none (por defecto) pasa las muestras sin modificar.
- Waterfall: matplotlib.specgram window Blackman, NFFT=256, overlap=NFFT-56, animación derecha→izquierda en tiempo real.

Ejemplo:
  txori-waterfall --source file --in ejemplos/mi_audio.wav --cpu none

Novedad (Build 003): Corrección de warnings Matplotlib en animación (cache_frame_data=False) y retención de la referencia anim.

Novedad (Build 004): Robustez en backends sin window manager de Matplotlib (evita crash en set_window_title).

Novedad (Build 005): Corrige crash en specgram por uso incorrecto de Blackman; se usa función compatible con mlab.

Novedad (Build 006): Corrige definitivamente el runtime pasando función Blackman compatible a specgram.

Novedad (Build 008): El título del espectrograma muestra el sample rate recibido (Fs en Hz).

Compatibilidad de formatos:
- WAV PCM de 8 o 16 bits, mono o estéreo (se mezcla a mono).

Nuevo argumento:
- --fft-window [Blackman|BlackmanHarris|FlatTop|Hamming|Hanning|Rectangular] (default: Blackman)

Nuevos argumentos:
- --fft-nfft INT (default: 256)
- --fft-overlap INT (default: NFFT-56)

Novedad (Build 012): Se evita RuntimeWarning divide by zero aplicando piso numérico antes de log10 del espectro.

Novedad (Build 013): Manejo de Ctrl+C sin excepción; muestra 'Programa terminado por el usuario'.

Novedad (Build 014): Corrige warning de log10 en runtime usando epsilon también en la animación.

Novedad (Build 015):
- Colormap por defecto: ocean.
- Nuevo argumento: --fft-cmap NOMBRE (default: ocean) para elegir el colormap.

Novedad (Build 016):
- Ancho por defecto ~640 px; ajustar con --fft-pixels INT.
- Las columnas de tiempo se escalan proporcionalmente (base 400 cols a 640 px).
- El título muestra Fs, píxeles y columnas actuales.
