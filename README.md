Txori
Programa para procesamiento de sonidos de señales CW

Versión 1.0 build 022


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

Novedad (Build 017): Agrega --wide para usar 4096 píxeles horizontales (mantiene píxeles verticales).
- Si se indica --wide, se ignora --fft-pixels y se usan 4096 px horizontales.

Novedad (Build 017): --wide ajusta el ancho real del lienzo a 4096 px (alto inalterado).

Novedad (Build 018): Escala de frecuencias movida al margen derecho del gráfico.

Novedad (Build 018): --fft-ema para suavizado temporal (EMA) y --vmin/--vmax para rango dB fijo.

Argumentos y valores por defecto:
- --source [file] (obligatorio)
- --in RUTA (obligatorio cuando --source file)
- --cpu [none] (default: none)
- --fft-window [Blackman|BlackmanHarris|FlatTop|Hamming|Hanning|Rectangular] (default: Blackman)
- --fft-nfft INT (default: 256)
- --fft-overlap INT (default: NFFT-56)
- --fft-cmap STR (default: ocean)
- --fft-pixels INT (default: 640)
- --wide (flag) si se indica, fija 4096 píxeles horizontales e ignora --fft-pixels
- --fft-ema FLOAT en (0,1) (default: desactivado)
- --vmin FLOAT (default: auto)
- --vmax FLOAT (default: auto)

Configuración de referencia:
  txori-waterfall --source file --in ./sounds/test.wav

Fuente adicional:
- --source tone: tono senoidal sintetizado
- --tone-freq FLOAT (default: 600)
- --tone-fsr INT (default: 4000)

Fuente adicional:
- --source tone: tono senoidal sintetizado
- --tone-freq FLOAT (default: 600)
- --tone-fsr INT (default: 4000)

Reproducción en vivo:
- --spkr: envía las muestras de --source a la salida de audio predeterminada (no altera CPU ni waterfall).
