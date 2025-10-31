Txori
Programa para procesamiento de sonidos de señales CW


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

Novedad (Build 007): Animación de espectrograma desplazándose de derecha a izquierda y nuevo argumento --fft-cols (default 800).
