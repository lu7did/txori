# txori
Programa en Python 3.12 que captura audio de la entrada predeterminada y lo representa en un gráfico waterfall (espectrograma).

Versión 1.0 build 001

Características:
- Paquete instalable con separación de lógica (OOP) y manejo de excepciones.
- CI: ruff, black, PEP8, PEP257, mypy, pyright, pytest (>=80%), bandit, trufflehog.
- Documentación automática con pdoc.

Instalación y uso rápido:
- Crear entorno y ejecutar:
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  python -m pip install -e .
  txori-waterfall  # por defecto corre continuo hasta Ctrl+C
  txori-waterfall --dur 5  # duración fija

Modo continuo (hasta Ctrl+C):
  txori-waterfall --continuous --rate 48000 --nfft 1024 --overlap 0.5

Fuentes disponibles:
- Micrófono/stream: --source stream (por defecto). En modo continuo procesa bloques; con --dur graba una vez y muestra estático.
- Tono 600 Hz: --source tone (opcionalmente emitir por parlante con --spkr).
- CW Morse 600 Hz: --source cw (mensaje por defecto "LU7DZ TEST     ", wpm=20). Puede emitirse por parlante con --spkr.

Orientación del gráfico:
- Frecuencia en eje vertical (Y)
- Tiempo en eje horizontal (X) de derecha a izquierda en segundos; ventana fija con desplazamiento

Argumentos:
- --dur float: Duración en segundos; si se especifica, desactiva el modo continuo.
- --rate int: Frecuencia de muestreo (Hz). Por defecto 48000.
- --nfft int: Tamaño de la FFT. Por defecto 4096; 2048 reduce costo computacional.
- --overlap float: Traslape entre ventanas en [0,1). Por defecto 0.5.
- --hop int: Paso entre frames en muestras (si se indica, reemplaza --overlap).
- --cmap str: Colormap de Matplotlib. Por defecto "viridis".
- --continuous: Visualización continua hasta interrupción; se ignora si se indica --dur.
- --max-frames int: Filas visibles en vivo (buffer rodante). Por defecto 400.
- --source [stream|tone|cw]: Fuente de datos. Por defecto stream.
- --spkr: Emitir señal por parlante (stream copia entrada a salida; tone/cw reproducen la señal generada).
- --time: Mostrar timeplot en vivo junto al waterfall.
- --window [hann|blackman|blackmanharris]: Ventana de análisis. Por defecto blackman.
- --vol int: Nivel máx. 0-100% de la señal (afecta stream/tone/cw y speaker/timeplot/waterfall). Por defecto 60.
- --row-median: Resta la mediana por fila del espectrograma para mejorar contraste.
- --db-range float: Rango dinámico del colormap en dB (ej. 80) para ver energía y silencios.

Ejemplos:
- txori-waterfall --source stream --time
- txori-waterfall --source tone --spkr
- txori-waterfall --source cw --time --spkr

Comando de referencia para CW (modo DSP a 6 kHz):
- txori-waterfall --source cw --dsp --spkr --fir-decim --bpf --cwkill --hop 16 --window blackmanharris --nfft 216 --row-median --db-range 60 --smooth 20
  Notas: --fir-decim aplica un FIR antes del 8:1 para mitigar transitorios; --smooth realiza un EMA de 20 columnas; --cwkill centra un BPF en 600 Hz con BW por defecto 20 Hz; el waterfall se dibuja a Fs'=6000 Hz.
