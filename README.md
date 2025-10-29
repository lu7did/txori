# txori
Programa en Python 3.12 que captura audio de la entrada predeterminada y lo representa en un gráfico waterfall (espectrograma).

Versión 1.0 build 000

Características:
- Paquete instalable con separación de lógica (OOP) y manejo de excepciones.
- CI: ruff, black, PEP8, PEP257, mypy, pyright, pytest (>=80%), bandit, trufflehog.
- Documentación automática con pdoc.

Instalación y uso rápido:
- Crear entorno y ejecutar:
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  python -m pip install -e .
  txori-waterfall --dur 5

Modo continuo (hasta Ctrl+C):
  txori-waterfall --continuous --rate 48000 --nfft 1024 --overlap 0.5

Fuentes disponibles:
- Micrófono/stream: --source stream (por defecto)
- Tono 600 Hz: --source tone

Orientación del gráfico:
- Frecuencia en eje vertical (Y)
- Tiempo en eje horizontal (X) de derecha a izquierda en segundos; ventana fija con desplazamiento
