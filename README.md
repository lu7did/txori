# Txori

Procesamiento de señales en tiempo casi-real con Python 3.12.

## Características
- Captura de señal (entrada de audio opcional o generador sintético).
- Filtro pasabajos (3 kHz) para limitar ancho de banda.
- Procesamiento (etapa configurable, inicialmente identidad).
- Análisis FFT con resolución de ~3 Hz y límite a 3 kHz.
- Visualización en espectrograma desplazable (al menos 1200px de ancho), promedio de hasta 100 columnas y actualización cada 5; escala de color logarítmica (dB).
- Ventana opcional de tiempo (--time) mostrando la señal cruda sin filtrar; duración igual al espectrograma y en modo --forever continúa indefinidamente.

## CLI y ayuda

- --audio: usar entrada de audio real
- --test: usar generador interno de senoidal
- --tone HZ: frecuencia del tono de test (default 1000 Hz)
- --time: abrir ventana separada con la señal temporal cruda
- --seconds, --forever, --out, --titulo, --width, --cutoff, --bin, --avg-samples

## Diseño
- Orientado a objetos y patrones de estrategia por subsistema.
- Separación de lógica de dominio y presentación.
- Empaquetado como librería (src layout) y CLI básica (`python -m txori`).
- Tratamiento de excepciones específico (`txori.exceptions`).

## Requisitos
- Python 3.12
- Dependencias: ver `requirements.txt`. Para visualización en vivo instalar `matplotlib` (extra `viz`). Para audio real, `sounddevice`.

## Uso rápido
```
# Modo en vivo (requiere matplotlib):
python -m txori --seconds 2
python -m txori --audio --seconds 5  # requiere hardware de audio

# Ventana de tiempo crudo separada
python -m txori --seconds 2 --time

# Modo test con tono configurable
python -m txori --test --tone 1000 --seconds 2

# Guardar a archivo (sin ventana):
python -m txori --seconds 2 --out spectrogram.png
```


## Desarrollo
- Formato y lint: Ruff + Black.
- Tipado: MyPy (modo estricto en módulos principales).
- Seguridad: Bandit.
- Tests: PyTest con cobertura objetivo >= 80%.
- Docs: pdoc (generadas a `docs/`).

## CI/CD
GitHub Actions ejecuta lint, tipado, pruebas, seguridad y publica artefacto de documentación en cada push/PR.

## Licencia
MIT. Ver `LICENSE`.
