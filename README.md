# Txori

Procesamiento de señales en tiempo casi-real con Python 3.12.

## Características
- Captura de señal (entrada de audio opcional o generador sintético).
- Filtro pasabajos (3 kHz) para limitar ancho de banda.
- Procesamiento (etapa configurable, inicialmente identidad).
- Análisis FFT con resolución de ~3 Hz y límite a 3 kHz.
- Visualización en espectrograma desplazable (al menos 1200px de ancho), promedio de hasta 100 columnas y actualización cada 5.

## Diseño
- Orientado a objetos y patrones de estrategia por subsistema.
- Separación de lógica de dominio y presentación.
- Empaquetado como librería (src layout) y CLI básica (`python -m txori`).
- Tratamiento de excepciones específico (`txori.exceptions`).

## Requisitos
- Python 3.12
- Dependencias: ver `requirements.txt`.

## Uso rápido
```
python -m txori --seconds 2 --out spectrogram.png  # señal sintética
python -m txori --audio --seconds 5 --out spec.png # requiere hardware de audio
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
