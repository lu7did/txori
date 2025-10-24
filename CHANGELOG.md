# Changelog

## [0.1.3] - 2025-10-24
### Added
- Nueva opción "--cw": al indicarla se ignoran "--tone" y "--audio" y la entrada se toma de un generador interno de tono senoidal de 600 Hz conmutado (señal/silencio) cada 57 ms; se agrega "--cw-tone" para variar la frecuencia del tono CW.

## [0.1.2] - 2025-10-24
### Added
- Se habilitará una opción "--test" que cuando sea indicada se tomará la señal a procesar por el sistema desde un generador interno de muestras que creará una señal sinusoidal de 1000 Hz por default, se agregará un parametro adicional "--tone" con la indicación de la frecuencia en hertz que tendrá la señal de test si es indicada.

## [0.1.1] - 2025-10-23
### Added
- Ventana de tiempo separada y opcional (`--time`) que muestra la señal cruda sin procesar, con duración igual al espectrograma y actualización independiente; en `--forever` sigue mostrando hasta interrupción.
- Modo de prueba (`--test`) que fuerza la señal desde un generador interno; frecuencia configurable con `--tone` (default 1000 Hz).
- Escala de color del espectrograma ahora usa asignación logarítmica (dB) para la energía por bucket.
- Opción --titulo para título externo en la UI.
- Ayuda/README actualizados con nuevas opciones.

### Fixed
- Errores de MyPy en live.py y capture.py: inicialización de `plt`, tipos `Any`, y eliminación de `type: ignore` innecesario.

## [0.1.0] - 2025-10-23
### Added
- Estructura de proyecto empaquetable (src layout) y configuración (Ruff, Black, MyPy, PyTest, Bandit).
- Subsistemas: captura (sintética y audio opcional), filtro pasabajos, procesamiento identidad, análisis FFT, visualización espectrograma.
- CLI mínima (`python -m txori`) con guardado de imagen.
- CI con GitHub Actions: lint, tipado, pruebas, seguridad y docs pdoc.
- Documentación básica inicial y CHANGELOG.

### Changed
- Visualización en vivo: se agrega traza superior de intensidad con fondo negro y píxeles color lime, sincronizada con el espectrograma.
- La traza superior usa el nivel instantáneo de la señal filtrada por pasabajos (sin promediado), con normalización dinámica.
- Espectrograma ajustado para ocupar todo el ancho configurable (`--width`, por defecto 1200 px); el eje de tiempo se escala usando `seconds_per_col`.
- Cadena de temporización: entrada 48 kHz → pasabajos `--cutoff` (default 3000 Hz) → diezmado 1:8 (6 kHz) → una columna cada `--avg-samples` (default 15) muestras diezmadas (0.0025 s/col en defecto).
- FFT y número de bins acordes al ancho de banda (0..cutoff) y `--bin` (default 3 Hz); el eje Y en Hz a la derecha.
- Textos de interfaz/título reposicionados para quedar más alejados del espectrograma.
- CLI: `--seconds` ahora corre indefinidamente cuando no se indica; `--forever` sigue disponible.
- Ayuda de ejecución actualizada para reflejar nuevos parámetros: `--width`, `--cutoff`, `--bin`, `--avg-samples`.

### Fixed
- Correcciones Ruff: docstrings ausentes, hints `X | None`, orden de imports, eliminación de `assert False` en tests.
- Ajustes menores de robustez y dependencias opcionales (import perezoso de matplotlib en vivo).
