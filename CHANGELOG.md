# CHANGELOG

- 2025-11-01 - Versión 1.0 ALPHA --BASELINE--: Documentación consolidada (README/BUILD/JOURNAL), lista completa de argumentos y valores por defecto, estructura del programa; CI actualizado (coverage >=80%).
- 2025-10-31 - Build 046: Ajustes Ruff: config lint.*, ignora E501; corrige imports, F821 y duplicados.
- 2025-10-31 - Build 045: Remueve trufflehog de requirements-dev y sincroniza CI/docs.
- 2025-10-31 - Build 044: Elimina trufflehog de requirements-dev y del workflow CI.
- 2025-10-31 - Build 043: Actualiza CONTEXT para no utilizar Trufflehog en controles de seguridad.

- 2025-10-31 - Build 042: Corrige callback de audio (usar buffers 1D en la cola) para evitar ValueError de dimensiones.
- 2025-10-31 - Build 041: Salida de audio con callback (cola + resample) para asegurar reproducción continua.
- 2025-10-31 - Build 040: --spkr reproduce post-CPU y hace fallback a 48 kHz con resample si el Fs del procesado no es soportado por el dispositivo.
- 2025-10-31 - Build 039: --spkr ahora reproduce las mismas muestras post-CPU que llegan al waterfall, usando el Fs actual del procesado.
- 2025-10-31 - Build 038: Nuevo argumento --fft-fpu y el título del waterfall muestra Fs, hop, cols y fpu.
- 2025-10-31 - Build 037: Acelera actualización del waterfall (frames_per_update=1) para acercar al ritmo de la fuente.
- 2025-10-31 - Build 036: Manejo limpio de Ctrl+C: detener productor/escritor y cerrar stream en finally.
- 2025-10-31 - Build 035: Evita crash en shutdown (threads no daemon y join limpio de productor/escritor).
- 2025-10-31 - Build 034: Corrige CLI agregando args --cwfilter/--cpu-bpf-freq/--cpu-bpf-bw al parser.
- 2025-10-31 - Build 033: Con --cpu lpf y --cwfilter aplica BPF CW (f0=600 Hz, BW=200 Hz) antes del waterfall; agrega --cpu-bpf-freq y --cpu-bpf-bw.
- 2025-10-31 - Build 032: --cpu lpf ahora: bypass si Fs<=4000; si Fs>4000 aplica LPF fc (default 2000 Hz) y remuestrea a 2*fc; agrega --cpu-lpf-freq.
- 2025-10-31 - Build 031: Nuevo CPU --cpu lpf (LPF 3 kHz + remuestreo a 6 kHz si Fs>6 kHz) y waterfall ajustado a Fs=6 kHz; pacing de productor según Fs de la fuente.
- 2025-10-31 - Build 030: Salida de speaker siempre en float32 para evitar distorsión en algunos hosts con 8/16 bits.
- 2025-10-31 - Build 029: Desacopla lectura de la fuente en hilo productor; speaker y time plot en tiempo real e independientes del render del waterfall.
- 2025-10-31 - Build 028: Evita lentitud cuando --spkr y --time están activos usando escritura asíncrona al dispositivo de audio.
- 2025-10-31 - Build 027: Ajusta reproducción en altavoz según 8/16 bits y Fs para evitar distorsión con WAV de 8 bits.
- 2025-10-31 - Build 026: FileSource soporta WAV PCM de 8 y 16 bits respetando su sample rate; resto del flujo (CPU, spkr, time plot) sin cambios.
- 2025-10-31 - Build 025: Reduce la escala de tiempo del time plot al usar --time; agrega --time-scale (default: 0.5) para ajustar.

- 2025-10-31 - Build 024: Agrega argumento --time para mostrar un gráfico de tiempo separado alimentado por la misma fuente y sample rate; documentación actualizada.


- 2025-10-31 - Build 023: Implementa --spkr enviando la salida de la fuente al audio predeterminado sin modificar waterfall; README actualizado con todos los argumentos y valores por defecto.

- 2025-10-31 - Build 022: Agrega reproducción por altavoz (--spkr) usando sounddevice.

- 2025-10-31 - Build 021: Implementa ToneSource (seno 600Hz @ fs 4000Hz) y corrige import.

- 2025-10-31 - Build 020: Nueva fuente --source tone (seno 600 Hz por defecto) con --tone-freq y --tone-fsr.

- 2025-10-31 - Build 019: README documenta todos los argumentos y valores por defecto.

- 2025-10-31 - Build 018: Agrega suavizado EMA del espectro y permite fijar rango dB con --vmin/--vmax.

- 2025-10-31 - Build 018: Frecuencia en eje derecho (tick_right y label a la derecha).

- 2025-10-31 - Build 017: Aplica ancho 4096 px cuando se usa --wide (ajuste de figura por DPI).

- 2025-10-31 - Build 017: Implementa flag --wide (4096 px horizontales, altura sin cambios).

- 2025-10-31 - Build 016: Agrega --fft-pixels, fija ~640 px por defecto y escala columnas proporcionalmente; título informativo.

- 2025-10-31 - Build 015: Default colormap ocean y argumento --fft-cmap configurable.

- 2025-10-31 - Build 014: Evita divide-by-zero en log10 durante animación (mlab.specgram + epsilon).

- 2025-10-31 - Build 013: Ctrl+C cierra limpiamente sin traceback y muestra mensaje al usuario.

- 2025-10-31 - Build 012: Evita warning divide-by-zero en specgram usando epsilon antes de log10.

- 2025-10-31 - Build 011: Añade --fft-nfft y --fft-overlap (por defecto NFFT-56).

- 2025-10-31 - Build 010: Agrega --fft-window con múltiples opciones de ventana (default Blackman).

- 2025-10-31 - Build 009: Soporte para WAV PCM de 8 bits en FileSource.

- 2025-10-31 - Build 008: Título del espectrograma ahora indica sample rate (Fs).

- 2025-10-31 - Build 007: README documenta configuración de referencia y argumentos (versión 1.0).

- 2025-10-31 - Build 006: Fix de runtime en animación (window Blackman ahora función compatible).

- 2025-10-31 - Build 005: Arreglo de runtime (window Blackman en specgram) usando función adecuada.

- 2025-10-31 - Build 004: Evita error de runtime cuando el backend no expone window manager en Matplotlib.

- 2025-10-31 - Build 003: Corrige warnings de Matplotlib en animación (cache_frame_data=False) y conserva referencia anim.

- 2025-10-31 - Build 002: Implementa --source file, --in, --cpu none y espectrograma en tiempo real (Blackman, NFFT=256, overlap=NFFT-56) con animación derecha→izquierda.
- 2025-10-31 - Build 001: Resync funcional.
