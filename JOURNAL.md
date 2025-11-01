Txori - JOURNAL (2025-11-01T04:27:14Z)

Resumen de interacciones recientes
- Ajuste de time plot con --time y --time-scale para ventana temporal más corta por unidad horizontal.
- Soporte de WAV 8/16 bits, sample rate nativo; mezcla a mono si es estéreo.
- Correcciones de distorsión en altavoz para fuentes de 8 bits y respeto de Fs en reproducción.
- Desacople: speaker y time plot en tiempo real e independientes del render del waterfall.
- Procesador CPU 'lpf' con fc configurable (--cpu-lpf-freq); bypass si Fs<=4000; si Fs>4000, LPF y diezmado a 2*fc.
- Opción --cwfilter agrega BPF CW (f0 y BW configurables con --cpu-bpf-freq/--cpu-bpf-bw) tras LPF antes del waterfall.
- Waterfall ajustado: frames_per_update configurable (--fft-fpu) y título con Fs, hop, width_cols y fpu.
- Múltiples fixes de runtime y shutdown limpio de hilos/productor/salida de audio.
- Correcciones de lint (ruff/flake8), formato (black), typing (mypy/pyright), seguridad (bandit), y cobertura >=80%.

Nota: Este journal resume los puntos clave; los logs de diálogo completos no están disponibles en este artefacto.
