"""CLI para el procesador de sonidos y espectrograma en tiempo real."""
from __future__ import annotations

import argparse
import sys

from .sources import FileSource, ToneSource, Source
from .cpu import NoOpProcessor, Processor, LpfProcessor
from .waterfall import SpectrogramAnimator
from . import waterfall as waterfall_mod


def build_parser() -> argparse.ArgumentParser:
    """Crea el parser de argumentos."""
    p = argparse.ArgumentParser(
        prog="txori-waterfall",
        description=(
            "Procesa una fuente de audio (--source) con un CPU (--cpu) y "
            "muestra un espectrograma en tiempo real."
        ),
    )
    p.add_argument("--source", choices=["file", "tone"], required=True, help="Tipo de fuente")
    p.add_argument(
        "--in",
        dest="infile",
        type=str,
        help="Ruta al archivo .WAV cuando --source=file",
    )
    p.add_argument(
        "--tone-freq",
        type=float,
        default=600.0,
        help="Frecuencia del tono (Hz) cuando --source=tone (default: 600)",
    )
    p.add_argument(
        "--tone-fsr",
        type=int,
        default=4000,
        help="Sample rate (Hz) del tono cuando --source=tone (default: 4000)",
    )
    p.add_argument(
        "--cpu",
        choices=["none", "lpf"],
        default="none",
        help="Procesador a aplicar (none|lpf)",
    )
    p.add_argument(
        "--cpu-lpf-freq",
        type=float,
        default=2000.0,
        help="Frecuencia de corte (Hz) para --cpu lpf; el diezmado resultante será 2*fc (default: 2000)",
    )
    p.add_argument(
        "--fft-window",
        type=str,
        choices=[
            "Blackman",
            "BlackmanHarris",
            "FlatTop",
            "Hamming",
            "Hanning",
            "Rectangular",
        ],
        default="Blackman",
        help="Función de ventana para reducir fuga espectral",
    )
    p.add_argument(
        "--fft-nfft",
        type=int,
        default=256,
        help="Tamaño de la FFT (NFFT)",
    )
    p.add_argument(
        "--fft-overlap",
        type=int,
        help="Traslape (noverlap) en muestras; por defecto NFFT-56",
    )
    p.add_argument(
        "--fft-cmap",
        type=str,
        default="ocean",
        help="Colormap para el espectrograma (default: ocean)",
    )
    p.add_argument(
        "--fft-pixels",
        type=int,
        default=640,
        help="Ancho del espectrograma en píxeles (default: 640)",
    )
    p.add_argument(
        "--wide",
        action="store_true",
        help="Usa 4096 píxeles horizontales (mantiene los píxeles verticales)",
    )
    p.add_argument(
        "--fft-ema",
        type=float,
        help="Factor EMA (0<ema<1) para suavizar el espectro en tiempo",
    )
    p.add_argument(
        "--vmin",
        type=float,
        help="Escala dB mínima para el colormap",
    )
    p.add_argument(
        "--vmax",
        type=float,
        help="Escala dB máxima para el colormap",
    )
    p.add_argument(
        "--spkr",
        action="store_true",
        help="Reproducir las muestras de la fuente por la salida de audio",
    )
    p.add_argument(
        "--time",
        action="store_true",
        help="Mostrar un gráfico de tiempo en ventana separada (misma fuente y Fs)",
    )
    p.add_argument(
        "--time-scale",
        type=float,
        default=0.5,
        help="Factor (0<scale<=1) para reducir la ventana temporal del time plot",
    )
    return p


def _make_source(kind: str, infile: str | None, tone_freq: float, tone_fsr: int) -> Source:
    if kind == "file":
        if not infile:
            raise SystemExit("--in es obligatorio cuando --source file")
        return FileSource(infile)
    if kind == "tone":
        return ToneSource(freq_hz=tone_freq, fs=tone_fsr)
    raise SystemExit(f"Fuente no soportada: {kind}")


def _make_cpu(kind: str, fs: int | None = None, lpf_fc: float | None = None) -> Processor:
    if kind in ("none", "noop"):
        return NoOpProcessor()
    if kind == "lpf":
        if fs is None:
            raise SystemExit("CPU lpf requiere conocer el sample rate de la fuente")
        return LpfProcessor(fs_in=int(fs), cutoff_hz=float(lpf_fc or 2000.0))
    raise SystemExit(f"CPU no soportada: {kind}")


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada principal."""
    args = build_parser().parse_args(argv)
    src = _make_source(args.source, args.infile, args.tone_freq, args.tone_fsr)
    cpu = _make_cpu(args.cpu, fs=src.sample_rate, lpf_fc=getattr(args, "cpu_lpf_freq", 2000.0))

    nfft = int(args.fft_nfft)
    overlap = int(args.fft_overlap) if getattr(args, "fft_overlap", None) is not None else max(0, nfft - 56)
    overlap = min(max(overlap, 0), nfft - 1)
    hop = max(1, nfft - overlap)
    pixels = 4096 if getattr(args, "wide", False) else int(args.fft_pixels)
    # Ajustar Fs del waterfall si CPU lpf aplica diezmado a 2*fc cuando Fs>4000
    anim_fs = int(2 * args.cpu_lpf_freq) if args.cpu == "lpf" and src.sample_rate > 4000 else src.sample_rate
    animator = SpectrogramAnimator(
        fs=anim_fs,
        nfft=nfft,
        hop=hop,
        frames_per_update=4,
        width_cols=400,
        fft_window=args.fft_window,
        cmap=args.fft_cmap,
        pixels=pixels,
        fft_ema=args.fft_ema,
        vmin=args.vmin,
        vmax=args.vmax,
    )
    # Exponer opcionalmente el backend de audio al módulo waterfall sin modificarlo
    try:
        import sounddevice as _sd  # type: ignore
    except Exception:
        _sd = None  # type: ignore
    try:
        waterfall_mod.sd = _sd  # type: ignore[attr-defined]
    except Exception:
        pass

    try:
        animator.run(
            src,
            cpu,
            spkr=bool(getattr(args, "spkr", False)),
            time_plot=bool(getattr(args, "time", False)),
            time_scale=float(getattr(args, "time_scale", 1.0)),
        )
    except KeyboardInterrupt:
        print("Programa terminado por el usuario")
    finally:
        src.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
