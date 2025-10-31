"""CLI para el procesador de sonidos y espectrograma en tiempo real."""
from __future__ import annotations

import argparse
import sys

from .sources import FileSource, ToneSource, Source
from .cpu import NoOpProcessor, Processor
from .waterfall import SpectrogramAnimator


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
        choices=["none"],
        default="none",
        help="Procesador a aplicar (por defecto: none)",
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
    return p


def _make_source(kind: str, infile: str | None, tone_freq: float, tone_fsr: int) -> Source:
    if kind == "file":
        if not infile:
            raise SystemExit("--in es obligatorio cuando --source file")
        return FileSource(infile)
    if kind == "tone":
        return ToneSource(freq_hz=tone_freq, fs=tone_fsr)
    raise SystemExit(f"Fuente no soportada: {kind}")


def _make_cpu(kind: str) -> Processor:
    if kind in ("none", "noop"):
        return NoOpProcessor()
    raise SystemExit(f"CPU no soportada: {kind}")


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada principal."""
    args = build_parser().parse_args(argv)
    src = _make_source(args.source, args.infile, args.tone_freq, args.tone_fsr)
    cpu = _make_cpu(args.cpu)

    nfft = int(args.fft_nfft)
    overlap = int(args.fft_overlap) if getattr(args, "fft_overlap", None) is not None else max(0, nfft - 56)
    overlap = min(max(overlap, 0), nfft - 1)
    hop = max(1, nfft - overlap)
    pixels = 4096 if getattr(args, "wide", False) else int(args.fft_pixels)
    animator = SpectrogramAnimator(
        fs=src.sample_rate,
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
    try:
        animator.run(src, cpu)
    except KeyboardInterrupt:
        print("Programa terminado por el usuario")
    finally:
        src.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
