"""CLI para capturar audio y mostrar un gráfico waterfall."""
from __future__ import annotations

import argparse
import sys

from . import __build__, __version__
from .audio import AudioError, DefaultAudioSource, StreamAudioSource
from .waterfall import WaterfallComputer, WaterfallRenderer, WaterfallLive


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Captura audio desde la entrada predeterminada y muestra un waterfall.",
    )
    parser.add_argument("--dur", type=float, default=5.0, help="Duración de captura en segundos.")
    parser.add_argument("--rate", type=int, default=48_000, help="Frecuencia de muestreo (Hz).")
    parser.add_argument("--nfft", type=int, default=1024, help="Tamaño de la FFT.")
    parser.add_argument(
        "--overlap", type=float, default=0.5, help="Traslape entre ventanas en [0,1)."
    )
    parser.add_argument("--cmap", type=str, default="viridis", help="Colormap de Matplotlib.")
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Visualización continua hasta interrupción (Ctrl+C).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=400,
        help="Cantidad de filas visibles en vivo (buffer rodante).",
    )
    return parser.parse_args()


def main() -> None:
    """Punto de entrada del ejecutable ``txori-waterfall``."""
    args = _parse_args()
    print(f"txori {__version__} build {__build__}")
    try:
        if args.continuous:
            # tamaño de bloque acorde al paso para actualizar por frame
            step = int(args.nfft * (1 - args.overlap)) or 1
            stream = StreamAudioSource(sample_rate=args.rate, channels=1, blocksize=step)
            live = WaterfallLive(
                nfft=args.nfft,
                overlap=args.overlap,
                cmap=args.cmap,
                max_frames=args.max_frames,
            )
            live.run(stream.blocks(), sample_rate=args.rate)
        else:
            source = DefaultAudioSource(sample_rate=args.rate, channels=1)
            data = source.record(args.dur)
            comp = WaterfallComputer(nfft=args.nfft, overlap=args.overlap)
            spec = comp.compute(data)
            WaterfallRenderer(cmap=args.cmap).show(spec, args.rate, args.nfft)
    except (AudioError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
