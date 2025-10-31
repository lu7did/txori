"""CLI para el procesador de sonidos y espectrograma en tiempo real."""
from __future__ import annotations

import argparse
import sys

from .sources import FileSource, Source
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
    p.add_argument("--source", choices=["file"], required=True, help="Tipo de fuente")
    p.add_argument(
        "--in",
        dest="infile",
        type=str,
        help="Ruta al archivo .WAV cuando --source=file",
    )
    p.add_argument(
        "--cpu",
        choices=["none"],
        default="none",
        help="Procesador a aplicar (por defecto: none)",
    )
    return p


def _make_source(kind: str, infile: str | None) -> Source:
    if kind == "file":
        if not infile:
            raise SystemExit("--in es obligatorio cuando --source file")
        return FileSource(infile)
    raise SystemExit(f"Fuente no soportada: {kind}")


def _make_cpu(kind: str) -> Processor:
    if kind in ("none", "noop"):
        return NoOpProcessor()
    raise SystemExit(f"CPU no soportada: {kind}")


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada principal."""
    args = build_parser().parse_args(argv)
    src = _make_source(args.source, args.infile)
    cpu = _make_cpu(args.cpu)

    animator = SpectrogramAnimator(
        fs=src.sample_rate, nfft=256, hop=56, frames_per_update=4, width_cols=400
    )
    try:
        animator.run(src, cpu)
    finally:
        src.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
