#!/usr/bin/env python3
"""Simple CLI to demonstrate AudioSource usage.

If --audio is specified, it captures from default audio input and forwards
128-frame blocks to a placeholder handler.
"""
from __future__ import annotations

import argparse
import sys
from typing import NoReturn

from txori.audio import AudioSource


def _print_handler(data: bytes, sample_width: int, channels: int, rate: int) -> None:
    # Placeholder handler: print basic info without flooding output
    frames = len(data) // (sample_width * channels)
    sys.stdout.write(f"frames={frames} ch={channels} sw={sample_width} rate={rate}\n")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> NoReturn:
    parser = argparse.ArgumentParser(description="Txori audio demo")
    parser.add_argument(
        "--audio",
        action="store_true",
        help="Activar captura desde la fuente de sonido predeterminada",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=3.0,
        help="Duración de captura en segundos (por defecto 3.0)",
    )
    args = parser.parse_args(argv)

    if not args.audio:
        print("--audio no especificado; no se inicia la fuente de audio")
        raise SystemExit(0)

    src = AudioSource(handler=_print_handler, frames_per_buffer=128)
    with src:
        src.run(seconds=args.seconds)
    raise SystemExit(0)


if __name__ == "__main__":  # pragma: no cover
    main()
