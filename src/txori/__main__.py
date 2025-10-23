"""CLI mínima para ejecutar la canalización."""

from __future__ import annotations

import argparse

from .config import SystemConfig
from .pipeline import Pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Txori: espectrograma en tiempo casi-real")
    parser.add_argument("--audio", action="store_true", help="Usar entrada de audio real")
    parser.add_argument("--seconds", type=float, default=2.0, help="Duración de la ejecución")
    parser.add_argument("--out", type=str, default="spectrogram.png", help="Archivo de salida PNG")
    args = parser.parse_args()

    cfg = SystemConfig(use_audio=bool(args.audio))
    pipe = Pipeline(cfg)
    pipe.run(seconds=float(args.seconds))
    pipe.renderer.save(args.out)


if __name__ == "__main__":  # pragma: no cover
    main()
