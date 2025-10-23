"""CLI mínima para ejecutar la canalización."""

from __future__ import annotations

import argparse

from .config import SystemConfig
from .pipeline import Pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Txori: espectrograma en tiempo casi-real")
    parser.add_argument("--audio", action="store_true", help="Usar entrada de audio real")
    parser.add_argument("--seconds", type=float, default=2.0, help="Duración de la ejecución")
    parser.add_argument("--forever", action="store_true", help="Ejecutar indefinidamente hasta interrupción (Ctrl+C)")
    parser.add_argument("--out", type=str, default=None, help="Archivo de salida PNG (opcional)")
    args = parser.parse_args()

    cfg = SystemConfig(use_audio=bool(args.audio))
    pipe = Pipeline(cfg)
    seconds = None if args.forever else float(args.seconds)
    if args.out:
        if seconds is None:
            raise SystemExit("--out no es compatible con --forever; use un tiempo finito")
        pipe.run(seconds=seconds)
        pipe.renderer.save(args.out)
    else:
        from .live import LiveViewer  # import perezoso

        viewer = LiveViewer(max_freq_hz=float(cfg.cutoff_hz), bin_hz=float(cfg.fft_bin_hz))
        try:
            pipe.run(seconds=seconds, on_frame=viewer.update)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":  # pragma: no cover
    main()
