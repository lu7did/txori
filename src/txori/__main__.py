"""CLI mínima para ejecutar la canalización."""

from __future__ import annotations

import argparse

from .config import SystemConfig
from .pipeline import Pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Txori: espectrograma + traza de intensidad en vivo (tiempo casi-real)")
    parser.add_argument("--audio", action="store_true", help="Usar entrada de audio real")
    parser.add_argument("--seconds", type=float, default=None, help="Duración de la ejecución; si no se indica, corre indefinidamente")
    parser.add_argument("--forever", action="store_true", help="Ejecutar indefinidamente hasta interrupción (Ctrl+C)")
    parser.add_argument("--out", type=str, default=None, help="Archivo de salida PNG (opcional)")
    parser.add_argument("--titulo", type=str, default=None, help="Texto a mostrar como título externo")
    parser.add_argument("--width", type=int, default=None, help="Ancho del espectrograma en píxeles (default 1200)")
    parser.add_argument("--cutoff", type=float, default=None, help="Frecuencia de corte del pasabajos en Hz (default 3000)")
    parser.add_argument("--bin", type=float, default=None, help="Ancho de bin FFT en Hz (default 3.0)")
    parser.add_argument("--avg-samples", type=int, default=None, help="Muestras diezmadas a promediar por columna (default 15)")  # promedio para traza superior y espectrograma
    args = parser.parse_args()

    cfg = SystemConfig(
        use_audio=bool(args.audio),
        image_width=(args.width if args.width is not None else 1200),
        cutoff_hz=(args.cutoff if args.cutoff is not None else 3000.0),
        fft_bin_hz=(args.bin if args.bin is not None else 3.0),
        samples_per_col=(args.avg_samples if args.avg_samples is not None else 15),
    )
    pipe = Pipeline(cfg)
    seconds = None if (args.forever or args.seconds is None) else float(args.seconds)
    if args.out:
        if seconds is None:
            raise SystemExit("--out no es compatible con --forever; use un tiempo finito")
        pipe.run(seconds=seconds)
        pipe.renderer.save(args.out)
    else:
        from .live import LiveViewer  # import perezoso

        decim_rate = cfg.sample_rate // max(1, cfg.sample_rate // 6000)
        seconds_per_col = float(cfg.samples_per_col) / float(decim_rate)
        viewer = LiveViewer(
            max_freq_hz=float(cfg.cutoff_hz),
            bin_hz=float(cfg.fft_bin_hz),
            seconds_per_col=seconds_per_col,
            title_text=args.titulo,
            device_text=getattr(pipe, "source_label", None),
        )
        try:
            pipe.run(seconds=seconds, on_frame=viewer.update)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":  # pragma: no cover
    main()
