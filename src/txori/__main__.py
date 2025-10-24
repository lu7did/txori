"""
CLI mínima para ejecutar la canalización.

(c) Dr. Pedro E. Colla 2020-2025 (LU7DZ).
"""

from __future__ import annotations

import argparse

from .config import SystemConfig
from .pipeline import Pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Txori: espectrograma + traza de intensidad en vivo (tiempo casi-real)"
    )
    parser.add_argument(
        "--audio", action="store_true", help="Usar entrada de audio real"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Usar generador interno de tono senoidal (por defecto 1000 Hz; ver --tone)",
    )
    parser.add_argument(
        "--tone",
        type=float,
        default=None,
        help="Frecuencia en Hz para el tono de test (requiere --test)",
    )
    parser.add_argument(
        "--cw",
        action="store_true",
        help=(
            "Usar generador CW: tono 600 Hz conmutado 57 ms ON/OFF; ignora --audio y --tone"
        ),
    )
    parser.add_argument(
        "--cw-tone",
        dest="cw_tone",
        type=float,
        default=None,
        help="Frecuencia del tono CW (default 600 Hz)",
    )
    parser.add_argument(
        "--time",
        action="store_true",
        help="Mostrar ventana separada con la señal temporal sin procesar",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="Duración de la ejecución; si no se indica, corre indefinidamente",
    )
    parser.add_argument(
        "--forever",
        action="store_true",
        help="Ejecutar indefinidamente hasta interrupción (Ctrl+C)",
    )
    parser.add_argument(
        "--out", type=str, default=None, help="Archivo de salida PNG (opcional)"
    )
    parser.add_argument(
        "--titulo", type=str, default=None, help="Texto a mostrar como título externo"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Ancho del espectrograma en píxeles (default 1200)",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=None,
        help="Frecuencia de corte del pasabajos en Hz (default 3000)",
    )
    parser.add_argument(
        "--bin", type=float, default=None, help="Ancho de bin FFT en Hz (default 6.0)"
    )
    parser.add_argument(
        "--spec-speed",
        type=float,
        default=None,
        help="Factor de velocidad del espectrograma relativo al tiempo (default 1.0)",
    )
    parser.add_argument(
        "--avg-samples",
        type=int,
        default=None,
        help="Muestras diezmadas a promediar por columna (default 15)",
    )  # promedio para traza superior y espectrograma
    parser.add_argument(
        "--time-speed",
        type=float,
        default=None,
        help="Factor de velocidad horizontal del gráfico de tiempo (por defecto 86.4)",
    )
    parser.add_argument(
        "--time-color",
        type=str,
        default=None,
        help="Color de la línea de tiempo (por defecto 'skyblue')",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Modo directo: FFT sobre muestras sin filtrar/decimar (default)",
    )
    parser.add_argument(
        "--dsp",
        action="store_true",
        help="Modo DSP: LPF 3 kHz + decimación 8:1 + AGC + DSP antes de FFT",
    )
    args = parser.parse_args()

    # Resolver modo: por defecto DSP; si ambos, prioriza DIRECT
    use_direct = True if args.direct else False

    cfg = SystemConfig(
        use_audio=bool(args.audio) and not bool(args.test) and not bool(args.cw),
        image_width=(args.width if args.width is not None else 1200),
        cutoff_hz=(args.cutoff if args.cutoff is not None else 3000.0),
        fft_bin_hz=(args.bin if args.bin is not None else 6.0),
        samples_per_col=(args.avg_samples if args.avg_samples is not None else 15),
        test_tone_hz=(args.tone if args.tone is not None else 1000.0),
        cw_mode=bool(args.cw),
        cw_tone_hz=(args.cw_tone if args.cw_tone is not None else 600.0),
        direct_mode=bool(use_direct),
        spec_speed_factor=(args.spec_speed if args.spec_speed is not None else 1.0),
    )
    pipe = Pipeline(cfg)
    seconds = None if (args.forever or args.seconds is None) else float(args.seconds)
    if args.out:
        if seconds is None:
            raise SystemExit(
                "--out no es compatible con --forever; use un tiempo finito"
            )
        pipe.run(seconds=seconds)
        pipe.renderer.save(args.out)
    else:
        from .live import LiveViewer, TimeViewer  # import perezoso

        if cfg.direct_mode:
            decim_rate = int(cfg.sample_rate)
            decim_factor = 1
        else:
            decim_factor = max(1, int(cfg.sample_rate // 6000))
            decim_rate = int(cfg.sample_rate // decim_factor)
        eff_spc = max(1, int(round(cfg.samples_per_col / max(cfg.spec_speed_factor, 1e-9))))
        seconds_per_col = float(eff_spc) / float(decim_rate)
        spp_target = int(decim_factor) * int(eff_spc)
        viewer = LiveViewer(
            max_freq_hz=float(cfg.cutoff_hz),
            bin_hz=float(cfg.fft_bin_hz),
            seconds_per_col=seconds_per_col,
            title_text=args.titulo,
            device_text=getattr(pipe, "source_label", None),
        )
        time_viewer = (
            TimeViewer(
                sample_rate=cfg.sample_rate,
                span_seconds=(seconds if seconds is not None else 30.0),
                speed_factor=(args.time_speed if args.time_speed is not None else 86.4),
                time_color=(args.time_color if args.time_color is not None else "skyblue"),
                sync_spp=spp_target,
            )
            if args.time
            else None
        )
        try:
            pipe.run(
                seconds=seconds,
                on_frame=viewer.update,
                on_time_sample=(time_viewer.push_sample if time_viewer else None),
                raw_input=True,
            )
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":  # pragma: no cover
    main()
