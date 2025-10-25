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
        "--cwbw",
        dest="cwbw",
        type=float,
        default=None,
        help="Semiancho de banda CW en Hz (default 20)",
    )
    parser.add_argument(
        "--qrn",
        action="store_true",
        help="Añade segunda portadora CW a 1000 Hz (solo con --cw)",
    )
    parser.add_argument(
        "--qrm",
        action="store_true",
        help="Añade QRM: portadoras CW en 200/400/800/1200 Hz (solo con --cw)",
    )
    parser.add_argument(
        "--noise",
        action="store_true",
        help="Añade ruido blanco en modo CW (default -60 dBFS; ver --noiselevel)",
    )
    parser.add_argument(
        "--noiselevel",
        type=float,
        default=None,
        help="Nivel relativo del ruido (dB por debajo del pico CW, default 20)",
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
        "--text",
        type=str,
        default=None,
        help="Título para el espectrómetro (solo en --dsp)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Ancho del espectrograma en píxeles (default 2400)",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=None,
        help="Frecuencia de corte del pasabajos en Hz (default 3000)",
    )
    parser.add_argument(
        "--bin", type=float, default=None, help="Ancho de bin FFT en Hz (default 10.0)"
    )
    parser.add_argument(
        "--att",
        type=float,
        default=None,
        help="Atenuación en dB previa a FFT (default -40; 0=sin atenuar)",
    )
    parser.add_argument(
        "--spec-speed",
        type=float,
        default=None,
        help="Factor de velocidad del espectrograma relativo al tiempo (default 4.0)",
    )
    parser.add_argument(
        "--ppb",
        type=int,
        default=None,
        help="Píxeles verticales por bin del espectrograma (default 4)",
    )
    parser.add_argument(
        "--pixbin",
        type=int,
        default=None,
        help="Píxeles por bin del espectrograma (default = actual*4, típicamente 16)",
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

    # Resolver pixeles por bin: --pixbin > --ppb > default actual*4
    _default_pixbin = SystemConfig().pixels_per_bin * 4
    _pixels_per_bin = (
        int(args.pixbin)
        if args.pixbin is not None
        else (int(args.ppb) if args.ppb is not None else int(_default_pixbin))
    )
    cfg = SystemConfig(
        use_audio=bool(args.audio) and not bool(args.test) and not bool(args.cw),
        image_width=(args.width if args.width is not None else 1200),
        cutoff_hz=(args.cutoff if args.cutoff is not None else 3000.0),
        fft_bin_hz=(args.bin if args.bin is not None else 10.0),
        samples_per_col=(args.avg_samples if args.avg_samples is not None else 15),
        test_tone_hz=(args.tone if args.tone is not None else 1000.0),
        cw_mode=bool(args.cw),
        cw_tone_hz=(args.cw_tone if args.cw_tone is not None else 600.0),
        direct_mode=bool(use_direct),
        spec_speed_factor=(args.spec_speed if args.spec_speed is not None else 4.0),
        att_db=(args.att if args.att is not None else None),
        pixels_per_bin=int(_pixels_per_bin),
    )

    # Propagar QRN dinámicamente
    cfg.qrn_mode = bool(getattr(args, "qrn", False))
    cfg.qrm_mode = bool(getattr(args, "qrm", False))
    cfg.noise_mode = bool(getattr(args, "noise", False))
    cfg.noise_level_db = float(args.noiselevel) if args.noiselevel is not None else 20.0
    pipe = Pipeline(cfg)
    seconds = None if (args.forever or args.seconds is None) else float(args.seconds)
    # Waterfall eliminado: no se genera ni guarda imagen
    from .live import TimeViewer, SpectrometerViewer  # import perezoso  # noqa: E402, I001

    # Configurar time viewer si se pide
    if cfg.direct_mode:
        decim_factor = 1
        decim_rate = int(cfg.sample_rate)
    else:
        decim_factor = max(1, int(cfg.sample_rate // 6000))
        decim_rate = int(cfg.sample_rate // decim_factor)
    # Espectrograma con librosa a 6000 SPS en modo --dsp
    dsp_spec = None
    spp_target = int(decim_factor)
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
    # Crear espectrómetro (área en navy) sólo en --dsp
    spectro_viewer = None
    if not cfg.direct_mode:
        spectro_viewer = SpectrometerViewer(
            title_text=(args.text or "Espectrómetro"),
            device_text=f"{getattr(pipe, 'source_label', 'Entrada')} @ {cfg.sample_rate} Hz, proc {decim_rate} SPS",
        )
        spectro_viewer.show()
        # Conectar DSP librosa para dibujar en el espectrómetro existente
        try:
            from .fft_live import DSPLibrosaSpectrogram
            dsp_spec = DSPLibrosaSpectrogram(
                sr=decim_rate,
                span_seconds=4.0,
                ext_ax=spectro_viewer.ax,
                device_name=getattr(pipe, 'source_label', 'Entrada'),
                decim_factor=int(decim_factor),
                user_text=(args.text or ''),
                device_sr=int(cfg.sample_rate),
                n_fft=2048,
                hop_length=(64 if bool(args.cw) else None),
                cw_mode=bool(args.cw),
                cw_center_hz=float(cfg.cw_tone_hz),
                cw_bw_hz=(float(args.cwbw) if args.cwbw is not None else 20.0),
                cw_extra_centers=(
                    ([1000.0] if bool(getattr(args, "qrn", False)) else [])
                    + ([200.0, 400.0, 800.0, 1200.0] if bool(getattr(args, "qrm", False)) else [])
                    or None
                ),
                noise_mode=bool(getattr(args, "noise", False)),
            )
            dsp_spec.show()
        except Exception:
            dsp_spec = None
    try:
        pipe.run(
            seconds=seconds,
            on_frame=None,
            on_time_sample=(time_viewer.push_sample if time_viewer else None),
            on_dsp_sample=(dsp_spec.push_sample if dsp_spec else None),
            raw_input=True,
        )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":  # pragma: no cover
    main()
