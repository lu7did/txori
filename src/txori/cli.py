#*--------------------------------------------------------------------------------------------------------*
#* txori
#* procesador de sonidos para ham radio sport contesting
#*
#* (c) Dr. Pedro E. Colla (LT7D/LU7DZ) 2009,2025
#*
#* Free for radioamateur uses - Commercial use requires licence
#*
#*--------------------------------------------------------------------------------------------------------*
"""CLI para el procesador de sonidos y espectrograma en tiempo real."""
from __future__ import annotations

import argparse
import queue
import time
from typing import Any, cast
import numpy as np

from .sources import FileSource, ToneSource, LineSource, Source
from .cpu import (
    NoOpProcessor,
    Processor,
    LpfProcessor,
    SkimmerProcessor,
    SkimmerWithBpf,
    BandPassProcessor,
    ChainProcessor,
)
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
    p.add_argument(
        "--source",
        choices=["file", "tone", "line"],
        required=True,
        help="Tipo de fuente (file|tone|line)",
    )
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
        choices=["none", "lpf", "skimmer", "bypass"],
        default="none",
        help="Procesador a aplicar (none|lpf|skimmer|bypass)",
    )
    p.add_argument(
        "--cpu-lpf-freq",
        type=float,
        default=2000.0,
        help=(
            "Frecuencia de corte (Hz) para --cpu lpf; "
            "el diezmado resultante será 2*fc (default: 2000)"
        ),
    )
    p.add_argument(
        "--cwfilter",
        action="store_true",
        help="Con --cpu lpf, aplica además un pasabanda CW (f0/BW configurables)",
    )
    p.add_argument(
        "--cpu-bpf-freq",
        type=float,
        default=600.0,
        help="Frecuencia central (Hz) del pasabanda CW (default: 600)",
    )
    p.add_argument(
        "--cpu-bpf-bw",
        type=float,
        default=200.0,
        help="Ancho de banda (Hz) del pasabanda CW (default: 200)",
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
        "--fft-fpu",
        type=int,
        default=1,
        help="Frames por actualización del waterfall (default: 1)",
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
        "--spkr-device",
        type=int,
        help="Índice del dispositivo de salida (sd.query_devices())",
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
    p.add_argument(
        "--line",
        dest="line_device",
        type=int,
        help="Indice del dispositivo de entrada para --source line (ignorado si no es line)",
    )
    # Para --cpu skimmer: habilita BPF 600Hz/200Hz post-diezmado
    p.add_argument(
        "--skimmer",
        action="store_true",
        help="Con --cpu skimmer agrega BPF centrado en 600 Hz BW=200 Hz tras el diezmado",
    )
    return p


def _make_source(
    kind: str, infile: str | None, tone_freq: float, tone_fsr: int, line_dev: int | None = None
) -> Source:
    if kind == "file":
        if not infile:
            raise SystemExit("--in es obligatorio cuando --source file")
        return FileSource(infile)
    if kind == "tone":
        return ToneSource(freq_hz=tone_freq, fs=tone_fsr)
    if kind == "line":
        try:
            return LineSource(device=line_dev)
        except RuntimeError as e:
            print(str(e))
            raise SystemExit(1)
    raise SystemExit(f"Fuente no soportada: {kind}")


def _make_cpu(
    kind: str,
    fs: int | None = None,
    lpf_fc: float | None = None,
    cwfilter: bool = False,
    bpf_f0: float = 600.0,
    bpf_bw: float = 200.0,
    skimmer_bpf: bool = False,
) -> Processor:
    if kind in ("none", "noop"):
        return NoOpProcessor()
    if kind == "lpf":
        if fs is None:
            raise SystemExit("CPU lpf requiere conocer el sample rate de la fuente")
        base = LpfProcessor(fs_in=int(fs), cutoff_hz=float(lpf_fc or 2000.0))
        if cwfilter:
            fs_bpf = int(2 * float(lpf_fc or 2000.0)) if int(fs) > 4000 else int(fs)
            bpf = BandPassProcessor(
                fs=fs_bpf, center_hz=float(bpf_f0), bw_hz=float(bpf_bw)
            )
            return ChainProcessor([base, bpf])
        return base
    if kind == "skimmer":
        if fs is None:
            raise SystemExit("CPU skimmer requiere conocer el sample rate de la fuente")
        if skimmer_bpf:
            return SkimmerWithBpf(fs_in=int(fs))
        return SkimmerProcessor(fs_in=int(fs))
    raise SystemExit(f"CPU no soportada: {kind}")


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada principal."""
    args = build_parser().parse_args(argv)
    src = _make_source(args.source, args.infile, args.tone_freq, args.tone_fsr, getattr(args, "line_device", None))

    # CPU bypass: enviar fuente a salida predeterminada, sin waterfall ni time plot
    if args.cpu == "bypass":
        try:
            import sounddevice as sd  # type: ignore
        except Exception:
            print("Audio backend no disponible para bypass")
            src.close()
            return 1
        # Reproduce toda la fuente directamente al dispositivo usando su Fs
        sr = int(getattr(src, "sample_rate", 48000)) or 48000
        try:
            data_chunks = []
            chunk = max(2048, sr // 10)
            while True:
                x = src.read(chunk)
                if x.size == 0:
                    break
                data_chunks.append(x.astype(np.float32))
            if not data_chunks:
                print("Fuente vacía")
                src.close()
                return 0
            data = np.concatenate(data_chunks)
            # Si se indicó dispositivo, configurarlo como salida por defecto
            dev = getattr(args, "spkr_device", None)
            if dev is not None:
                try:
                    sd.default.device = (None, int(dev))  # type: ignore[assignment]
                except Exception:
                    pass
            sd.play(data, sr)
            sd.wait()
        except KeyboardInterrupt:
            pass
        finally:
            src.close()
        return 0

    cpu = _make_cpu(
        args.cpu,
        fs=src.sample_rate,
        lpf_fc=getattr(args, "cpu_lpf_freq", 2000.0),
        cwfilter=bool(getattr(args, "cwfilter", False)),
        bpf_f0=getattr(args, "cpu_bpf_freq", 600.0),
        bpf_bw=getattr(args, "cpu_bpf_bw", 200.0),
        skimmer_bpf=bool(getattr(args, "skimmer", False)),
    )

    nfft = int(args.fft_nfft)
    overlap = (
        int(args.fft_overlap)
        if getattr(args, "fft_overlap", None) is not None
        else max(0, nfft - 56)
    )
    overlap = min(max(overlap, 0), nfft - 1)
    hop = max(1, nfft - overlap)
    pixels = 4096 if getattr(args, "wide", False) else int(args.fft_pixels)
    # Ajustar Fs del waterfall según CPU seleccionada
    if args.cpu == "lpf" and src.sample_rate > 4000:
        anim_fs = int(2 * args.cpu_lpf_freq)
    elif args.cpu == "skimmer" and src.sample_rate > 8000:
        anim_fs = 8000
    else:
        anim_fs = src.sample_rate
    animator = SpectrogramAnimator(
        fs=anim_fs,
        nfft=nfft,
        hop=hop,
        frames_per_update=int(getattr(args, "fft_fpu", 1)),
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
        import sounddevice as _sd
    except Exception:
        _sd = None
    try:
        waterfall_mod.sd = _sd
        # Pasar dispositivo de salida elegido al módulo waterfall
        waterfall_mod._SPKR_DEVICE = getattr(args, "spkr_device", None)
    except Exception:  # nosec B110 - optional audio backend assignment, safe to ignore
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
