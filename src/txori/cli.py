"""CLI para capturar audio y mostrar un gráfico waterfall."""
from __future__ import annotations

import argparse
import sys

import numpy as np
import sounddevice as sd
import time
from collections import deque

from . import __build__, __version__
from .audio import AudioError, DefaultAudioSource, StreamAudioSource
from .waterfall import WaterfallComputer, WaterfallRenderer, WaterfallLive


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Captura audio desde la entrada predeterminada y muestra un waterfall.",
    )
    parser.add_argument("--dur", type=float, default=None, help="Duración en segundos (si se especifica, desactiva el modo continuo).")
    parser.add_argument("--rate", type=int, default=48_000, help="Frecuencia de muestreo (Hz).")
    parser.add_argument("--nfft", type=int, default=4096, help="Tamaño de la FFT (por defecto 4096; 2048 reduce costo computacional).")
    parser.add_argument(
        "--overlap", type=float, default=0.5, help="Traslape entre ventanas en [0,1)."
    )
    parser.add_argument("--cmap", type=str, default="viridis", help="Colormap de Matplotlib.")
    parser.add_argument(
        "--window",
        type=str,
        choices=["hann", "blackman", "blackmanharris"],
        default="blackman",
        help="Ventana de análisis del espectrograma (por defecto blackman).",
    )
    parser.add_argument("--vol", type=float, default=60.0, help="Volumen de salida 0-100% (tone/cw). Por defecto 60%.")
    parser.add_argument(
        "--continuous",
        action="store_true",
        default=True,
        help="Visualización continua hasta interrupción (Ctrl+C). Por defecto es continuo salvo que se indique --dur.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=400,
        help="Cantidad de filas visibles en vivo (buffer rodante).",
    )
    parser.add_argument(
        "--source",
        choices=["stream", "tone", "cw"],
        default="stream",
        help="Fuente de datos: stream=entrada de audio, tone=generador 600 Hz, cw=Morse 600 Hz.",
    )
    parser.add_argument(
        "--spkr",
        action="store_true",
        help="Emitir tonos por parlante (solo con --source tone).",
    )
    parser.add_argument(
        "--time",
        action="store_true",
        help="Mostrar timeplot en vivo.",
    )
    parser.add_argument("--hop", type=int, default=None, help="Paso entre frames en muestras (overrides --overlap si se especifica).")
    parser.add_argument("--row-median", action="store_true", help="Restar mediana por fila para mejorar contraste.")
    parser.add_argument("--db-range", type=float, default=None, help="Rango dinámico en dB para el colormap (p.ej. 80).")
    return parser.parse_args()


def _filter_kwargs(callable_obj, kwargs: dict) -> dict:
    try:
        sig = inspect.signature(callable_obj)
        params = sig.parameters
        return {k: v for k, v in kwargs.items() if k in params}
    except Exception:
        return kwargs


def main() -> None:
    """Punto de entrada del ejecutable ``txori-waterfall``."""
    args = _parse_args()
    print(f"txori {__version__} build {__build__}")
    amp = max(0.0, min(1.0, float(getattr(args, "vol", 60.0)) / 100.0))
    try:
        # Modo continuo por defecto salvo que se especifique --dur
        if args.dur is None and args.continuous:
            # tamaño de bloque acorde al paso para actualizar por frame
            step = int(args.nfft * (1 - args.overlap)) or 1
            if args.source == "stream":
                if args.spkr:
                    def _speaker_stream_blocks():
                        buf = deque(maxlen=200)
                        def _cb(indata, outdata, frames, t, status):  # noqa: ANN001
                            outdata[:, 0] = indata[:, 0]
                            try:
                                buf.append(indata.copy().reshape(-1))
                            except Exception:
                                pass
                        with sd.Stream(
                            samplerate=args.rate,
                            channels=1,
                            dtype="float32",
                            blocksize=step,
                            callback=_cb,
                        ):
                            while True:
                                if buf:
                                    yield buf.popleft()
                                else:
                                    time.sleep(0.001)
                    blocks = _speaker_stream_blocks()
                else:
                    stream = StreamAudioSource(sample_rate=args.rate, channels=1, blocksize=step)
                    blocks = (amp * b for b in stream.blocks())
            elif args.source == "tone":
                from .audio import ToneAudioSource
                tone = ToneAudioSource(sample_rate=args.rate, blocksize=step)
                amp = max(0.0, min(1.0, float(getattr(args, "vol", 60.0)) / 100.0))
                out = None
                ring = deque(maxlen=50)

                def _blocks_and_spkr(src_iter):
                    for b in src_iter:
                        bb = (amp * b).astype(np.float32, copy=False)
                        if args.spkr:
                            ring.append(bb.copy())
                        yield bb

                if args.spkr:
                    def _cb(outdata, frames, t, status):  # noqa: ANN001
                        try:
                            blk = ring.popleft()
                        except Exception:
                            blk = np.zeros(frames, dtype=np.float32)
                        if blk.shape[0] < frames:
                            tmp = np.zeros(frames, dtype=np.float32)
                            tmp[: blk.shape[0]] = blk
                            blk = tmp
                        outdata[:, 0] = blk[:frames]

                    out = sd.OutputStream(
                        samplerate=args.rate,
                        channels=1,
                        dtype="float32",
                        blocksize=step,
                        latency="high",
                        callback=_cb,
                    )
                    out.start()
                blocks = _blocks_and_spkr(tone.blocks())
            elif args.source == "cw":
                from .audio import MorseAudioSource
                cw = MorseAudioSource(sample_rate=args.rate, frequency=600.0, wpm=20.0, message="LU7DZ TEST     ", blocksize=step)
                amp = max(0.0, min(1.0, float(getattr(args, "vol", 60.0)) / 100.0))
                out = None
                ring = deque(maxlen=50)

                def _blocks_and_spkr(src_iter):
                    for b in src_iter:
                        bb = (amp * b).astype(np.float32, copy=False)
                        if args.spkr:
                            ring.append(bb.copy())
                        yield bb

                if args.spkr:
                    def _cb(outdata, frames, t, status):  # noqa: ANN001
                        try:
                            blk = ring.popleft()
                        except Exception:
                            blk = np.zeros(frames, dtype=np.float32)
                        if blk.shape[0] < frames:
                            tmp = np.zeros(frames, dtype=np.float32)
                            tmp[: blk.shape[0]] = blk
                            blk = tmp
                        outdata[:, 0] = blk[:frames]

                    out = sd.OutputStream(
                        samplerate=args.rate,
                        channels=1,
                        dtype="float32",
                        blocksize=step,
                        latency="high",
                        callback=_cb,
                    )
                    out.start()
                blocks = _blocks_and_spkr(cw.blocks())
            # Construcción compatible hacia atrás: setear atributos opcionales si existen
            live = WaterfallLive(
                nfft=args.nfft,
                overlap=args.overlap,
                cmap=args.cmap,
                max_frames=args.max_frames,
                enable_timeplot=getattr(args, "time", False),
                window=args.window,
            )
            if hasattr(live, "hop"):
                setattr(live, "hop", getattr(args, "hop", None))
            if hasattr(live, "row_median"):
                setattr(live, "row_median", getattr(args, "row_median", False))
            if hasattr(live, "db_range"):
                setattr(live, "db_range", getattr(args, "db_range", None))
            try:
                live.run(blocks, sample_rate=args.rate)
            finally:
                try:
                    if 'out' in locals() and out is not None:
                        out.stop()
                        out.close()
                except Exception:
                    pass
        else:
            comp = WaterfallComputer(nfft=args.nfft, overlap=args.overlap, window=args.window)
            if hasattr(comp, "hop"):
                setattr(comp, "hop", getattr(args, "hop", None))
            if hasattr(comp, "row_median"):
                setattr(comp, "row_median", getattr(args, "row_median", False))
            if args.source == "stream":
                source = DefaultAudioSource(sample_rate=args.rate, channels=1)
                data = source.record(args.dur)
                if args.spkr:
                    sd.play(data.reshape(-1, 1), samplerate=args.rate, blocking=False)
            elif args.source == "tone":
                from .audio import ToneAudioSource
                data = ToneAudioSource(sample_rate=args.rate).record(args.dur)
                if args.spkr:
                    amp = max(0.0, min(1.0, float(getattr(args, "vol", 60.0)) / 100.0))
                    sd.play((amp * data).reshape(-1, 1), samplerate=args.rate, blocking=False)
            elif args.source == "cw":
                from .audio import MorseAudioSource
                data = MorseAudioSource(sample_rate=args.rate, frequency=600.0, wpm=20.0, message="LU7DZ TEST     ").record(args.dur)
                if args.spkr:
                    amp = max(0.0, min(1.0, float(getattr(args, "vol", 60.0)) / 100.0))
                    sd.play((amp * data).reshape(-1, 1), samplerate=args.rate, blocking=False)
            spec = comp.compute(data)
            WaterfallRenderer(cmap=args.cmap).show(
                spec, args.rate, args.nfft, args.overlap
            )
    except (AudioError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
