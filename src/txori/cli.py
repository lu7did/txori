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
    parser.add_argument("--nfft", type=int, default=1024, help="Tamaño de la FFT.")
    parser.add_argument(
        "--overlap", type=float, default=0.5, help="Traslape entre ventanas en [0,1)."
    )
    parser.add_argument("--cmap", type=str, default="viridis", help="Colormap de Matplotlib.")
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
    return parser.parse_args()


def main() -> None:
    """Punto de entrada del ejecutable ``txori-waterfall``."""
    args = _parse_args()
    print(f"txori {__version__} build {__build__}")
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
                    blocks = stream.blocks()
            elif args.source == "tone":
                from .audio import ToneAudioSource
                tone = ToneAudioSource(sample_rate=args.rate, blocksize=step)
                out = None
                if args.spkr:
                    phase = 0.0

                    def _cb(outdata, frames, t, status):  # noqa: ANN001
                        nonlocal phase
                        idx = np.arange(frames, dtype=np.float32)
                        tt = (phase + idx) / float(args.rate)
                        outdata[:, 0] = np.sin(2 * np.pi * 600.0 * tt).astype(np.float32)
                        phase += frames

                    out = sd.OutputStream(
                        samplerate=args.rate,
                        channels=1,
                        dtype="float32",
                        blocksize=step,
                        callback=_cb,
                    )
                    out.start()
                blocks = tone.blocks()
            elif args.source == "cw":
                from .audio import MorseAudioSource
                cw = MorseAudioSource(sample_rate=args.rate, frequency=600.0, wpm=20.0, message="LU7DZ TEST     ", blocksize=step)
                out = None
                if args.spkr:
                    phase = 0.0
                    def _cb(outdata, frames, t, status):  # noqa: ANN001
                        nonlocal phase
                        # Parlante recibe mismo CW: modular con puerta generada por la fuente
                        outdata[:, 0] = cw.record(frames / float(args.rate))[:frames]
                        phase += frames
                    out = sd.OutputStream(
                        samplerate=args.rate,
                        channels=1,
                        dtype="float32",
                        blocksize=step,
                        callback=_cb,
                    )
                    out.start()
                blocks = cw.blocks()
            live = WaterfallLive(
                nfft=args.nfft,
                overlap=args.overlap,
                cmap=args.cmap,
                max_frames=args.max_frames,
                enable_timeplot=getattr(args, "time", False),
            )
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
            comp = WaterfallComputer(nfft=args.nfft, overlap=args.overlap)
            if args.source == "stream":
                source = DefaultAudioSource(sample_rate=args.rate, channels=1)
                data = source.record(args.dur)
                if args.spkr:
                    sd.play(data.reshape(-1, 1), samplerate=args.rate, blocking=False)
            elif args.source == "tone":
                from .audio import ToneAudioSource
                data = ToneAudioSource(sample_rate=args.rate).record(args.dur)
                if args.spkr:
                    sd.play(data.reshape(-1, 1), samplerate=args.rate, blocking=False)
            elif args.source == "cw":
                from .audio import MorseAudioSource
                data = MorseAudioSource(sample_rate=args.rate, frequency=600.0, wpm=20.0, message="LU7DZ TEST     ").record(args.dur)
                if args.spkr:
                    sd.play(data.reshape(-1, 1), samplerate=args.rate, blocking=False)
            spec = comp.compute(data)
            WaterfallRenderer(cmap=args.cmap).show(
                spec, args.rate, args.nfft, args.overlap
            )
    except (AudioError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
