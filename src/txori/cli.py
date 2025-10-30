"""CLI para capturar audio y mostrar un gráfico waterfall."""
from __future__ import annotations

import argparse
import sys

import numpy as np
import sounddevice as sd
import time
from collections import deque


class BiquadBandpass:
    def __init__(self, fs: float, f0: float = 600.0, bw: float = 100.0) -> None:
        Q = max(1e-6, float(f0) / float(bw))
        w0 = 2.0 * np.pi * float(f0) / float(fs)
        cosw = float(np.cos(w0))
        sinw = float(np.sin(w0))
        alpha = sinw / (2.0 * Q)
        b0 = alpha
        b1 = 0.0
        b2 = -alpha
        a0 = 1.0 + alpha
        a1 = -2.0 * cosw
        a2 = 1.0 - alpha
        self.b0 = b0 / a0
        self.b1 = b1 / a0
        self.b2 = b2 / a0
        self.a1 = a1 / a0
        self.a2 = a2 / a0
        self.x1 = 0.0
        self.x2 = 0.0
        self.y1 = 0.0
        self.y2 = 0.0

    def process_block(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        y = np.empty_like(x)
        b0, b1, b2, a1, a2 = self.b0, self.b1, self.b2, self.a1, self.a2
        x1, x2, y1, y2 = self.x1, self.x2, self.y1, self.y2
        for i in range(x.size):
            xi = float(x[i])
            yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            y[i] = yi
            x2, x1 = x1, xi
            y2, y1 = y1, yi
        self.x1, self.x2, self.y1, self.y2 = x1, x2, y1, y2
        return y.astype(np.float32, copy=False)

class OnePoleLowpass:
    def __init__(self, fs: float, cutoff_hz: float = 3000.0) -> None:
        self.fs = float(fs)
        self.set_cutoff(cutoff_hz)
        self.y1 = 0.0
    def set_cutoff(self, cutoff_hz: float) -> None:
        fc = max(1.0, float(cutoff_hz))
        dt = 1.0 / self.fs
        rc = 1.0 / (2.0 * np.pi * fc)
        self.alpha = dt / (rc + dt)
    def process_block(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        y = np.empty_like(x)
        a = self.alpha
        y1 = self.y1
        for i in range(x.size):
            xi = float(x[i])
            y1 = y1 + a * (xi - y1)
            y[i] = y1
        self.y1 = y1
        return y.astype(np.float32, copy=False)

from . import __build__, __version__
from .audio import AudioError, DefaultAudioSource, StreamAudioSource
from .waterfall import WaterfallComputer, WaterfallRenderer, WaterfallLive, TimePlotLive


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
    parser.add_argument("--vol", type=float, default=60.0, help="Volumen de salida 0-100%% (tone/cw). Por defecto 60%%.")
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
    parser.add_argument("--bpf", action="store_true", help="Activar filtro pasabanda 600Hz±50Hz antes de visualizar/reproducir.")
    parser.add_argument("--dsp", action="store_true", help="Modo DSP: aplica LPF y usa su salida para waterfall; opcionalmente envía a parlante/timeplot.")
    parser.add_argument("--cutoff", type=float, default=3000.0, help="Frecuencia de corte del LPF en Hz (solo con --dsp). Por defecto 3000.")
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
        # Modo DSP: salida LPF al waterfall y opcionalmente a parlante/timeplot
        if getattr(args, "dsp", False):
            # Ajustes de nfft/hop por defecto en DSP (SR efectivo 6000 Hz)
            nfft_eff = args.nfft if args.nfft != 4096 else 1024
            step = int(nfft_eff * (1 - args.overlap)) or 1
            lpf = OnePoleLowpass(args.rate, getattr(args, "cutoff", 3000.0))
            if args.dur is None and args.continuous:
                # Fuente por bloques
                if args.source == "stream":
                    src = StreamAudioSource(sample_rate=args.rate, channels=1, blocksize=step)
                    blocks = src.blocks()
                elif args.source == "tone":
                    from .audio import ToneAudioSource
                    src = ToneAudioSource(sample_rate=args.rate, blocksize=step)
                    blocks = src.blocks()
                elif args.source == "cw":
                    from .audio import MorseAudioSource
                    src = MorseAudioSource(sample_rate=args.rate, frequency=600.0, wpm=20.0, message="LU7DZ TEST     ", blocksize=step)
                    blocks = src.blocks()
                out = None
                dec_sr = 6000
                dec_step = max(1, step // 8)
                if args.spkr:
                    out = sd.OutputStream(samplerate=dec_sr, channels=1, dtype="float32", blocksize=dec_step)
                    out.start()
                def _blocks_lpf():
                    for b in blocks:
                        y = lpf.process_block(amp * b)
                        z = y[::8]
                        if out is not None:
                            try:
                                out.write(z.reshape(-1, 1))
                            except Exception:
                                pass
                        yield z
                live = WaterfallLive(
                    nfft=nfft_eff,
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
                    live.run(_blocks_lpf(), sample_rate=6000)
                finally:
                    try:
                        if out is not None:
                            out.stop(); out.close()
                    except Exception:
                        pass
            else:
                # Modo fijo (duración)
                if args.source == "stream":
                    source = DefaultAudioSource(sample_rate=args.rate, channels=1)
                    data = source.record(args.dur)
                elif args.source == "tone":
                    from .audio import ToneAudioSource
                    data = ToneAudioSource(sample_rate=args.rate).record(args.dur)
                elif args.source == "cw":
                    from .audio import MorseAudioSource
                    data = MorseAudioSource(sample_rate=args.rate, frequency=600.0, wpm=20.0, message="LU7DZ TEST     ").record(args.dur)
                y = lpf.process_block(amp * data)
                z = y[::8]
                if args.spkr:
                    sd.play(z.reshape(-1, 1), samplerate=6000, blocking=False)
                if getattr(args, "time", False):
                    from matplotlib import pyplot as plt
                    tplot = TimePlotLive()
                    tplot.update(z)
                    tplot.redraw()
                    plt.ioff(); plt.show()
                comp = WaterfallComputer(nfft=nfft_eff, overlap=args.overlap, window=args.window)
                if hasattr(comp, "hop"):
                    setattr(comp, "hop", getattr(args, "hop", None))
                if hasattr(comp, "row_median"):
                    setattr(comp, "row_median", getattr(args, "row_median", False))
                spec = comp.compute(z)
                WaterfallRenderer(cmap=args.cmap).show(
                    spec, 6000, args.nfft, args.overlap
                )
            return
        # Modo continuo por defecto salvo que se especifique --dur
        if args.dur is None and args.continuous:
            # tamaño de bloque acorde al paso para actualizar por frame
            step = int(args.nfft * (1 - args.overlap)) or 1
            if args.source == "stream":
                if args.spkr:
                    def _speaker_stream_blocks():
                        buf = deque(maxlen=200)
                        def _cb(indata, outdata, frames, t, status):  # noqa: ANN001
                            x = indata[:, 0].astype(np.float32, copy=False)
                            y = bpf_spkr.process_block(amp * x)
                            outdata[:, 0] = y
                            try:
                                buf.append(y.copy())
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
                    if args.bpf:
                        bpf_spkr = BiquadBandpass(args.rate, 600.0, 100.0)
                    blocks = _speaker_stream_blocks()
                else:
                    stream = StreamAudioSource(sample_rate=args.rate, channels=1, blocksize=step)
                    if args.bpf:
                        bpf_blocks = BiquadBandpass(args.rate, 600.0, 100.0)
                        blocks = (bpf_blocks.process_block(amp * b) for b in stream.blocks())
                    else:
                        blocks = (amp * b for b in stream.blocks())
            elif args.source == "tone":
                from .audio import ToneAudioSource
                tone = ToneAudioSource(sample_rate=args.rate, blocksize=step)
                out = None
                if args.spkr:
                    phase = 0.0
                    amp = max(0.0, min(1.0, float(getattr(args, "vol", 60.0)) / 100.0))

                    def _cb(outdata, frames, t, status):  # noqa: ANN001
                        nonlocal phase
                        idx = np.arange(frames, dtype=np.float32)
                        tt = (phase + idx) / float(args.rate)
                        raw = np.sin(2 * np.pi * 600.0 * tt).astype(np.float32)
                        outdata[:, 0] = amp * bpf_spkr.process_block(raw)
                        phase += frames

                    out = sd.OutputStream(
                        samplerate=args.rate,
                        channels=1,
                        dtype="float32",
                        blocksize=step,
                        callback=_cb,
                    )
                    out.start()
                if args.bpf:
                    bpf_blocks = BiquadBandpass(args.rate, 600.0, 100.0)
                    blocks = (bpf_blocks.process_block(amp * b) for b in tone.blocks())
                else:
                    blocks = (amp * b for b in tone.blocks())
            elif args.source == "cw":
                from .audio import MorseAudioSource
                cw = MorseAudioSource(sample_rate=args.rate, frequency=600.0, wpm=20.0, message="LU7DZ TEST     ", blocksize=step)
                out = None
                if args.spkr:
                    phase = 0.0
                    amp = max(0.0, min(1.0, float(getattr(args, "vol", 60.0)) / 100.0))
                    def _cb(outdata, frames, t, status):  # noqa: ANN001
                        nonlocal phase
                        # Parlante recibe mismo CW: modular con puerta generada por la fuente
                        raw = cw.record(frames / float(args.rate))[:frames]
                        outdata[:, 0] = amp * bpf_spkr.process_block(raw)
                        phase += frames
                    out = sd.OutputStream(
                        samplerate=args.rate,
                        channels=1,
                        dtype="float32",
                        blocksize=step,
                        callback=_cb,
                    )
                    out.start()
                if args.bpf:
                    bpf_blocks = BiquadBandpass(args.rate, 600.0, 100.0)
                    blocks = (bpf_blocks.process_block(amp * b) for b in cw.blocks())
                else:
                    blocks = (amp * b for b in cw.blocks())
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
                    bpf = BiquadBandpass(args.rate, 600.0, 100.0)
                    y = bpf.process_block(amp * data)
                    sd.play(y.reshape(-1, 1), samplerate=args.rate, blocking=False)
            elif args.source == "tone":
                from .audio import ToneAudioSource
                data = ToneAudioSource(sample_rate=args.rate).record(args.dur)
                if args.spkr:
                    bpf = BiquadBandpass(args.rate, 600.0, 100.0)
                    y = bpf.process_block(amp * data)
                    sd.play(y.reshape(-1, 1), samplerate=args.rate, blocking=False)
            elif args.source == "cw":
                from .audio import MorseAudioSource
                data = MorseAudioSource(sample_rate=args.rate, frequency=600.0, wpm=20.0, message="LU7DZ TEST     ").record(args.dur)
                if args.spkr:
                    y = (BiquadBandpass(args.rate, 600.0, 100.0).process_block(amp * data)) if args.bpf else (amp * data)
                    sd.play(y.reshape(-1, 1), samplerate=args.rate, blocking=False)
            if args.bpf:
                data = BiquadBandpass(args.rate, 600.0, 100.0).process_block(amp * data)
            else:
                data = (amp * data)
            spec = comp.compute(data)
            WaterfallRenderer(cmap=args.cmap).show(
                spec, args.rate, args.nfft, args.overlap
            )
    except (AudioError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
