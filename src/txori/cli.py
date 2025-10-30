"""CLI para capturar audio y mostrar un gráfico waterfall."""
from __future__ import annotations

import argparse
import sys

import numpy as np
import sounddevice as sd
import time
import random
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
    parser.add_argument("--cwspeed", type=float, default=None, help="Velocidad CW en palabras por minuto (WPM); por defecto 20.")
    parser.add_argument("--hop", type=int, default=None, help="Paso entre frames en muestras (overrides --overlap si se especifica).")
    parser.add_argument("--row-median", action="store_true", help="Restar mediana por fila para mejorar contraste.")
    parser.add_argument("--db-range", type=float, default=None, help="Rango dinámico en dB para el colormap (p.ej. 80).")
    parser.add_argument("--bpf", action="store_true", help="Activar filtro pasabanda 600Hz±50Hz antes de visualizar/reproducir.")
    parser.add_argument("--cwfilter", action="store_true", help="Con --bpf: aplicar 20 BPF de 10 Hz entre 500–700 Hz (10 por debajo y 10 por encima de 600 Hz) y enviar cada salida al waterfall/speaker/timeplot.")
    parser.add_argument("--dsp", action="store_true", help="Modo DSP: aplica LPF y usa su salida para waterfall; opcionalmente envía a parlante/timeplot.")
    parser.add_argument("--cutoff", type=float, default=3000.0, help="Frecuencia de corte del LPF en Hz (solo con --dsp). Por defecto 3000.")
    parser.add_argument("--fir-decim", action="store_true", help="En --dsp: usar decimador FIR (129 taps, Fc~0.4*Fs') antes del 8:1 para reducir spikes.")
    parser.add_argument("--smooth", type=int, default=0, help="Suavizado temporal del waterfall (EMA N columnas, N=1..10) solo en --dsp.")
    parser.add_argument("--cwkill", nargs="?", const=20.0, type=float, help="Suprime tono CW: BPF en 600 Hz con ancho de banda (Hz), por defecto 20 Hz.")
    parser.add_argument("--qrm", nargs="?", const=5, type=int, help="Con --source cw: agrega N (1-10, por defecto 5) señales CW interferentes con licencias y tonos aleatorios (200–2500 Hz).")
    parser.add_argument("--qrn", action="store_true", help="Con --source cw: agrega ruido blanco con SNR (dB) respecto a la señal CW principal.")
    parser.add_argument("--snr", type=float, default=None, help="SNR del ruido en dB (negativos). Por defecto -18 dB.")
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
            dec_sr = 6000
            cutoff_eff = min(float(getattr(args, "cutoff", 3000.0)), 0.45 * dec_sr)
            lpf_stages = [OnePoleLowpass(args.rate, cutoff_eff) for _ in range(3)]
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
                    main_cw = MorseAudioSource(sample_rate=args.rate, frequency=600.0, wpm=(getattr(args, "cwspeed", 20.0) or 20.0), message="LU7DZ TEST     ", blocksize=step)
                    qrm_n = int(max(1, min(10, getattr(args, "qrm", 0) or 0))) if getattr(args, "qrm", None) else 0
                    if qrm_n > 0:
                        calls_all = ["LS1D","LP1H","CX6V","PT2T","CE2CE","9A9","K1TTT","K3LR","AO3O","EA7A"]
                        calls = random.sample(calls_all, k=min(qrm_n, len(calls_all)))
                        qrm_srcs = [
                            MorseAudioSource(
                                sample_rate=args.rate,
                                frequency=float(random.uniform(200.0, 2500.0)),
                                wpm=float(random.uniform(10.0, 40.0)),
                                message=f"{c} TEST    ",
                                blocksize=step,
                            ) for c in calls
                        ]
                        def _sum_blocks():
                            cw_it = main_cw.blocks()
                            qrm_its = [s.blocks() for s in qrm_srcs]
                            snr_db = float(getattr(args, "snr", -18.0) or -18.0)
                            pr = 10.0 ** (snr_db / 10.0)
                            eps = 1e-12
                            while True:
                                main_b = next(cw_it)
                                tot = main_b.copy()
                                for it in qrm_its:
                                    tot = tot + next(it)
                                if getattr(args, "qrn", False):
                                    p = float(np.mean(main_b.astype(np.float32) ** 2)) + eps
                                    noise_std = np.sqrt(max(0.0, p * pr))
                                    n = (np.random.randn(tot.size).astype(np.float32) * noise_std)
                                    tot = (tot + n).astype(np.float32)
                                yield tot
                        blocks = _sum_blocks()
                    else:
                        blocks = main_cw.blocks()
                out = None
                dec_sr = 6000
                dec_step = max(1, step // 8)
                obuf = None
                if args.spkr:
                    from collections import deque as _dq
                    obuf = _dq(maxlen=50)
                    def _cb(outdata, frames, t, status):  # noqa: ANN001
                        # Consume desde el buffer; si no hay datos, salida en cero
                        n = frames
                        outdata[:, 0] = 0.0
                        if obuf:
                            filled = 0
                            while filled < n and obuf:
                                v = obuf.popleft()
                                m = min(n - filled, v.size)
                                outdata[filled:filled+m, 0] = v[:m]
                                if m < v.size:
                                    # sobró: reinyectar remanente
                                    obuf.appendleft(v[m:])
                                filled += m
                    out = sd.OutputStream(samplerate=dec_sr, channels=1, dtype="float32", blocksize=dec_step, callback=_cb)
                    out.start()
                def _blocks_lpf():
                    for b in blocks:
                        tmp = amp * b
                        for _s in lpf_stages:
                            tmp = _s.process_block(tmp)
                        y = tmp
                        z = y[::8]
                        # cwkill en vivo (Fs'=6000)
                        if getattr(args, "cwkill", None):
                            bw = max(1.0, float(args.cwkill))
                            z = BiquadBandpass(6000, 600.0, bw).process_block(z)
                        if obuf is not None:
                            try:
                                obuf.append(z.astype(np.float32, copy=False))
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
                if getattr(args, "smooth", 0) > 0 and hasattr(live, "smooth"):
                    live.smooth = int(args.smooth)
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
                    main_cw = MorseAudioSource(sample_rate=args.rate, frequency=600.0, wpm=(getattr(args, "cwspeed", 20.0) or 20.0), message="LU7DZ TEST     ")
                    data = main_cw.record(args.dur)
                    qrm_n = int(max(1, min(10, getattr(args, "qrm", 0) or 0))) if getattr(args, "qrm", None) else 0
                    if qrm_n > 0:
                        calls_all = ["LS1D","LP1H","CX6V","PT2T","CE2CE","9A9","K1TTT","K3LR","AO3O","EA7A"]
                        calls = random.sample(calls_all, k=min(qrm_n, len(calls_all)))
                        for c in calls:
                            f = float(random.uniform(200.0, 2500.0))
                            w = float(random.uniform(10.0, 40.0))
                            data = data + MorseAudioSource(sample_rate=args.rate, frequency=f, wpm=w, message=f"{c} TEST    ").record(args.dur)
                    if getattr(args, "qrn", False):
                        snr_db = float(getattr(args, "snr", -18.0) or -18.0)
                        pr = 10.0 ** (snr_db / 10.0)
                        p = float(np.mean(main_cw.record(args.dur).astype(np.float32) ** 2)) + 1e-12
                        noise_std = np.sqrt(max(0.0, p * pr))
                        data = data + (np.random.randn(data.size).astype(np.float32) * noise_std)
                tmp = amp * data
                # FIR decimator opcional
                if getattr(args, "fir_decim", False):
                    # 129 taps lowpass con Fc 0.4 * 3000 (norm al Fs de entrada)
                    taps = 129
                    fc = 0.4 * 3000.0
                    nyq = args.rate / 2.0
                    m = np.arange(taps, dtype=np.float32) - (taps - 1) / 2.0
                    h = np.sinc((fc / nyq) * m)
                    h *= np.hamming(taps).astype(np.float32)
                    h /= np.sum(h)
                    tmp = np.convolve(tmp, h, mode="same").astype(np.float32)
                else:
                    for _s in lpf_stages:
                        tmp = _s.process_block(tmp)
                y = tmp
                z = y[::8]
                # cwfilter solo con --dsp (a 6 kHz) y modo fijo
                mix = z
                # cwkill: BPF centrado 600 Hz con BW configurable
                if getattr(args, "cwkill", None):
                    bw = max(1.0, float(args.cwkill))
                    mix = BiquadBandpass(6000, 600.0, bw).process_block(mix)
                if args.bpf and getattr(args, "cwfilter", False):
                    base = BiquadBandpass(6000, 600.0, 100.0).process_block(mix)
                    centers = [505.0 + 10.0 * k for k in range(10)] + [605.0 + 10.0 * k for k in range(10)]
                    outs = [BiquadBandpass(6000, cf, 10.0).process_block(base) for cf in centers]
                    mix = np.sum(np.stack(outs, axis=0), axis=0) / float(len(outs))
                if args.spkr:
                    mx = float(np.max(np.abs(mix))) or 1.0
                    sd.play((mix / mx).reshape(-1, 1), samplerate=6000, blocking=False)
                if getattr(args, "time", False):
                    from matplotlib import pyplot as plt
                    tplot = TimePlotLive()
                    tplot.update(mix)
                    tplot.redraw()
                    plt.ioff(); plt.show()
                comp = WaterfallComputer(nfft=nfft_eff, overlap=args.overlap, window=args.window)
                if hasattr(comp, "hop"):
                    setattr(comp, "hop", getattr(args, "hop", None))
                if hasattr(comp, "row_median"):
                    setattr(comp, "row_median", getattr(args, "row_median", False))
                spec = comp.compute(mix)
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
                        bpf_spkr = BiquadBandpass(args.rate, 600.0, 100.0) if args.bpf else None
                        def _cb(indata, outdata, frames, t, status):  # noqa: ANN001
                            x = indata[:, 0].astype(np.float32, copy=False)
                            y = amp * x
                            if bpf_spkr is not None:
                                y = bpf_spkr.process_block(y)
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
                    bpf_spkr = BiquadBandpass(args.rate, 600.0, 100.0) if args.bpf else None

                    def _cb(outdata, frames, t, status):  # noqa: ANN001
                        nonlocal phase
                        idx = np.arange(frames, dtype=np.float32)
                        tt = (phase + idx) / float(args.rate)
                        raw = np.sin(2 * np.pi * 600.0 * tt).astype(np.float32)
                        if bpf_spkr is not None:
                            raw = bpf_spkr.process_block(raw)
                        outdata[:, 0] = amp * raw
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
                cw = MorseAudioSource(sample_rate=args.rate, frequency=600.0, wpm=(getattr(args, "cwspeed", 20.0) or 20.0), message="LU7DZ TEST     ", blocksize=step)
                # QRM fuentes adicionales (solo si se indicó --qrm)
                qrm_n = int(max(1, min(10, getattr(args, "qrm", 0) or 0))) if getattr(args, "qrm", None) else 0
                qrm_srcs = []
                if qrm_n > 0:
                    calls_all = ["LS1D","LP1H","CX6V","PT2T","CE2CE","9A9","K1TTT","K3LR","AO3O","EA7A"]
                    calls = random.sample(calls_all, k=min(qrm_n, len(calls_all)))
                    for c in calls:
                        qrm_srcs.append(MorseAudioSource(sample_rate=args.rate, frequency=float(random.uniform(200.0, 2500.0)), wpm=float(random.uniform(10.0, 40.0)), message=f"{c} TEST    ", blocksize=step))
                out = None
                if args.spkr:
                    phase = 0.0
                    amp = max(0.0, min(1.0, float(getattr(args, "vol", 60.0)) / 100.0))
                    bpf_spkr = BiquadBandpass(args.rate, 600.0, 100.0) if args.bpf else None
                    def _cb(outdata, frames, t, status):  # noqa: ANN001
                        nonlocal phase
                        # Parlante recibe CW + QRM sumados (+QRN)
                        raw_main = cw.record(frames / float(args.rate))[:frames]
                        raw = raw_main.copy()
                        if qrm_srcs:
                            for s in qrm_srcs:
                                raw = raw + s.record(frames / float(args.rate))[:frames]
                        if getattr(args, "qrn", False):
                            snr_db = float(getattr(args, "snr", -18.0) or -18.0)
                            pr = 10.0 ** (snr_db / 10.0)
                            p = float(np.mean(raw_main.astype(np.float32) ** 2)) + 1e-12
                            noise_std = np.sqrt(max(0.0, p * pr))
                            raw = raw + (np.random.randn(frames).astype(np.float32) * noise_std)
                        if bpf_spkr is not None:
                            raw = bpf_spkr.process_block(raw)
                        outdata[:, 0] = amp * raw
                        phase += frames
                    out = sd.OutputStream(
                        samplerate=args.rate,
                        channels=1,
                        dtype="float32",
                        blocksize=step,
                        callback=_cb,
                    )
                    out.start()
                def _cw_blocks_sum():
                    cw_it = cw.blocks()
                    qrm_its = [s.blocks() for s in qrm_srcs] if qrm_srcs else []
                    snr_db = float(getattr(args, "snr", -18.0) or -18.0)
                    pr = 10.0 ** (snr_db / 10.0)
                    while True:
                        main_b = next(cw_it)
                        tot = main_b.copy()
                        for it in qrm_its:
                            tot = tot + next(it)
                        if getattr(args, "qrn", False):
                            p = float(np.mean(main_b.astype(np.float32) ** 2)) + 1e-12
                            noise_std = np.sqrt(max(0.0, p * pr))
                            tot = tot + (np.random.randn(tot.size).astype(np.float32) * noise_std)
                        yield tot.astype(np.float32)
                base_blocks = _cw_blocks_sum()
                if args.bpf:
                    bpf_blocks = BiquadBandpass(args.rate, 600.0, 100.0)
                    blocks = (bpf_blocks.process_block(amp * b) for b in base_blocks)
                else:
                    blocks = (amp * b for b in base_blocks)
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
            elif args.source == "tone":
                from .audio import ToneAudioSource
                data = ToneAudioSource(sample_rate=args.rate).record(args.dur)
            elif args.source == "cw":
                from .audio import MorseAudioSource
                main_cw = MorseAudioSource(sample_rate=args.rate, frequency=600.0, wpm=(getattr(args, "cwspeed", 20.0) or 20.0), message="LU7DZ TEST     ")
                data = main_cw.record(args.dur)
                qrm_n = int(max(1, min(10, getattr(args, "qrm", 0) or 0))) if getattr(args, "qrm", None) else 0
                if qrm_n > 0:
                    calls_all = ["LS1D","LP1H","CX6V","PT2T","CE2CE","9A9","K1TTT","K3LR","AO3O","EA7A"]
                    calls = random.sample(calls_all, k=min(qrm_n, len(calls_all)))
                    for c in calls:
                        f = float(random.uniform(200.0, 2500.0))
                        data = data + MorseAudioSource(sample_rate=args.rate, frequency=f, wpm=(getattr(args, "cwspeed", 20.0) or 20.0), message=f"{c} TEST    ").record(args.dur)
            data = (amp * data)
            if args.bpf and getattr(args, "cwfilter", False):
                base = BiquadBandpass(args.rate, 600.0, 100.0).process_block(data)
                centers = [505.0 + 10.0 * k for k in range(10)] + [605.0 + 10.0 * k for k in range(10)]
                outs = [BiquadBandpass(args.rate, cf, 10.0).process_block(base) for cf in centers]
                # Mezcla promedio para una sola salida común
                mix = np.sum(np.stack(outs, axis=0), axis=0) / float(len(outs))
                # Speaker: mezcla normalizada
                if args.spkr:
                    mx = float(np.max(np.abs(mix))) or 1.0
                    sd.play((mix / mx).reshape(-1, 1), samplerate=args.rate, blocking=False)
                # Timeplot: único
                if getattr(args, "time", False):
                    from matplotlib import pyplot as plt
                    tplot = TimePlotLive()
                    tplot.update(mix)
                    tplot.redraw()
                    plt.ioff(); plt.show()
                # Waterfall: único
                spec = comp.compute(mix)
                WaterfallRenderer(cmap=args.cmap).show(spec, args.rate, args.nfft, args.overlap)
            else:
                if args.spkr:
                    if args.bpf:
                        y = BiquadBandpass(args.rate, 600.0, 100.0).process_block(data)
                    else:
                        y = data
                    sd.play(y.reshape(-1, 1), samplerate=args.rate, blocking=False)
                if args.bpf:
                    data = BiquadBandpass(args.rate, 600.0, 100.0).process_block(data)
                spec = comp.compute(data)
                WaterfallRenderer(cmap=args.cmap).show(spec, args.rate, args.nfft, args.overlap)
    except (AudioError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
