#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

try:
    import sounddevice as sd  # type: ignore
except Exception:  # pragma: no cover
    sd = None

try:
    from txori.capture import (  # type: ignore
        SyntheticCWToneCapture,
        SyntheticSineCapture,
    )
    from txori.config import SystemConfig  # type: ignore
except Exception:
    SyntheticCWToneCapture = None  # type: ignore
    SyntheticSineCapture = None  # type: ignore
    SystemConfig = None  # type: ignore


def make_source(args, fs: int):
    if args.audio:
        if sd is None:
            raise SystemExit("--audio requiere 'sounddevice' instalado")
        stream = sd.InputStream(samplerate=fs, channels=1, dtype="float32", blocksize=1024)
        stream.start()

        def next_chunk(n: int) -> np.ndarray:
            frames, _ = stream.read(n)
            return np.asarray(frames[:, 0], dtype=np.float32)

        return next_chunk
    cfg = SystemConfig(sample_rate=fs) if SystemConfig is not None else None
    if args.cw:
        src = SyntheticCWToneCapture(freq_hz=float(args.cw_tone or 600.0), cfg=cfg)  # type: ignore
    else:
        freq = float(args.tone or 1000.0)
        src = SyntheticSineCapture(freq_hz=freq, cfg=cfg)  # type: ignore

    def next_chunk(n: int) -> np.ndarray:
        out = np.empty(n, dtype=np.float32)
        for i in range(n):
            out[i] = float(src.next_sample())  # type: ignore
        return out

    return next_chunk


def main() -> None:
    p = argparse.ArgumentParser(description="Txori testwave live time plot (30s window)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--audio", action="store_true", help="usar entrada de audio")
    g.add_argument("--cw", action="store_true", help="usar tono CW conmutado (57 ms)")
    p.add_argument("--cw-tone", type=float, default=600.0, help="frecuencia CW (Hz)")
    p.add_argument("--tone", type=float, default=1000.0, help="frecuencia seno (Hz)")
    p.add_argument("--seconds", type=float, help="tiempo total de ejecución")
    p.add_argument("--forever", action="store_true", help="ejecutar indefinidamente")
    p.add_argument("--samplerate", type=int, default=48000, help="Fs (Hz)")
    args = p.parse_args()

    fs = int(args.samplerate)
    span = 30.0
    run_until = None if args.forever or args.seconds is None else float(args.seconds)

    next_chunk = make_source(args, fs)

    N = max(1, int(fs * span))
    buf = np.zeros(N, dtype=np.float32)
    idx = -1

    plt.ion()
    fig, ax = plt.subplots(1, 1)
    fig.canvas.manager.set_window_title("Txori - Tiempo (testwave)")
    fig.canvas.draw()
    width_px = int(getattr(ax, "bbox", None).width) if hasattr(ax, "bbox") else 1200
    M = min(N, max(1000, width_px))
    dec = max(1, N // M)
    x = np.linspace(-span, 0.0, M)
    y = np.zeros(M, dtype=np.float32)
    (line,) = ax.plot(x, y, color="lime")
    ax.set_xlim(-span, 0.0)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Amplitud (V)")
    ax.grid(True, alpha=0.2)
    plt.show(block=False)

    t0 = time.perf_counter()
    chunk = max(256, dec)
    while True:
        s = next_chunk(chunk)
        n = int(s.shape[0])
        np.clip(s, -1.0, 1.0, out=s)
        for i in range(n):
            idx = (idx + 1) % N
            buf[idx] = s[i]
        now = time.perf_counter()
        if now - t0 >= 1.0 / 40.0:
            start = (idx + 1 - dec * M) % N
            idxs = (start + dec * np.arange(M, dtype=int)) % N
            y[:] = buf[idxs]
            line.set_data(x, y)
            fig.canvas.draw_idle()
            plt.pause(0.001)
            t0 = now
        time.sleep(max(0.0, n / fs - 0.0005))
        if run_until is not None:
            run_until -= n / fs
            if run_until <= 0:
                break

    plt.ioff()


if __name__ == "__main__":  # pragma: no cover
    main()
