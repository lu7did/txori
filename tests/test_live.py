import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from txori.waterfall import WaterfallLive


def _blocks(step: int, count: int):
    t = 0
    sr = 8000
    for _ in range(count):
        n = step
        idx = np.arange(n, dtype=np.float32)
        x = np.sin(2 * np.pi * 440.0 * (t + idx) / sr).astype(np.float32)
        t += n
        yield x


def test_waterfall_live_runs(monkeypatch):
    called = {"pause": 0, "show": 0}
    monkeypatch.setattr(plt, "pause", lambda *_a, **_k: called.__setitem__("pause", called["pause"] + 1))
    monkeypatch.setattr(plt, "show", lambda *_a, **_k: called.__setitem__("show", called["show"] + 1))
    live = WaterfallLive(nfft=16, overlap=0.5, cmap="viridis", max_frames=10)
    step = int(16 * (1 - 0.5))
    live.run(_blocks(step, count=20), sample_rate=8000)
    assert called["show"] >= 1
