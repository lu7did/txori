import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend no interactivo
from matplotlib import pyplot as plt

from txori.waterfall import WaterfallRenderer


def test_renderer_show(monkeypatch):
    spec = np.random.rand(10, 17).astype(np.float32)
    renderer = WaterfallRenderer(cmap="viridis")
    called = {"v": False}

    def _no_show():
        called["v"] = True

    monkeypatch.setattr(plt, "show", _no_show)
    renderer.show(spec, sample_rate=48000, nfft=32, overlap=0.5)
    assert called["v"]
