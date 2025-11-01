"""Additional tests for cpu processors."""
from __future__ import annotations

import numpy as np

from txori.cpu import NoOpProcessor, LpfProcessor, BandPassProcessor, ChainProcessor


def test_processors_shape_and_finiteness() -> None:
    fs = 8000
    t = np.arange(0, fs // 10) / fs
    x = (np.sin(2 * np.pi * 400 * t) + 0.5 * np.sin(2 * np.pi * 1200 * t)).astype(
        np.float32
    )

    y0 = NoOpProcessor().process(x)
    assert y0.shape == x.shape and np.isfinite(y0).all()

    lp = LpfProcessor(fs_in=fs, cutoff_hz=2000.0)
    y1 = lp.process(x)
    assert y1.ndim == 1 and y1.size > 0 and np.isfinite(y1).all()

    bp = BandPassProcessor(fs=fs, center_hz=600.0, bw_hz=200.0)
    y2 = bp.process(x)
    assert y2.shape == x.shape and np.isfinite(y2).all()

    ch = ChainProcessor([lp, bp])
    y3 = ch.process(x)
    assert y3.ndim == 1 and y3.size > 0 and np.isfinite(y3).all()
