"""Cover LPF bypass and decimation branches."""
from __future__ import annotations

import numpy as np

from txori.cpu import LpfProcessor


def test_lpf_bypass_when_fs_le_4000() -> None:
    fs = 4000
    t = np.arange(800) / fs
    x = np.sin(2 * np.pi * 300 * t).astype(np.float32)
    lp = LpfProcessor(fs_in=fs, cutoff_hz=2000.0)
    y = lp.process(x)
    assert y.shape == x.shape and np.allclose(y, x)


def test_lpf_decimates_when_fs_gt_4000() -> None:
    fs = 12000
    t = np.arange(2400) / fs
    x = (np.sin(2 * np.pi * 1000 * t) + 0.25 * np.sin(2 * np.pi * 2500 * t)).astype(np.float32)
    lp = LpfProcessor(fs_in=fs, cutoff_hz=2000.0)
    y = lp.process(x)
    assert y.ndim == 1 and 0 < y.size < x.size
