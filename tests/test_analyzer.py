# (c) Dr. Pedro E. Colla 2020-2025 (LU7DZ)
from __future__ import annotations

import numpy as np

from txori.config import SystemConfig
from txori.pipeline import Pipeline


def test_pipeline_analyze_bins_shape() -> None:
    cfg = SystemConfig(use_audio=False, window_size=128)
    pipe = Pipeline(cfg)
    # Construir una ventana con tono 1 kHz
    fs = float(cfg.sample_rate)
    t = np.arange(cfg.window_size) / fs
    x = np.sin(2 * np.pi * 1000.0 * t)
    spec = pipe._analyze(x)  # type: ignore[attr-defined]
    assert spec.ndim == 1
    assert spec.size > 0
