# (c) Dr. Pedro E. Colla 2020-2025 (LU7DZ)
from __future__ import annotations

import numpy as np

from txori.config import SystemConfig
from txori.pipeline import Pipeline


def test_analyze_energy_bins_mapping() -> None:
    cfg = SystemConfig(use_audio=False, window_size=256)
    p = Pipeline(cfg)
    fs = float(cfg.sample_rate)
    t = np.arange(cfg.window_size) / fs
    # Dos tonos: 1 kHz y 2 kHz
    x = np.sin(2 * np.pi * 1000.0 * t) + 0.5 * np.sin(2 * np.pi * 2000.0 * t)
    spec = p._analyze(x)  # type: ignore[attr-defined]
    assert spec.ndim == 1 and spec.size > 0
    # Las energdas no deben ser todas cero
    assert float(np.sum(spec)) > 0.0
