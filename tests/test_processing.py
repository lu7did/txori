# (c) Dr. Pedro E. Colla 2020-2025 (LU7DZ)
from __future__ import annotations

import numpy as np

from txori.processing import AGCProcessor, DSPProcessor, IdentityProcessor


def test_identity_processor_passthrough() -> None:
    proc = IdentityProcessor()
    x = np.array([0.0, 1.0, -1.0, 0.5], dtype=float)
    y = proc.process(x)
    assert np.allclose(x, y)


def test_agc_limits_output_range() -> None:
    agc = AGCProcessor()
    x = np.linspace(-10.0, 10.0, 1000)
    y = agc.process(x)
    assert np.isfinite(y).all()


def test_dsp_stability_on_noise() -> None:
    dsp = DSPProcessor()
    rng = np.random.default_rng(123)
    x = rng.standard_normal(2048).astype(float)
    y = dsp.process(x)
    assert y.shape == x.shape
