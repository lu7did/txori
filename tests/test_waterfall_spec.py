"""Tests for SpectrogramAnimator._compute_spec to increase coverage."""
from __future__ import annotations

import numpy as np

from txori.waterfall import SpectrogramAnimator


def test_compute_spec_runs_and_shapes() -> None:
    anim = SpectrogramAnimator(
        fs=4000,
        width_cols=8,
        nfft=128,
        hop=64,
        window="Hamming",
        ema=None,
    )
    # Preload buffer with random samples
    anim._buffer[:] = np.random.randn(anim._buffer.size).astype(np.float32)
    Pxx, freqs, bins = anim._compute_spec()
    assert Pxx.ndim == 2 and freqs.ndim == 1 and bins.ndim == 1
    assert Pxx.shape[0] == freqs.size
