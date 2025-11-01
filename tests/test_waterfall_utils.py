"""Cover SpectrogramAnimator utilities: window functions and push."""
from __future__ import annotations

import numpy as np

from txori.waterfall import SpectrogramAnimator


def test_window_variants_return_expected_shapes() -> None:
    anim = SpectrogramAnimator(fs=4000, nfft=64, hop=32, width_cols=4)
    N = 64
    for w in [
        "blackman",
        "hanning",
        "hann",
        "hamming",
        "rect",
        "rectangular",
        "boxcar",
        "none",
        "blackmanharris",
        "blackman-harris",
        "bh4",
        "flattop",
        "flat-top",
        "flat_top",
        "unknown",
    ]:
        anim._win = w
        out = anim._window_fn(np.zeros(N))
        assert isinstance(out, np.ndarray)
        assert out.shape == (N,)


def test_push_replaces_or_appends_buffer() -> None:
    anim = SpectrogramAnimator(fs=4000, nfft=64, hop=32, width_cols=4)
    # Large push replaces buffer
    big = np.arange(anim._buf_len + 10, dtype=np.float32)
    anim._push(big)
    assert np.allclose(anim._buffer, big[-anim._buf_len:])
    # Small push appends
    small = np.ones(20, dtype=np.float32)
    before = anim._buffer.copy()
    anim._push(small)
    assert anim._buffer.size == before.size and np.allclose(anim._buffer[-20:], small)
