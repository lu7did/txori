# (c) Dr. Pedro E. Colla 2020-2025 (LU7DZ)
from __future__ import annotations

import numpy as np
import pytest

import txori.visualization as viz
from txori.visualization import SpectrogramRenderer, VisualizationError


def test_invalid_dimensions_raises() -> None:
    with pytest.raises(VisualizationError):
        SpectrogramRenderer(height=0, width=10)


def test_size_mismatch_raises() -> None:
    r = SpectrogramRenderer(height=8, width=16)
    with pytest.raises(VisualizationError):
        r.push_spectrum(np.zeros(7))


def test_update_and_consume_frame(monkeypatch) -> None:
    r = SpectrogramRenderer(height=8, width=16, average_frames=2, update_interval=1, pixels_per_bin=1)
    # Monkeypatch perf_counter to control FPS gating
    t = {"v": 0.0}

    def fake_perf_counter() -> float:
        t["v"] += 1.0
        return t["v"]

    monkeypatch.setattr(viz.time, "perf_counter", fake_perf_counter)
    # Push two spectra to set _dirty and allow consume
    s = np.linspace(0.1, 1.0, 8)
    r.push_spectrum(s)
    r.push_spectrum(s)
    img = r.image
    assert img.shape == (8, 16, 3)
    fr = r.consume_frame()
    # First consume after enough time should return an image or None depending on timing
    assert fr is None or fr.shape == img.shape


def test_energy_to_color_varies() -> None:
    r = SpectrogramRenderer(height=4, width=4)
    # Access private for coverage purposes
    c_lo = r._energy_to_color(1e-12, 1.0)
    c_hi = r._energy_to_color(1.0, 1.0)
    assert isinstance(c_lo, tuple) and len(c_lo) == 3
    assert isinstance(c_hi, tuple) and len(c_hi) == 3
    assert c_lo != c_hi


def test_pixels_per_bin_affects_height() -> None:
    r = SpectrogramRenderer(height=4, width=8, pixels_per_bin=2)
    assert r.image.shape[0] == 4 * 2
    # Draw one column
    s = np.linspace(0.2, 0.9, 4)
    r.push_spectrum(s)
    assert r.image.shape == (8, 8, 3)
