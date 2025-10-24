# (c) Dr. Pedro E. Colla 2020-2025 (LU7DZ)
from __future__ import annotations

import numpy as np

from txori.capture import CaptureController, SlidingWindow, SyntheticSineCapture
from txori.config import SystemConfig


def test_sliding_window_push_order() -> None:
    w = SlidingWindow(5)
    for i in range(7):
        w.push(float(i))
    # Último insertado (6) debe estar en posición 0 y anteriores desplazados
    assert np.allclose(w.array[:5], np.array([6, 5, 4, 3, 2], dtype=float))


def test_capture_controller_steps() -> None:
    cap = SyntheticSineCapture(freq_hz=1000.0, cfg=SystemConfig())
    ctl = CaptureController(cap, window_size=4)
    a = ctl.step()
    b = ctl.step()
    assert a.shape == (4,)
    assert b[0] != a[0]
