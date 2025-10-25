# (c) Dr. Pedro E. Colla 2020-2025 (LU7DZ)
from __future__ import annotations

import numpy as np

from txori.visualization import SpectrogramRenderer


def test_renderer_draws_and_saves(tmp_path) -> None:
    h, w = 16, 32
    rend = SpectrogramRenderer(height=h, width=w, average_frames=4, update_interval=1, pixels_per_bin=1)
    # Push varios espectros válidos (1-D, tamaño=h)
    for _ in range(5):
        spec = np.linspace(1.0, 2.0, h).astype(float)
        rend.push_spectrum(spec)
    # La imagen debe tener el tamaño esperado y contener columnas actualizadas
    img = rend.image
    assert img.shape == (h, w, 3)
    # Debe poder entregar un frame y guardarlo
    frame = rend.consume_frame()
    assert frame is None or frame.shape == img.shape
    p = tmp_path / "spec.png"
    rend.save(str(p))
    assert p.exists()
