from __future__ import annotations

import numpy as np

from txori.visualization import SpectrogramRenderer


def test_renderer_draws_and_saves(tmp_path) -> None:
    h, w = 16, 32
    renderer = SpectrogramRenderer(height=h, width=w, average_frames=2, update_interval=1)
    s1 = np.zeros(h, dtype=float)
    s1[0] = 1.0
    renderer.push_spectrum(s1)
    img1 = renderer.image.copy()

    s2 = np.linspace(0.0, 1.0, h, dtype=float)
    renderer.push_spectrum(s2)
    img2 = renderer.image
    assert not np.array_equal(img1, img2)

    # Conversiones y guardado
    pil = renderer.to_pil()
    out = tmp_path / "viz.png"
    pil.save(out)
    assert out.exists() and out.stat().st_size > 0


def test_renderer_input_validation() -> None:
    r = SpectrogramRenderer(height=8, width=8)
    bad = np.ones((2, 2))
    try:
        r.push_spectrum(bad)  # debe lanzar por espectro no 1-D
    except Exception as e:  # noqa: BLE001
        assert "1-D" in str(e)
    else:
        assert False, "Se esperaba excepción por espectro no 1-D"

    wrong_size = np.ones(7)
    try:
        r.push_spectrum(wrong_size)
    except Exception as e:  # noqa: BLE001
        assert "altura" in str(e).lower()
    else:
        assert False, "Se esperaba excepción por altura incompatible"