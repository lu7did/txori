from __future__ import annotations

import numpy as np

from txori.cpu import SkimmerProcessor
from txori import cli as cli_mod


def test_skimmer_bypass_and_decimate() -> None:
    # Bypass when fs <= 8000
    fs1 = 8000
    t1 = np.arange(800) / fs1
    x1 = np.sin(2 * np.pi * 300 * t1).astype(np.float32)
    sk1 = SkimmerProcessor(fs_in=fs1)
    y1 = sk1.process(x1)
    assert y1.shape == x1.shape and np.allclose(y1, x1)

    # LPF+resample to 8000 when fs > 8000
    fs2 = 12000
    t2 = np.arange(2400) / fs2
    x2 = (np.sin(2 * np.pi * 1000 * t2) + 0.25 * np.sin(2 * np.pi * 2500 * t2)).astype(np.float32)
    sk2 = SkimmerProcessor(fs_in=fs2)
    y2 = sk2.process(x2)
    assert y2.ndim == 1 and 0 < y2.size < x2.size and np.isfinite(y2).all()


def test_cli_make_cpu_and_anim_fs_with_skimmer(monkeypatch) -> None:
    captured = {}

    class DummyAnim:
        def __init__(self, *, fs: int, **kwargs) -> None:
            captured["fs"] = fs
        def run(self, *args, **kwargs):
            return None

    monkeypatch.setattr(cli_mod, "SpectrogramAnimator", DummyAnim, raising=True)
    # Run main with tone source at 12 kHz and skimmer CPU; expect animator fs = 8000
    cli_mod.main(["--source", "tone", "--tone-fsr", "12000", "--cpu", "skimmer"])  # type: ignore[arg-type]
    assert captured.get("fs") == 8000


def test_cli_skimmer_with_bpf_flag(monkeypatch) -> None:
    captured = {}

    class DummyAnim:
        def __init__(self, *, fs: int, **kwargs) -> None:
            captured["fs"] = fs
        def run(self, *args, **kwargs):
            return None

    monkeypatch.setattr(cli_mod, "SpectrogramAnimator", DummyAnim, raising=True)
    # With --skimmer flag active, still expect animator fs = 8000
    cli_mod.main(["--source", "tone", "--tone-fsr", "16000", "--cpu", "skimmer", "--skimmer"])  # type: ignore[arg-type]
    assert captured.get("fs") == 8000
