import sys
import numpy as np

from txori import cli as cli_mod


class _FakeSource:
    def __init__(self, sample_rate: int, channels: int) -> None:  # noqa: D401
        self.sample_rate = sample_rate
        self.channels = channels

    def record(self, duration_s: float):  # noqa: D401
        n = max(1, int(self.sample_rate * duration_s))
        t = np.arange(n, dtype=np.float32) / float(self.sample_rate)
        return np.sin(2 * np.pi * 440.0 * t).astype(np.float32)


def test_cli_main(monkeypatch):
    monkeypatch.setattr(cli_mod, "DefaultAudioSource", _FakeSource)
    monkeypatch.setattr(cli_mod.WaterfallRenderer, "show", lambda *a, **k: None)
    argv = [
        "txori-waterfall",
        "--dur",
        "0.05",
        "--rate",
        "8000",
        "--nfft",
        "128",
        "--overlap",
        "0.5",
        "--cmap",
        "viridis",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cli_mod.main()
