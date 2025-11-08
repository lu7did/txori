from __future__ import annotations

import types
import numpy as np
import sys

import txori.cli as cli


def test_main_bypass_8k(monkeypatch) -> None:
    # Fake sounddevice module
    mod = types.SimpleNamespace()
    started = {}
    class _OS:
        def __init__(self, samplerate: int, channels: int, dtype: str, callback, blocksize: int, latency: str) -> None:  # noqa: D401
            started["sr"] = samplerate
            self._cb = callback
        def start(self) -> None:
            return
        def stop(self) -> None:
            return
        def close(self) -> None:
            return
    mod.OutputStream = _OS
    monkeypatch.setitem(sys.modules, "sounddevice", mod)

    class FakeSrc(cli.Source):  # type: ignore[misc]
        @property
        def sample_rate(self) -> int:
            return 8000
        def read(self, n: int):
            return np.zeros(min(n, 64), dtype=np.float32)
        def close(self) -> None:
            return

    monkeypatch.setattr(cli, "_make_source", lambda *a, **kw: FakeSrc(), raising=True)
    rc = cli.main(["--source","tone","--cpu","bypass"])  # type: ignore[arg-type]
    assert rc == 0 and started.get("sr") in (8000, 44100, 48000)
