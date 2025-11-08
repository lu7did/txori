from __future__ import annotations

import types
import numpy as np
import sys

import txori.cli as cli


class FakeSource(cli.Source):  # type: ignore[misc]
    def __init__(self, sr: int = 8000) -> None:
        self._sr = sr
        self._calls = 0
    @property
    def sample_rate(self) -> int:
        return self._sr
    def read(self, n: int) -> np.ndarray:
        self._calls += 1
        if self._calls <= 2:
            return np.zeros(min(n, 256), dtype=np.float32)
        return np.array([], dtype=np.float32)
    def close(self) -> None:
        return


def test_parser_has_bypass_option() -> None:
    p = cli.build_parser()
    args = p.parse_args(["--source","tone","--cpu","bypass"])  # noqa: E501
    assert args.cpu == "bypass"


def test_main_bypass_runs(monkeypatch) -> None:
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

    # Patch source factory to return finite source
    monkeypatch.setattr(cli, "_make_source", lambda *a, **kw: FakeSource(sr=48000), raising=True)

    rc = cli.main(["--source","tone","--cpu","bypass"])  # type: ignore[arg-type]
    assert rc == 0 and started.get("sr") in (48000, 44100, 8000)
