from __future__ import annotations

import types
import sys
import txori.cli as cli


def test_bypass_tone_48k(monkeypatch) -> None:
    # Fake sounddevice capturing samplerate
    started = {}
    class _OS:
        def __init__(self, samplerate: int, channels: int, dtype: str, callback, blocksize: int, latency: str) -> None:  # noqa: D401
            started['sr'] = samplerate
            self._cb = callback
        def start(self) -> None: return
        def stop(self) -> None: return
        def close(self) -> None: return
    mod = types.SimpleNamespace(OutputStream=_OS)
    monkeypatch.setitem(sys.modules, 'sounddevice', mod)
    rc = cli.main(['--source','tone','--tone-fsr','48000','--cpu','bypass'])  # type: ignore[arg-type]
    assert rc == 0 and started.get('sr') == 48000
