import sys
import types
import pytest

from txori.audio import AudioSource


class _DummyStream:
    def __init__(self):
        self._active = False

    def start_stream(self):
        self._active = True

    def is_active(self):
        return self._active

    def stop_stream(self):
        self._active = False

    def close(self):
        pass


class _DummyPA:
    paInt16 = object()
    paContinue = object()
    paAbort = object()

    def __init__(self):
        self.open_calls = []

    def get_sample_size(self, fmt):
        return 2

    def get_default_input_device_info(self):
        return {"defaultSampleRate": 8000.0, "maxInputChannels": 1}

    def open(self, **kwargs):  # type: ignore[no-untyped-def]
        self.open_calls.append(kwargs)
        return _DummyStream()

    def terminate(self):
        pass


@pytest.fixture(autouse=True)
def patch_pyaudio(monkeypatch):
    mod = types.SimpleNamespace(
        PyAudio=lambda: _DummyPA(),
        paInt16=object(),
        paContinue=object(),
        paAbort=object(),
    )
    sys.modules["pyaudio"] = mod  # type: ignore[assignment]
    yield
    sys.modules.pop("pyaudio", None)


def test_audio_source_start_stop_calls_handler_once(monkeypatch):
    calls = []

    def handler(data: bytes, sw: int, ch: int, rate: int):
        calls.append((len(data), sw, ch, rate))

    src = AudioSource(handler=handler, frames_per_buffer=128)

    # Monkeypatch AudioSource.start to emulate one callback
    def fake_start(self):  # type: ignore[no-redef]
        self._pa = _DummyPA()
        sw = 2
        ch = 1
        rate = 8000
        data = b"\x00\x00" * 128
        handler(data, sw, ch, rate)
        self._stream = _DummyStream()
        self._stream.start_stream()

    monkeypatch.setattr(AudioSource, "start", fake_start)
    monkeypatch.setattr(AudioSource, "stop", lambda self: None)

    with src:
        pass

    assert calls and calls[0][0] == 256
