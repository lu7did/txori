"""Audio input source using PyAudio.

Provides AudioSource to stream audio from the default input device and deliver
blocks of samples to a handler.
"""
from __future__ import annotations

from dataclasses import dataclass
from threading import Event
from typing import Callable, Optional

try:  # Defer hard dependency until runtime
    import pyaudio as _pyaudio  # type: ignore
except Exception as _err:  # pragma: no cover - handled at runtime
    _pyaudio = None  # type: ignore[assignment]
    _PYAUDIO_IMPORT_ERROR = _err
else:
    _PYAUDIO_IMPORT_ERROR = None


SampleHandler = Callable[[bytes, int, int, int], None]


@dataclass
class AudioSource:
    """Stream audio from default input device.

    Args:
      handler: Callback invoked with (data, sample_width, channels, rate) per block.
      frames_per_buffer: Frames per callback. Defaults to 128.
      channels: Channels to open. If None, uses device default/max.
      rate: Sample rate (Hz). If None, uses device default.

    Notes:
      - Uses 16-bit PCM (paInt16) format.
      - Call start() to begin streaming and stop() to end, or use as context manager.
    """

    handler: SampleHandler
    frames_per_buffer: int = 128
    channels: Optional[int] = None
    rate: Optional[int] = None

    # Internal runtime fields (initialized in start)
    _pa: object | None = None
    _stream: object | None = None
    _stopped: Event = Event()

    def _ensure_pyaudio(self) -> None:
        if _pyaudio is None:  # pragma: no cover - env dependent
            raise RuntimeError(
                "PyAudio is not available: " f"{_PYAUDIO_IMPORT_ERROR}"
            )

    def start(self) -> None:
        """Start capturing from default input and delivering to handler."""
        self._ensure_pyaudio()
        assert _pyaudio is not None

        self._stopped.clear()
        pa = _pyaudio.PyAudio()

        dev_info = pa.get_default_input_device_info()
        rate = int(self.rate or dev_info.get("defaultSampleRate", 16000))
        max_ch = int(dev_info.get("maxInputChannels", 1) or 1)
        channels = int(self.channels or min(1, max_ch) or 1)
        fmt = _pyaudio.paInt16
        sw = pa.get_sample_size(fmt)

        def _callback(in_data, frame_count, time_info, status_flags):  # type: ignore[no-untyped-def]
            try:
                if in_data:
                    # Deliver raw bytes and metadata
                    self.handler(in_data, sw, channels, rate)
            except Exception:
                # Stop stream on handler error to avoid tight error loop
                return (None, _pyaudio.paAbort)
            return (None, _pyaudio.paContinue)

        stream = pa.open(
            format=fmt,
            channels=channels,
            rate=rate,
            input=True,
            frames_per_buffer=self.frames_per_buffer,
            stream_callback=_callback,
        )
        self._pa = pa
        self._stream = stream
        stream.start_stream()

    def running(self) -> bool:
        """Return whether the stream is active.

        Returns:
          True if PyAudio stream is active and running.
        """
        s = self._stream
        return bool(s and s.is_active())

    def stop(self) -> None:
        """Stop streaming and release resources."""
        s = self._stream
        pa = self._pa
        self._stream = None
        self._pa = None
        try:
            if s is not None:
                try:
                    s.stop_stream()
                finally:
                    s.close()
        finally:
            if pa is not None:
                pa.terminate()
        self._stopped.set()

    # Context manager helpers
    def __enter__(self) -> "AudioSource":  # noqa: D401 short
        """Enter context and start streaming."""
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401 short
        """Exit context and stop streaming."""
        self.stop()

    def run(self, seconds: Optional[float] = None) -> None:
        """Run until Ctrl+C or for given seconds if provided."""
        import time

        try:
            if seconds is None:
                while self.running():
                    time.sleep(0.05)
            else:
                end = time.time() + max(0.0, seconds)
                while self.running() and time.time() < end:
                    time.sleep(0.01)
        except KeyboardInterrupt:  # pragma: no cover - interactive
            pass
        finally:
            self.stop()
