"""Integration-like tests for CLI main with patched Animator to boost coverage."""
from __future__ import annotations

import os
import wave
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

import txori.cli as cli


class _FakeAnimator:
    def __init__(self, **kwargs: Any) -> None:  # noqa: D401
        self.kwargs = kwargs
        _FAKES.append(self)

    def run(self, src: Any, cpu: Any, *, spkr: bool = False, time_plot: bool = False, time_scale: float = 1.0) -> None:  # noqa: D401
        self.src = src
        self.cpu = cpu
        self.spkr = spkr
        self.time_plot = time_plot
        self.time_scale = time_scale


_FAKES: list[_FakeAnimator] = []


def _write_wav_mono_8bit(path: str, sr: int) -> None:
    # 8-bit unsigned PCM ramp
    n = sr // 2
    x = (np.sin(2 * np.pi * 440.0 * (np.arange(n) / sr)) * 0.45 + 0.5)
    d8 = np.clip((x * 255.0).astype(np.uint8), 0, 255)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(1)
        wf.setframerate(sr)
        wf.writeframes(d8.tobytes())


def test_cli_main_tone_lpf_chain_invokes_animator(monkeypatch) -> None:
    monkeypatch.setattr(cli, "SpectrogramAnimator", _FakeAnimator, raising=True)
    _FAKES.clear()
    argv = [
        "--source",
        "tone",
        "--tone-freq",
        "600",
        "--tone-fsr",
        "8000",
        "--cpu",
        "lpf",
        "--cpu-lpf-freq",
        "2200",
        "--cwfilter",
        "--cpu-bpf-freq",
        "650",
        "--cpu-bpf-bw",
        "300",
        "--fft-window",
        "Hamming",
        "--fft-nfft",
        "128",
        "--fft-overlap",
        "64",
        "--fft-pixels",
        "800",
    ]
    rc = cli.main(argv)
    assert rc == 0 and _FAKES, "Animator should have been constructed"
    anim = _FAKES[-1]
    # anim_fs should be 2*fc = 4400 because tone fs=8000 (>4000) and cpu lpf freq=2200
    assert anim.kwargs["fs"] == 4400
    assert anim.kwargs["fft_window"] == "Hamming"


def test_cli_make_source_and_cpu_error_branches(monkeypatch) -> None:
    # Unknown source
    try:
        cli._make_source("unknown", None, 0.0, 0)
        raise AssertionError("_make_source should exit on unknown kind")
    except SystemExit:
        pass
    # lpf without fs
    try:
        cli._make_cpu("lpf", fs=None)
        raise AssertionError("_make_cpu should exit when fs is None for lpf")
    except SystemExit:
        pass


def test_cli_main_file_8bit_builds_animator(monkeypatch) -> None:
    monkeypatch.setattr(cli, "SpectrogramAnimator", _FakeAnimator, raising=True)
    _FAKES.clear()
    with TemporaryDirectory() as td:
        wav = os.path.join(td, "mono8.wav")
        _write_wav_mono_8bit(wav, 8000)
        argv = [
            "--source",
            "file",
            "--in",
            wav,
            "--cpu",
            "lpf",
            "--cpu-lpf-freq",
            "1500",
        ]
        rc = cli.main(argv)
        assert rc == 0 and _FAKES, "Animator should have been constructed for file source"
        anim = _FAKES[-1]
        # anim_fs should be 2*fc = 3000 because file fs=8000 (>4000) and cpu lpf freq=1500
        assert anim.kwargs["fs"] == 3000
