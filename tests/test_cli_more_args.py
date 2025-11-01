"""Additional CLI parsing coverage for optional switches."""
from __future__ import annotations

from txori.cli import build_parser


def test_cli_parse_fft_and_flags() -> None:
    p = build_parser()
    args = p.parse_args([
        "--source",
        "tone",
        "--tone-freq",
        "600",
        "--tone-fsr",
        "4000",
        "--cpu",
        "lpf",
        "--cpu-lpf-freq",
        "2200",
        "--fft-window",
        "Hamming",
        "--fft-nfft",
        "128",
        "--fft-overlap",
        "64",
        "--fft-pixels",
        "800",
        "--spkr",
        "--time",
        "--time-scale",
        "0.5",
    ])
    assert args.fft_window == "Hamming" and args.spkr and args.time and args.time_scale == 0.5
