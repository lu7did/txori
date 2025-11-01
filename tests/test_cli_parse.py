"""Lightweight tests for CLI argument parsing."""
from __future__ import annotations

from txori.cli import build_parser


def test_cli_parse_tone() -> None:
    p = build_parser()
    args = p.parse_args(["--source", "tone", "--tone-freq", "700", "--tone-fsr", "4000"])  # noqa: E501
    assert args.source == "tone" and args.tone_freq == 700.0 and args.tone_fsr == 4000


def test_cli_parse_file() -> None:
    p = build_parser()
    args = p.parse_args(["--source", "file", "--in", "dummy.wav"])  # path not used here
    assert args.source == "file" and args.infile == "dummy.wav"
