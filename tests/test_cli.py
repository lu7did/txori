from __future__ import annotations

import sys

from txori.__main__ import main


def test_cli_main_runs(tmp_path, monkeypatch) -> None:
    out = tmp_path / "spec.png"
    monkeypatch.setattr(sys, "argv", ["txori", "--seconds", "0.01", "--out", str(out)])
    main()
    assert out.exists() and out.stat().st_size > 0