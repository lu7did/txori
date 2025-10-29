import sys
from txori import cli as cli_mod

def test_cli_continuous(monkeypatch):
    monkeypatch.setattr(cli_mod, "WaterfallLive", lambda *a, **k: type("_L", (), {"run": lambda *_a, **_k: None})())
    argv = [
        "txori-waterfall",
        "--continuous",
        "--rate",
        "8000",
        "--nfft",
        "128",
        "--overlap",
        "0.25",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cli_mod.main()
