from __future__ import annotations

import sys

from txori.__main__ import main


def test_cli_main_runs(tmp_path, monkeypatch) -> None:
    out = tmp_path / "spec.png"
    monkeypatch.setattr(sys, "argv", ["txori", "--seconds", "0.01", "--out", str(out)])
    main()
    assert out.exists() and out.stat().st_size > 0


def test_cli_live_mode_no_crash(monkeypatch):
    # Forzamos modo live pero sin matplotlib instalado: debe fallar con mensaje claro
    monkeypatch.setattr(sys, "argv", ["txori", "--seconds", "0.0"])  # segundos 0 => termina enseguida
    try:
        main()
    except Exception as e:  # noqa: BLE001
        # Aceptamos que falle si no está matplotlib, pero el mensaje debe ser claro
        assert "matplotlib" in str(e).lower() or "requires" in str(e).lower()
    else:
        # Si pasa sin excepción, también es válido
        assert True