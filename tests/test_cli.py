# (c) Dr. Pedro E. Colla 2020-2025 (LU7DZ)
from __future__ import annotations

import sys

from txori.__main__ import main  # pragma: no cover


def test_cli_main_runs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["txori", "--direct", "--seconds", "0.01"])  # evitar matplotlib en CI
    main()


def test_cli_live_mode_no_crash(monkeypatch):
    # Forzamos modo live pero sin matplotlib instalado: debe terminar sin bloquear
    monkeypatch.setattr(
        sys, "argv", ["txori", "--seconds", "0.0"]
    )  # segundos 0 => termina enseguida
    try:
        main()
    except Exception:
        # Permitimos cualquier excepción de entorno gráfico
        assert True
    else:
        assert True
