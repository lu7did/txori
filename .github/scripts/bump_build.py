import datetime as _dt
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INIT = ROOT / "src" / "txori" / "__init__.py"
README = ROOT / "README.md"
CHANGELOG = ROOT / "CHANGELOG.md"


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _write_text(p: Path, s: str) -> None:
    p.write_text(s, encoding="utf-8")


def bump_build_in_init() -> str:
    s = _read_text(INIT)
    m = re.search(r"__build__\s*=\s*\"(\d{3})\"", s)
    if not m:
        raise RuntimeError("No se encontró __build__ en __init__.py")
    build = int(m.group(1)) + 1
    new_build = f"{build:03d}"
    s = re.sub(r"(__build__\s*=\s*\")\d{3}(\")", rf"\g<1>{new_build}\2", s)
    _write_text(INIT, s)
    return new_build


def update_readme(new_build: str) -> None:
    s = _read_text(README)
    s = re.sub(r"build\s+\d{3}", f"build {new_build}", s)
    _write_text(README, s)


def update_changelog(new_build: str) -> None:
    today = _dt.date.today().isoformat()
    line = f"- {today} Versión 1.0 build {new_build}: actualización automática de build.\n"
    s = _read_text(CHANGELOG)
    s = s.rstrip() + "\n" + line
    _write_text(CHANGELOG, s)


if __name__ == "__main__":
    new_build = bump_build_in_init()
    update_readme(new_build)
    update_changelog(new_build)
