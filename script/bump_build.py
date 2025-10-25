#!/usr/bin/env python3
"""Bump build number and update files.

- Increments BUILD file (3-digit, zero-padded)
- Updates README.md line with "Versión 1.0 build XXX"
- Updates CHANGELOG.md by appending new build entry
- Updates src/txori/__init__.py __build__ constant
"""
from __future__ import annotations
import pathlib
import re
from datetime import datetime

ROOT = pathlib.Path(__file__).resolve().parents[1]
build_file = ROOT / "BUILD"
readme = ROOT / "README.md"
changelog = ROOT / "CHANGELOG.md"
init_py = ROOT / "src" / "txori" / "__init__.py"

current = int(build_file.read_text().strip()) if build_file.exists() else 0
new = current + 1
new_str = f"{new:03d}"
build_file.write_text(new_str + "\n")

# README
readme_text = readme.read_text(encoding="utf-8")
readme_text = re.sub(r"Versión 1.0 build \d{3}", f"Versión 1.0 build {new_str}", readme_text)
readme.write_text(readme_text, encoding="utf-8")

# CHANGELOG
now = datetime.utcnow().strftime("%Y-%m-%d")
with changelog.open("a", encoding="utf-8") as fh:
  fh.write(f"\n- Versión 1.0 build {new_str}: bump automático en {now}.\n")

# __init__.py
init_text = init_py.read_text(encoding="utf-8")
init_text = re.sub(r"__build__ = \"\d{3}\"", f"__build__ = \"{new_str}\"", init_text)
init_py.write_text(init_text, encoding="utf-8")
