#!/usr/bin/env bash
set -euo pipefail
# Script de prueba/ejecución de Txori
# Uso:
#  - En vivo (ventana): ./scripts/txori.sh --seconds 30
#  - Indefinido:        ./scripts/txori.sh --forever
#  - Guardar PNG:       ./scripts/txori.sh --seconds 5 --out spectrogram.png
if [ $# -eq 0 ]; then
  exec python -m txori --seconds 5
else
  exec python -m txori "$@"
fi
