Txori - BUILD

Prerequisitos
- Python 3.12
- pip install -r requirements.txt
- pip install -r requirements-dev.txt

Chequeos locales
- Formato: black . --check --exclude '/(ejemplos|scripts|\.github/scripts)/'
- Ruff: ruff check .
- Flake8: flake8 --exclude scripts,ejemplos --max-line-length 120 --extend-ignore E203,E501
- Pydocstyle: pydocstyle src --ignore D102,D107,D203
- Mypy: mypy src
- Pyright: pyright
- Bandit: bandit -r src
- Tests: pytest -q --cov=src --cov-report=term-missing --cov-fail-under=80

Build de distribución
- python -m build

Ejecución
- txori-waterfall --source file --in ./sounds/contest.wav
