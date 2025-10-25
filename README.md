# txori

Versión 1.0 build 000

Paquete Python (>=3.12) estructurado tipo cookiecutter con CI (lint, type-check, tests, seguridad) y documentación automática.

## Uso rápido

```python
from txori import Txori, add, reverse_string

print(add(2, 3))               # 5
print(reverse_string("txori"))  # "iroxt"
print(Txori().greet("Mundo"))   # "Hola, Mundo!"
```

Consulta CHANGELOG.md para trazabilidad y mira artifacts de CI para la documentación generada (pdoc).
