"""txori: Captura audio y muestra un gráfico waterfall.

Este paquete implementa la captura desde la entrada predeterminada de audio y la
representación tipo waterfall separando la lógica de negocio de la presentación
(OOP) y utilizando manejo de excepciones.
"""

from . import audio as audio  # reexport para __all__ correcto
from . import waterfall as waterfall

__all__ = [
    "audio",
    "waterfall",
]

__version__ = "1.0.0"
__build__ = "002"  # Actualizado automáticamente por CI
