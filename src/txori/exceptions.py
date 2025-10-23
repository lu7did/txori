"""Excepciones específicas del dominio."""

class TxoriError(Exception):
    """Error base del paquete."""


class AudioUnavailableError(TxoriError):
    """No se pudo inicializar la entrada de audio."""


class VisualizationError(TxoriError):
    """Error en la etapa de visualización."""
