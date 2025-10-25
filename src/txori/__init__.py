"""txori package.

Provides simple, well-typed utilities and an example class.
"""
from .audio import AudioSource
from .core import Txori, add, reverse_string

__all__ = ["Txori", "add", "reverse_string", "AudioSource"]
__version__ = "1.0.0"
__build__ = "000"
VERSION_BUILD = f"1.0 build {__build__}"
