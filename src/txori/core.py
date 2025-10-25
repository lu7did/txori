"""Core functionality for txori.

This module demonstrates simple, typed, and documented functions/classes.
"""
from dataclasses import dataclass


def add(a: int, b: int) -> int:
  """Add two integers.

  Args:
    a: First integer.
    b: Second integer.

  Returns:
    The sum of a and b.
  """
  return a + b


def reverse_string(text: str) -> str:
  """Return the reverse of the given string.

  Args:
    text: Input string.

  Returns:
    The reversed string.
  """
  return text[::-1]


@dataclass(slots=True)
class Txori:
  """Simple greeter class."""

  greeting: str = "Hola"

  def greet(self, name: str) -> str:
    """Return a friendly greeting.

    Args:
      name: Person or entity to greet.

    Returns:
      Greeting string in the form "Hola, <name>!".
    """
    if not name:
      raise ValueError("name must be non-empty")
    return f"{self.greeting}, {name}!"
