from hypothesis import given, strategies as st
import pytest

from txori import add, reverse_string, Txori


@given(st.integers(), st.integers())
def test_add_commutative(a: int, b: int) -> None:
  assert add(a, b) == add(b, a)


def test_reverse_roundtrip() -> None:
  s = "txori"
  assert reverse_string(reverse_string(s)) == s


def test_greet_success() -> None:
  assert Txori().greet("Mundo") == "Hola, Mundo!"


def test_greet_empty_raises() -> None:
  with pytest.raises(ValueError):
    Txori().greet("")
