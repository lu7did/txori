# (c) Dr. Pedro E. Colla 2020-2025 (LU7DZ)
from __future__ import annotations

import pytest


@pytest.mark.skip(reason="FFT module removed; analyzer now inlined in Pipeline")
def test_fft_bins_count_and_energy() -> None:
    assert True
