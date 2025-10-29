import numpy as np
from hypothesis import given, strategies as st, settings, HealthCheck

from txori.waterfall import WaterfallComputer


def test_spectrogram_shape():
    sr = 48_000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    x = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    comp = WaterfallComputer(nfft=1024, overlap=0.5)
    spec = comp.compute(x)
    assert spec.ndim == 2
    assert spec.shape[1] == 1024 // 2 + 1
    assert np.isfinite(spec).all()


def test_tone_peak_bin():
    sr = 48_000
    f0 = 2000
    n = sr
    t = np.arange(n, dtype=np.float32) / sr
    x = np.sin(2 * np.pi * f0 * t).astype(np.float32)
    nfft = 2048
    comp = WaterfallComputer(nfft=nfft, overlap=0.75)
    spec = comp.compute(x)
    avg = spec.mean(axis=0)
    peak_bin = int(np.argmax(avg))
    freq_res = sr / nfft
    peak_freq = peak_bin * freq_res
    assert abs(peak_freq - f0) < 2 * freq_res


@settings(suppress_health_check=[HealthCheck.large_base_example])
@given(st.lists(st.floats(min_value=-1.0, max_value=1.0), min_size=300, max_size=2000))
def test_compute_properties(data):
    x = np.array(data, dtype=np.float32)
    comp = WaterfallComputer(nfft=256, overlap=0.5)
    spec = comp.compute(x)
    assert spec.shape[1] == 256 // 2 + 1
    assert np.isfinite(spec).all()


def test_invalid_overlap():
    sr = 8000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    x = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    comp = WaterfallComputer(nfft=256, overlap=1.1)
    try:
        comp.compute(x)
    except ValueError:
        pass
    else:
        raise AssertionError("Se esperaba ValueError por overlap inválido")


def test_short_signal():
    x = np.zeros(128, dtype=np.float32)
    comp = WaterfallComputer(nfft=256, overlap=0.5)
    try:
        comp.compute(x)
    except ValueError:
        pass
    else:
        raise AssertionError("Se esperaba ValueError por señal corta")


def test_non_1d_signal():
    x = np.zeros((10, 10), dtype=np.float32)
    comp = WaterfallComputer(nfft=16, overlap=0.5)
    try:
        comp.compute(x)  # type: ignore[arg-type]
    except ValueError:
        pass
    else:
        raise AssertionError("Se esperaba ValueError por señal no mono (1D)")
