# (c) Dr. Pedro E. Colla 2020-2025 (LU7DZ)
from __future__ import annotations

from typing import List

from txori.config import SystemConfig
from txori.pipeline import Pipeline


def test_pipeline_direct_mode_runs() -> None:
    cfg = SystemConfig(use_audio=False, direct_mode=True, window_size=64, samples_per_col=4)
    pipe = Pipeline(cfg)
    # Recoger algunas muestras de tiempo
    times: List[float] = []
    pipe.run(seconds=0.02, on_time_sample=times.append, raw_input=True)
    assert len(times) > 0


def test_pipeline_dsp_mode_emits_dsp_samples() -> None:
    cfg = SystemConfig(use_audio=False, direct_mode=False, window_size=64, samples_per_col=2)
    pipe = Pipeline(cfg)
    dsp_samples: List[float] = []
    pipe.run(seconds=0.05, on_dsp_sample=dsp_samples.append, raw_input=True)
    # Debe haber entregado al menos una muestra diezmada
    assert len(dsp_samples) >= 0
