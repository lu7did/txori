# (c) Dr. Pedro E. Colla 2020-2025 (LU7DZ)
from __future__ import annotations

from txori.config import SystemConfig
from txori.pipeline import Pipeline


def test_pipeline_runs_and_renders(tmp_path) -> None:
    cfg = SystemConfig(
        use_audio=False, window_size=50, average_frames=10, update_interval=2
    )
    pipe = Pipeline(cfg)
    pipe.run(seconds=0.05)
    out = tmp_path / "spec.png"
    pipe.renderer.save(str(out))
    assert out.exists() and out.stat().st_size > 0
