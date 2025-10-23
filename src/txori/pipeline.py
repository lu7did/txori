"""Orquestación de la canalización de procesamiento."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .capture import AudioInputCapture, CaptureController, SyntheticSineCapture
from .config import SystemConfig
from .fft_analysis import FFTAnalyzer
from .filtering import OnePoleLowPass
from .processing import IdentityProcessor
from .visualization import SpectrogramRenderer


@dataclass(slots=True)
class Pipeline:
    """Canalización completa: captura -> filtro -> proceso -> FFT -> visual."""

    cfg: SystemConfig
    capture: CaptureController = field(init=False)
    lpf: OnePoleLowPass = field(init=False)
    proc: IdentityProcessor = field(init=False)
    fft: FFTAnalyzer = field(init=False)
    renderer: SpectrogramRenderer = field(init=False)

    def __post_init__(self) -> None:
        cap = AudioInputCapture(self.cfg) if self.cfg.use_audio else SyntheticSineCapture(cfg=self.cfg)
        self.capture = CaptureController(cap, window_size=self.cfg.window_size)
        self.lpf = OnePoleLowPass(fs=float(self.cfg.sample_rate), fc=float(self.cfg.cutoff_hz))
        self.proc = IdentityProcessor()
        n_bins = int(self.cfg.cutoff_hz // self.cfg.fft_bin_hz) + 1
        self.fft = FFTAnalyzer(fs=float(self.cfg.sample_rate), fc=float(self.cfg.cutoff_hz), bin_hz=float(self.cfg.fft_bin_hz))
        self.renderer = SpectrogramRenderer(height=n_bins, width=int(self.cfg.image_width), average_frames=int(self.cfg.average_frames), update_interval=int(self.cfg.update_interval))

    def step(self) -> None:
        window = self.capture.step()
        filtered = self.lpf.process_window(window)
        processed = self.proc.process(filtered)
        spectrum = self.fft.analyze(processed)
        self.renderer.push_spectrum(spectrum)

    def run(self, seconds: float | None = None) -> None:
        """Ejecuta indefinidamente o durante `seconds` segundos (si se indica)."""
        start = time.perf_counter()
        period = 1.0 / float(self.cfg.sample_rate)
        while True:
            self.step()
            if seconds is not None and (time.perf_counter() - start) >= seconds:
                break
            # Bucle ligero: no dormimos exactamente period para no frenar tests
            # En uso real, capturaría por bloques; aquí se simula muestra a muestra.
            # time.sleep(period)  # opcional
