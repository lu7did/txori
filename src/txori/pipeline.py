"""Orquestación de la canalización de procesamiento."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from .capture import AudioInputCapture, CaptureController, SyntheticSineCapture
from .config import SystemConfig
from .fft_analysis import FFTAnalyzer
from .filtering import OnePoleLowPass
from .processing import IdentityProcessor
from .visualization import SpectrogramRenderer


@dataclass(slots=True)
class Pipeline:
    """Canalización completa: captura -> filtro -> diezmado -> proceso -> FFT -> visual."""

    cfg: SystemConfig
    capture: CaptureController = field(init=False)
    lpf: OnePoleLowPass = field(init=False)
    proc: IdentityProcessor = field(init=False)
    fft: FFTAnalyzer = field(init=False)
    renderer: SpectrogramRenderer = field(init=False)
    source_label: str = field(init=False)
    # Estado de temporización/diezmado
    _decim_factor: int = field(init=False, repr=False)
    _decim_rate: int = field(init=False, repr=False)
    _step_count: int = field(default=0, init=False, repr=False)
    _col_count: int = field(default=0, init=False, repr=False)
    _last_level: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        cap = (
            AudioInputCapture(self.cfg)
            if self.cfg.use_audio
            else SyntheticSineCapture(freq_hz=float(self.cfg.test_tone_hz), cfg=self.cfg)
        )
        self.capture = CaptureController(cap, window_size=self.cfg.window_size)
        self.lpf = OnePoleLowPass(
            fs=float(self.cfg.sample_rate), fc=float(self.cfg.cutoff_hz)
        )
        self.proc = IdentityProcessor()
        # Diezmado 48 kHz -> 6 kHz (1 de cada 8)
        self._decim_factor = max(1, int(self.cfg.sample_rate // 6000))
        self._decim_rate = int(self.cfg.sample_rate // self._decim_factor)
        n_bins = int(self.cfg.cutoff_hz // self.cfg.fft_bin_hz) + 1
        # FFT en dominio diezmado
        self.fft = FFTAnalyzer(
            fs=float(self._decim_rate),
            fc=float(self.cfg.cutoff_hz),
            bin_hz=float(self.cfg.fft_bin_hz),
        )
        # Render: una columna por actualización
        self.renderer = SpectrogramRenderer(
            height=n_bins,
            width=int(self.cfg.image_width),
            average_frames=int(self.cfg.average_frames),
            update_interval=1,
        )
        self.source_label = getattr(cap, "label", lambda: "Entrada")()

    def step(self) -> npt.NDArray[np.float64]:
        window = self.capture.step()
        filtered = self.lpf.process_window(window)
        # Nivel instantáneo tras pasabajos (tiempo real): módulo de la muestra más reciente
        self._last_level = float(abs(filtered[0]))
        # Diezmado para columna de espectrograma
        self._step_count += 1
        if self._step_count % self._decim_factor == 0:
            self._col_count += 1
            if self._col_count >= int(self.cfg.samples_per_col):
                self._col_count = 0
                # Ventana diezmada para análisis
                decimated_window = filtered[:: self._decim_factor]
                processed = self.proc.process(decimated_window)
                spectrum = self.fft.analyze(processed)
                self.renderer.push_spectrum(spectrum)
        return window

    def run(
        self,
        seconds: float | None = None,
        on_frame: Callable[[npt.NDArray[np.uint8], float | None], None] | None = None,
        on_time_sample: Callable[[float], None] | None = None,
        raw_input: bool = False,
    ) -> None:
        """Ejecuta; llama on_frame(img, nivel) con nuevas columnas y on_time_sample(sample) por muestra.

        raw_input: si True, on_time_sample recibe la muestra de ENTRADA sin filtrar.
        """
        start = time.perf_counter()
        while True:
            window = self.step()
            if on_time_sample is not None:
                sample = float(window[0] if raw_input else self._last_level or 0.0)
                on_time_sample(sample)
            if on_frame is not None:
                img = self.renderer.consume_frame()
                if img is not None:
                    on_frame(img, self._last_level)
            if seconds is not None and (time.perf_counter() - start) >= seconds:
                break
            # time.sleep(1.0 / self.cfg.sample_rate)  # opcional para uso real
