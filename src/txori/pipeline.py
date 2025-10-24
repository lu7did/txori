"""
Orquestación de la canalización de procesamiento.

(c) Dr. Pedro E. Colla 2020-2025 (LU7DZ).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from .capture import (
    AudioInputCapture,
    BaseCapture,
    CaptureController,
    SyntheticCWToneCapture,
    SyntheticSineCapture,
)
from .config import SystemConfig
from .fft_analysis import FFTAnalyzer
from .filtering import OnePoleLowPass
from .processing import AGCProcessor, DSPProcessor, IdentityProcessor
from .visualization import SpectrogramRenderer


@dataclass(slots=True)
class Pipeline:
    """Canalización completa: captura -> filtro -> diezmado -> proceso -> FFT -> visual."""

    cfg: SystemConfig
    capture: CaptureController = field(init=False)
    lpf: OnePoleLowPass = field(init=False)
    proc: IdentityProcessor = field(init=False)
    agc: AGCProcessor = field(init=False)
    dsp: DSPProcessor = field(init=False)
    fft: FFTAnalyzer = field(init=False)
    renderer: SpectrogramRenderer = field(init=False)
    source_label: str = field(init=False)
    # Estado de temporización/diezmado
    _decim_factor: int = field(init=False, repr=False)
    _decim_rate: int = field(init=False, repr=False)
    _step_count: int = field(default=0, init=False, repr=False)
    _col_count: int = field(default=0, init=False, repr=False)
    _last_level: float | None = field(default=None, init=False, repr=False)
    _dsp_buf: npt.NDArray[np.float64] = field(init=False, repr=False)
    _direct: bool = field(init=False, repr=False)
    _samples_per_col_eff: int = field(init=False, repr=False)
    _att_gain: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        cap: BaseCapture
        if self.cfg.cw_mode:
            cap = SyntheticCWToneCapture(
                freq_hz=float(self.cfg.cw_tone_hz), cfg=self.cfg
            )
        else:
            cap = (
                AudioInputCapture(self.cfg)
                if self.cfg.use_audio
                else SyntheticSineCapture(
                    freq_hz=float(self.cfg.test_tone_hz), cfg=self.cfg
                )
            )
        # Modo de procesamiento
        self._direct = bool(self.cfg.direct_mode)
        # Ventana de captura (en directo usamos al menos 6000)
        win_size = max(self.cfg.window_size, 6000) if self._direct else self.cfg.window_size
        self.capture = CaptureController(cap, window_size=win_size)
        self.lpf = OnePoleLowPass(
            fs=float(self.cfg.sample_rate), fc=float(self.cfg.cutoff_hz)
        )
        self.proc = IdentityProcessor()
        self.agc = AGCProcessor()
        self.dsp = DSPProcessor()
        # Diezmado: directo=1; DSP=48 kHz -> 6 kHz (1 de cada 8)
        self._decim_factor = 1 if self._direct else max(1, int(self.cfg.sample_rate // 6000))
        self._decim_rate = int(self.cfg.sample_rate // self._decim_factor)
        # Velocidad de columnas controlada por spec_speed_factor
        self._samples_per_col_eff = max(1, int(round(self.cfg.samples_per_col / max(self.cfg.spec_speed_factor, 1e-9))))
        # Atenuación previa a FFT en ganancia lineal (10^(dB/10))
        att = float(getattr(self.cfg, "att_db", -40.0))
        self._att_gain = 1.0 if abs(att) < 1e-12 else float(10.0 ** (att / 10.0))
        # Aplicar atenuación por muestra en captura
        try:
            setattr(self.capture, "gain", float(self._att_gain))
        except Exception:
            pass
        n_bins = int(self.cfg.cutoff_hz // self.cfg.fft_bin_hz) + 1
        vbf = int(getattr(self.cfg, "vertical_bins_factor", 1))
        view_bins = max(1, int(n_bins) * max(1, vbf))
        # FFT
        self.fft = FFTAnalyzer(
            fs=float(self._decim_rate),
            fc=float(self.cfg.cutoff_hz),
            bin_hz=float(self.cfg.fft_bin_hz),
        )
        # Render: una columna por actualización
        self.renderer = SpectrogramRenderer(
            height=view_bins,
            width=int(max(self.cfg.image_width * 10, 1200)),

            average_frames=int(1 if self.cfg.cw_mode else self.cfg.average_frames),
            update_interval=1,
            pixels_per_bin=int(getattr(self.cfg, "pixels_per_bin", 1)),
        )
        # Buffer DSP de 6000 muestras (en dominio diezmado si aplica)
        import numpy as _np

        self._dsp_buf = _np.zeros(6000, dtype=_np.float64)
        self.source_label = getattr(cap, "label", lambda: "Entrada")()

    def step(self) -> npt.NDArray[np.float64]:  # pragma: no cover
        window = self.capture.step()
        if self._direct:
            # Modo directo: sin filtro ni decimación
            self._last_level = float(abs(window[0]))
            self._step_count += 1
            if self._step_count % self._decim_factor == 0:  # 1:1 por muestra
                self._col_count += 1
                if self._col_count >= int(self.cfg.samples_per_col):
                    self._col_count = 0
                    processed = self.proc.process(window)
                    spectrum = self.fft.analyze(processed)
                    h = int(self.renderer.height)
                    if spectrum.size != h:
                        import numpy as _np

                        spectrum = (
                            spectrum[:h]
                            if spectrum.size >= h
                            else _np.pad(spectrum, (0, h - spectrum.size))
                        )
                    self.renderer.push_spectrum(spectrum)
        else:
            # DSP: LPF -> decimación 8:1 -> AGC -> DSP -> FFT
            filtered = self.lpf.process_window(window)
            self._last_level = float(abs(filtered[0]))
            self._step_count += 1
            if self._step_count % self._decim_factor == 0:
                # Nueva muestra diezmada: tomar la más reciente
                import numpy as _np

                decim_sample = float(filtered[0])
                self._dsp_buf = _np.roll(self._dsp_buf, 1)
                self._dsp_buf[0] = decim_sample
                self._col_count += 1
                if self._col_count >= int(self.cfg.samples_per_col):
                    self._col_count = 0
                    agc_out = self.agc.process(self._dsp_buf)
                    dsp_out = self.dsp.process(agc_out)
                    spectrum = self.fft.analyze(dsp_out)
                    h = int(self.renderer.height)
                    if spectrum.size != h:
                        spectrum = spectrum[:h]
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
        chunk = 1 if self.cfg.use_audio else max(256, int(self.cfg.sample_rate // 40))  # audio: 1 muestra/iteración
        while True:
            # Procesar un bloque de muestras
            for _ in range(chunk):
                window = self.step()
                if on_time_sample is not None:
                    sample = float(window[0] if raw_input else self._last_level or 0.0)
                    on_time_sample(sample)
            # Actualizar espectrograma si hay nueva columna
            if on_frame is not None:
                img = self.renderer.consume_frame()
                if img is not None:
                    on_frame(img, self._last_level)
            if seconds is not None and (time.perf_counter() - start) >= seconds:
                break
