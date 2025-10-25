"""
Orquestación de la canalización de procesamiento.

(c) Dr. Pedro E. Colla 2020-2025 (LU7DZ).
"""

# ruff: noqa: I001
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import time

import numpy as np
import numpy.typing as npt

from .capture import (
    AudioInputCapture,
    BaseCapture,
    CaptureController,
    SyntheticCWToneCapture,
    SyntheticCWToneGroupCapture,
    SyntheticSineCapture,
)
from .config import SystemConfig
from .filtering import OnePoleLowPass
from .processing import AGCProcessor, DSPProcessor, IdentityProcessor



@dataclass(slots=True)
class Pipeline:
    """Canalización completa: captura -> filtro -> diezmado -> proceso -> FFT -> visual."""

    cfg: SystemConfig
    capture: CaptureController = field(init=False)
    lpf: OnePoleLowPass = field(init=False)
    proc: IdentityProcessor = field(init=False)
    agc: AGCProcessor = field(init=False)
    dsp: DSPProcessor = field(init=False)

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
    _fs: float = field(init=False, repr=False)
    _fc: float = field(init=False, repr=False)
    _bin_hz: float = field(init=False, repr=False)
    _last_dsp_sample: float | None = field(default=None, init=False, repr=False)
    _new_dsp_sample: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        cap: BaseCapture
        if self.cfg.cw_mode:
            # Construir lista de portadoras y decidir si usar grupo
            freqs: list[float] = [float(self.cfg.cw_tone_hz)]
            if getattr(self.cfg, "qrn_mode", False):
                freqs.append(1000.0)
            if getattr(self.cfg, "qrm_mode", False):
                freqs.extend([200.0, 400.0, 800.0, 1200.0])
            if len(freqs) > 1 or getattr(self.cfg, "noise_mode", False):
                cap = SyntheticCWToneGroupCapture(
                    freqs_hz=freqs,
                    cfg=self.cfg,
                    noise_db=float(getattr(self.cfg, "noise_level_db", 20.0)),
                    with_noise=bool(getattr(self.cfg, "noise_mode", False)),
                )
            else:
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
        # Atenuación previa a FFT en ganancia lineal (10^(dB/10)) si fue indicada
        att_opt = getattr(self.cfg, "att_db", None)
        if att_opt is None:
            self._att_gain = 1.0
        else:
            att = float(att_opt)
            self._att_gain = 1.0 if abs(att) < 1e-12 else float(10.0 ** (att / 10.0))
        # Aplicar atenuación por muestra en captura
        try:
            self.capture.gain = float(self._att_gain)  # type: ignore[attr-defined]
        except Exception:
            pass
        # Especificación de bins/altura
        fmax = 24_000.0 if self._direct else 3_000.0
        # target_bins removed (unused)
        # FFT con tamaño de bin fijo: 30 Hz (direct) o 3.75 Hz (dsp)
        bin_hz_eff = 30.0 if self._direct else 3.75
        self._fs = float(self._decim_rate)
        self._fc = float(fmax)
        self._bin_hz = float(bin_hz_eff)
        # Waterfall eliminado: no se crea renderer
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
                    # Waterfall eliminado: no renderizamos espectro
        else:
            # DSP: LPF -> decimación 8:1 -> AGC -> DSP -> FFT
            filtered = self.lpf.process_window(window)
            # Pasar por AGC y DSP antes de exponer la muestra a la FFT
            proc_agc = self.agc.process(filtered)
            proc_dsp = self.dsp.process(proc_agc)
            self._last_level = float(abs(proc_dsp[0]))
            self._step_count += 1
            if self._step_count % self._decim_factor == 0:
                # Nueva muestra diezmada: tomar la más reciente
                import numpy as _np

                decim_sample = float(proc_dsp[0])
                self._dsp_buf = _np.roll(self._dsp_buf, 1)
                self._dsp_buf[0] = decim_sample
                # Exponer muestra diezmada para consumidores (espectrómetro en vivo)
                self._last_dsp_sample = float(decim_sample)
                self._new_dsp_sample = True
                self._col_count += 1
                if self._col_count >= int(self.cfg.samples_per_col):
                    self._col_count = 0
                    # Waterfall eliminado: no renderizamos espectro
        return window

    def _analyze(self, window: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        import numpy as _np
        n = int(window.size)
        if n <= 0:
            n_bins = int(_np.floor(float(self._fc) / max(float(self._bin_hz), 1e-9))) + 1
            return _np.zeros(n_bins, dtype=_np.float64)
        nfft = 1 << (max(1, n) - 1).bit_length()
        w = _np.hanning(n)
        xw = (window * w).astype(_np.float64)
        if nfft > n:
            xw = _np.pad(xw, (0, nfft - n))
        X = _np.fft.rfft(xw, n=nfft)
        freqs = _np.fft.rfftfreq(nfft, d=1.0 / float(self._fs))
        mask = freqs < (float(self._fc) - 1e-9)
        freqs = freqs[mask]
        power = (X.real**2 + X.imag**2)[mask]
        n_bins = int(_np.floor(float(self._fc) / max(float(self._bin_hz), 1e-9))) + 1
        out = _np.zeros(n_bins, dtype=_np.float64)
        idx = _np.floor(freqs / max(float(self._bin_hz), 1e-9)).astype(int)
        idx = _np.clip(idx, 0, n_bins - 1)
        _np.add.at(out, idx, power)
        return out

    def run(
        self,
        seconds: float | None = None,
        on_frame: Callable[[npt.NDArray[np.uint8], float | None], None] | None = None,
        on_time_sample: Callable[[float], None] | None = None,
        on_dsp_sample: Callable[[float], None] | None = None,
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
                if on_dsp_sample is not None and getattr(self, "_new_dsp_sample", False):
                    on_dsp_sample(float(getattr(self, "_last_dsp_sample", 0.0)))
                    self._new_dsp_sample = False
                if on_time_sample is not None:
                    sample = float(window[0] if raw_input else self._last_level or 0.0)
                    on_time_sample(sample)
            # Waterfall eliminado: no hay frames para consumir
            if seconds is not None and (time.perf_counter() - start) >= seconds:
                break
