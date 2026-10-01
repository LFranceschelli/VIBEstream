"""
EBIV Release 5.2 — Centralised configuration.

All Release 5.0 parameters live here as plain dataclasses so that nothing new
is scattered as magic numbers through the processing loop.  Release 4.0's
convention (flat variables edited at the top of EBIV_Main.py) is preserved:
EBIV_Main.py still exposes flat, commented parameters and simply packs them
into these objects before calling into the library.

Nothing in this module imports hardware, numpy-heavy code, or the camera SDK,
so it can be imported by the offline test-suite on a machine with no camera,
no Analog Discovery and no CUDA.
"""

import os
from dataclasses import dataclass, field, asdict, is_dataclass
from typing import Optional, Tuple, List


# ==========================================================================
#  Version — SINGLE SOURCE
#
#  Everything that states a version reads this: the console banner, the live
#  stream header, and the 'release' field written into every
#  <run>_config.json.  Do not write the number anywhere else.  Before this
#  was centralised, VibeStream carried code well past Release 5.1 while still
#  stamping '5.1' into experiment logs, which makes a run unidentifiable
#  after the fact.
#
#  History:
#    5.0  closed-loop control, CPU-only pipeline, Release 4.0 audit
#    5.1  parameter GUI; laser starts in every run mode
#    5.2  'live' reference (operator sets the target with [+]/[-] while the
#         loop is closed), GPU correlation restored for the live stream and
#         fixed for the offline pyramid (audit B7/B8), single-file entry
#         point + launcher, DLL-dialog guard, single-source version
# ==========================================================================

__version__ = "1.0.0"          # VIBEstream release (evolution of rt-EBIV Release 5.2)


# ==========================================================================
#  Regions of interest
# ==========================================================================

@dataclass
class ROIConfig:
    """
    Two nested regions, both in FULL-SENSOR pixel coordinates.

    display_roi : [start_x, end_x, start_y, end_y] or None
        The region actually handed to the PIV engine and shown on screen.
        This is Release 4.0's `processing_roi`; the name is kept as an alias
        in EBIV_Main.py.  None = full sensor.

    control_roi : [start_x, end_x, start_y, end_y] or None
        The sub-region whose vectors feed the controller.  MUST be contained
        inside display_roi.  None = use the whole display_roi (and a warning
        is logged, because feeding the controller the entire field including
        the quiescent surroundings is rarely what you want).
    """
    display_roi: Optional[List[int]] = None
    control_roi: Optional[List[int]] = None

    def validate(self, sensor_w: int, sensor_h: int) -> Tuple[List[int], List[int]]:
        """Clamp both ROIs to the sensor and check containment.

        Returns (display_roi, control_roi) as concrete 4-lists in full-sensor
        coordinates.  Raises ValueError if control_roi is not inside display_roi.
        """
        d = list(self.display_roi) if self.display_roi is not None else [0, sensor_w, 0, sensor_h]
        d = [max(0, d[0]), min(sensor_w, d[1]), max(0, d[2]), min(sensor_h, d[3])]
        if d[1] <= d[0] or d[3] <= d[2]:
            raise ValueError(f"display_roi is empty after clamping: {d}")

        if self.control_roi is None:
            c = list(d)
        else:
            c = list(self.control_roi)

        if not (d[0] <= c[0] < c[1] <= d[1] and d[2] <= c[2] < c[3] <= d[3]):
            raise ValueError(
                f"control_roi {c} must be a non-empty region contained inside "
                f"display_roi {d} (both in full-sensor pixel coordinates)."
            )
        return d, c


# ==========================================================================
#  Feedback velocity definition
# ==========================================================================

@dataclass
class MeasurementConfig:
    """
    How the scalar feedback quantity U_control is distilled from the vector
    field inside the ControlROI.

    component :
        'u'      streamwise = +U   (jet flowing left -> right in the image)
        '-u'     streamwise = -U   (right -> left)
        'v'      streamwise = +V   (downward;  V is positive DOWN in this code)
        '-v'     streamwise = -V   (upward)
        'magnitude'  sqrt(U^2+V^2) — NOT recommended for a jet: it is
                 positive-definite, so it cannot distinguish forward from
                 reverse flow and it biases high in noisy regions.

    statistic : 'mean' or 'median'.  Median is more robust to surviving
        outliers but is slightly more expensive and has a mildly non-linear
        response to the vector population.

    min_valid_fraction :
        Fraction of the ControlROI nodes that must be valid for the
        measurement to be declared usable.  Below this the measurement is
        flagged INVALID and never reaches the PID.

    min_valid_vectors :
        Absolute floor on the number of valid vectors, applied in addition to
        min_valid_fraction.

    use_correlation_gate / min_correlation :
        Reject individual vectors whose normalised correlation peak is below
        min_correlation.  This is what catches the Release 4.0 failure mode
        in which a flat (all-zero) correlation plane silently produces a
        displacement of -window_size/2 px.  See AUDIT_R4_to_R5.md, bug B2.

    max_abs_displacement_px :
        Physically implausible displacements are rejected.  Set to None to
        disable.  A sensible value is slightly below window_size/2.

    calibration_px_per_mm :
        Optical resolution of the imaging system, in pixels per millimetre —
        the number you read off a calibration target.  Set it and EVERYTHING
        (set-point, measurement, logs, plots, PID gains, filter, HUD) is in
        m/s.  Leave it None and everything stays in px/frame.

            U [m/s] = U [px/pulse] / calibration_px_per_mm * 1e-3 / dt

        with dt the pulse separation (below).  Nothing else in the code
        changes: the conversion is applied ONCE, at the measurement, so the
        whole loop remains internally consistent in whichever unit you chose.

    pulse_separation_s :
        The dt used above, in seconds.  Leave it None and it is derived from
        the acquisition setting: consecutive pseudo-frames are one
        accumulation window apart, so dt = accum_time_us * 1e-6 = 1/f_acq.
        With the laser running at f_acq — one pulse per pseudo-frame — that
        is exactly 1/f_laser.

        Set it explicitly ONLY if the frame separation is not 1/f_acq, for
        example if you illuminate with pulse pairs rather than one pulse per
        accumulation window.  If laser_frequency_hz and f_acq disagree,
        start-up says so, because then one of the two is misconfigured.

    uncalibrated_units :
        What the feedback quantity means when calibration_px_per_mm is NOT
        set.  Only relevant in that case: with a calibration everything is
        m/s regardless.

            'px/s'      (default) divide the per-frame displacement by dt, so
                        the number is a VELOCITY.  Changing f_acq then leaves
                        the measurement, the set-point and the PID gains
                        meaning the same thing.
            'px/frame'  the raw correlation displacement, as Release 4.0 and
                        Release 5.0/5.1 reported it.  Use it to reproduce
                        older runs; be aware that it scales with 1/f_acq, so
                        a gain set tuned at one acquisition rate is wrong at
                        another.

    velocity_scale, velocity_units :
        The low-level form of the same thing.  U_control is computed in
        px/frame and multiplied by velocity_scale once, at the measurement.

        velocity_scale = 1.0 keeps everything in px/frame (the default: no
        calibration is assumed anywhere).  If calibration_px_per_mm is set,
        resolve_units() overwrites velocity_scale and velocity_units from it,
        and the resolved values are what get written into the run header.

        NOTHING in the control law depends on this choice.  But note that the
        PID gains are in volts per (velocity unit), so switching units
        rescales them: with px/mm = 20 and f = 500 Hz the scale is 0.025, and
        gains expressed in m/s must be 40x larger than the same controller
        expressed in px/frame.  Start-up prints the factor.
    """
    component: str = 'u'
    statistic: str = 'mean'
    min_valid_fraction: float = 0.5
    min_valid_vectors: int = 4
    use_correlation_gate: bool = True
    min_correlation: float = 0.05
    max_abs_displacement_px: Optional[float] = None
    uncalibrated_units: str = 'px/s'      # 'px/s' | 'px/frame'
    calibration_px_per_mm: Optional[float] = None
    pulse_separation_s: Optional[float] = None
    velocity_scale: float = 1.0
    velocity_units: str = 'px/frame'

    def resolve_units(self, frame_dt_s, laser_frequency_hz=None):
        """
        Turn calibration_px_per_mm into velocity_scale, once, at start-up.

        frame_dt_s : the separation between the two correlated pseudo-frames,
            in seconds (accum_time_us * 1e-6 for the real-time path).

        Returns a dict describing the resolved conversion, for logging.  Safe
        to call when no calibration is configured: it then reports the
        px/frame identity scaling and changes nothing.
        """
        dt = float(self.pulse_separation_s) if self.pulse_separation_s else float(frame_dt_s)
        if dt <= 0:
            raise ValueError(f"pulse separation must be > 0 s, got {dt}")

        info = {
            'dt_s': dt,
            'dt_source': ('pulse_separation_s (explicit)'
                          if self.pulse_separation_s else
                          'accumulation window (1/f_acq)'),
            'px_per_mm': self.calibration_px_per_mm,
            'laser_frequency_hz': laser_frequency_hz,
            'f_from_dt_hz': 1.0 / dt,
            'laser_mismatch': False,
        }

        if laser_frequency_hz and self.pulse_separation_s is None:
            # One pulse per pseudo-frame means 1/f_laser == dt.  A mismatch is
            # not automatically wrong (pulse-pair illumination is legitimate)
            # but it does mean the derived dt is a guess, so say so.
            if abs(laser_frequency_hz - 1.0 / dt) > 0.01 * (1.0 / dt):
                info['laser_mismatch'] = True

        # ALWAYS derive from the DECLARED intent, never from the previously
        # resolved value.  resolve_units() is called again on every run, and
        # the GUI keeps ONE Session alive across runs: without this, changing
        # f_acq and pressing Run a second time left the old scale in place and
        # every velocity was silently wrong by the ratio of the two rates.
        declared = self._declared_scale()

        if self.calibration_px_per_mm is None:
            # No optical calibration, so we cannot reach m/s.  We CAN still
            # report a velocity rather than a per-frame displacement, and that
            # matters: px/frame is not a property of the flow, it is a
            # property of how fast you happen to be sampling it.  Change f_acq
            # and every px/frame number, set-point and PID gain silently
            # changes meaning.  px/s does not.
            if declared != 1.0:
                self.velocity_scale = declared          # explicit override
            elif self.uncalibrated_units == 'px/s':
                self.velocity_scale = 1.0 / dt
                self.velocity_units = 'px/s'
            else:
                self.velocity_scale = 1.0
                self.velocity_units = 'px/frame'
            info['scale'] = self.velocity_scale
            info['units'] = self.velocity_units
            info['px_per_unit'] = 1.0 / self.velocity_scale
            info['uncalibrated_basis'] = self.uncalibrated_units
            return info

        if self.calibration_px_per_mm <= 0:
            raise ValueError("calibration_px_per_mm must be > 0")

        # px/frame -> mm/frame -> m/frame -> m/s
        self.velocity_scale = 1e-3 / float(self.calibration_px_per_mm) / dt
        self.velocity_units = 'm/s'
        info['scale'] = self.velocity_scale
        info['units'] = self.velocity_units
        info['overrode_velocity_scale'] = (declared != 1.0)
        info['px_per_unit'] = 1.0 / self.velocity_scale
        return info

    # ------------------------------------------------------------------
    #  Declared intent vs resolved value
    #
    #  velocity_scale and velocity_units are OVERWRITTEN by resolve_units, so
    #  the field alone cannot say what the user asked for.  The first
    #  resolution snapshots the declared values; every later one derives from
    #  that snapshot, and restore_declared() puts them back before a preset is
    #  written.  Without the latter, saving a preset after a run stored the
    #  RESOLVED scale, which reloaded as an explicit override and pinned the
    #  units to whatever f_acq happened to be that day.
    # ------------------------------------------------------------------

    def _declared_scale(self):
        if not hasattr(self, '_scale_declared'):
            self._scale_declared = float(self.velocity_scale)
            self._units_declared = self.velocity_units
        return self._scale_declared

    def restore_declared(self):
        """Undo resolve_units(), leaving the user's declared intent."""
        if hasattr(self, '_scale_declared'):
            self.velocity_scale = self._scale_declared
            self.velocity_units = self._units_declared
        return self

    def to_px(self, value_in_display_units):
        """Inverse conversion: a set-point in display units -> px/frame."""
        return float(value_in_display_units) / self.velocity_scale

    def __post_init__(self):
        if self.uncalibrated_units not in ('px/s', 'px/frame'):
            raise ValueError("uncalibrated_units must be 'px/s' or 'px/frame', "
                             f"got {self.uncalibrated_units!r}")
        allowed = ('u', '-u', 'v', '-v', 'magnitude')
        if self.component not in allowed:
            raise ValueError(f"component must be one of {allowed}, got {self.component!r}")
        if self.statistic not in ('mean', 'median'):
            raise ValueError("statistic must be 'mean' or 'median'")


# ==========================================================================
#  Temporal filtering
# ==========================================================================

@dataclass
class FilterConfig:
    """
    Temporal filter applied to U_control_raw before the PID sees it.

    kind :
        'none'  no filtering.  U_control_filtered == U_control_raw.
        'ma'    moving average over the last `window` VALID samples.
        'ema'   first-order low-pass (exponential moving average).

    window : int
        Moving-average length, in samples.  A length-N moving average delays
        the signal by roughly (N-1)/2 samples.  At a control rate f_c this is
        a phase lag of about (N-1)/(2*f_c) seconds, which eats directly into
        your phase margin.  The value is written into every log header so
        that the delay you chose is never invisible.

    tau_s : float
        EMA time constant in SECONDS (not a dimensionless alpha).  The filter
        recomputes alpha = dt/(tau + dt) from the MEASURED dt at every update,
        so the cut-off stays correct even when the control loop jitters.
        Group delay is approximately tau seconds.
    """
    kind: str = 'ema'
    window: int = 5
    tau_s: float = 0.10

    def __post_init__(self):
        if self.kind not in ('none', 'ma', 'ema'):
            raise ValueError("filter kind must be 'none', 'ma' or 'ema'")
        if self.window < 1:
            raise ValueError("filter window must be >= 1")
        if self.tau_s <= 0:
            raise ValueError("tau_s must be > 0")

    def describe(self) -> str:
        if self.kind == 'none':
            return "none (no added phase lag)"
        if self.kind == 'ma':
            return (f"moving average, window={self.window} samples "
                    f"(~{(self.window - 1) / 2.0:.1f} samples of group delay)")
        return f"EMA, tau={self.tau_s:.3f} s (~{self.tau_s:.3f} s of group delay)"


# ==========================================================================
#  Reference / set-point generator
# ==========================================================================

@dataclass
class ReferenceConfig:
    """
    kind :
        'constant'  U_target = value
        'live'      U_target starts at value and is then moved by the operator
                    with [+] and [-] WHILE THE LOOP IS CLOSED.  The keys move
                    the TARGET VELOCITY, not the pump voltage: the PID decides
                    the voltage.  (Moving the voltage by hand is what run_mode
                    'manual' is for, and that has no controller at all.)
        'step'      U_target = value before t_step, then value + step_amplitude
        'sine'      U_target = value + amplitude*sin(2*pi*freq_hz*t + phase)
        'file'      U_target interpolated from a two-column time series
        'random'    band-limited pseudo-random walk (reproducible via seed)

    All times are seconds measured from the moment closed-loop control is
    ARMED, not from program start.
    """
    kind: str = 'constant'
    value: float = 0.0

    # live: operator-driven target.  Step and limits are in DISPLAY UNITS
    # (m/s if a px/mm calibration is set, otherwise px/frame), the same units
    # as `value`.  live_min/live_max are the range the operator is allowed to
    # ask for; leaving them None means unbounded, which is legal but means a
    # held-down key can walk the target far past anything the pump can reach.
    live_step: float = 0.1
    live_min: Optional[float] = None
    live_max: Optional[float] = None

    # step
    t_step_s: float = 5.0
    step_amplitude: float = 0.0

    # sine
    amplitude: float = 0.0
    freq_hz: float = 0.1
    phase_rad: float = 0.0

    # file: two columns (time_s, target).  .csv, .txt or .npy
    filepath: Optional[str] = None
    file_loop: bool = True

    # band-limited random
    random_seed: int = 12345
    random_cutoff_hz: float = 0.05     # bandwidth of the reference
    random_std: float = 0.0            # standard deviation about `value`
    random_clip: Optional[Tuple[float, float]] = None

    def __post_init__(self):
        allowed = ('constant', 'live', 'step', 'sine', 'file', 'random')
        if self.kind not in allowed:
            raise ValueError(f"reference kind must be one of {allowed}")
        if self.kind == 'live':
            if self.live_step == 0:
                raise ValueError("reference kind 'live' needs a non-zero live_step")
            if (self.live_min is not None and self.live_max is not None
                    and self.live_max < self.live_min):
                raise ValueError("reference live_max must be >= live_min")
        # NOTE: kind='file' also needs a filepath, but that is checked in
        # Session.validate() rather than here.  A GUI user selects the kind
        # from a combo box and only then browses for the file, so rejecting
        # the intermediate state at construction time would make the control
        # unusable.  Validation at Run time is both later and stricter: it
        # checks the file actually exists.


# ==========================================================================
#  PID
# ==========================================================================

@dataclass
class PIDConfig:
    """
    Discrete-time PID acting on  error = U_target - U_control_filtered  and
    producing a pump voltage.

    Gains are in VOLTS per (velocity unit), i.e. they depend on
    MeasurementConfig.velocity_scale.  Determine them from an open-loop
    calibration run (FLAG_PUMP_CALIBRATION); do not guess.

    kp, ki, kd :
        Proportional [V / unit], integral [V / (unit*s)], derivative
        [V*s / unit].

    v_min, v_max :
        Hard output saturation, volts.  Also enforced independently in the
        pump backend, which will refuse to output anything outside its own
        configured limits regardless of what the PID asks for.

    slew_rate_v_per_s :
        Maximum rate of change of the commanded voltage.  None disables.
        The EcoDrift is a magnetically-coupled propeller pump; slewing its
        command faster than it can physically follow only injects noise.

    derivative_on_measurement :
        True  -> D acts on -d(measurement)/dt   (no set-point kick on steps)
        False -> D acts on  d(error)/dt

    derivative_filter_tau_s :
        First-order filter on the derivative term, seconds.  0 disables.
        Differentiating a noisy velocity estimate without this is usually
        useless.

    anti_windup :
        'clamp'      stop integrating while saturated and pushing further out
        'back_calc'  back-calculation with gain `back_calc_gain`
        'none'       no protection (not recommended)

    integral_limit :
        Optional hard bound on the integral term's contribution, volts.
    """
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0

    v_min: float = 0.0
    v_max: float = 5.0

    slew_rate_v_per_s: Optional[float] = 2.0
    derivative_on_measurement: bool = True
    derivative_filter_tau_s: float = 0.05

    anti_windup: str = 'clamp'
    back_calc_gain: float = 1.0
    integral_limit: Optional[float] = None

    def __post_init__(self):
        if self.v_min >= self.v_max:
            raise ValueError("v_min must be < v_max")
        if self.anti_windup not in ('clamp', 'back_calc', 'none'):
            raise ValueError("anti_windup must be 'clamp', 'back_calc' or 'none'")


# ==========================================================================
#  Supervisor: rates, dropouts, safe state
# ==========================================================================

@dataclass
class SupervisorConfig:
    """
    control_rate_hz :
        Rate at which the controller runs and a new voltage is issued.  This
        is INDEPENDENT of the EBIV field rate.  The EcoDrift's hydraulic time
        constant is of order seconds, so there is no reason to command it at
        the EBIV rate; 5-20 Hz is a sensible starting range.  Set it from the
        measured step response, not from the camera rate.

    visualization_rate_hz :
        Main image window refresh rate.  Purely cosmetic.

    plot_rate_hz :
        Time-history strip-chart refresh rate.

    hold_timeout_s :
        How long the controller keeps holding its last output when the
        measurement is invalid or stale.  Short dropouts (a few bad frames)
        are normal in EBIV and should not disturb the pump.

    safe_timeout_s :
        After this long without a valid measurement the controller gives up
        and drives the pump to safe_pump_voltage.  MUST be > hold_timeout_s.

    max_measurement_age_s :
        A measurement whose DATA is older than this is treated as stale even
        if it was valid when produced.  Age is counted from when the light
        arrived, not from when the correlation finished, so it covers the
        whole chain: sensor -> Metavision SDK buffer -> event accumulation ->
        queue -> correlation.  That matters because the dominant term is
        usually the SDK buffer: if the acquisition loop cannot keep up with
        f_acq, events queue inside the SDK and every frame is older than the
        last, without bound.  The run header reports the acquisition lag.

    safe_pump_voltage :
        The voltage commanded on shutdown, on an unrecoverable fault, and
        after safe_timeout_s.  DO NOT assume 0 V is safe or appropriate:
        determine experimentally what your facility should do when the
        controller loses sight of the flow.  Holding a low but non-zero flow
        is often preferable to stopping the pump entirely.

    startup_pump_voltage :
        Voltage applied when the pump output is first enabled, before any
        control action.  Also the manual-mode starting point.
    """
    control_rate_hz: float = 10.0
    visualization_rate_hz: float = 20.0
    plot_rate_hz: float = 5.0

    hold_timeout_s: float = 0.5
    safe_timeout_s: float = 3.0
    max_measurement_age_s: float = 0.5

    safe_pump_voltage: float = 0.0
    startup_pump_voltage: float = 0.0

    def __post_init__(self):
        if self.safe_timeout_s <= self.hold_timeout_s:
            raise ValueError("safe_timeout_s must be greater than hold_timeout_s")
        for r in (self.control_rate_hz, self.visualization_rate_hz, self.plot_rate_hz):
            if r <= 0:
                raise ValueError("all rates must be > 0 Hz")


# ==========================================================================
#  Hardware (Analog Discovery 3)
# ==========================================================================

@dataclass
class AD3Config:
    """
    Analog Discovery 3 channel allocation.

    IMPORTANT — device ownership.  The WaveForms desktop application takes an
    EXCLUSIVE lock on the device.  Release 5.0 opens the AD3 itself, so
    WaveForms must be CLOSED (or the device released from it) before starting
    a run.  Because Python then owns the device, Release 5.0 must also
    generate the laser waveform: if the GUI is closed and Python does not
    drive the laser, there is no laser.

    Channel allocation (defaults):
        laser  -> Analog Out channel 0  (W1) : continuous square/pulse train
        pump   -> Analog Out channel 1  (W2) : slowly varying DC, 0..5 V

    If your laser is driven from a DIO pin instead, set
    laser_backend='pattern' and laser_dio_channel; the pump then defaults to
    Analog Out channel 0.

    The laser channel is configured ONCE at start-up and never touched again.
    Pump updates use FDwfAnalogOutNodeOffsetSet followed by
    FDwfAnalogOutConfigure(hdwf, pump_channel, 3) — 'apply dynamically without
    changing the state of the instrument', per the WaveForms SDK reference.
    idxChannel is ALWAYS the specific channel; -1 (all channels) is never used
    anywhere in this codebase.
    """
    enabled: bool = False                 # False -> mock backend, no hardware
    device_index: int = -1                # -1 = first available

    # --- laser ---
    laser_backend: str = 'analog'         # 'analog' | 'pattern' | 'external'
    laser_channel: int = 0                # analog-out channel index (W1)
    laser_dio_channel: int = 0            # used when laser_backend='pattern'
    # LASER RATE.  None (the default) means "follow the acquisition
    # frequency": the laser fires once per pseudo-frame, so 1/f_laser is
    # exactly the frame separation dt the velocity is computed from.  That is
    # what you want in almost every run, and keeping it derived removes a
    # whole class of silent errors — two numbers that must agree, edited in
    # two places, with only a warning if they drift apart.
    #
    # Set a NUMBER only when the laser genuinely is not one pulse per frame:
    #   * laser_backend='external' — the AD3 does not drive the laser and this
    #     field just records what the external source is doing;
    #   * pulse-pair illumination — two pulses per accumulation window, in
    #     which case also set MeasurementConfig.pulse_separation_s, because dt
    #     is then NOT 1/f_acq and nothing can infer it.
    # A number that merely disagrees with f_acq by accident still gets the
    # start-up warning it always did.
    laser_frequency_hz: Optional[float] = None
    laser_amplitude_v: float = 2.5        # square wave amplitude (half p-p)
    laser_offset_v: float = 2.5           # -> 0..5 V TTL-ish square
    laser_duty_percent: float = 10.0      # pulse width as % of the period
    # What the laser pin does when the channel is NOT running (between runs,
    # after close, after a crash).  True selects DwfAnalogOutIdleDisable: the
    # AD3 stops driving the pin.  False parks it at the configured offset,
    # which with offset=2.5 V is a continuously-triggered laser.
    #
    # Until Release 5.2 True selected DwfAnalogOutIdleInitial, meaning the
    # first sample of the waveform — and a WaveForms square starts HIGH, so
    # "idle low" actually parked the output at offset+amplitude = 5 V and the
    # laser went to full power on every close.
    #
    # Even with True the pin is left high-impedance, which a pulled-up laser
    # input reads as HIGH.  close() therefore drives a real 0 V first.  For a
    # hard guarantee, fit a pull-down resistor at the laser trigger input.
    laser_idle_low: bool = True

    # --- pump ---
    pump_channel: int = 1                 # analog-out channel index (W2)
    pump_v_min: float = 0.0               # hardware-enforced floor
    pump_v_max: float = 5.0               # hardware-enforced ceiling
    #  The EcoDrift 4.3 accepts 0-10 V on its auxiliary control input.  We
    #  deliberately command only 0-5 V.  These limits are enforced inside the
    #  pump backend and are the LAST line of defence: a command outside them
    #  is clamped and logged, never passed to the device.

    def __post_init__(self):
        if self.laser_backend not in ('analog', 'pattern', 'external'):
            raise ValueError("laser_backend must be 'analog', 'pattern' or 'external'")
        if self.laser_backend == 'analog' and self.laser_channel == self.pump_channel:
            raise ValueError(
                f"laser_channel and pump_channel are both {self.pump_channel}. "
                "The laser and the pump must use different Analog Out channels."
            )
        if not (0.0 <= self.pump_v_min < self.pump_v_max <= 10.0):
            raise ValueError("require 0 <= pump_v_min < pump_v_max <= 10 V")
        if not (0.0 < self.laser_duty_percent < 100.0):
            raise ValueError("laser_duty_percent must be in (0, 100)")


# ==========================================================================
#  Open-loop calibration sweep
# ==========================================================================

@dataclass
class CalibrationConfig:
    """
    Open-loop characterisation of  V_command -> U_control.

    mode :
        'manual'  drive the voltage from the keyboard ([+]/[-]) while rtEBIV
                  keeps measuring.  Nothing is automated.
        'steps'   hold each voltage in `voltages` for dwell_s seconds.
        'sweep'   linear ramp from v_start to v_stop over sweep_duration_s,
                  optionally back down again.

    Every mode logs timestamp, commanded voltage, raw and filtered velocity
    and the valid-vector count, so that both the static curve and the
    transient response can be extracted afterwards.
    """
    mode: str = 'manual'
    voltages: List[float] = field(default_factory=lambda: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    dwell_s: float = 20.0
    settle_s: float = 10.0        # discarded head of each dwell, for the static curve
    v_start: float = 0.0
    v_stop: float = 5.0
    sweep_duration_s: float = 120.0
    sweep_return: bool = True     # sweep back down to expose hysteresis
    manual_step_v: float = 0.1    # keyboard increment

    def __post_init__(self):
        if self.mode not in ('manual', 'steps', 'sweep'):
            raise ValueError("calibration mode must be 'manual', 'steps' or 'sweep'")
        if self.settle_s >= self.dwell_s:
            raise ValueError("settle_s must be shorter than dwell_s")


# ==========================================================================
#  Logging
# ==========================================================================

@dataclass
class LoggingConfig:
    """
    enabled :
        Master switch for the per-control-step experiment log.

    formats :
        Any of 'csv', 'mat'.  CSV is written incrementally (crash-safe, and
        readable while the run is in progress); the .mat is written once at
        the end using scipy.io.savemat, matching Release 4.0's convention so
        the same MATLAB tooling reads it.

    flush_every :
        CSV rows buffered between flushes.  0 flushes every row (safest,
        slightly more I/O).

    latency_report_period_s :
        How often the periodic timing/latency summary is printed.  Per-frame
        printing is never done.

    (Release 5.1 removed save_field_snapshots / field_snapshot_period_s: they
    were declared here but never read by anything, so they were a knob that
    silently did nothing.  Full velocity fields are saved by the separate
    RT-PIV .mat path, RunConfig.save_rt_piv or the [f] key.)
    """
    enabled: bool = True
    formats: Tuple[str, ...] = ('csv', 'mat')
    flush_every: int = 50
    latency_report_period_s: float = 10.0


# ==========================================================================
#  PIV engine (Release 5.0 CPU correlator)
# ==========================================================================

@dataclass
class PIVConfig:
    """
    window_size, node_distance, triple_corr :
        As Release 4.0.

    fft_workers :
        Threads used by scipy.fft for the batched transform.  1 reproduces
        Release 4.0 exactly.  >1 is numerically identical (the batch is split
        across threads) and helps on large grids; on small grids the thread
        overhead makes it slower.  Measure with bench_release5.py.

    subpixel :
        Release 4.0's real-time path returns INTEGER pixel displacements —
        neither the CPU nor the GPU real-time backend performs sub-pixel
        interpolation (only the offline pyramidal path does).  At a typical
        5 px displacement that is a 20% quantisation of the control signal.
        Release 5.0 enables the same 3-point Gaussian estimator used offline.
        Set False to reproduce Release 4.0's real-time output bit-for-bit.

    compute_quality :
        Compute the normalised correlation coefficient per vector.  Needed by
        the ControlROI correlation gate.  Costs one extra pass over the
        window buffers.

    fast_mean_removal :
        Remove each window's mean by zeroing the DC bin of its FFT instead of
        subtracting it in the spatial domain.  Mathematically identical, but
        NOT bit-identical in float32: measured max relative difference on the
        correlation plane ~1.5e-7, with up to ~0.3% of integer peak positions
        flipping on tie-prone sparse data.  Measured 1.2-1.4x faster on the
        full pipeline.  OFF by default; turn it on only if you have satisfied
        yourself that the difference is irrelevant for your case.

    min_frames_before_output :
        Number of frames that must have been accumulated before any velocity
        field is published.  Release 4.0 correlates against a zero-filled
        buffer on the first iteration, which produces a spurious
        (-ws/2, -ws/2) displacement.  See AUDIT_R4_to_R5.md, bug B1.

    use_gpu :
        Run the correlation on the GPU through PyTorch.  Applies to BOTH the
        live stream (including closed-loop runs) and the offline pyramidal
        PIV.  Needs torch built with CUDA; without it the run falls back to
        the CPU and says so, so leaving this True on a machine with no GPU is
        harmless.

        Live stream.  GPUCorrelator.correlate() returns (U, V, CC) with
        sub-pixel interpolation, exactly like CPUCorrelator, so the ControlROI
        estimator and the PID cannot tell the difference: measured agreement
        is ~3e-6 px on the displacement and ~3e-6 relative on CC, and the
        integer peaks are identical.  Release 4.0's GPU real-time kernel
        (batch_cross_correlate) did NOT do either and is not used.

        What is NOT established is latency.  A GPU buys throughput; a control
        loop cares about per-field delay and its jitter, and the host-device
        round trip adds both.  Compare the latency summary of a GPU run
        against a CPU run before trusting PID gains tuned on one of them.

        Offline pyramidal PIV.  Release 4.0's version was wrong twice (audit
        B7, B8) and could not complete a single snapshot; both are fixed in
        Release 5.2.  CC is NOT computed on this path and is written as NaN.

        All of the above was verified on CPU tensors, since the machine that
        wrote it has no CUDA.  Run tools/check_gpu.py on YOUR machine once.
    """
    window_size: int = 48
    node_distance: int = 24
    triple_corr: bool = False
    fft_workers: int = 1
    subpixel: bool = True
    compute_quality: bool = True
    fast_mean_removal: bool = False
    use_gpu: bool = False

    # Cap on how often the real-time PIV thread is given work, in Hz.
    # None = as fast as the correlator can go, which is what Release 4.0 did.
    #
    # Why you might want a cap.  The acquisition loop and the PIV worker are
    # threads in ONE process, so they share the GIL.  scipy's FFT releases it,
    # but window extraction, the mean removal, argmax and the sub-pixel fit do
    # not.  A PIV worker running flat out at 40 Hz therefore holds the GIL for
    # a large fraction of every second, and the acquisition loop -- which must
    # come round every 1/f_acq to keep up with the sensor -- gets starved.
    # Events then queue inside the Metavision SDK and the whole display falls
    # progressively behind reality.
    #
    # The controller does not need 40 Hz.  It runs at control_rate_hz, and a
    # measurement it never looks at is pure GIL contention.  Two to four times
    # the control rate is plenty.
    rt_max_rate_hz: Optional[float] = None
    validation: bool = True
    val_threshold: float = 2.0
    val_epsilon: float = 0.1
    min_frames_before_output: int = 2


# ==========================================================================
#  Top-level container
# ==========================================================================

@dataclass
class ControlSystemConfig:
    """Everything Release 5.0 adds on top of Release 4.0, in one object."""
    roi: ROIConfig = field(default_factory=ROIConfig)
    measurement: MeasurementConfig = field(default_factory=MeasurementConfig)
    filt: FilterConfig = field(default_factory=FilterConfig)
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    pid: PIDConfig = field(default_factory=PIDConfig)
    supervisor: SupervisorConfig = field(default_factory=SupervisorConfig)
    ad3: AD3Config = field(default_factory=AD3Config)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    piv: PIVConfig = field(default_factory=PIVConfig)

    def cross_check(self):
        """Consistency checks that span more than one sub-config."""
        problems = []
        if self.pid.v_min < self.ad3.pump_v_min or self.pid.v_max > self.ad3.pump_v_max:
            problems.append(
                f"PID output range [{self.pid.v_min}, {self.pid.v_max}] V is not "
                f"contained in the pump hardware range "
                f"[{self.ad3.pump_v_min}, {self.ad3.pump_v_max}] V."
            )
        sv = self.supervisor.safe_pump_voltage
        if not (self.ad3.pump_v_min <= sv <= self.ad3.pump_v_max):
            problems.append(
                f"safe_pump_voltage {sv} V is outside the pump hardware range "
                f"[{self.ad3.pump_v_min}, {self.ad3.pump_v_max}] V."
            )
        if self.supervisor.control_rate_hz > 100:
            problems.append(
                f"control_rate_hz={self.supervisor.control_rate_hz} Hz is very high "
                "for a propeller pump; the actuator cannot follow it."
            )
        if problems:
            raise ValueError("Configuration problems:\n  - " + "\n  - ".join(problems))
        return True

    def to_dict(self):
        """Flat-ish dict for logging into the experiment header."""
        return asdict(self)


# ==========================================================================
#  Acquisition and offline-pipeline parameters
# ==========================================================================

@dataclass
class HRConfig:
    """
    High-resolution estimation (vibe_hr / vibe_train): the LR rt-EBIV field is
    turned into an HR field in POD space by one of the paper's three
    estimators.  The LR settings are NOT repeated here: training uses the live
    PIV & ROI settings, so the model always matches the live stream.

    Live (stream mode)
        enabled, model_path, method, steady_state, show_hr
    Where the parameters come from (source)
        'raw'        a .raw recording: LR and HR PIV computed here, then POD and
                     operators (below)
        'fields'     LR / HR fields computed elsewhere (ext_lr_file, ext_hr_file,
                     MATLAB layout) or a saved training set .npz: POD and
                     operators computed here
        'operators'  operators computed elsewhere (ext_model_file, written by
                     tools/matlab/vibe_export_model.m): only converted
        external conventions: ext_y_up, ext_velocity_units, ext_f_hz,
        ext_sensor_w / _h (see vibe_import)
    Training (offline step 'train HR model')
        HR processing  : hr_window, hr_step, hr_levels, hr_stencil,
                         hr_combine, hr_predictor
        frames         : train_raw (blank = the Run tab's raw file),
                         train_start_frame, train_n_frames (0 = all)
        split          : n_train | gap | n_val | gap | n_test (fields)
        POD / operators: rank (0 = elbow), truncate_lr, elbow_threshold,
                         elbow_smooth, elbow_span, lambda_c, vr_max_gain
        estimator      : tune_q, tune_r (Q, R multipliers; live and test)
        output         : model_name (in Out/<acq name>/), u_ref (0 = auto),
                         save_training_set, export_matlab
    """
    # --- live ---------------------------------------------------------
    enabled: bool = False
    model_path: Optional[str] = None
    method: str = 'kf'                    # 'kf' | 'lse' | 'lse_vr'
    steady_state: bool = True
    show_hr: bool = True
    # --- where the parameters come from --------------------------------
    source: str = 'raw'                   # 'raw' | 'fields' | 'operators'
    ext_lr_file: Optional[str] = None     # 'fields': LR.mat, or a training set .npz
    ext_hr_file: Optional[str] = None     # 'fields': HR.mat
    ext_model_file: Optional[str] = None  # 'operators': .mat of vibe_export_model.m
    ext_y_up: bool = True                 # external Y axis points up (v positive up)
    ext_velocity_units: str = 'm/s'       # 'px/frame' | 'm/s' (X, Y in mm) | 'unit/s'
    ext_f_hz: float = 0.0                 # rate of the external data; 0 = f_acq
    ext_sensor_w: int = 1280              # sensor size, used only with an empty ROI
    ext_sensor_h: int = 720
    # --- training: HR processing --------------------------------------
    hr_window: int = 32
    hr_step: int = 8
    hr_levels: int = 2
    hr_stencil: str = 'centred'           # 'centred' | 'forward'
    hr_combine: str = 'peaks'             # 'peaks' | 'planes'
    hr_predictor: bool = True
    # --- training: frames -----------------------------------------------
    train_raw: Optional[str] = None
    train_start_frame: int = 0
    train_n_frames: int = 0
    # --- training: split --------------------------------------------------
    n_train: int = 4500
    n_val: int = 1500
    n_test: int = 2000
    gap: int = 500
    # --- training: POD / operators ------------------------------------------
    vr_max_gain: float = 10.0             # cap on the VR gain (Proc.MaxGain)
    tune_q: float = 1.0                   # Q multiplier (Proc.tuneQ), live + test
    tune_r: float = 1.0                   # R multiplier (Proc.tuneR), live + test
    rank: int = 0                         # 0 = elbow criterion
    truncate_lr: bool = True              # FlagLRRankLimit
    elbow_threshold: float = 0.999
    elbow_smooth: str = 'none'            # 'none' | 'movmean' | 'fir'
    elbow_span: int = 25
    lambda_c: float = 1e-12
    # --- training: output ---------------------------------------------------
    model_name: str = "hr_model.npz"
    u_ref: float = 0.0                    # for delta; 0 = mean speed of the HR mean
    save_training_set: bool = True
    export_matlab: bool = False

    def validate(self, need_model=False):
        p = []
        if self.source not in ('raw', 'fields', 'operators'):
            p.append("hr.source must be 'raw', 'fields' or 'operators'")
        if self.ext_velocity_units not in ('px/frame', 'm/s', 'unit/s'):
            p.append("hr.ext_velocity_units must be 'px/frame', 'm/s' or 'unit/s'")
        if self.ext_f_hz < 0 or self.ext_sensor_w < 1 or self.ext_sensor_h < 1:
            p.append("hr.ext_f_hz >= 0 and a positive sensor size")
        if self.method not in ('kf', 'lse', 'lse_vr'):
            p.append("hr.method must be 'kf', 'lse' or 'lse_vr'")
        if self.hr_stencil not in ('centred', 'forward'):
            p.append("hr.hr_stencil must be 'centred' or 'forward'")
        if self.hr_combine not in ('peaks', 'planes'):
            p.append("hr.hr_combine must be 'peaks' or 'planes'")
        if self.elbow_smooth not in ('none', 'movmean', 'fir'):
            p.append("hr.elbow_smooth must be 'none', 'movmean' or 'fir'")
        if self.hr_window < 8 or self.hr_step < 1 or self.hr_levels < 1:
            p.append("HR window >= 8, step >= 1, levels >= 1")
        if min(self.n_train, self.n_val) < 10 or self.n_test < 0 or self.gap < 0:
            p.append("HR split: n_train, n_val >= 10; n_test, gap >= 0")
        if not (0.0 < self.elbow_threshold < 1.1):
            p.append("hr.elbow_threshold must be in (0, 1.1)")
        if self.vr_max_gain < 1.0:
            p.append("hr.vr_max_gain must be >= 1")
        if self.tune_q <= 0 or self.tune_r <= 0:
            p.append("hr.tune_q and hr.tune_r must be > 0")
        if need_model:
            import os
            if not str(self.model_path or '').strip() or not os.path.exists(self.model_path):
                p.append(f"HR estimation is enabled but the model file does not exist: "
                         f"{self.model_path!r}")
        return p


@dataclass
class RunConfig:
    """
    The parameters that are not part of the control system: acquisition,
    the offline pipeline, output paths and display cosmetics.

    Through Release 5.0 these lived as loose local variables at the top of
    EBIV_Main.main(), which is fine for editing a file by hand but leaves the
    GUI with nothing to bind to.  Collecting them here means there is exactly
    one object to edit, validate, save as a preset and hand to run_session().

    mode :
        'stream'   live camera: EBIV, and optionally the controller.
        'offline'  the file-based chain: record a .raw, generate pseudo-images
                   from it (one per laser pulse), and run the pyramidal PIV.  Any subset of
                   the four steps can be selected; they run in order.

    run_mode : only meaningful when mode == 'stream'
        'ebiv'        measure and display only (Release 4.0 behaviour)
        'manual'      + open-loop pump control from the keyboard
        'calibration' + the programmed open-loop voltage schedule
        'closed_loop' + the PID (starts disarmed; press [c])
    """
    # --- what to run -------------------------------------------------
    mode: str = 'stream'
    run_mode: str = 'ebiv'

    do_record: bool = False
    do_playback: bool = False
    do_image_gen: bool = True
    do_piv_process: bool = True
    do_hr_train: bool = False

    # --- acquisition -------------------------------------------------
    f_acq: float = 500.0                  # Hz; accumulation is 1/f_acq
    max_events_per_pixel: int = 1         # contrast clamp
    duration_sec: float = 2.0             # .raw recording length
    trigger_mode: str = 'none'            # 'none' | 'external' | 'auto'
    trigger_duty_cycle: float = 0.5

    flip_x: bool = False
    flip_y: bool = False
    # Gaussian smoothing (std, px) of every live pseudo-frame before the
    # correlation; 0 = off.  HR training applies the same value, so a model
    # always sees frames like the live ones.
    frame_smooth_sigma: float = 0.0

    camera_biases: dict = field(default_factory=lambda: {
        'bias_diff_on': 60, 'bias_diff_off': 140, 'bias_hpf': 70,
        'bias_fo': 0, 'bias_refr': 90, 'bias_diff': 0,
    })

    # --- paths -------------------------------------------------------
    output_base_folder: str = os.path.join(os.path.expanduser("~"), "VIBEstream_data")
    acq_name: str = "BoardOn_D3"
    raw_filename: str = "BoardOn_D3_cam00050876_2026-01-27_15-21-12.raw"

    # --- offline image generation ------------------------------------
    n_images: int = 50
    img_prefix: str = "frame_"
    apply_gaussian: bool = True
    gaussian_kernel: int = 5              # square kernel side, odd
    gaussian_sigma: float = 1.0
    burst_search_max_sec: float = 1.0

    # --- offline PIV --------------------------------------------------
    pyramid_levels: int = 3

    # --- display ------------------------------------------------------
    arrow_skip: int = 1
    arrow_scale: float = 4.0
    blackout_background: bool = True
    show_strip_chart: bool = True
    plot_history_s: float = 60.0          # time span drawn on the strip chart
    plot_zoom_window_s: float = 12.0      # recent slice that sets the y-scale.
                                          # <= 0 scales on the whole history.

    # --- high-resolution estimation -------------------------------------
    hr: HRConfig = field(default_factory=HRConfig)

    # --- diagnostics / saving ----------------------------------------
    profile: bool = False
    profile_report_every: int = 200
    save_rt_piv: bool = False

    # ------------------------------------------------------------------

    @property
    def accum_time_us(self) -> int:
        return int(round(1e6 / self.f_acq))

    @property
    def trigger_period_us(self) -> int:
        return self.accum_time_us

    def validate(self):
        problems = []
        if self.mode not in ('stream', 'offline'):
            problems.append("mode must be 'stream' or 'offline'")
        if self.run_mode not in ('ebiv', 'manual', 'calibration', 'closed_loop'):
            problems.append("run_mode must be ebiv / manual / calibration / closed_loop")
        if self.trigger_mode not in ('none', 'external', 'auto'):
            problems.append("trigger_mode must be 'none', 'external' or 'auto'")
        if self.f_acq <= 0:
            problems.append("f_acq must be > 0 Hz")
        if self.max_events_per_pixel < 1:
            problems.append("max_events_per_pixel must be >= 1")
        if self.gaussian_kernel < 1 or self.gaussian_kernel % 2 == 0:
            problems.append("gaussian_kernel must be a positive odd number")
        if not (0.0 < self.trigger_duty_cycle <= 1.0):
            problems.append("trigger_duty_cycle must be in (0, 1]")
        if self.pyramid_levels < 1:
            problems.append("pyramid_levels must be >= 1")
        if self.n_images < 1:
            problems.append("n_images must be >= 1")
        if self.arrow_skip < 1:
            problems.append("arrow_skip must be >= 1")
        if self.plot_history_s <= 0:
            problems.append("plot_history_s must be > 0 s")
        if not str(self.output_base_folder).strip():
            problems.append("output_base_folder must be set")
        if self.mode == 'offline' and not any(
                (self.do_record, self.do_playback, self.do_image_gen,
                 self.do_piv_process, self.do_hr_train)):
            problems.append("offline mode: select at least one step to run")
        if self.frame_smooth_sigma < 0:
            problems.append("frame_smooth_sigma must be >= 0")
        problems += self.hr.validate(
            need_model=(self.mode == 'stream' and self.hr.enabled))
        if self.mode == 'offline' and self.do_hr_train and self.hr.source == 'raw' \
                and self.trigger_mode == 'none':
            problems.append("HR training builds one pseudo-image per laser pulse; the "
                            "live stream must then use trigger mode 'auto' or 'external', "
                            "not 'none'")
        if (self.do_playback or self.do_image_gen) and not \
                str(self.raw_filename).strip():
            problems.append("a raw filename is required to play back or "
                            "generate frames")
        if problems:
            raise ValueError("Acquisition problems:\n  - " + "\n  - ".join(problems))
        return True

    def to_dict(self):
        return asdict(self)


# ==========================================================================
#  One object for the whole run
# ==========================================================================

@dataclass
class Session:
    """
    Everything one run needs: the acquisition/offline parameters and the
    control-system parameters.  This is what the GUI edits, what a preset
    file contains, and what run_session() consumes.
    """
    run: RunConfig = field(default_factory=RunConfig)
    control: ControlSystemConfig = field(default_factory=ControlSystemConfig)

    SCHEMA = 1

    def resolve_laser_frequency(self):
        """Lock the laser to the acquisition rate unless explicitly overridden.

        Returns (frequency_hz, was_derived).  Idempotent, so it is safe to
        call from several entry points — EBIV_Main, the GUI and run_session
        all reach the hardware by different routes.
        """
        ad3 = self.control.ad3
        # Snapshot the DECLARED value once: resolve() writes the resolved
        # number into the same field, so after the first call the field can no
        # longer say whether the user asked to follow f_acq.  The GUI keeps
        # one Session across runs, so this must re-derive every time.
        if not hasattr(ad3, '_laser_hz_declared'):
            ad3._laser_hz_declared = ad3.laser_frequency_hz

        if ad3._laser_hz_declared is None:
            ad3.laser_frequency_hz = float(self.run.f_acq)
            ad3._laser_freq_derived = True
            return ad3.laser_frequency_hz, True

        ad3.laser_frequency_hz = float(ad3._laser_hz_declared)
        ad3._laser_freq_derived = False
        return ad3.laser_frequency_hz, False

    def forget_resolution(self):
        """Treat the CURRENT field values as the user's declared intent.

        Call this after writing new values into the config from outside — the
        GUI does it every time it reads its widgets.  Without it the snapshot
        taken by the first resolve() outlives the edit, so a later resolve
        re-derives from a value the user has since changed.
        """
        ad3 = self.control.ad3
        if hasattr(ad3, '_laser_hz_declared'):
            del ad3._laser_hz_declared
        if hasattr(ad3, '_laser_freq_derived'):
            del ad3._laser_freq_derived
        m = self.control.measurement
        for attr in ('_scale_declared', '_units_declared'):
            if hasattr(m, attr):
                delattr(m, attr)
        return self

    def restore_declared(self):
        """Put every resolved-in-place value back to what the user declared.

        Called before a preset is serialised.  A preset that stores resolved
        values is a trap: it reloads as a set of explicit overrides, so the
        laser stops following f_acq and the velocity scale is frozen at
        whatever the acquisition rate was when it was saved.
        """
        ad3 = self.control.ad3
        if hasattr(ad3, '_laser_hz_declared'):
            ad3.laser_frequency_hz = ad3._laser_hz_declared
        self.control.measurement.restore_declared()
        return self

    def validate(self):
        """Validate both halves.  Raises ValueError listing every problem."""
        import os as _os
        self.resolve_laser_frequency()
        errors = []
        try:
            self.run.validate()
        except ValueError as exc:
            errors.append(str(exc))
        try:
            self.control.cross_check()
        except ValueError as exc:
            errors.append(str(exc))

        # Cross-cutting checks that need more than one sub-config, or that are
        # deliberately deferred from __post_init__ so a GUI can hold an
        # intermediate state while the user fills the rest in.
        cross = []
        ref = self.control.reference
        if ref.kind == 'file':
            if not ref.filepath:
                cross.append("the reference is set to 'file' but no time-series "
                             "file has been chosen")
            elif not _os.path.exists(ref.filepath):
                cross.append(f"the reference time-series file does not exist: "
                             f"{ref.filepath}")
        # ROI sanity.  Containment against the sensor can only be checked once
        # the camera reports its size, but a self-inconsistent rectangle can
        # and should be rejected here rather than at the rig.
        d, c = self.control.roi.display_roi, self.control.roi.control_roi
        for name, r in (("DisplayROI", d), ("ControlROI", c)):
            if not r:
                continue
            if len(r) != 4:
                cross.append(f"the {name} needs four values [x0, x1, y0, y1]")
            elif any(v < 0 for v in r):
                cross.append(f"the {name} {r} has a negative coordinate")
            elif r[1] <= r[0] or r[3] <= r[2]:
                cross.append(f"the {name} {r} is empty: it needs x1 > x0 and "
                             f"y1 > y0")
        if d and c and len(d) == 4 and len(c) == 4 and \
                not (d[0] <= c[0] < c[1] <= d[1] and d[2] <= c[2] < c[3] <= d[3]):
            cross.append(f"the ControlROI {c} must be a non-empty region inside "
                         f"the DisplayROI {d}")
        ws = self.control.piv.window_size
        if d and len(d) == 4 and (d[1] - d[0] < ws or d[3] - d[2] < ws):
            cross.append(f"the DisplayROI {d} is smaller than the interrogation "
                         f"window ({ws} px) in at least one direction")
        if cross:
            errors.append("Configuration problems:\n  - " + "\n  - ".join(cross))

        if errors:
            raise ValueError("\n\n".join(errors))
        return True

    def to_dict(self):
        # Serialise DECLARED intent, not resolved values.  Deep-copied so a
        # save cannot disturb a run that is already using the resolved ones.
        import copy as _copy
        snap = _copy.deepcopy(self)
        snap.restore_declared()
        return {'schema': self.SCHEMA,
                'run': snap.run.to_dict(),
                'control': snap.control.to_dict()}

    # ------------------------------------------------------------------

    @staticmethod
    def _apply(obj, data, path=""):
        """
        Recursively set dataclass fields from a plain dict.

        Unknown keys are collected and returned rather than raising, so a
        preset written by an older or newer version still loads and the
        caller can report what was ignored.  Missing keys keep their default.
        """
        ignored = []
        for key, value in (data or {}).items():
            if not hasattr(obj, key):
                ignored.append(f"{path}{key}")
                continue
            current = getattr(obj, key)
            if is_dataclass(current) and isinstance(value, dict):
                ignored += Session._apply(current, value, f"{path}{key}.")
            elif isinstance(current, tuple) and isinstance(value, list):
                setattr(obj, key, tuple(value))
            else:
                setattr(obj, key, value)
        return ignored

    @classmethod
    def from_dict(cls, data):
        """Build a Session from a preset dict.  Returns (session, ignored_keys)."""
        s = cls()
        payload = data.get('run') if isinstance(data, dict) else None
        ignored = []
        if payload is not None:
            ignored += cls._apply(s.run, data.get('run'), 'run.')
            ignored += cls._apply(s.control, data.get('control'), 'control.')
        else:
            # tolerate a bare ControlSystemConfig dump
            ignored += cls._apply(s.control, data, 'control.')
        # re-run __post_init__ validation on the leaf dataclasses
        for sub in (s.control.measurement, s.control.filt, s.control.reference,
                    s.control.pid, s.control.supervisor, s.control.ad3,
                    s.control.calibration):
            if hasattr(sub, '__post_init__'):
                sub.__post_init__()
        return s, ignored

    def save(self, path):
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return path

    @classmethod
    def load(cls, path):
        import json
        with open(path, encoding='utf-8') as f:
            return cls.from_dict(json.load(f))
