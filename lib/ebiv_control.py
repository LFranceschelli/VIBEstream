"""
EBIV Release 5.0 — Control law.

Pure computation: no camera, no Analog Discovery, no OpenCV, no plotting.
Everything in this module can be unit-tested and run in closed loop against a
simulated pump/jet on a machine with no hardware attached.

Components
    Measurement          immutable snapshot published by the PIV thread
    ControlROIEstimator  vector field  ->  scalar U_control_raw
    TemporalFilter       none / moving average / exponential moving average
    ReferenceGenerator   constant / step / sine / file / band-limited random
    PIDController        discrete-time PID with anti-windup and slew limiting
    ControlSupervisor    state machine: MANUAL / CLOSED_LOOP / HOLD / SAFE
"""

import os
import time
import logging
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ==========================================================================
#  Controller states
# ==========================================================================

class ControlState:
    """Controller states.  Plain strings so they log and serialise cleanly."""
    MANUAL = "MANUAL"            # open loop; voltage set by the operator
    CLOSED_LOOP = "CLOSED_LOOP"  # PID active on a fresh, valid measurement
    HOLD = "HOLD"                # armed, but measurement invalid/stale: output frozen
    SAFE = "SAFE"                # fault or shutdown: driven to safe_pump_voltage

    ALL = (MANUAL, CLOSED_LOOP, HOLD, SAFE)


# ==========================================================================
#  Measurement snapshot
# ==========================================================================

@dataclass(frozen=True)
class Measurement:
    """
    One EBIV velocity estimate as seen by the controller.

    Published by the PIV worker as a SINGLE object assignment so the control
    thread can never observe a half-updated set of fields (Release 4.0 wrote
    U, V and M into a shared dict field-by-field, which allows a torn read —
    see AUDIT_R4_to_R5.md, bug B3).

    t_frame      perf_counter() when the newest contributing frame was
                 finalised in the acquisition loop.  This is the timestamp
                 the latency budget is measured against.
    t_measured   perf_counter() when this estimate finished being computed.
    u_raw        scalar control velocity, already in display units
    valid        whether u_raw may be used by the controller
    n_valid      number of vectors that passed all validity criteria
    n_total      number of ControlROI grid nodes
    seq          monotonically increasing sequence number
    """
    u_raw: float
    valid: bool
    n_valid: int
    n_total: int
    t_frame: float
    t_measured: float
    seq: int
    reason: str = ""

    @property
    def valid_fraction(self) -> float:
        return self.n_valid / self.n_total if self.n_total else 0.0


class MeasurementHolder:
    """
    Single-slot mailbox between the PIV worker and the control thread.

    Publishing and reading are single attribute accesses, which are atomic
    under the CPython GIL, so no lock is needed and the control thread never
    blocks behind the PIV thread.
    """
    __slots__ = ('latest',)

    def __init__(self):
        self.latest: Optional[Measurement] = None

    def publish(self, m: Measurement):
        self.latest = m

    def read(self) -> Optional[Measurement]:
        return self.latest


# ==========================================================================
#  ControlROI  ->  scalar
# ==========================================================================

class ControlROIEstimator:
    """
    Reduces a PIV vector field to the single scalar the controller acts on.

    The PIV grid is defined on the DisplayROI.  Grid node (gy, gx) has its
    interrogation-window centre at, in full-sensor pixel coordinates:

        x = display_roi[0] + gx * node_distance + window_size // 2
        y = display_roi[2] + gy * node_distance + window_size // 2

    Because the grid is regular and the ControlROI is a rectangle, the
    selected nodes form a contiguous block, so selection is a slice rather
    than a boolean mask — computed once at construction, not per frame.

    Terminology: this is a CONTROL-REGION velocity estimate, i.e. the spatial
    statistic of the streamwise component over whatever part of the flow the
    ControlROI happens to cover.  If the ControlROI sits on the jet core it
    is a core velocity, which is NOT the cross-sectional bulk velocity and is
    NOT the nozzle exit velocity.  It is called U_control internally for that
    reason; displaying it as "Uj" is a labelling convenience only.
    """

    def __init__(self, cfg_measurement, display_roi, control_roi,
                 window_size, node_distance, grid_y, grid_x):
        self.cfg = cfg_measurement
        self.window_size = window_size
        self.node_distance = node_distance
        self.grid_shape = (grid_y, grid_x)

        half = window_size // 2
        # Node-centre coordinates in full-sensor pixels
        xc = display_roi[0] + np.arange(grid_x) * node_distance + half
        yc = display_roi[2] + np.arange(grid_y) * node_distance + half

        in_x = np.nonzero((xc >= control_roi[0]) & (xc < control_roi[1]))[0]
        in_y = np.nonzero((yc >= control_roi[2]) & (yc < control_roi[3]))[0]

        if in_x.size == 0 or in_y.size == 0:
            raise ValueError(
                f"ControlROI {control_roi} contains no PIV grid nodes.\n"
                f"  Grid node centres span x={xc[0]}..{xc[-1]} step {node_distance}, "
                f"y={yc[0]}..{yc[-1]} step {node_distance}.\n"
                f"  Make the ControlROI larger than one node spacing "
                f"({node_distance} px) in both directions, or reduce node_distance."
            )

        self.sy = slice(int(in_y[0]), int(in_y[-1]) + 1)
        self.sx = slice(int(in_x[0]), int(in_x[-1]) + 1)
        self.n_total = int(in_y.size * in_x.size)

        # Pixel extent actually covered by the selected nodes (for the overlay)
        self.node_extent = (int(xc[in_x[0]]), int(xc[in_x[-1]]),
                            int(yc[in_y[0]]), int(yc[in_y[-1]]))

        self._min_valid = max(int(self.cfg.min_valid_vectors),
                              int(np.ceil(self.cfg.min_valid_fraction * self.n_total)))

        logging.info(
            "ControlROI: %d PIV nodes (%d x %d) selected out of %d in the DisplayROI; "
            "need >= %d valid for a usable measurement.",
            self.n_total, in_y.size, in_x.size, grid_y * grid_x, self._min_valid)

    # ------------------------------------------------------------------

    def _component(self, U, V):
        c = self.cfg.component
        if c == 'u':
            return U
        if c == '-u':
            return -U
        if c == 'v':
            return V
        if c == '-v':
            return -V
        return np.hypot(U, V)

    def estimate(self, U, V, valid_mask=None, CC=None):
        """
        Returns (u_raw, valid, n_valid, reason).

        u_raw is in display units (px/frame multiplied by velocity_scale).
        A NaN u_raw is never returned as valid.
        """
        Us = U[self.sy, self.sx]
        Vs = V[self.sy, self.sx]

        good = np.isfinite(Us) & np.isfinite(Vs)

        if valid_mask is not None:
            good &= valid_mask[self.sy, self.sx]

        if self.cfg.use_correlation_gate and CC is not None:
            ccs = CC[self.sy, self.sx]
            good &= np.isfinite(ccs) & (ccs >= self.cfg.min_correlation)

        if self.cfg.max_abs_displacement_px is not None:
            lim = self.cfg.max_abs_displacement_px
            good &= (np.abs(Us) <= lim) & (np.abs(Vs) <= lim)

        n_valid = int(np.count_nonzero(good))
        if n_valid < self._min_valid:
            return (float('nan'), False, n_valid,
                    f"only {n_valid}/{self.n_total} valid vectors "
                    f"(need {self._min_valid})")

        comp = self._component(Us, Vs)[good]
        if self.cfg.statistic == 'median':
            u = float(np.median(comp))
        else:
            u = float(np.mean(comp))

        if not np.isfinite(u):
            return float('nan'), False, n_valid, "non-finite spatial statistic"

        return u * self.cfg.velocity_scale, True, n_valid, ""


# ==========================================================================
#  Temporal filtering
# ==========================================================================

class TemporalFilter:
    """
    Moving average or first-order low-pass on the scalar control velocity.

    Only VALID samples are fed in; a dropout does not inject a zero or a NaN
    into the filter memory, it simply does not advance it.

    The EMA recomputes its coefficient from the MEASURED dt at every update:

        alpha = dt / (tau + dt)
        y    += alpha * (x - y)

    so the -3 dB corner stays at 1/(2*pi*tau) even when the control loop
    jitters.  A moving average of length N delays the signal by about
    (N-1)/2 samples; an EMA by about tau seconds.  Neither is free, and both
    are recorded in the run header.
    """

    def __init__(self, cfg_filter):
        self.cfg = cfg_filter
        self._buf = deque(maxlen=max(1, cfg_filter.window))
        self._y = None
        self._n = 0

    def reset(self):
        self._buf.clear()
        self._y = None
        self._n = 0

    @property
    def initialised(self) -> bool:
        return self._n > 0

    def update(self, x: float, dt: float) -> float:
        """Feed one valid sample; returns the filtered value."""
        if not np.isfinite(x):
            raise ValueError("TemporalFilter.update() called with a non-finite sample")

        self._n += 1
        kind = self.cfg.kind

        if kind == 'none':
            self._y = float(x)
        elif kind == 'ma':
            self._buf.append(float(x))
            self._y = float(sum(self._buf) / len(self._buf))
        else:  # ema
            if self._y is None or dt <= 0.0:
                self._y = float(x)
            else:
                alpha = dt / (self.cfg.tau_s + dt)
                self._y += alpha * (float(x) - self._y)
        return self._y

    @property
    def value(self) -> Optional[float]:
        return self._y


# ==========================================================================
#  Reference generators
# ==========================================================================

class ReferenceGenerator:
    """Base class.  t is seconds since the controller was armed."""

    def __call__(self, t: float) -> float:
        raise NotImplementedError

    def describe(self) -> str:
        return self.__class__.__name__


class ConstantReference(ReferenceGenerator):
    def __init__(self, value):
        self.value = float(value)

    def __call__(self, t):
        return self.value

    def describe(self):
        return f"constant {self.value:g}"


class StepReference(ReferenceGenerator):
    def __init__(self, value, t_step_s, step_amplitude):
        self.v0 = float(value)
        self.t_step = float(t_step_s)
        self.da = float(step_amplitude)

    def __call__(self, t):
        return self.v0 + (self.da if t >= self.t_step else 0.0)

    def describe(self):
        return f"step {self.v0:g} -> {self.v0 + self.da:g} at t={self.t_step:g} s"


class SineReference(ReferenceGenerator):
    def __init__(self, value, amplitude, freq_hz, phase_rad=0.0):
        self.v0 = float(value)
        self.a = float(amplitude)
        self.f = float(freq_hz)
        self.p = float(phase_rad)

    def __call__(self, t):
        return self.v0 + self.a * np.sin(2.0 * np.pi * self.f * t + self.p)

    def describe(self):
        return f"sine {self.v0:g} + {self.a:g}*sin(2*pi*{self.f:g}*t)"


class FileReference(ReferenceGenerator):
    """
    Two-column (time_s, target) time series, linearly interpolated.
    Accepts .npy or any delimited text file readable by np.loadtxt
    (comma or whitespace separated, '#' comments).
    """

    def __init__(self, filepath, loop=True):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Reference time series not found: {filepath}")
        if filepath.lower().endswith('.npy'):
            data = np.load(filepath)
        else:
            try:
                data = np.loadtxt(filepath, delimiter=',')
            except ValueError:
                data = np.loadtxt(filepath)
        data = np.atleast_2d(data)
        if data.shape[1] < 2:
            raise ValueError(f"{filepath}: expected 2 columns (time_s, target), "
                             f"got shape {data.shape}")
        order = np.argsort(data[:, 0])
        self.t = np.ascontiguousarray(data[order, 0], dtype=np.float64)
        self.y = np.ascontiguousarray(data[order, 1], dtype=np.float64)
        self.loop = bool(loop)
        self.duration = float(self.t[-1] - self.t[0])
        self.path = filepath

    def __call__(self, t):
        if self.loop and self.duration > 0:
            t = self.t[0] + ((t - self.t[0]) % self.duration)
        return float(np.interp(t, self.t, self.y))

    def describe(self):
        return (f"file {os.path.basename(self.path)} "
                f"({len(self.t)} points, {self.duration:g} s, loop={self.loop})")


class BandLimitedRandomReference(ReferenceGenerator):
    """
    Reproducible, band-limited random reference.

    A white sequence is generated on a FIXED internal time grid, low-pass
    filtered to `cutoff_hz`, normalised to `std`, and then interpolated in
    time.  Because the sequence lives on a fixed grid rather than being
    advanced once per control step, the trajectory depends only on the seed
    and NOT on the loop timing — the same seed replays the same reference
    even if the control rate or the jitter changes between runs.

    This deliberately avoids the usual mistake of drawing an independent
    random value at every control step, which would demand infinite actuator
    bandwidth and tell you nothing about the plant.
    """

    def __init__(self, value, std, cutoff_hz, seed=12345,
                 duration_s=3600.0, clip=None, oversample=20):
        self.v0 = float(value)
        self.std = float(std)
        self.fc = float(cutoff_hz)
        self.seed = int(seed)
        self.clip = clip

        fs = max(4.0 * self.fc * oversample / oversample, self.fc * oversample)
        fs = max(fs, 1.0)
        n = int(np.ceil(duration_s * fs)) + 2
        self.t_grid = np.arange(n) / fs

        rng = np.random.default_rng(self.seed)
        w = rng.standard_normal(n)

        # Two-pole low-pass (first order applied forward twice) -> ~ -12 dB/oct.
        # Applied causally in both passes so the shape is well defined; the
        # result is then normalised, so the exact filter gain is irrelevant.
        alpha = (2.0 * np.pi * self.fc / fs) / (1.0 + 2.0 * np.pi * self.fc / fs)
        y = np.empty(n)
        acc = 0.0
        for i in range(n):
            acc += alpha * (w[i] - acc)
            y[i] = acc
        acc = 0.0
        for i in range(n):
            acc += alpha * (y[i] - acc)
            y[i] = acc

        y -= y.mean()
        s = y.std()
        self.y_grid = (y / s * self.std) if s > 0 else y
        self.duration = float(self.t_grid[-1])

    def __call__(self, t):
        tt = t % self.duration if self.duration > 0 else 0.0
        v = self.v0 + float(np.interp(tt, self.t_grid, self.y_grid))
        if self.clip is not None:
            v = float(np.clip(v, self.clip[0], self.clip[1]))
        return v

    def describe(self):
        return (f"band-limited random about {self.v0:g}, std={self.std:g}, "
                f"fc={self.fc:g} Hz, seed={self.seed}")


class LiveReference(ReferenceGenerator):
    """
    Target set by the operator DURING the run, with [+] and [-].

    This is not the same thing as 'manual' run mode.  There, [+]/[-] move the
    pump VOLTAGE and there is no controller.  Here the loop stays closed and
    [+]/[-] move the TARGET VELOCITY: the PID works out what voltage that
    needs.  You steer the quantity you care about and let the controller deal
    with the actuator, which is the whole point of having a controller.

    Thread safety: the control thread reads this through __call__ while the
    key handler writes it.  A float read is atomic under the GIL, so the lock
    is not about tearing; it is because bump() is a read-modify-write and
    because a reader must never see a value that has been added to but not
    yet clamped.
    """

    def __init__(self, value, step, lo=None, hi=None):
        self.step = abs(float(step))
        if self.step == 0.0:
            raise ValueError("live reference: step must be non-zero")
        self.lo = -np.inf if lo is None else float(lo)
        self.hi = np.inf if hi is None else float(hi)
        if self.hi < self.lo:
            raise ValueError("live reference: live_max must be >= live_min")
        self._lock = threading.Lock()
        self._value = self._clamp(float(value))

    def _clamp(self, v):
        return float(min(max(v, self.lo), self.hi))

    def __call__(self, t):
        with self._lock:
            return self._value

    @property
    def value(self):
        with self._lock:
            return self._value

    def set(self, v):
        """Absolute set.  A non-finite request is refused, not clamped: a NaN
        target would propagate straight into the PID error."""
        v = float(v)
        if not np.isfinite(v):
            return self.value
        with self._lock:
            self._value = self._clamp(v)
            return self._value

    def bump(self, n_steps=1):
        """Relative set, in units of `step`.  Returns the new target."""
        with self._lock:
            self._value = self._clamp(self._value + float(n_steps) * self.step)
            return self._value

    def describe(self):
        lo = "-inf" if not np.isfinite(self.lo) else f"{self.lo:g}"
        hi = "+inf" if not np.isfinite(self.hi) else f"{self.hi:g}"
        return f"live target, step {self.step:g}, range [{lo}, {hi}]"


def make_reference(cfg_reference) -> ReferenceGenerator:
    """Factory.  Adding a new trajectory means adding a class and one line here."""
    k = cfg_reference.kind
    if k == 'constant':
        return ConstantReference(cfg_reference.value)
    if k == 'live':
        return LiveReference(cfg_reference.value, cfg_reference.live_step,
                             cfg_reference.live_min, cfg_reference.live_max)
    if k == 'step':
        return StepReference(cfg_reference.value, cfg_reference.t_step_s,
                             cfg_reference.step_amplitude)
    if k == 'sine':
        return SineReference(cfg_reference.value, cfg_reference.amplitude,
                             cfg_reference.freq_hz, cfg_reference.phase_rad)
    if k == 'file':
        return FileReference(cfg_reference.filepath, cfg_reference.file_loop)
    if k == 'random':
        return BandLimitedRandomReference(
            cfg_reference.value, cfg_reference.random_std,
            cfg_reference.random_cutoff_hz, cfg_reference.random_seed,
            clip=cfg_reference.random_clip)
    raise ValueError(f"unknown reference kind {k!r}")


# ==========================================================================
#  PID
# ==========================================================================

@dataclass
class PIDDiagnostics:
    """Per-step internals, for logging and for tuning."""
    error: float = 0.0
    p_term: float = 0.0
    i_term: float = 0.0
    d_term: float = 0.0
    u_unsaturated: float = 0.0
    u_saturated: float = 0.0
    u_command: float = 0.0
    saturated: bool = False
    slew_limited: bool = False
    dt: float = 0.0


class PIDController:
    """
    Discrete-time PID.

    error = target - measurement, output = pump voltage.

    Features:
      - uses the MEASURED dt of every step, clamped to `max_dt` so that a
        stalled loop cannot inject an enormous integral or derivative step;
      - derivative on measurement (default) to avoid set-point kick;
      - first-order filtered derivative;
      - output saturation to [v_min, v_max];
      - slew-rate limiting on the commanded voltage;
      - anti-windup evaluated against the FINAL command, so that both
        saturation and slew limiting stop the integrator;
      - bumpless transfer via set_bumpless().

    The controller is entirely unaware of how the voltage reaches the pump.
    """

    def __init__(self, cfg_pid, max_dt=1.0):
        self.cfg = cfg_pid
        self.max_dt = float(max_dt)
        self.reset()

    def reset(self, output=None):
        self._i = 0.0
        self._d_state = 0.0
        self._prev_meas = None
        self._prev_error = None
        self._u_prev = float(self.cfg.v_min if output is None else output)
        self._started = False

    def set_bumpless(self, current_output, measurement, target):
        """
        Prepare a jump-free MANUAL -> CLOSED_LOOP transition.

        The integral term is preloaded so that the controller's first output
        equals the voltage the pump is already receiving.  Derivative state is
        cleared, so the first step contributes no D action.
        """
        e = target - measurement
        p = self.cfg.kp * e
        self._i = float(current_output) - p
        if self.cfg.integral_limit is not None:
            self._i = float(np.clip(self._i, -self.cfg.integral_limit,
                                    self.cfg.integral_limit))
        self._d_state = 0.0
        self._prev_meas = float(measurement)
        self._prev_error = float(e)
        self._u_prev = float(current_output)
        self._started = True

    # ------------------------------------------------------------------

    def update(self, target, measurement, dt) -> PIDDiagnostics:
        c = self.cfg
        dt = float(min(max(dt, 1e-6), self.max_dt))
        e = float(target) - float(measurement)

        d = PIDDiagnostics(error=e, dt=dt)

        # --- proportional ---
        d.p_term = c.kp * e

        # --- derivative (before integrating, so both see the same state) ---
        if not self._started or self._prev_meas is None:
            d_raw = 0.0
        elif c.derivative_on_measurement:
            d_raw = -c.kd * (float(measurement) - self._prev_meas) / dt
        else:
            d_raw = c.kd * (e - self._prev_error) / dt

        if c.derivative_filter_tau_s > 0.0:
            a = dt / (c.derivative_filter_tau_s + dt)
            self._d_state += a * (d_raw - self._d_state)
            d.d_term = self._d_state
        else:
            self._d_state = d_raw
            d.d_term = d_raw

        # --- integral: computed tentatively, committed conditionally ---
        #
        # Conditional integration.  The output is computed ONCE, from the
        # tentative integral, and then limited.  The anti-windup decision only
        # controls whether the integral update is COMMITTED; it never
        # recomputes or discards the command.
        #
        # Getting this wrong is easy and the failure is silent.  An earlier
        # version of this function reverted the integral AND recomputed the
        # command from the reverted state.  Combined with the bumpless preload
        # (which sets I so that the first output equals the current voltage)
        # and the slew limiter (which holds the first move to slew*dt), that
        # dead-locked the controller at its starting voltage forever: every
        # step proposed a move, saw the move limited, reverted, and recomputed
        # exactly the starting value.  test_release5.py section 7 catches it.
        i_prev = self._i
        i_new = self._i + c.ki * e * dt
        if c.integral_limit is not None:
            i_new = float(np.clip(i_new, -c.integral_limit, c.integral_limit))

        u_unsat = d.p_term + i_new + d.d_term
        u_sat = float(np.clip(u_unsat, c.v_min, c.v_max))

        u_cmd = u_sat
        if c.slew_rate_v_per_s is not None and self._started:
            max_step = c.slew_rate_v_per_s * dt
            u_cmd = float(np.clip(u_sat, self._u_prev - max_step, self._u_prev + max_step))
        d.slew_limited = (u_cmd != u_sat)
        d.saturated = (u_sat != u_unsat)

        # The command is held back when it is limited by amplitude saturation
        # or by the slew rate.  In either case adding more integral action
        # cannot make the actuator move any further or any faster, so the
        # integral update is dropped.
        held_back = (u_unsat - u_cmd) * (c.ki * e) > 0.0

        if c.anti_windup == 'clamp':
            self._i = i_prev if held_back else i_new
        elif c.anti_windup == 'back_calc':
            self._i = i_new + c.back_calc_gain * (u_cmd - u_unsat) * dt
            if c.integral_limit is not None:
                self._i = float(np.clip(self._i, -c.integral_limit, c.integral_limit))
        else:
            self._i = i_new

        # i_term reports the COMMITTED integral; u_unsaturated reports what the
        # controller asked for, i.e. the value built from the tentative one.
        d.i_term = self._i
        d.u_unsaturated = u_unsat
        d.u_saturated = u_sat
        d.u_command = u_cmd

        self._prev_meas = float(measurement)
        self._prev_error = e
        self._u_prev = u_cmd
        self._started = True
        return d

    def hold(self, u_current):
        """
        Freeze the controller at the current output without integrating.
        Used while the measurement is invalid: the loop keeps its last action
        but accumulates no error against a measurement it cannot trust.
        """
        self._u_prev = float(u_current)
        self._prev_meas = None      # forces D to restart cleanly on resume
        self._d_state = 0.0


# ==========================================================================
#  Supervisor
# ==========================================================================

class ControlSupervisor:
    """
    Ties measurement -> filter -> reference -> PID -> pump together and owns
    the fault logic.

    Dropout strategy (documented, configurable, and logged):

        valid measurement            -> CLOSED_LOOP, PID runs normally
        invalid / stale, < hold_timeout_s
                                     -> HOLD.  The last commanded voltage is
                                        maintained.  The PID does not
                                        integrate and its derivative state is
                                        cleared, so no phantom error
                                        accumulates against data that does not
                                        exist.  This covers the ordinary case
                                        of a few bad EBIV frames.
        invalid / stale, > safe_timeout_s
                                     -> SAFE.  The pump is driven to
                                        safe_pump_voltage and the loop
                                        DISARMS.  Recovery requires an
                                        explicit re-arm by the operator, so a
                                        flapping measurement can never
                                        silently re-engage the controller.

    Every state change is logged once, at the transition, never per step.
    """

    def __init__(self, cfg, pump, reference=None, logger=None):
        self.cfg = cfg
        self.sup = cfg.supervisor
        self.pump = pump
        self.reference = reference if reference is not None else make_reference(cfg.reference)
        self.filter = TemporalFilter(cfg.filt)
        self.pid = PIDController(cfg.pid, max_dt=5.0 / max(self.sup.control_rate_hz, 1e-6))
        self.logger = logger

        self.state = ControlState.MANUAL
        self.manual_voltage = float(self.sup.startup_pump_voltage)
        self.v_command = float(self.sup.startup_pump_voltage)

        self.t_armed = None
        self._t_last_valid = None
        self._t_last_update = None
        self._last_seq = -1

        self.u_raw = float('nan')
        self.u_filtered = float('nan')
        self.u_target = float('nan')
        self.measurement_age_s = float('nan')
        self.measurement_stale = False
        self.last_diag = PIDDiagnostics()
        self.n_valid = 0
        self.n_total = 0
        self.invalid_reason = ""

        self.counters = {'steps': 0, 'valid': 0, 'hold': 0, 'safe': 0,
                         'stale': 0, 'invalid': 0}

    # ------------------------------------------------------------------
    #  Mode switching
    # ------------------------------------------------------------------

    def arm(self, now=None):
        """MANUAL -> CLOSED_LOOP, bumplessly."""
        if self.state == ControlState.CLOSED_LOOP:
            return
        now = time.perf_counter() if now is None else now
        self.t_armed = now
        self._t_last_valid = now
        self._t_last_update = now

        meas = self.u_filtered if np.isfinite(self.u_filtered) else 0.0
        tgt = self.reference(0.0)
        self.pid.set_bumpless(self.v_command, meas, tgt)
        self._set_state(ControlState.CLOSED_LOOP,
                        f"armed at V={self.v_command:.3f} V (bumpless), "
                        f"target={tgt:.4g}")

    def disarm(self, reason="operator"):
        """CLOSED_LOOP -> MANUAL, holding the present voltage."""
        if self.state == ControlState.MANUAL:
            return
        self.manual_voltage = self.v_command
        self._set_state(ControlState.MANUAL,
                        f"disarmed ({reason}); holding {self.v_command:.3f} V")

    def set_manual_voltage(self, v):
        """Open-loop / calibration command.  Ignored while closed loop."""
        self.manual_voltage = float(np.clip(v, self.cfg.pid.v_min, self.cfg.pid.v_max))
        return self.manual_voltage

    # ------------------------------------------------------------------
    #  Live target  (reference kind 'live')
    #
    #  These are the ONLY way the rest of the program touches the reference,
    #  so a non-live reference degrades to None rather than to a silent
    #  no-op that looks like it worked.
    # ------------------------------------------------------------------

    @property
    def live_target(self):
        """The LiveReference, or None when the reference is not operator-driven."""
        return self.reference if isinstance(self.reference, LiveReference) else None

    def bump_target(self, n_steps):
        """Move the live target by n_steps increments.  None if not live."""
        ref = self.live_target
        return None if ref is None else ref.bump(n_steps)

    def set_target(self, value):
        """Set the live target absolutely.  None if not live."""
        ref = self.live_target
        return None if ref is None else ref.set(value)

    def target_to_measurement(self):
        """Snap the live target onto the current filtered velocity.

        Useful just before arming: it makes the initial error zero, so the
        loop closes without a transient and the only thing that then moves
        the jet is your own [+]/[-].  Returns None if the reference is not
        live, or if there is no finite measurement to snap to.
        """
        ref = self.live_target
        if ref is None or not np.isfinite(self.u_filtered):
            return None
        return ref.set(self.u_filtered)

    def go_safe(self, reason):
        self.state = ControlState.SAFE
        self.v_command = float(self.sup.safe_pump_voltage)
        try:
            self.pump.set_voltage(self.v_command)
        except Exception as exc:                              # noqa: BLE001
            logging.error("Failed to command the safe voltage: %s", exc)
        self.counters['safe'] += 1
        logging.warning("CONTROLLER -> SAFE (%s). Pump driven to %.3f V.",
                        reason, self.v_command)
        if self.logger is not None:
            self.logger.log_event('SAFE', reason)

    def _set_state(self, new, reason=""):
        if new != self.state:
            logging.info("Controller state: %s -> %s%s",
                         self.state, new, f" ({reason})" if reason else "")
            if self.logger is not None:
                self.logger.log_event(new, reason)
            self.state = new

    # ------------------------------------------------------------------
    #  One control step
    # ------------------------------------------------------------------

    def step(self, measurement: Optional[Measurement], now=None):
        """
        Execute one control update.  Called by the control thread at
        control_rate_hz.  Returns the commanded voltage.
        """
        now = time.perf_counter() if now is None else now
        self.counters['steps'] += 1

        dt = (now - self._t_last_update) if self._t_last_update is not None else \
            1.0 / self.sup.control_rate_hz
        self._t_last_update = now

        # --- ingest the measurement -----------------------------------
        #
        # TWO SEPARATE QUESTIONS, and conflating them was a mistake:
        #
        #   1. is this measurement GOOD?      (enough valid vectors, finite)
        #   2. is it FRESH ENOUGH TO CONTROL ON?   (age vs max_measurement_age_s)
        #
        # A good but late measurement is still the best knowledge we have of
        # the flow, and it belongs on the display, in the filter and in the
        # log.  It must NOT drive the PID.  Gating both on the age froze the
        # readout at whatever value happened to arrive before the acquisition
        # lag grew past the limit -- with 93% valid vectors on screen and a
        # flat line on the strip chart, which reads as a broken measurement
        # rather than a late one.
        #
        # Age is measured from t_frame, i.e. when the LIGHT ARRIVED, so it
        # covers sensor -> SDK buffer -> accumulation -> queue -> correlation.
        fresh = (measurement is not None and measurement.seq != self._last_seq)
        quality_ok = False
        stale = False
        usable = False
        age = float('nan')
        if measurement is not None:
            age = now - measurement.t_frame
            if not measurement.valid:
                self.invalid_reason = measurement.reason or "measurement flagged invalid"
                self.counters['invalid'] += 1
            elif not np.isfinite(measurement.u_raw):
                # Defence in depth: the estimator never returns valid=True with
                # a non-finite value, but a NaN or Inf must not be able to reach
                # the filter or the PID even if some future producer gets this
                # wrong.  A single NaN entering an integrator poisons the output
                # permanently.
                self.invalid_reason = "measurement value is not finite"
                self.counters['invalid'] += 1
            else:
                quality_ok = True
                stale = age > self.sup.max_measurement_age_s
                if stale:
                    self.invalid_reason = (
                        f"measurement is {age * 1e3:.0f} ms old "
                        f"(limit {self.sup.max_measurement_age_s * 1e3:.0f} ms): "
                        f"shown, but NOT used for control")
                    self.counters['stale'] += 1
                elif not fresh:
                    self.invalid_reason = "no new measurement since last step"
                else:
                    usable = True
            self.n_valid, self.n_total = measurement.n_valid, measurement.n_total
        else:
            self.invalid_reason = "no measurement available"

        self.measurement_age_s = age
        self.measurement_stale = bool(stale)

        # The DISPLAY/filter path: a good measurement is ingested whatever its
        # age, so the operator always sees the most recent thing EBIV actually
        # measured.  This is what makes 'manual' and 'calibration' usable while
        # an acquisition problem is being chased.
        if quality_ok and fresh:
            self._last_seq = measurement.seq
            self.u_raw = measurement.u_raw
            self.u_filtered = self.filter.update(measurement.u_raw, dt)
            if usable:
                # ONLY a fresh measurement resets the fault timers.  If a stale
                # one did, a permanently lagging system would never reach SAFE.
                self._t_last_valid = now
                self.counters['valid'] += 1
                self.invalid_reason = ""
        elif fresh and measurement is not None:
            # consume the sequence so the same bad frame is not re-counted
            self._last_seq = measurement.seq
            self.u_raw = float('nan')

        # --- MANUAL: straight through, no PID -------------------------
        if self.state == ControlState.MANUAL:
            self.u_target = float('nan')
            self.v_command = self._apply(self.manual_voltage)
            return self.v_command

        if self.state == ControlState.SAFE:
            self.v_command = self._apply(self.sup.safe_pump_voltage)
            return self.v_command

        # --- armed: CLOSED_LOOP or HOLD -------------------------------
        t_rel = now - self.t_armed if self.t_armed is not None else 0.0
        self.u_target = self.reference(t_rel)

        since_valid = now - (self._t_last_valid if self._t_last_valid is not None else now)

        if usable:
            self._set_state(ControlState.CLOSED_LOOP)
            self.last_diag = self.pid.update(self.u_target, self.u_filtered, dt)
            self.v_command = self._apply(self.last_diag.u_command)
            return self.v_command

        if since_valid > self.sup.safe_timeout_s:
            self.go_safe(f"no valid measurement for {since_valid:.2f} s "
                         f"(last reason: {self.invalid_reason})")
            return self.v_command

        # The measurement is unusable.  In every case the last commanded
        # voltage is held and the PID neither integrates nor differentiates,
        # so no phantom error accumulates against data that does not exist.
        #
        # hold_timeout_s decides only whether this is ANNOUNCED.  Below it the
        # controller stays in CLOSED_LOOP: a single dropped EBIV field is
        # normal and should not flip the displayed state or write a log line
        # every time.  Above it the state becomes HOLD, which is visible on
        # the HUD, recorded in the event log and counted.
        #
        # (Until Release 5.1 this branch read `since_valid > hold_timeout or
        # not usable`.  The second clause is always true here, so the
        # threshold was unreachable and hold_timeout_s did nothing at all —
        # a documented parameter with no effect.)
        self.counters['hold'] += 1
        self.pid.hold(self.v_command)
        if since_valid > self.sup.hold_timeout_s:
            if self.state != ControlState.HOLD:
                self._set_state(ControlState.HOLD, self.invalid_reason)
        self.v_command = self._apply(self.v_command)
        return self.v_command

    # ------------------------------------------------------------------

    def _apply(self, v):
        """
        Send a voltage to the pump backend.  The backend clamps again, and
        refuses a non-finite command outright.

        A failure here is treated as a fault: the controller transitions to
        SAFE and immediately ATTEMPTS the safe voltage rather than waiting for
        the next control step, because whatever the pump is currently holding
        is by definition not what the controller intended.
        """
        try:
            return self.pump.set_voltage(v)
        except Exception as exc:                              # noqa: BLE001
            logging.error("Pump command failed (%s); going to SAFE.", exc)
            if self.state != ControlState.SAFE:
                self.state = ControlState.SAFE
                self.counters['safe'] += 1
                if self.logger is not None:
                    self.logger.log_event('SAFE', f"pump command failed: {exc}")
            safe = float(self.sup.safe_pump_voltage)
            try:
                return self.pump.set_voltage(safe)
            except Exception:                                 # noqa: BLE001
                logging.error("The safe voltage could not be commanded either. "
                              "The pump is not under software control.")
            return safe

    def shutdown(self, reason="shutdown"):
        """Controlled transition to the configured safe state."""
        try:
            self.go_safe(reason)
        finally:
            logging.info("Controller counters: %s", self.counters)


# ==========================================================================
#  Simulated plant, for testing the loop with no hardware
# ==========================================================================

class FirstOrderJetModel:
    """
    Crude stand-in for pump + hydraulics, used ONLY by the test-suite and by
    the simulation run mode.

    U_dot = (K * f(V - V_deadband) - U) / tau,  evaluated with a fixed step,
    plus an optional pure transport delay and measurement noise.

    f() is a mildly non-linear (square-root-like) static map, deliberately NOT
    the linear relation the task warns against assuming.  It exists so the
    controller can be exercised; it is NOT a model of the EcoDrift 4.3 and no
    parameter in it should be taken as physical.  The real static and dynamic
    response must come from the open-loop calibration run.
    """

    def __init__(self, gain=4.0, tau=1.5, deadband=0.6, delay_s=0.2,
                 noise_std=0.0, seed=0, exponent=0.5):
        self.K, self.tau, self.db = gain, tau, deadband
        self.delay_s, self.exponent = delay_s, exponent
        self.rng = np.random.default_rng(seed)
        self.noise_std = noise_std
        self.u = 0.0
        self._buf = deque()

    def _static(self, v):
        x = max(0.0, v - self.db)
        return self.K * (x ** self.exponent)

    def step(self, v, dt):
        target = self._static(v)
        self.u += (target - self.u) * (dt / (self.tau + dt))
        self._buf.append(self.u)
        n_delay = max(1, int(round(self.delay_s / max(dt, 1e-9))))
        while len(self._buf) > n_delay:
            self._buf.popleft()
        out = self._buf[0]
        if self.noise_std > 0:
            out += self.rng.normal(0.0, self.noise_std)
        return out
