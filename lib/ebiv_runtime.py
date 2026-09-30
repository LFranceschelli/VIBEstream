"""
EBIV Release 5.0 — Runtime plumbing.

Three logically separate loops, deliberately decoupled:

    ACQUISITION (main thread)
        EventsIterator -> event accumulation -> pseudo-frame -> PIV queue.
        Also owns every OpenCV GUI call, because HighGUI is not thread-safe.

    PIV WORKER (one thread)
        correlation -> validation -> ControlROI statistic -> publishes an
        immutable Measurement.  Runs as fast as it can; the queue has
        maxsize 1 so it always works on the freshest available pair.

    CONTROL (one thread)
        wakes at control_rate_hz, reads the latest Measurement, runs the
        filter / reference / PID, and commands the pump.

Why a separate control thread rather than folding the controller into the
acquisition loop:  the control rate must be settable independently of both
the camera rate and the display rate, and the PID needs a reasonably regular
dt.  The controller's own work is microseconds of scalar arithmetic plus one
ctypes call into the WaveForms runtime, which releases the GIL for the
duration of the USB transaction, so it does not meaningfully contend with the
numpy work in the other threads.  The loop nonetheless measures its ACTUAL dt
and feeds that to the PID rather than assuming the nominal period — the
measured jitter is reported at the end of every run.

The GIL is a real constraint and is respected rather than papered over: the
expensive stage (FFT correlation) is a single numpy/scipy call that releases
the GIL internally, so one worker thread genuinely overlaps with acquisition.
Adding more Python-level PIV threads would not help, and multiprocessing was
not adopted because shipping ~14 MB of window buffers per field across a pipe
would cost more than the correlation itself.
"""

import time
import logging
import threading

import numpy as np

from ebiv_control import Measurement, ControlState


# ==========================================================================
#  Latest-value mailboxes
# ==========================================================================

class FieldHolder:
    """
    Latest full vector field, for the display only.

    Release 4.0 published U, V and M as three separate writes into a shared
    dict, so the drawing thread could read U from one field and V from the
    next (AUDIT_R4_to_R5.md, bug B3).  Here the whole field is one tuple
    published by a single attribute assignment, which is atomic under the GIL.
    """
    __slots__ = ('latest',)

    def __init__(self):
        self.latest = None      # (U, V, M, CC, seq, t_frame)

    def publish(self, U, V, M, CC, seq, t_frame):
        self.latest = (U, V, M, CC, seq, t_frame)

    def read(self):
        return self.latest


# ==========================================================================
#  PIV worker
# ==========================================================================

def rt_piv_worker(in_queue, field_holder, meas_holder, correlator, estimator,
                  cfg, prof, logger=None, stats=None, hr=None):
    """
    Background PIV thread.

    Robustness changes versus Release 4.0:
      * the whole body is wrapped so that an exception logs a traceback and
        the thread keeps running.  In Release 4.0 an exception killed the
        thread silently and the display simply froze on the last field — with
        a controller attached that would mean feeding a stale velocity to the
        pump forever.  Here a failed field publishes an INVALID measurement,
        which the supervisor sees and handles.
      * the per-frame rate log is gone.  It formatted a string and took the
        logging lock on every tenth field in the middle of the hot loop; rate
        and latency now go to the periodic summary instead.
    """
    from ebiv_utils import universal_outlier_detection

    seq = 0
    while True:
        task = in_queue.get()
        if task is None:
            break

        frames, t_frame, apply_val = task[:3]
        frame_idx = task[3] if len(task) > 3 else None
        t0 = time.perf_counter()
        try:
            with prof.measure("total_correlation"):
                U, V, CC = correlator.correlate(frames, prof=prof)

            valid_mask = None
            U_disp, V_disp = U, V
            if apply_val:
                with prof.measure("validation"):
                    U_disp, V_disp, valid_mask = universal_outlier_detection(
                        U, V, cfg.piv.val_threshold, cfg.piv.val_epsilon)

            with prof.measure("magnitude_calc"):
                M = np.hypot(U_disp, V_disp)

            seq += 1
            field_holder.publish(U_disp, V_disp, M, CC, seq, t_frame)

            # HR estimation (vibe_hr_live.HRLive): same field as displayed, i.e.
            # validated and median-replaced when validation is on, like the
            # LR training data.  Frame index -> dropped pairs are bridged.
            if hr is not None and frame_idx is not None:
                with prof.measure("hr_estimation"):
                    hr.update(U_disp, V_disp, frame_idx)

            if estimator is not None:
                with prof.measure("control_roi_average"):
                    # NOTE: the ControlROI statistic uses the RAW U, V together
                    # with the outlier mask, so rejected vectors are EXCLUDED
                    # rather than replaced by a local median.  Release 4.0
                    # discarded the mask entirely and kept the substituted
                    # values (AUDIT_R4_to_R5.md, bug B4).
                    u_raw, ok, n_valid, reason = estimator.estimate(
                        U, V, valid_mask, CC)
                t_meas = time.perf_counter()
                meas_holder.publish(Measurement(
                    u_raw=u_raw, valid=bool(ok), n_valid=int(n_valid),
                    n_total=int(estimator.n_total), t_frame=t_frame,
                    t_measured=t_meas, seq=seq, reason=reason))

            if stats is not None:
                stats.record('piv_field_total', time.perf_counter() - t0)
                stats.record('piv_input_age', t0 - t_frame)

        except Exception:                                      # noqa: BLE001
            logging.exception("RT-PIV worker failed on field %d; continuing.", seq)
            seq += 1
            if estimator is not None:
                now = time.perf_counter()
                meas_holder.publish(Measurement(
                    u_raw=float('nan'), valid=False, n_valid=0,
                    n_total=int(estimator.n_total), t_frame=t_frame,
                    t_measured=now, seq=seq,
                    reason="PIV worker exception"))

    logging.info("RT-PIV worker exiting after %d fields.", seq)


# ==========================================================================
#  Lightweight timing accumulator
# ==========================================================================

class LatencyStats:
    """
    Ring-buffered timing accumulator with a periodic, rate-limited report.

    Deliberately not the profiler: the profiler keeps every sample for the
    offline CSV, which is fine for a 50-frame diagnostic run but grows without
    bound over a long closed-loop experiment.
    """

    def __init__(self, maxlen=4000, report_period_s=10.0):
        from collections import deque, defaultdict
        self._d = defaultdict(lambda: deque(maxlen=maxlen))
        self.report_period_s = float(report_period_s)
        self._t_last = time.perf_counter()

    def record(self, key, seconds):
        self._d[key].append(float(seconds))

    def series(self):
        """(stage, samples) pairs, so the logger can adopt them at shutdown."""
        return self._d.items()

    def snapshot(self):
        out = []
        for k, v in self._d.items():
            if not v:
                continue
            a = np.fromiter(v, dtype=np.float64) * 1e3
            out.append((k, a.size, a.mean(), np.percentile(a, 50),
                        np.percentile(a, 95), a.max()))
        out.sort(key=lambda r: -r[2])
        return out

    def maybe_report(self, extra_lines=()):
        now = time.perf_counter()
        if now - self._t_last < self.report_period_s:
            return False
        self._t_last = now
        rows = self.snapshot()
        if not rows:
            return False
        lines = ["", "-" * 72, "  Timing (rolling window)", "-" * 72,
                 f"  {'stage':<28s} {'n':>6s} {'mean':>8s} {'p50':>8s} "
                 f"{'p95':>8s} {'max':>8s}"]
        for k, n, m, p50, p95, mx in rows:
            lines.append(f"  {k:<28s} {n:>6d} {m:>7.2f}ms {p50:>7.2f}ms "
                         f"{p95:>7.2f}ms {mx:>7.2f}ms")
        for e in extra_lines:
            lines.append(f"  {e}")
        lines.append("-" * 72)
        logging.info("\n".join(lines))
        return True


# ==========================================================================
#  Asynchronous file saver
# ==========================================================================

class AsyncFileSaver:
    """
    Off-loads .tif and .mat writes from the acquisition loop.

    Release 4.0 called cv2.imwrite and scipy.io.savemat INSIDE the fast loop,
    once per camera chunk, and re-created the output directory with
    os.makedirs on every single frame (AUDIT_R4_to_R5.md, bug B5).  At 500 Hz
    a savemat of a full vector field cannot possibly keep up, so the loop
    stalls behind the filesystem and the camera buffer backs up.

    Here the write is queued to a daemon thread with a bounded queue.  If the
    disk cannot keep up the newest item is DROPPED and counted, rather than
    blocking acquisition — a dropped snapshot is a much smaller problem than a
    stalled control loop, and the drop count is reported at the end.
    """

    def __init__(self, maxsize=64):
        import queue
        self.q = queue.Queue(maxsize=maxsize)
        self.dropped = 0
        self.written = 0
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        import cv2
        from scipy.io import savemat
        while True:
            item = self.q.get()
            if item is None:
                break
            kind, path, payload = item
            try:
                if kind == 'tif':
                    cv2.imwrite(path, payload)
                else:
                    savemat(path, payload)
                self.written += 1
            except Exception:                                  # noqa: BLE001
                logging.exception("Failed to write %s", path)

    def submit(self, kind, path, payload):
        try:
            self.q.put_nowait((kind, path, payload))
            return True
        except Exception:                                      # noqa: BLE001
            self.dropped += 1
            return False

    def close(self, timeout=10.0):
        try:
            self.q.put_nowait(None)
        except Exception:                                      # noqa: BLE001
            pass
        self._t.join(timeout=timeout)
        if self.dropped:
            logging.warning("AsyncFileSaver: %d writes dropped (disk could not "
                            "keep up), %d written.", self.dropped, self.written)
        else:
            logging.info("AsyncFileSaver: %d files written, none dropped.",
                         self.written)


# ==========================================================================
#  Open-loop calibration driver
# ==========================================================================

class CalibrationDriver:
    """
    Prescribes the pump voltage in open loop while rtEBIV keeps measuring.

    Runs inside the control thread so that it shares the same clock, the same
    logging and the same safety handling as closed-loop operation; the only
    difference is that the voltage comes from a schedule instead of the PID.

    voltage_at(t) returns the commanded voltage for elapsed time t, or None
    when the programme has finished.
    """

    def __init__(self, cfg_cal, v_min, v_max, force_manual=False):
        """
        force_manual :
            Ignore cfg_cal.mode and always return the operator's value.

            RELEASE 5.2 BUG FIX.  STREAM_MODE = 'manual' and CAL_MODE are two
            different settings, and the driver used to obey CAL_MODE in both
            cases.  With the shipped CAL_MODE = 'steps' that meant a 'manual'
            run silently ran the programmed voltage schedule: the control
            thread called set_manual_voltage(schedule) every 100 ms, so a
            [+] press was overwritten before the next one arrived and the
            voltage appeared stuck at one step above the schedule value
            forever.  CAL_MODE now governs STREAM_MODE = 'calibration' only.
        """
        self.cfg = cfg_cal
        self.v_min, self.v_max = float(v_min), float(v_max)
        self._manual = float(v_min)
        self.force_manual = bool(force_manual)
        self.finished = False

    @property
    def mode(self):
        """The mode actually in force, which is not always cfg.mode."""
        return 'manual' if self.force_manual else self.cfg.mode

    # --- manual mode -------------------------------------------------
    def nudge(self, delta):
        self._manual = float(np.clip(self._manual + delta, self.v_min, self.v_max))
        return self._manual

    def set_manual(self, v):
        self._manual = float(np.clip(v, self.v_min, self.v_max))
        return self._manual

    # --- programme ---------------------------------------------------
    def voltage_at(self, t):
        c = self.cfg
        if self.mode == 'manual':
            return self._manual

        if self.mode == 'steps':
            n = len(c.voltages)
            total = n * c.dwell_s
            if t >= total:
                self.finished = True
                return float(np.clip(c.voltages[-1], self.v_min, self.v_max))
            i = int(t // c.dwell_s)
            return float(np.clip(c.voltages[min(i, n - 1)], self.v_min, self.v_max))

        # sweep
        d = c.sweep_duration_s
        if c.sweep_return:
            if t >= 2 * d:
                self.finished = True
                return float(np.clip(c.v_start, self.v_min, self.v_max))
            frac = (t / d) if t < d else (2.0 - t / d)
        else:
            if t >= d:
                self.finished = True
                return float(np.clip(c.v_stop, self.v_min, self.v_max))
            frac = t / d
        v = c.v_start + frac * (c.v_stop - c.v_start)
        return float(np.clip(v, self.v_min, self.v_max))

    def describe(self):
        c = self.cfg
        if self.mode == 'manual':
            return (f"manual, [+]/[-] in {c.manual_step_v:g} V steps, "
                    f"range [{self.v_min:g}, {self.v_max:g}] V")
        if c.mode == 'steps':
            return (f"steps {c.voltages} V, {c.dwell_s:g} s each "
                    f"(first {c.settle_s:g} s treated as transient), "
                    f"total {len(c.voltages) * c.dwell_s:g} s")
        return (f"sweep {c.v_start:g} -> {c.v_stop:g} V over {c.sweep_duration_s:g} s"
                + (" and back" if c.sweep_return else ""))


# ==========================================================================
#  Control thread
# ==========================================================================

def control_thread_main(supervisor, meas_holder, cfg, logger, chart,
                        stop_event, stats, calibration=None,
                        on_step=None):
    """
    Fixed-rate control loop.

    Timing: the next wake-up is advanced by exactly one period each iteration
    so that scheduling jitter does not accumulate into drift.  If the loop
    falls more than a few periods behind (a long GC pause, a system hiccup)
    the schedule is re-based on the current time rather than firing a burst of
    catch-up steps into the pump.

    Every step uses the MEASURED interval, not the nominal one.
    """
    period = 1.0 / cfg.supervisor.control_rate_hz
    t_next = time.perf_counter()
    t_prev = None
    t_start = time.perf_counter()
    n_late = 0

    logging.info("Control thread started at %.2f Hz (period %.1f ms).",
                 cfg.supervisor.control_rate_hz, period * 1e3)

    while not stop_event.is_set():
        now = time.perf_counter()
        if now < t_next:
            stop_event.wait(min(t_next - now, 0.010))
            continue

        t_next += period
        if t_next < now - 3 * period:
            n_late += 1
            t_next = now + period       # re-base instead of bursting

        t_step0 = time.perf_counter()
        m = meas_holder.read()

        # open-loop programme drives the manual set-point
        if calibration is not None and supervisor.state == ControlState.MANUAL:
            v = calibration.voltage_at(now - t_start)
            if v is not None:
                supervisor.set_manual_voltage(v)

        _valid_before = supervisor.counters['valid']
        try:
            supervisor.step(m, now)
        except Exception:                                      # noqa: BLE001
            logging.exception("Control step failed; transitioning to SAFE.")
            supervisor.go_safe("control step raised an exception")
        # The supervisor increments 'valid' exactly when it accepts a fresh,
        # in-date measurement, so this is the authoritative "was it used".
        used_this_step = supervisor.counters['valid'] > _valid_before

        t_now = time.perf_counter()
        dt_ctl = (now - t_prev) if t_prev is not None else period
        t_prev = now

        # LATENCY vs AGE -- two different things, and recording them as one
        # made the headline number nonsense.
        #
        # `t_now - m.t_frame` was recorded on EVERY control step, including the
        # ones where the controller looked at a held measurement and refused
        # it.  During a gap in the PIV stream the same Measurement object is
        # re-read each cycle and its age keeps growing, so the metric reported
        # 2.7 s mean / 8.5 s max "latency" for a pipeline whose real latency
        # was ~35 ms.  Nothing was ever actuated on those samples.
        #
        #   latency_frame_to_actuation : only when the controller USED the
        #                                measurement.  The real loop latency.
        #   measurement_age_at_step    : every step, whatever it looked at.
        #                                Says how stale things get in the gaps.
        latency = (t_now - m.t_frame) if m is not None else None
        stats.record('control_step', t_now - t_step0)
        stats.record('control_interval', dt_ctl)
        if latency is not None:
            stats.record('measurement_age_at_step', latency)
            if used_this_step:
                stats.record('latency_frame_to_actuation', latency)

        if logger is not None:
            logger.log_step(supervisor, m, dt_ctl, latency)
        if chart is not None:
            chart.append(now - t_start, supervisor.u_target, supervisor.u_filtered,
                         supervisor.u_raw, supervisor.v_command, supervisor.state)
        if on_step is not None:
            on_step(supervisor, m)

    logging.info("Control thread stopping (%d re-base events).", n_late)


# ==========================================================================
#  Simulated closed-loop run (no camera, no Analog Discovery)
# ==========================================================================

def simulate_closed_loop(cfg, plant, duration_s=60.0, dt=None, pump=None,
                         measurement_noise=0.0, dropout_windows=(), seed=0):
    """
    Drive the real ControlSupervisor against a simulated plant on a synthetic
    clock.  Used by the test-suite and by FLAG_SIMULATE to check gains,
    reference trajectories and fault handling before touching hardware.

    dropout_windows : iterable of (t_start, t_end) during which the
        measurement is reported INVALID, so the HOLD / SAFE logic can be
        exercised deterministically.

    Returns a dict of numpy arrays.
    """
    from ebiv_control import ControlSupervisor, Measurement
    from ebiv_hardware import MockPump

    if pump is None:
        pump = MockPump(cfg.ad3.pump_v_min, cfg.ad3.pump_v_max,
                        cfg.supervisor.startup_pump_voltage)
    sup = ControlSupervisor(cfg, pump)
    dt = dt or 1.0 / cfg.supervisor.control_rate_hz
    rng = np.random.default_rng(seed)

    n = int(duration_s / dt)
    t = np.arange(n) * dt
    out = {k: np.full(n, np.nan) for k in
           ('u_target', 'u_raw', 'u_filtered', 'v_command', 'u_plant', 'error')}
    out['state'] = np.empty(n, dtype=object)
    out['t'] = t

    u_plant = 0.0
    sup.arm(now=0.0)
    for i in range(n):
        now = t[i]
        u_plant = plant.step(pump.voltage, dt)
        y = u_plant + (rng.normal(0.0, measurement_noise) if measurement_noise else 0.0)
        bad = any(a <= now < b for a, b in dropout_windows)
        m = Measurement(u_raw=float(y), valid=not bad,
                        n_valid=0 if bad else 100, n_total=100,
                        t_frame=now, t_measured=now, seq=i,
                        reason="simulated dropout" if bad else "")
        sup.step(m, now=now)
        out['u_target'][i] = sup.u_target
        out['u_raw'][i] = sup.u_raw
        out['u_filtered'][i] = sup.u_filtered
        out['v_command'][i] = sup.v_command
        out['u_plant'][i] = u_plant
        out['error'][i] = sup.last_diag.error
        out['state'][i] = sup.state
    out['supervisor'] = sup
    out['pump'] = pump
    return out
