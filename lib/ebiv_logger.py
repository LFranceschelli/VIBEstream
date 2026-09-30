"""
EBIV Release 5.2 — Structured experiment logging.

One run produces, in <output_dir>/Control/<run_name>/ :

    <run_name>_control.csv    one row per control step, written incrementally
    <run_name>_events.csv     state transitions and faults, with timestamps
    <run_name>_config.json    the complete configuration, human-readable
    <run_name>_control.mat    the same columns as MATLAB arrays, plus the
                              config as a struct and the timing summary
    <run_name>_timing.csv     per-stage latency statistics

The .mat is written with scipy.io.savemat, the format Release 4.0 already
uses for velocity fields, so the same MATLAB tooling reads it.  The CSV is
written as the run proceeds so that a crash or a power cut still leaves
usable data, and so a run can be watched from another process.

Nothing here prints per step.  The periodic summary is on a timer.
"""

import os
import csv
import json
import time
import logging
from collections import deque, defaultdict

import numpy as np
from scipy.io import savemat

from ebiv_config import __version__


CONTROL_COLUMNS = [
    't_wall',              # seconds since the epoch (absolute, for cross-referencing)
    't_rel',               # seconds since the logger started
    'seq',                 # control-step counter
    'state',               # MANUAL / CLOSED_LOOP / HOLD / SAFE
    'u_target',            # reference, display units
    'u_raw',               # ControlROI spatial statistic, unfiltered
    'u_filtered',          # after the temporal filter; this is the PID input
    'v_command',           # voltage actually sent to the pump backend
    'error',
    'p_term', 'i_term', 'd_term',
    'u_unsaturated',       # PID output before saturation and slew limiting
    'saturated', 'slew_limited',
    'n_valid', 'n_total', 'valid_fraction',
    'measurement_valid',
    # Age of the FLOW information at the moment the controller looked at it:
    # newest contributing frame -> now.  This is the quantity the staleness
    # gate tests against max_measurement_age_s and the one quoted in
    # invalid_reason, so a reader comparing the two sees the same number.
    'measurement_age_ms',
    # How long the finished PIV result sat in the holder before this step
    # picked it up.  Small and uninteresting while PIV outruns the control
    # rate; it grows when the PIV rate cap is set below the control rate.
    'piv_output_age_ms',
    'latency_ms',          # newest contributing frame -> voltage applied
    'dt_control_ms',       # measured control-step interval
    'invalid_reason',
]


class ExperimentLogger:
    """Per-control-step logger plus a lightweight latency accumulator."""

    def __init__(self, output_dir, run_name, cfg, extra_header=None):
        self.cfg = cfg
        self.lcfg = cfg.logging
        self.enabled = bool(self.lcfg.enabled)
        self.dir = os.path.join(output_dir, "Control", run_name)
        self.run_name = run_name
        self.t0_wall = time.time()
        self.t0_perf = time.perf_counter()
        self.seq = 0

        self._rows = []
        self._events = []
        self._csv_file = None
        self._csv_writer = None
        self._since_flush = 0
        self._lat = defaultdict(lambda: deque(maxlen=4000))

        if not self.enabled:
            logging.info("Experiment logging disabled.")
            return

        os.makedirs(self.dir, exist_ok=True)

        header = {
            'run_name': run_name,
            'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'release': __version__,
            'config': cfg.to_dict(),
        }
        if extra_header:
            header.update(extra_header)
        with open(os.path.join(self.dir, f"{run_name}_config.json"), 'w') as f:
            json.dump(header, f, indent=2, default=str)

        if 'csv' in self.lcfg.formats:
            path = os.path.join(self.dir, f"{run_name}_control.csv")
            self._csv_file = open(path, 'w', newline='')
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(CONTROL_COLUMNS)

        logging.info("Experiment log -> %s", self.dir)
        # Make the choices that add phase lag impossible to overlook.
        logging.info("  temporal filter : %s", cfg.filt.describe())
        logging.info("  control rate    : %.2f Hz", cfg.supervisor.control_rate_hz)
        logging.info("  reference       : %s", cfg.reference.kind)
        logging.info("  PID             : Kp=%g Ki=%g Kd=%g, out [%.2f, %.2f] V, "
                     "slew=%s V/s", cfg.pid.kp, cfg.pid.ki, cfg.pid.kd,
                     cfg.pid.v_min, cfg.pid.v_max, cfg.pid.slew_rate_v_per_s)
        logging.info("  safe voltage    : %.3f V after %.1f s without a valid "
                     "measurement", cfg.supervisor.safe_pump_voltage,
                     cfg.supervisor.safe_timeout_s)

    # ------------------------------------------------------------------

    def log_step(self, sup, measurement, dt_control, latency_s=None):
        """Record one control step.  Cheap: appends and (rarely) flushes."""
        if not self.enabled:
            return
        self.seq += 1
        now_perf = time.perf_counter()
        d = sup.last_diag
        # NOT t_measured: that is when PIV finished, which says nothing about
        # how old the flow it describes is.  Before 5.2 this column carried
        # the t_measured age, so it read ~20 ms while the gate was rejecting
        # the same measurement as 3 s stale.
        if measurement is not None:
            age = now_perf - measurement.t_frame
            held = now_perf - measurement.t_measured
        else:
            age = held = float('nan')

        row = [
            self.t0_wall + (now_perf - self.t0_perf),
            now_perf - self.t0_perf,
            self.seq,
            sup.state,
            sup.u_target,
            sup.u_raw,
            sup.u_filtered,
            sup.v_command,
            d.error, d.p_term, d.i_term, d.d_term,
            d.u_unsaturated,
            int(d.saturated), int(d.slew_limited),
            sup.n_valid, sup.n_total,
            (sup.n_valid / sup.n_total) if sup.n_total else 0.0,
            int(measurement.valid) if measurement is not None else 0,
            age * 1e3,
            held * 1e3,
            (latency_s * 1e3) if latency_s is not None else float('nan'),
            dt_control * 1e3,
            sup.invalid_reason,
        ]
        self._rows.append(row)
        if self._csv_writer is not None:
            self._csv_writer.writerow(row)
            self._since_flush += 1
            if self._since_flush >= max(1, self.lcfg.flush_every):
                self._csv_file.flush()
                self._since_flush = 0

    def log_event(self, kind, detail=""):
        if not self.enabled:
            return
        self._events.append((time.time(), time.perf_counter() - self.t0_perf,
                             str(kind), str(detail)))

    def record_latency(self, stage, seconds):
        self._lat[stage].append(float(seconds))

    def adopt_stats(self, stats):
        """
        Merge a runtime LatencyStats accumulator into the logger's tables.

        The live loops record their timings into LatencyStats, which is a
        bounded ring buffer used for the periodic console summary.  Without
        this call those numbers would never reach <run>_timing.csv or the
        .mat, which is exactly what the integration test caught.
        """
        if stats is None:
            return
        try:
            for stage, samples in stats.series():
                self._lat[stage].extend(samples)
        except Exception:                                      # noqa: BLE001
            logging.exception("Could not adopt runtime timing statistics.")

    # ------------------------------------------------------------------

    def latency_summary(self):
        """(stage, n, mean_ms, p50, p95, max) rows, sorted by mean descending."""
        out = []
        for k, v in self._lat.items():
            if not v:
                continue
            a = np.fromiter(v, dtype=np.float64) * 1e3
            out.append((k, a.size, a.mean(), np.percentile(a, 50),
                        np.percentile(a, 95), a.max()))
        out.sort(key=lambda r: -r[2])
        return out

    def report_latency(self, title="Closed-loop timing"):
        rows = self.latency_summary()
        if not rows:
            return
        lines = ["", "=" * 74, f"  {title}", "=" * 74,
                 f"  {'stage':<30s} {'n':>7s} {'mean':>8s} {'p50':>8s} "
                 f"{'p95':>8s} {'max':>8s}", "  " + "-" * 70]
        for k, n, m, p50, p95, mx in rows:
            lines.append(f"  {k:<30s} {n:>7d} {m:>7.2f}ms {p50:>7.2f}ms "
                         f"{p95:>7.2f}ms {mx:>7.2f}ms")
        lines.append("=" * 74)
        logging.info("\n".join(lines))

    # ------------------------------------------------------------------

    def close(self, sup=None, extra=None):
        if not self.enabled:
            return
        if self._csv_file is not None:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file = None

        if self._events:
            with open(os.path.join(self.dir, f"{self.run_name}_events.csv"),
                      'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['t_wall', 't_rel', 'event', 'detail'])
                w.writerows(self._events)

        lat = self.latency_summary()
        if lat:
            with open(os.path.join(self.dir, f"{self.run_name}_timing.csv"),
                      'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['stage', 'n', 'mean_ms', 'p50_ms', 'p95_ms', 'max_ms'])
                w.writerows(lat)

        if 'mat' in self.lcfg.formats and self._rows:
            arr = list(zip(*self._rows))
            md = {}
            for name, col in zip(CONTROL_COLUMNS, arr):
                if name in ('state', 'invalid_reason'):
                    md[name] = np.array(col, dtype=object)
                else:
                    md[name] = np.asarray(col, dtype=np.float64)
            md['config_json'] = json.dumps(self.cfg.to_dict(), default=str)
            md['columns'] = np.array(CONTROL_COLUMNS, dtype=object)
            md['n_steps'] = len(self._rows)
            if lat:
                md['timing_stage'] = np.array([r[0] for r in lat], dtype=object)
                md['timing_mean_ms'] = np.asarray([r[2] for r in lat])
                md['timing_p95_ms'] = np.asarray([r[4] for r in lat])
            if sup is not None:
                md['counters'] = {k: int(v) for k, v in sup.counters.items()}
            if extra:
                md.update(extra)
            savemat(os.path.join(self.dir, f"{self.run_name}_control.mat"), md,
                    do_compression=True)

        logging.info("Experiment log closed: %d control steps, %d events -> %s",
                     len(self._rows), len(self._events), self.dir)
