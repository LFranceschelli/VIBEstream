"""
EBIV Release 5.1 — end-to-end integration test with a FAKE camera.

Exercises the real stream_camera(): the acquisition loop, the PIV worker
thread, the control thread, the ControlROI, the supervisor state machine, the
logger and the visualisation code paths — all of it, with no camera, no
Analog Discovery and no display.

A synthetic event stream is injected in place of metavision's EventsIterator,
carrying particles that translate by a known displacement each frame, so the
recovered velocity can be checked against ground truth through the whole
pipeline rather than only at the correlator.

    python test_integration.py
"""

# Run from anywhere: put the parent folder (the library) on the path.
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)                      # EBIV_Main.py
_sys.path.insert(0, _os.path.join(_ROOT, 'lib'))  # the library


import os
import sys
import types
import time
import tempfile
import logging
import threading

import numpy as np

_PASS, _FAIL = [], []


def check(name, cond, detail=""):
    (_PASS if cond else _FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
    return bool(cond)


# ==========================================================================
#  Fake camera
# ==========================================================================

SENSOR_W, SENSOR_H = 640, 480
TRUE_DX, TRUE_DY = 4.0, 0.0        # px per frame, +x  -> component 'u'


class _FakeEventsIterator:
    """Yields structured event chunks with particles translating by (dx, dy)."""

    def __init__(self, n_chunks, n_particles=9000, seed=0, dx=TRUE_DX, dy=TRUE_DY,
                 chunk_dt_us=2000):
        self.n_chunks = n_chunks
        self.dx, self.dy = dx, dy
        self.chunk_dt_us = chunk_dt_us
        rng = np.random.default_rng(seed)
        self.x0 = rng.uniform(0, SENSOR_W, n_particles)
        self.y0 = rng.uniform(0, SENSOR_H, n_particles)
        self.i = 0

    @staticmethod
    def from_device(device=None, delta_t=None):
        raise NotImplementedError            # replaced per-test

    def get_size(self):
        return SENSOR_H, SENSOR_W

    def __iter__(self):
        return self

    def __next__(self):
        if self.i >= self.n_chunks:
            raise StopIteration
        k = self.i
        self.i += 1
        x = (self.x0 + k * self.dx) % SENSOR_W
        y = (self.y0 + k * self.dy) % SENSOR_H
        n = x.size
        evs = np.zeros(n, dtype=[('x', '<u2'), ('y', '<u2'), ('t', '<i8'), ('p', 'i1')])
        evs['x'] = x.astype(np.uint16)
        evs['y'] = y.astype(np.uint16)
        evs['t'] = k * self.chunk_dt_us + np.arange(n) % self.chunk_dt_us
        # a small real-time-ish pacing so the control thread gets to run
        time.sleep(0.002)
        return evs


class _FakeStream:
    def log_raw_data(self, p):
        pass

    def stop_log_raw_data(self):
        pass


class _FakeDevice:
    def get_i_events_stream(self):
        return _FakeStream()

    def get_i_trigger_in(self):
        return None

    def get_i_ll_biases(self):
        return None


def install_fakes():
    """Inject fake metavision modules and headless OpenCV GUI stubs."""
    mv = types.ModuleType("metavision_hal")

    class _DD:
        @staticmethod
        def list():
            return ["fake"]

        @staticmethod
        def open(d):
            return _FakeDevice()

    mv.DeviceDiscovery = _DD
    sys.modules["metavision_hal"] = mv

    core = types.ModuleType("metavision_core")
    eio = types.ModuleType("metavision_core.event_io")
    eio.EventsIterator = _FakeEventsIterator
    core.event_io = eio
    sys.modules["metavision_core"] = core
    sys.modules["metavision_core.event_io"] = eio

    import cv2
    cv2.imshow = lambda *a, **k: None
    cv2.destroyAllWindows = lambda *a, **k: None
    cv2.destroyWindow = lambda *a, **k: None
    return cv2


# ==========================================================================

def run():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    cv2 = install_fakes()

    import ebiv_utils
    from ebiv_config import (ControlSystemConfig, ROIConfig, MeasurementConfig,
                             FilterConfig, ReferenceConfig, PIDConfig,
                             SupervisorConfig, AD3Config, CalibrationConfig,
                             LoggingConfig, PIVConfig)
    from ebiv_control import ControlState

    N_CHUNKS = 260
    # ebiv_utils imports the camera SDK at module load and degrades to
    # mv = None when it is absent.  It may already have been imported by an
    # earlier test, in which case injecting into sys.modules is not enough:
    # rebind the module attributes directly so this test is independent of
    # import order.
    ebiv_utils.mv = sys.modules["metavision_hal"]
    ebiv_utils._HAS_METAVISION = True
    ebiv_utils.EventsIterator = _FakeEventsIterator
    _FakeEventsIterator.from_device = staticmethod(
        lambda device=None, delta_t=None: _FakeEventsIterator(N_CHUNKS))

    display_roi = [64, 576, 64, 416]
    control_roi = [200, 400, 160, 320]

    cfg = ControlSystemConfig(
        roi=ROIConfig(display_roi=display_roi, control_roi=control_roi),
        measurement=MeasurementConfig(component='u', statistic='mean',
                                      min_valid_fraction=0.4,
                                      min_valid_vectors=3,
                                      use_correlation_gate=True,
                                      min_correlation=0.01,
                                      max_abs_displacement_px=15.0),
        filt=FilterConfig(kind='ema', tau_s=0.05),
        reference=ReferenceConfig(kind='constant', value=TRUE_DX),
        pid=PIDConfig(kp=0.2, ki=0.4, kd=0.0, v_min=0.0, v_max=5.0,
                      slew_rate_v_per_s=10.0),
        supervisor=SupervisorConfig(control_rate_hz=40.0,
                                    visualization_rate_hz=25.0,
                                    plot_rate_hz=5.0,
                                    hold_timeout_s=0.3, safe_timeout_s=1.5,
                                    max_measurement_age_s=0.5,
                                    safe_pump_voltage=0.75,
                                    startup_pump_voltage=1.0),
        ad3=AD3Config(enabled=False),
        calibration=CalibrationConfig(mode='manual'),
        logging=LoggingConfig(enabled=True, formats=('csv', 'mat'),
                              flush_every=1, latency_report_period_s=1.0),
        piv=PIVConfig(window_size=32, node_distance=16, fft_workers=1,
                      subpixel=True, compute_quality=True, validation=True),
    )

    # Press [c] once, a little way in, to arm the loop.  waitKey is only
    # called on display iterations (every skip_frames frames), so this fires
    # after a handful of them, once a velocity measurement exists.
    state = {'n': 0}

    def fake_waitkey(_):
        state['n'] += 1
        return ord('c') if state['n'] == 4 else 0xFF

    cv2.waitKey = fake_waitkey

    with tempfile.TemporaryDirectory() as out:
        print(f"\n{'=' * 74}\n  running stream_camera() against the fake camera\n{'=' * 74}")
        t0 = time.perf_counter()
        ebiv_utils.stream_camera(
            output_dir=out, accum_time_us=2000, biases=None,
            max_events_per_pixel=1, trigger_mode='none',
            prof=ebiv_utils._NULL_PROF, cfg=cfg,
            run_mode='closed_loop', run_name='integration',
            arrow_skip=1, arrow_scale=3, save_rt_piv=True,
            show_strip_chart=True)
        elapsed = time.perf_counter() - t0

        print(f"\n{'=' * 74}\n  checks\n{'=' * 74}")
        check("stream_camera() ran to completion against a synthetic stream",
              True, f"{N_CHUNKS} chunks in {elapsed:.1f} s")

        d = os.path.join(out, "Control", "integration")
        files = sorted(os.listdir(d)) if os.path.isdir(d) else []
        for want in ('integration_control.csv', 'integration_config.json',
                     'integration_control.mat', 'integration_timing.csv',
                     'integration_history.png'):
            check(f"wrote {want}", want in files, str(files))

        import csv
        with open(os.path.join(d, 'integration_control.csv')) as f:
            rows = list(csv.DictReader(f))
        check("control log has rows", len(rows) > 20, f"{len(rows)} control steps")

        states = [r['state'] for r in rows]
        check("the controller reached CLOSED_LOOP via the [c] key",
              ControlState.CLOSED_LOOP in states,
              f"states seen: {sorted(set(states))}")
        check("the controller never entered SAFE during a healthy run",
              ControlState.SAFE not in states[:-2],
              f"{states.count(ControlState.SAFE)} SAFE rows")

        closed = [r for r in rows if r['state'] == ControlState.CLOSED_LOOP]
        u = np.array([float(r['u_raw']) for r in closed if r['u_raw'] not in ('', 'nan')])
        u = u[np.isfinite(u)]
        # u_raw is logged in DISPLAY units, which are px/s by default now.
        # Convert back to px/frame with the same scale the estimator applied,
        # so this asserts the recovery of the injected displacement rather
        # than whichever unit the display happens to be using.
        _scale = cfg.measurement.velocity_scale
        u_px = u / _scale
        check("ControlROI velocity recovers the injected displacement "
              "end-to-end through the real pipeline",
              u.size > 5 and abs(np.median(u_px) - TRUE_DX) < 0.35,
              f"median U_control = {np.median(u):+.3f} "
              f"{cfg.measurement.velocity_units} = {np.median(u_px):+.3f} "
              f"px/frame, injected {TRUE_DX:+.1f} (n={u.size})")

        # The latency metric must count only measurements the controller
        # actually USED.  Recording it on every step meant a held measurement's
        # ever-growing age was logged as "latency": a real run reported 2.7 s
        # mean for a pipeline whose true latency was ~35 ms.
        import ebiv_runtime as _rt
        src = open(_rt.__file__, encoding='utf-8').read()
        check("latency is recorded only when the measurement was used",
              "if used_this_step:" in src
              and "stats.record('latency_frame_to_actuation', latency)" in src,
              "otherwise the headline latency is the age of data nobody acted on")
        check("the age of every looked-at measurement is kept separately",
              "stats.record('measurement_age_at_step', latency)" in src,
              "the staleness information is still worth having, under an "
              "honest name")

        nv = np.array([float(r['valid_fraction']) for r in closed])
        check("most ControlROI vectors were valid",
              nv.size and nv.mean() > 0.8, f"mean valid fraction {nv.mean():.2f}")

        v = np.array([float(r['v_command']) for r in rows])
        check("the pump command stayed inside [v_min, v_max] at all times",
              v.min() >= 0.0 - 1e-9 and v.max() <= 5.0 + 1e-9,
              f"range [{v.min():.3f}, {v.max():.3f}] V")
        check("the controller actually moved the command (the loop did work)",
              v.max() - v.min() > 1e-6, f"span {v.max() - v.min():.3f} V")

        lat_all = np.array([float(r['latency_ms']) for r in rows])
        lat = lat_all[np.isfinite(lat_all)]
        check("measurement-to-actuation latency was recorded",
              lat.size > 10 and lat.max() < 5000,
              f"mean {lat.mean():.1f} ms, p95 {np.percentile(lat, 95):.1f} ms")

        # measurement_age_ms must be the quantity the staleness gate tests, or
        # a reader chasing a lag problem reads ~20 ms in the CSV while
        # invalid_reason on the same row says the measurement was seconds old.
        # Before 5.2 this column held the time since PIV finished instead.
        age = np.array([float(r['measurement_age_ms']) for r in rows])
        held = np.array([float(r['piv_output_age_ms']) for r in rows])
        ok = np.isfinite(age) & np.isfinite(lat_all)
        check("measurement_age_ms is the frame age, not the time since PIV "
              "finished",
              ok.sum() > 10 and np.allclose(age[ok], lat_all[ok], atol=1.0),
              f"agrees with latency_ms to {np.abs(age[ok] - lat_all[ok]).max():.3f} ms")
        okh = np.isfinite(held) & np.isfinite(age)
        check("piv_output_age_ms keeps the holder dwell time separately",
              okh.sum() > 10 and np.all(held[okh] <= age[okh] + 1e-6),
              f"holder dwell median {np.median(held[okh]):.1f} ms vs frame age "
              f"median {np.median(age[okh]):.1f} ms")

        dtc = np.array([float(r['dt_control_ms']) for r in rows])
        check("the control loop ran near its configured rate",
              abs(np.median(dtc) - 25.0) < 15.0,
              f"median interval {np.median(dtc):.1f} ms (nominal 25.0 ms at 40 Hz)")

        # the run must end in the configured safe state, not at 0 V by accident
        check("the run finished in the configured safe state",
              abs(v[-1] - 0.75) < 1e-9, f"final command {v[-1]:.3f} V")

        piv_dir = os.path.join(out, "RT_PIV")
        n_mat = len(os.listdir(piv_dir)) if os.path.isdir(piv_dir) else 0
        check("RT-PIV snapshots were written asynchronously", n_mat > 0,
              f"{n_mat} .mat files")
        if n_mat:
            from scipy.io import loadmat
            md = loadmat(os.path.join(piv_dir, sorted(os.listdir(piv_dir))[0]))
            check("snapshots carry the ROIs and the correlation quality",
                  'control_roi' in md and 'display_roi' in md and 'CC' in md,
                  sorted(k for k in md if not k.startswith('__')))

        with open(os.path.join(d, 'integration_timing.csv')) as f:
            trows = list(csv.DictReader(f))
        stages = {r['stage'] for r in trows}
        for s in ('control_step', 'control_interval', 'latency_frame_to_actuation',
                  'piv_field_total'):
            check(f"timing recorded for '{s}'", s in stages, str(sorted(stages)))

        threads = [t.name for t in threading.enumerate() if t.is_alive()]
        check("no worker threads were left running after shutdown",
              len(threads) <= 2, str(threads))

    print(f"\n{'=' * 74}\n  {len(_PASS)} passed, {len(_FAIL)} failed")
    if _FAIL:
        for n in _FAIL:
            print(f"    - {n}")
    print('=' * 74)
    return 1 if _FAIL else 0


if __name__ == "__main__":
    sys.exit(run())
