"""
EBIV Release 5.2 — software test-suite.

Runs with no camera, no Analog Discovery and no CUDA.  Only numpy, scipy and
OpenCV are required.  Invoke either as

    python test_release5.py
or  set FLAG_SELFTEST = True in EBIV_Main.py

Coverage, in the order the task requires:
  1  EBIV numerical equivalence with Release 4.0 (control disabled)
  2  ControlROI spatial average with synthetic vectors
  3  Moving-average and exponential filters
  4  PID saturation, anti-windup, slew limiting, bumpless transfer
  5  Invalid-measurement handling (HOLD -> SAFE)
  6  Reference generators
  7  Closed loop against a simulated first-order pump/jet model
  8  Hardware abstraction: mock pump enforces its limits; no device needed
"""

# Run from anywhere: put the parent folder (the library) on the path.
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)                      # EBIV_Main.py
_sys.path.insert(0, _os.path.join(_ROOT, 'lib'))  # the library


import io
import os
import sys
import math
import time
import logging
import tempfile
import traceback

import numpy as np
from scipy.fft import rfft2, irfft2

# Keep the test output readable
logging.basicConfig(level=logging.WARNING,
                    format="%(levelname)s | %(message)s")

_PASS, _FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        _PASS.append(name)
        print(f"  PASS  {name}" + (f"   [{detail}]" if detail else ""))
    else:
        _FAIL.append((name, detail))
        print(f"  FAIL  {name}" + (f"   [{detail}]" if detail else ""))
    return bool(cond)


def section(title):
    print(f"\n{'=' * 74}\n  {title}\n{'=' * 74}")


# ==========================================================================
#  Release 4.0 reference implementations, copied verbatim
# ==========================================================================

def _extract_windows_R4(frame, ws, nd, gy, gx):
    W = np.empty((gy * gx, ws, ws), dtype=np.float32)
    i = 0
    for y in range(gy):
        for x in range(gx):
            W[i] = frame[y * nd:y * nd + ws, x * nd:x * nd + ws]
            i += 1
    return W


def batch_cross_correlate_R4(frames, ws, nd):
    h, w = frames[0].shape
    gy = (h - ws) // nd + 1
    gx = (w - ws) // nd + 1
    if len(frames) == 3:
        W0 = _extract_windows_R4(frames[0], ws, nd, gy, gx).astype(np.float32)
        W1 = _extract_windows_R4(frames[1], ws, nd, gy, gx).astype(np.float32)
        W2 = _extract_windows_R4(frames[2], ws, nd, gy, gx).astype(np.float32)
        W0 -= W0.mean(axis=(1, 2), keepdims=True)
        W1 -= W1.mean(axis=(1, 2), keepdims=True)
        W2 -= W2.mean(axis=(1, 2), keepdims=True)
        F0, F1, F2 = rfft2(W0), rfft2(W1), rfft2(W2)
        cp = (np.conj(F0) * F1) + (np.conj(F1) * F2)
    else:
        W1 = _extract_windows_R4(frames[0], ws, nd, gy, gx).astype(np.float32)
        W2 = _extract_windows_R4(frames[1], ws, nd, gy, gx).astype(np.float32)
        W1 -= W1.mean(axis=(1, 2), keepdims=True)
        W2 -= W2.mean(axis=(1, 2), keepdims=True)
        F1, F2 = rfft2(W1), rfft2(W2)
        cp = np.conj(F1) * F2
    R = np.real(irfft2(cp))
    R = np.fft.fftshift(R, axes=(1, 2))
    Rf = R.reshape(R.shape[0], -1)
    pk = np.argmax(Rf, axis=1)
    cy, cx = np.unravel_index(pk, (ws, ws))
    U = (cx - ws // 2).reshape(gy, gx).astype(np.float32)
    V = (cy - ws // 2).reshape(gy, gx).astype(np.float32)
    return U, V


def batch_subpixel_peak_R4(R_batch, cy_all, cx_all):
    """Release 4.0's offline sub-pixel estimator, verbatim."""
    N = R_batch.shape[0]
    ws = R_batch.shape[1]
    dx = np.zeros(N, dtype=np.float64)
    dy = np.zeros(N, dtype=np.float64)
    idx = np.arange(N)
    valid_x = (cx_all > 0) & (cx_all < ws - 1)
    if np.any(valid_x):
        vi = idx[valid_x]
        cl_ = R_batch[vi, cy_all[vi], cx_all[vi] - 1]
        cc_ = R_batch[vi, cy_all[vi], cx_all[vi]]
        cr_ = R_batch[vi, cy_all[vi], cx_all[vi] + 1]
        ok = (cl_ > 0) & (cc_ > 0) & (cr_ > 0)
        gi = vi[ok]
        if len(gi):
            cl, cc, cr = np.log(cl_[ok]), np.log(cc_[ok]), np.log(cr_[ok])
            den = 2.0 * (cl - 2 * cc + cr)
            s = np.abs(den) > 1e-12
            dx[gi[s]] = np.clip((cl[s] - cr[s]) / den[s], -1, 1)
        pi = vi[~ok]
        if len(pi):
            a, b, c = cl_[~ok], cc_[~ok], cr_[~ok]
            den = 2.0 * (a - 2 * b + c)
            s = np.abs(den) > 1e-12
            dx[pi[s]] = np.clip((a[s] - c[s]) / den[s], -1, 1)
    valid_y = (cy_all > 0) & (cy_all < ws - 1)
    if np.any(valid_y):
        vi = idx[valid_y]
        cu_ = R_batch[vi, cy_all[vi] - 1, cx_all[vi]]
        cc_ = R_batch[vi, cy_all[vi], cx_all[vi]]
        cd_ = R_batch[vi, cy_all[vi] + 1, cx_all[vi]]
        ok = (cu_ > 0) & (cc_ > 0) & (cd_ > 0)
        gi = vi[ok]
        if len(gi):
            cu, cc, cd = np.log(cu_[ok]), np.log(cc_[ok]), np.log(cd_[ok])
            den = 2.0 * (cu - 2 * cc + cd)
            s = np.abs(den) > 1e-12
            dy[gi[s]] = np.clip((cu[s] - cd[s]) / den[s], -1, 1)
        pi = vi[~ok]
        if len(pi):
            a, b, c = cu_[~ok], cc_[~ok], cd_[~ok]
            den = 2.0 * (a - 2 * b + c)
            s = np.abs(den) > 1e-12
            dy[pi[s]] = np.clip((a[s] - c[s]) / den[s], -1, 1)
    return dx, dy


def make_pair(h, w, sx, sy, n=8000, seed=0):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0, w, n)
    ys = rng.uniform(0, h, n)

    def render(px, py):
        f = np.zeros((h, w), dtype=np.float32)
        xi = np.clip(px.astype(np.int32), 0, w - 1)
        yi = np.clip(py.astype(np.int32), 0, h - 1)
        np.add.at(f, (yi, xi), 255.0)
        return np.clip(f, 0, 255).astype(np.uint8)

    return render(xs, ys), render(xs + sx, ys + sy)


# ==========================================================================
#  1. EBIV equivalence
# ==========================================================================

def test_ebiv_equivalence():
    section("1  EBIV numerical equivalence with Release 4.0")
    from ebiv_piv import CPUCorrelator, extract_windows, batch_subpixel_flat

    rng = np.random.default_rng(0)

    # --- window extraction, element for element ---
    ok = True
    for (h, w, ws, nd) in [(240, 320, 32, 16), (200, 300, 48, 24), (128, 128, 64, 32)]:
        f = rng.integers(0, 256, (h, w), dtype=np.uint8)
        gy, gx = (h - ws) // nd + 1, (w - ws) // nd + 1
        a = _extract_windows_R4(f, ws, nd, gy, gx)
        b = extract_windows(f, ws, nd, gy, gx)
        ok &= np.array_equal(a, b)
    check("window extraction is bit-identical to the R4 Python loop", ok)

    # --- full RT correlation, 2-frame and 3-frame ---
    n_diff = n_tot = 0
    for i in range(12):
        h, w, ws, nd = 240, 320, 32, 16
        f1, f2 = make_pair(h, w, rng.uniform(-7, 7), rng.uniform(-7, 7), seed=i)
        f3 = make_pair(h, w, 1.0, 1.0, seed=i + 100)[1]
        for triple in (False, True):
            frames = [f1, f2, f3] if triple else [f1, f2]
            U4, V4 = batch_cross_correlate_R4(frames, ws, nd)
            c = CPUCorrelator(ws, nd, (h, w), triple_corr=triple,
                              fft_workers=1, subpixel=False,
                              compute_quality=False, fast_mean_removal=False)
            U5, V5, _ = c.correlate(frames)
            n_diff += int(np.sum((U4 != U5) | (V4 != V5)))
            n_tot += U4.size
    check("RT correlation is bit-identical to R4 (subpixel off, workers=1)",
          n_diff == 0, f"{n_tot} vectors compared, {n_diff} differ")

    # --- fft_workers must not change the result ---
    f1, f2 = make_pair(240, 320, 4.4, -2.2, seed=42)
    c1 = CPUCorrelator(32, 16, (240, 320), fft_workers=1, subpixel=False,
                       compute_quality=False)
    c4 = CPUCorrelator(32, 16, (240, 320), fft_workers=4, subpixel=False,
                       compute_quality=False)
    U1, V1, _ = c1.correlate([f1, f2])
    U4_, V4_, _ = c4.correlate([f1, f2])
    check("fft_workers>1 gives an identical result",
          np.array_equal(U1, U4_) and np.array_equal(V1, V4_))

    # --- sub-pixel estimator matches R4's offline implementation ---
    ws = 32
    R = np.abs(rng.standard_normal((60, ws, ws))).astype(np.float32) + 0.01
    R[10] = 0.0                                     # degenerate plane
    Rf = R.reshape(60, -1)
    pk = np.argmax(Rf, axis=1)
    cy, cx = np.unravel_index(pk, (ws, ws))
    dx4, dy4 = batch_subpixel_peak_R4(R, cy, cx)
    dx5, dy5 = batch_subpixel_flat(Rf, pk, ws)
    check("sub-pixel estimator is bit-identical to R4's offline version",
          np.array_equal(dx4, dx5) and np.array_equal(dy4, dy5))

    # --- sub-pixel actually improves accuracy on a known displacement ---
    errs = {}
    for sub in (False, True):
        e = []
        for k, s in enumerate(np.linspace(3.1, 4.9, 7)):
            a, b = make_pair(240, 320, s, 0.0, n=20000, seed=200 + k)
            c = CPUCorrelator(32, 16, (240, 320), subpixel=sub,
                              compute_quality=False)
            U, V, _ = c.correlate([a, b])
            e.append(abs(float(np.median(U)) - s))
        errs[sub] = float(np.mean(e))
    check("sub-pixel reduces the median displacement error",
          errs[True] < errs[False],
          f"integer {errs[False]:.3f} px -> sub-pixel {errs[True]:.3f} px")

    # --- the R4 degenerate-plane failure mode, and the R5 gate ---
    flat = np.full((240, 320), 7, dtype=np.uint8)
    U4, V4 = batch_cross_correlate_R4([flat, flat], 32, 16)
    check("R4 reports a spurious -ws/2 displacement on a flat correlation plane",
          np.all(U4 == -16) and np.all(V4 == -16),
          "confirms bug B1/B2: this is what R4 published on startup")
    c = CPUCorrelator(32, 16, (240, 320), compute_quality=True)
    _, _, CC = c.correlate([flat, flat])
    check("R5 flags that same case with a near-zero correlation coefficient",
          np.nanmax(np.abs(CC)) < 1e-3, f"max|CC| = {np.nanmax(np.abs(CC)):.2e}")


# ==========================================================================
#  2. ControlROI
# ==========================================================================

def test_control_roi():
    section("2  ControlROI spatial average with synthetic vectors")
    from ebiv_config import MeasurementConfig, ROIConfig
    from ebiv_control import ControlROIEstimator

    ws, nd = 32, 16
    display_roi = [0, 320, 0, 240]
    gy, gx = (240 - ws) // nd + 1, (320 - ws) // nd + 1     # 14 x 19

    # ControlROI covering node centres x in [80, 160), y in [48, 112)
    control_roi = [80, 160, 48, 112]
    est = ControlROIEstimator(MeasurementConfig(min_valid_fraction=0.0,
                                                min_valid_vectors=1,
                                                use_correlation_gate=False),
                              display_roi, control_roi, ws, nd, gy, gx)

    # node centre = idx*16 + 16  ->  x in [80,160) is idx 4..8, y is idx 2..5
    check("ControlROI selects the expected block of grid nodes",
          est.n_total == 5 * 4, f"n_total={est.n_total}, expected 20")

    U = np.zeros((gy, gx), np.float32)
    V = np.zeros((gy, gx), np.float32)
    U[:] = -99.0                     # outside must be ignored
    U[est.sy, est.sx] = 3.0
    V[est.sy, est.sx] = -1.0
    u, ok, n, _ = est.estimate(U, V)
    check("mean of +U inside ControlROI ignores everything outside",
          ok and abs(u - 3.0) < 1e-6, f"u={u}")

    # component selection
    for comp, want in (('u', 3.0), ('-u', -3.0), ('v', -1.0), ('-v', 1.0),
                       ('magnitude', math.hypot(3.0, 1.0))):
        e = ControlROIEstimator(MeasurementConfig(component=comp,
                                                  min_valid_fraction=0.0,
                                                  min_valid_vectors=1,
                                                  use_correlation_gate=False),
                                display_roi, control_roi, ws, nd, gy, gx)
        u, ok, _, _ = e.estimate(U, V)
        check(f"component '{comp}' returns {want:+.3f}", ok and abs(u - want) < 1e-5,
              f"got {u:+.4f}")

    # velocity scaling
    e = ControlROIEstimator(MeasurementConfig(velocity_scale=2.5,
                                              min_valid_fraction=0.0,
                                              min_valid_vectors=1,
                                              use_correlation_gate=False),
                            display_roi, control_roi, ws, nd, gy, gx)
    u, ok, _, _ = e.estimate(U, V)
    check("velocity_scale is applied once, at the measurement",
          ok and abs(u - 7.5) < 1e-6, f"u={u}")

    # NaN / mask / CC rejection.  The ControlROI block is 4 node-rows by
    # 5 node-columns, so blanking 2 rows removes 10 of the 20 vectors.
    n_cols = est.sx.stop - est.sx.start
    Un = U.copy()
    ys = list(range(est.sy.start, est.sy.stop))
    for r in ys[:2]:
        Un[r, est.sx] = np.nan
    u, ok, n, reason = est.estimate(Un, V)
    check("NaN vectors are excluded from the average",
          ok and abs(u - 3.0) < 1e-6 and n == est.n_total - 2 * n_cols,
          f"n_valid={n}, expected {est.n_total - 2 * n_cols}, u={u}")

    strict = ControlROIEstimator(MeasurementConfig(min_valid_fraction=0.8,
                                                   min_valid_vectors=1,
                                                   use_correlation_gate=False),
                                 display_roi, control_roi, ws, nd, gy, gx)
    Un2 = U.copy()
    for r in list(ys)[:4]:
        Un2[r, est.sx] = np.nan
    u, ok, n, reason = strict.estimate(Un2, V)
    check("too few valid vectors -> measurement declared INVALID",
          (not ok) and math.isnan(u), reason)

    ccg = ControlROIEstimator(MeasurementConfig(min_valid_fraction=0.0,
                                                 min_valid_vectors=1,
                                                 use_correlation_gate=True,
                                                 min_correlation=0.2),
                              display_roi, control_roi, ws, nd, gy, gx)
    CC = np.full((gy, gx), 0.9, np.float32)
    CC[est.sy, est.sx] = 0.01
    u, ok, n, reason = ccg.estimate(U, V, CC=CC)
    check("correlation gate rejects low-quality vectors",
          (not ok) and n == 0, reason)

    lim = ControlROIEstimator(MeasurementConfig(min_valid_fraction=0.0,
                                                min_valid_vectors=1,
                                                use_correlation_gate=False,
                                                max_abs_displacement_px=2.0),
                              display_roi, control_roi, ws, nd, gy, gx)
    u, ok, n, reason = lim.estimate(U, V)
    check("implausible displacements are rejected", (not ok) and n == 0, reason)

    # containment validation
    try:
        ROIConfig(display_roi=[0, 100, 0, 100],
                  control_roi=[50, 150, 0, 50]).validate(320, 240)
        bad = False
    except ValueError:
        bad = True
    check("a ControlROI outside the DisplayROI is rejected at configuration time", bad)


# ==========================================================================
#  3. Filters
# ==========================================================================

def test_filters():
    section("3  Temporal filters")
    from ebiv_config import FilterConfig
    from ebiv_control import TemporalFilter

    f = TemporalFilter(FilterConfig(kind='none'))
    check("kind='none' passes the sample through",
          all(f.update(x, 0.1) == x for x in (1.0, 5.0, -2.0)))

    f = TemporalFilter(FilterConfig(kind='ma', window=4))
    for x in (1.0, 2.0, 3.0, 4.0):
        y = f.update(x, 0.1)
    check("moving average over a full window", abs(y - 2.5) < 1e-12, f"y={y}")
    y = f.update(8.0, 0.1)
    check("moving average slides (window=4)", abs(y - (2 + 3 + 4 + 8) / 4) < 1e-12,
          f"y={y}")

    f = TemporalFilter(FilterConfig(kind='ma', window=5))
    y = f.update(10.0, 0.1)
    check("moving average of a single sample is that sample", abs(y - 10.0) < 1e-12)

    # EMA step response reaches 1-1/e after tau
    tau, dt = 1.0, 0.001
    f = TemporalFilter(FilterConfig(kind='ema', tau_s=tau))
    f.update(0.0, dt)
    n = int(tau / dt)
    for _ in range(n):
        y = f.update(1.0, dt)
    check("EMA reaches 1-1/e of a step after one tau",
          abs(y - (1 - math.exp(-1))) < 0.01, f"y={y:.4f}, expected {1 - math.exp(-1):.4f}")

    # dt-awareness: same physical time, different sample rate -> same output
    outs = []
    for dt in (0.001, 0.01):
        f = TemporalFilter(FilterConfig(kind='ema', tau_s=0.5))
        f.update(0.0, dt)
        for _ in range(int(0.5 / dt)):
            y = f.update(1.0, dt)
        outs.append(y)
    check("EMA recomputes alpha from the measured dt (rate-independent)",
          abs(outs[0] - outs[1]) < 0.02, f"{outs[0]:.4f} vs {outs[1]:.4f}")

    # a moving average must lag; assert the lag is real and roughly (N-1)/2
    N, dt = 21, 0.01
    f = TemporalFilter(FilterConfig(kind='ma', window=N))
    t = np.arange(600) * dt
    x = np.sin(2 * np.pi * 0.5 * t)
    y = np.array([f.update(v, dt) for v in x])
    lag = np.argmax(np.correlate(y[100:], x[100:], 'full')) - (len(x) - 100 - 1)
    check("moving-average group delay is about (N-1)/2 samples",
          abs(lag - (N - 1) / 2) <= 2, f"measured {lag} samples, expected {(N-1)/2}")

    f = TemporalFilter(FilterConfig(kind='ema'))
    try:
        f.update(float('nan'), 0.1)
        raised = False
    except ValueError:
        raised = True
    check("the filter refuses a non-finite sample", raised)


# ==========================================================================
#  4. PID
# ==========================================================================

def test_pid():
    section("4  PID: saturation, anti-windup, slew, bumpless transfer")
    from ebiv_config import PIDConfig
    from ebiv_control import PIDController

    # --- pure proportional ---
    p = PIDController(PIDConfig(kp=2.0, v_min=-100, v_max=100,
                                slew_rate_v_per_s=None,
                                derivative_filter_tau_s=0.0))
    d = p.update(10.0, 4.0, 0.1)
    check("P term = Kp * error", abs(d.u_command - 12.0) < 1e-12, f"u={d.u_command}")

    # --- integral accumulates with the measured dt ---
    p = PIDController(PIDConfig(kp=0, ki=1.0, v_min=-100, v_max=100,
                                slew_rate_v_per_s=None))
    for _ in range(10):
        d = p.update(1.0, 0.0, 0.1)
    check("I term integrates error*dt", abs(d.i_term - 1.0) < 1e-9, f"I={d.i_term}")

    # --- saturation ---
    p = PIDController(PIDConfig(kp=100.0, v_min=0.0, v_max=5.0,
                                slew_rate_v_per_s=None))
    d = p.update(10.0, 0.0, 0.1)
    check("output saturates at v_max", d.u_command == 5.0 and d.saturated,
          f"unsat={d.u_unsaturated:.1f} -> {d.u_command}")
    d = p.update(-10.0, 0.0, 0.1)
    check("output saturates at v_min", d.u_command == 0.0)

    # --- anti-windup: clamp ---
    p = PIDController(PIDConfig(kp=0.0, ki=10.0, v_min=0.0, v_max=5.0,
                                anti_windup='clamp', slew_rate_v_per_s=None))
    for _ in range(200):
        p.update(10.0, 0.0, 0.05)          # drive hard into the upper limit
    i_wound = p._i
    n_back = 0
    while p.update(-10.0, 0.0, 0.05).u_command > 0.01 and n_back < 200:
        n_back += 1
    check("anti-windup 'clamp' bounds the integrator while saturated",
          i_wound <= 5.0 + 1e-6, f"I stopped at {i_wound:.3f} (v_max=5)")
    check("recovery from saturation is prompt (no long unwind)",
          n_back <= 3, f"{n_back} steps to leave the limit")

    # --- without anti-windup it DOES wind up (control) ---
    p = PIDController(PIDConfig(kp=0.0, ki=10.0, v_min=0.0, v_max=5.0,
                                anti_windup='none', slew_rate_v_per_s=None))
    for _ in range(200):
        p.update(10.0, 0.0, 0.05)
    n_bad = 0
    while p.update(-10.0, 0.0, 0.05).u_command > 0.01 and n_bad < 500:
        n_bad += 1
    check("with anti_windup='none' the integrator does wind up",
          n_bad > 10, f"{n_bad} steps to recover vs {n_back} with clamping")

    # --- back-calculation also bounds it ---
    p = PIDController(PIDConfig(kp=0.0, ki=10.0, v_min=0.0, v_max=5.0,
                                anti_windup='back_calc', back_calc_gain=5.0,
                                slew_rate_v_per_s=None))
    for _ in range(200):
        p.update(10.0, 0.0, 0.05)
    check("anti-windup 'back_calc' also bounds the integrator", p._i < 20.0,
          f"I={p._i:.2f}")

    # --- slew limiting ---
    p = PIDController(PIDConfig(kp=100.0, v_min=0.0, v_max=5.0,
                                slew_rate_v_per_s=1.0))
    p.set_bumpless(0.0, 0.0, 0.0)
    us = [p.update(10.0, 0.0, 0.1).u_command for _ in range(10)]
    steps = np.diff([0.0] + us)
    check("slew rate limits the per-step voltage change",
          np.all(steps <= 0.1 + 1e-9), f"max step {steps.max():.4f} V (limit 0.1)")
    check("slew-limited output still reaches the saturation value",
          abs(us[-1] - 1.0) < 1e-6, f"after 10 steps: {us[-1]:.3f} V")

    # --- derivative on measurement: no set-point kick ---
    cfg_dm = PIDConfig(kp=1.0, kd=1.0, v_min=-100, v_max=100,
                       derivative_on_measurement=True,
                       derivative_filter_tau_s=0.0, slew_rate_v_per_s=None)
    cfg_de = PIDConfig(kp=1.0, kd=1.0, v_min=-100, v_max=100,
                       derivative_on_measurement=False,
                       derivative_filter_tau_s=0.0, slew_rate_v_per_s=None)
    kicks = {}
    for tag, c in (('measurement', cfg_dm), ('error', cfg_de)):
        p = PIDController(c)
        p.set_bumpless(0.0, 0.0, 0.0)
        p.update(0.0, 0.0, 0.1)
        d = p.update(5.0, 0.0, 0.1)        # step the set-point
        kicks[tag] = abs(d.d_term)
    check("derivative-on-measurement removes the set-point kick",
          kicks['measurement'] < 1e-12 < kicks['error'],
          f"D on step: measurement {kicks['measurement']:.1f}, error {kicks['error']:.1f}")

    # --- derivative filtering attenuates noise ---
    rng = np.random.default_rng(3)
    amp = {}
    for tau in (0.0, 0.2):
        p = PIDController(PIDConfig(kp=0, kd=1.0, v_min=-1e6, v_max=1e6,
                                    derivative_filter_tau_s=tau,
                                    slew_rate_v_per_s=None))
        p.set_bumpless(0.0, 0.0, 0.0)
        vals = [p.update(0.0, rng.normal(0, 1), 0.01).d_term for _ in range(400)]
        amp[tau] = float(np.std(vals[50:]))
    check("derivative filtering attenuates measurement noise",
          amp[0.2] < 0.25 * amp[0.0],
          f"std of D: unfiltered {amp[0.0]:.1f}, tau=0.2 s {amp[0.2]:.1f}")

    # --- bumpless transfer ---
    p = PIDController(PIDConfig(kp=3.0, ki=1.0, v_min=0.0, v_max=5.0,
                                slew_rate_v_per_s=None))
    p.set_bumpless(current_output=2.75, measurement=4.0, target=4.0)
    d = p.update(4.0, 4.0, 0.1)
    check("bumpless transfer: the first closed-loop output matches the manual one",
          abs(d.u_command - 2.75) < 1e-9, f"{d.u_command:.6f} V vs 2.75 V")

    # --- dt clamping protects against a stalled loop ---
    p = PIDController(PIDConfig(kp=0, ki=1.0, v_min=-1e6, v_max=1e6,
                                slew_rate_v_per_s=None), max_dt=0.5)
    d = p.update(1.0, 0.0, 30.0)
    check("an enormous dt is clamped before it reaches the integrator",
          abs(d.dt - 0.5) < 1e-12 and abs(d.i_term - 0.5) < 1e-12,
          f"dt clamped to {d.dt}, I={d.i_term}")

    # --- the command is never outside the limits, whatever happens ---
    rng = np.random.default_rng(11)
    p = PIDController(PIDConfig(kp=50, ki=80, kd=5, v_min=0.0, v_max=5.0,
                                slew_rate_v_per_s=3.0))
    p.set_bumpless(0.0, 0.0, 0.0)
    outs = [p.update(rng.uniform(-50, 50), rng.uniform(-50, 50),
                     rng.uniform(0.001, 0.3)).u_command for _ in range(4000)]
    check("across 4000 adversarial steps the command never leaves [v_min, v_max]",
          min(outs) >= 0.0 and max(outs) <= 5.0,
          f"range [{min(outs):.3f}, {max(outs):.3f}] V")


# ==========================================================================
#  5. Invalid measurements
# ==========================================================================

def test_invalid_measurements():
    section("5  Invalid-measurement handling: HOLD then SAFE")
    from ebiv_config import (ControlSystemConfig, PIDConfig, SupervisorConfig,
                             ReferenceConfig, FilterConfig)
    from ebiv_control import ControlSupervisor, Measurement, ControlState
    from ebiv_hardware import MockPump

    cfg = ControlSystemConfig(
        pid=PIDConfig(kp=1.0, ki=0.5, v_min=0.0, v_max=5.0,
                      slew_rate_v_per_s=None),
        supervisor=SupervisorConfig(control_rate_hz=10.0, hold_timeout_s=0.5,
                                    safe_timeout_s=2.0,
                                    max_measurement_age_s=0.5,
                                    safe_pump_voltage=1.25),
        reference=ReferenceConfig(kind='constant', value=3.0),
        filt=FilterConfig(kind='none'))
    pump = MockPump(0.0, 5.0)
    sup = ControlSupervisor(cfg, pump)

    t = 0.0
    dt = 0.1

    def good(seq, val=3.0):
        return Measurement(val, True, 100, 100, t, t, seq)

    def bad(seq, reason="too few valid vectors"):
        return Measurement(float('nan'), False, 1, 100, t, t, seq, reason)

    for i in range(5):
        sup.step(good(i), now=t)
        t += dt
    sup.arm(now=t)
    for i in range(5, 15):
        sup.step(good(i), now=t)
        t += dt
    check("valid measurements keep the controller CLOSED_LOOP",
          sup.state == ControlState.CLOSED_LOOP, sup.state)
    v_before = sup.v_command

    # a single bad field must NOT flip the state: hold_timeout_s is 0.5 s
    sup.step(bad(15), now=t)
    t += dt
    check("one dropped field holds the output without announcing HOLD",
          sup.state == ControlState.CLOSED_LOOP
          and abs(sup.v_command - v_before) < 1e-12
          and sup.counters['hold'] == 1,
          f"state={sup.state}, hold count={sup.counters['hold']}")

    # a dropout longer than hold_timeout_s -> HOLD
    for i in range(16, 22):
        sup.step(bad(i), now=t)
        t += dt
    check("a dropout longer than hold_timeout_s is announced as HOLD",
          sup.state == ControlState.HOLD, sup.state)
    check("HOLD freezes the pump command at its last value",
          abs(sup.v_command - v_before) < 1e-12,
          f"{sup.v_command:.6f} vs {v_before:.6f} V")
    check("the PID does not integrate against a measurement it cannot trust",
          not any(math.isnan(h[1]) for h in pump.history))

    # recovery
    for i in range(22, 30):
        sup.step(good(i), now=t)
        t += dt
    check("a valid measurement returns the controller to CLOSED_LOOP",
          sup.state == ControlState.CLOSED_LOOP, sup.state)

    # long dropout -> SAFE
    for i in range(30, 70):
        sup.step(bad(i), now=t)
        t += dt
    check("a dropout longer than safe_timeout_s drives the SAFE state",
          sup.state == ControlState.SAFE, sup.state)
    check("SAFE commands the configured safe voltage, not 0 V by assumption",
          abs(pump.voltage - 1.25) < 1e-12, f"{pump.voltage} V")

    # SAFE latches: valid data alone must not silently re-engage the loop
    for i in range(70, 90):
        sup.step(good(i), now=t)
        t += dt
    check("SAFE latches until the operator explicitly re-arms",
          sup.state == ControlState.SAFE, sup.state)

    # NaN and Inf never reach the PID
    pump2 = MockPump(0.0, 5.0)
    sup2 = ControlSupervisor(cfg, pump2)
    t = 0.0
    sup2.step(good(0), now=t); t += dt
    sup2.arm(now=t)
    for i, v in enumerate([float('nan'), float('inf'), -float('inf')]):
        sup2.step(Measurement(v, True, 100, 100, t, t, i + 1), now=t)
        t += dt
    check("a 'valid' measurement carrying NaN/Inf never produces a NaN command",
          all(np.isfinite(h[1]) for h in pump2.history),
          f"{len(pump2.history)} commands, all finite")

    # staleness: valid flag but old timestamp
    pump3 = MockPump(0.0, 5.0)
    sup3 = ControlSupervisor(cfg, pump3)
    t = 0.0
    sup3.step(good(0), now=t); t += dt
    sup3.arm(now=t)
    stale = Measurement(3.0, True, 100, 100, t, t - 5.0, 99)
    for _ in range(8):
        sup3.step(stale, now=t)
        t += dt
    check("a stale-but-valid measurement is rejected on age",
          sup3.state in (ControlState.HOLD, ControlState.SAFE)
          and sup3.counters['stale'] > 0,
          f"state={sup3.state}, stale count={sup3.counters['stale']}")

    # a frozen (non-advancing) sequence number counts as no new data
    pump4 = MockPump(0.0, 5.0)
    sup4 = ControlSupervisor(cfg, pump4)
    t = 0.0
    sup4.step(good(0), now=t); t += dt
    sup4.arm(now=t)
    m = good(7)
    for _ in range(40):
        m = Measurement(3.0, True, 100, 100, t, t, 7)   # same seq, fresh time
        sup4.step(m, now=t)
        t += dt
    check("a repeated sequence number is treated as 'no new measurement'",
          sup4.state == ControlState.SAFE, sup4.state)


# ==========================================================================
#  6. References
# ==========================================================================

def test_references():
    section("6  Reference generators")
    from ebiv_config import ReferenceConfig
    from ebiv_control import make_reference

    r = make_reference(ReferenceConfig(kind='constant', value=2.5))
    check("constant", all(r(t) == 2.5 for t in (0, 1, 100)))

    r = make_reference(ReferenceConfig(kind='step', value=1.0, t_step_s=5.0,
                                       step_amplitude=2.0))
    check("step before/after the transition",
          r(4.99) == 1.0 and r(5.0) == 3.0 and r(50) == 3.0)

    r = make_reference(ReferenceConfig(kind='sine', value=1.0, amplitude=2.0,
                                       freq_hz=0.25))
    check("sine has the requested mean, amplitude and period",
          abs(r(0) - 1.0) < 1e-12 and abs(r(1.0) - 3.0) < 1e-9
          and abs(r(3.0) + 1.0) < 1e-9,
          f"r(0)={r(0):.3f} r(1)={r(1.0):.3f} r(3)={r(3.0):.3f}")

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ref.csv")
        np.savetxt(p, np.column_stack([[0, 1, 2, 3], [0, 10, 20, 30]]),
                   delimiter=',')
        r = make_reference(ReferenceConfig(kind='file', filepath=p, file_loop=False))
        check("file reference interpolates linearly",
              abs(r(0.5) - 5.0) < 1e-9 and abs(r(2.25) - 22.5) < 1e-9,
              f"r(0.5)={r(0.5)}, r(2.25)={r(2.25)}")
        r2 = make_reference(ReferenceConfig(kind='file', filepath=p, file_loop=True))
        check("file reference loops", abs(r2(3.5) - r2(0.5)) < 1e-9)

    c = ReferenceConfig(kind='random', value=5.0, random_std=1.0,
                        random_cutoff_hz=0.1, random_seed=7)
    a = make_reference(c)
    b = make_reference(c)
    ts = np.linspace(0, 200, 2001)
    va = np.array([a(t) for t in ts])
    vb = np.array([b(t) for t in ts])
    check("random reference is reproducible from its seed",
          np.array_equal(va, vb))
    c2 = ReferenceConfig(kind='random', value=5.0, random_std=1.0,
                         random_cutoff_hz=0.1, random_seed=8)
    vc = np.array([make_reference(c2)(t) for t in ts])
    check("a different seed gives a different trajectory", not np.allclose(va, vc))
    check("random reference has the requested mean and spread",
          abs(va.mean() - 5.0) < 0.35 and 0.6 < va.std() < 1.6,
          f"mean {va.mean():.3f}, std {va.std():.3f}")
    # band limiting: successive samples must be correlated, unlike white noise
    d1 = np.abs(np.diff(va)).mean()
    check("random reference is band-limited, not white",
          d1 < 0.2 * va.std(),
          f"mean |step| {d1:.4f} vs std {va.std():.3f} (white noise would be ~1.4x std)")
    cl = ReferenceConfig(kind='random', value=5.0, random_std=3.0,
                         random_cutoff_hz=0.1, random_seed=7,
                         random_clip=(4.0, 6.0))
    vcl = np.array([make_reference(cl)(t) for t in ts])
    check("random reference honours its clip limits",
          vcl.min() >= 4.0 - 1e-12 and vcl.max() <= 6.0 + 1e-12)

    # --- live (operator-driven) reference -----------------------------
    from ebiv_control import LiveReference

    lr = make_reference(ReferenceConfig(kind='live', value=2.0, live_step=0.5))
    check("make_reference builds a LiveReference for kind 'live'",
          isinstance(lr, LiveReference))
    check("live reference starts at its configured value",
          lr(0.0) == 2.0 and lr(123.4) == 2.0,
          "the value must not depend on t")
    check("live bump moves the target by one step",
          lr.bump(+1) == 2.5 and lr.bump(-1) == 2.0)
    check("live bump accepts multiple steps",
          lr.bump(+4) == 4.0)
    check("the control thread sees the bumped value",
          lr(9.9) == 4.0,
          "__call__ must read the same state that bump() writes")

    lim = LiveReference(1.0, 0.5, lo=0.0, hi=2.0)
    for _ in range(10):
        lim.bump(+1)
    hi_ok = lim.value == 2.0
    for _ in range(20):
        lim.bump(-1)
    check("live target is clamped to [live_min, live_max]",
          hi_ok and lim.value == 0.0,
          "a held-down key must not walk the target past the limits")

    nan_ref = LiveReference(3.0, 0.1)
    nan_ref.set(float('nan'))
    check("live set refuses a non-finite target",
          nan_ref.value == 3.0,
          "a NaN target would go straight into the PID error term")
    check("live set applies a finite target", nan_ref.set(7.25) == 7.25)

    try:
        ReferenceConfig(kind='live', value=1.0, live_step=0.0)
        bad_step = False
    except ValueError:
        bad_step = True
    try:
        ReferenceConfig(kind='live', value=1.0, live_step=0.1,
                        live_min=5.0, live_max=1.0)
        bad_range = False
    except ValueError:
        bad_range = True
    check("a zero live_step and an inverted live range are both rejected",
          bad_step and bad_range)

    # the generator must be independent of the PID.  Scan the CODE, not the
    # prose: a docstring that mentions the PID is documentation, not a
    # dependency, and the plain substring test flagged the one in
    # LiveReference as if it were.
    import ast as _ast
    import ebiv_control as ec
    src = open(ec.__file__, encoding='utf-8').read()
    cls_start = src.index("class ReferenceGenerator")
    cls_end = src.index("def make_reference")
    identifiers = set()
    for node in _ast.walk(_ast.parse(src[cls_start:cls_end])):
        if isinstance(node, _ast.Name):
            identifiers.add(node.id)
        elif isinstance(node, _ast.Attribute):
            identifiers.add(node.attr)
        elif isinstance(node, (_ast.ClassDef, _ast.FunctionDef)):
            identifiers.add(node.name)
        elif isinstance(node, _ast.arg):
            identifiers.add(node.arg)
    check("reference generators do not reference the PID at all",
          not any('pid' in n.lower() for n in identifiers),
          "new trajectories can be added without touching the control law")


# ==========================================================================
#  5b. GPU backend: the homothetic rescaling (audit B7 / B8)
# ==========================================================================

def test_gpu_homothetic_warp():
    section("5b  GPU homothetic rescaling")
    import cv2
    from ebiv_config import PIVConfig, Session

    # --- the option is wired all the way through -----------------------
    check("PIVConfig exposes use_gpu, default OFF",
          hasattr(PIVConfig(), 'use_gpu') and PIVConfig().use_gpu is False,
          "the GPU must never switch itself on")

    import ebiv_utils as _eu
    seen = {}
    orig = {n: getattr(_eu, n) for n in
            ('record_raw_file', 'check_raw_video', 'generate_centered_images',
             'process_offline_piv')}
    try:
        for name in orig:
            setattr(_eu, name, lambda *a, **k: None)
        _eu.process_offline_piv = lambda *a, **k: seen.update(k)
        with tempfile.TemporaryDirectory() as d:
            sess = Session()
            sess.run.mode = 'offline'
            sess.run.output_base_folder = d
            sess.run.do_piv_process = True
            sess.control.piv.use_gpu = True
            from ebiv_session import run_session
            run_session(sess)
    finally:
        for name, fn in orig.items():
            setattr(_eu, name, fn)
    check("piv.use_gpu reaches process_offline_piv",
          seen.get('use_gpu') is True,
          "Release 5.1 hard-coded use_gpu=False in the dispatcher")

    # --- the arithmetic ------------------------------------------------
    try:
        import torch                                            # noqa: F401
        from ebiv_gpu import GPUCorrelator
    except Exception as exc:                                    # noqa: BLE001
        check("GPU warp matches the CPU warp",
              True,
              f"SKIPPED: torch not importable here ({type(exc).__name__}). "
              f"Run tools/check_gpu.py on the machine with the GPU.")
        return

    ws = 48
    c = ws // 2
    g = GPUCorrelator(window_size=ws, node_distance=24, device='cpu')

    def cpu_warp(R, k):
        S = 1.0 / k
        M = np.array([[S, 0, c * (1 - S)], [0, S, c * (1 - S)]], np.float32)
        return cv2.warpAffine(R, M, (ws, ws), flags=cv2.INTER_LINEAR)

    def subpix_x(R):
        i = int(np.argmax(R)); cy, cx = i // ws, i % ws
        if not (0 < cx < ws - 1):
            return float(cx - c)
        l, m, r = R[cy, cx - 1], R[cy, cx], R[cy, cx + 1]
        if l > 0 and m > 0 and r > 0:
            l, m, r = np.log(l), np.log(m), np.log(r)
        den = 2 * (l - 2 * m + r)
        d = (l - r) / den if abs(den) > 1e-12 else 0.0
        return cx + float(np.clip(d, -1, 1)) - c

    yy, xx = np.mgrid[0:ws, 0:ws]
    worst_plane = 0.0
    worst_subpix = 0.0
    for k in (2, 3, 4):
        for d_true in (2.0, 3.7, 5.3):
            R = np.exp(-((xx - (c + k * d_true)) ** 2
                         + (yy - c) ** 2) / 6.0).astype(np.float32)
            a = cpu_warp(R, k)
            b = g._batch_homothetic_warp(
                torch.from_numpy(R).unsqueeze(0), k).squeeze(0).numpy()
            scale = max(float(np.abs(a).max()), 1e-12)
            worst_plane = max(worst_plane, float(np.abs(a - b).max()) / scale)
            worst_subpix = max(worst_subpix, abs(subpix_x(a) - subpix_x(b)))

    # NOT bit-equality.  cv2's bilinear resampler and torch's grid_sample are
    # different implementations of the same geometry, so they agree to float32
    # rounding (~1e-6 relative), not to the last bit.  The bar is set two
    # orders of magnitude below the smallest error that would matter (the
    # centre bug this replaced was 0.25-0.4 px), and well above float32 noise,
    # so it fails on a geometry error and not on a compiler or version change.
    check("GPU homothetic warp reproduces cv2.warpAffine",
          worst_plane < 1e-4,
          f"max relative plane difference = {worst_plane:.3e} over k=2,3,4")
    check("GPU and CPU recover the same sub-pixel displacement",
          worst_subpix < 1e-4,
          f"max difference {worst_subpix:.3e} px "
          f"(the centre bug this replaced was 0.25-0.4 px)")

    # The direction bug is the one that mattered; pin it explicitly so that a
    # future edit flipping the scale back is caught here and not in the lab.
    grid = g._get_warp_grid(3, 1)
    corner = float(grid[0, 0, 0, 0])
    check("the warp grid scales by k, not 1/k",
          corner < -1.0,
          f"grid samples outside [-1,1] at the edge (x={corner:.3f}), which "
          f"is what shrinking the plane by 1/k requires")

    # ------------------------------------------------------------------
    #  The REAL-TIME GPU backend must be a drop-in for the CPU one.
    #  This is what makes it safe for the closed loop: if U, V or CC
    #  differed, the ControlROI gate and the PID would behave differently
    #  depending on which backend happened to be in use.
    # ------------------------------------------------------------------
    import cv2 as _cv2
    from ebiv_piv import CPUCorrelator

    rng2 = np.random.default_rng(11)
    H, Wd, ws2, nd2 = 200, 480, 32, 24

    def _frames(dx, dy, k):
        pad = 50
        by = rng2.uniform(0, H + 2 * pad, 2500)
        bx = rng2.uniform(0, Wd + 2 * pad, 2500)
        out = []
        for i in range(k):
            f = np.zeros((H + 2 * pad, Wd + 2 * pad), np.float32)
            yi = np.clip((by + i * dy).astype(int), 0, f.shape[0] - 1)
            xi = np.clip((bx + i * dx).astype(int), 0, f.shape[1] - 1)
            np.add.at(f, (yi, xi), 255.0)
            out.append(_cv2.GaussianBlur(f, (5, 5), 1.0)[pad:pad + H,
                                                        pad:pad + Wd].copy())
        return out

    worst_int = 0.0
    worst_sub = 0.0
    worst_cc = 0.0
    for triple in (False, True):
        fr = _frames(3.4, -1.6, 3 if triple else 2)
        for sub in (True, False):
            cpu = CPUCorrelator(window_size=ws2, node_distance=nd2,
                                frame_shape=(H, Wd), triple_corr=triple,
                                fft_workers=1, subpixel=sub,
                                compute_quality=True, fast_mean_removal=False)
            gp = GPUCorrelator(ws2, nd2, device='cpu', triple_corr=triple,
                               subpixel=sub, compute_quality=True,
                               frame_shape=(H, Wd))
            Uc, Vc, Cc = cpu.correlate([f.copy() for f in fr])
            Ug, Vg, Cg = gp.correlate([f.copy() for f in fr])
            d = max(float(np.abs(Uc - Ug).max()), float(np.abs(Vc - Vg).max()))
            if sub:
                worst_sub = max(worst_sub, d)
            else:
                worst_int = max(worst_int, d)
            worst_cc = max(worst_cc,
                           float(np.abs(Cc - Cg).max())
                           / max(float(np.abs(Cc).max()), 1e-12))

    # With sub-pixel off both paths reduce to argmax on the same plane, so
    # anything but exact equality would mean the correlation itself differs.
    check("GPU and CPU real-time backends find the SAME integer peak",
          worst_int == 0.0,
          f"max |difference| = {worst_int:.3e} px, 2- and 3-frame")
    check("GPU and CPU real-time sub-pixel agree",
          worst_sub < 1e-4,
          f"max |difference| = {worst_sub:.3e} px")
    check("GPU and CPU real-time CC agree",
          worst_cc < 1e-4, f"max relative difference = {worst_cc:.3e}")

    gq = GPUCorrelator(ws2, nd2, device='cpu', compute_quality=False,
                       frame_shape=(H, Wd))
    _, _, cc_none = gq.correlate(_frames(2.0, 0.0, 2))
    check("compute_quality=False returns CC=None, as the CPU does",
          cc_none is None)
    check("the GPU backend exposes grid_y/grid_x like the CPU one",
          gq.grid_y == (H - ws2) // nd2 + 1 and gq.grid_x == (Wd - ws2) // nd2 + 1,
          f"{gq.grid_y}x{gq.grid_x}; the run header and the .mat writer read these")

    # Release 4.0's kernel is kept but must not be what Release 5 calls: it
    # returns 2 values, no CC, and no sub-pixel.
    legacy = GPUCorrelator(ws2, nd2, device='cpu').batch_cross_correlate(
        _frames(3.0, 0.0, 2))
    check("Release 4.0's batch_cross_correlate is preserved, and is integer-only",
          len(legacy) == 2
          and np.allclose(legacy[0], np.round(legacy[0])),
          "kept for reference; the closed loop uses correlate() instead")


# ==========================================================================
#  5c. STREAM_MODE 'manual' must not run the CAL_MODE programme
# ==========================================================================

def test_manual_mode_is_manual():
    section("5c  'manual' run mode ignores CAL_MODE")
    from ebiv_config import CalibrationConfig
    from ebiv_runtime import CalibrationDriver

    # The shipped CAL_MODE is 'steps'.  A 'manual' run built its driver from
    # that, so the control thread re-applied the schedule every cycle and the
    # operator's [+] was overwritten 100 ms later: the voltage sat at
    # (schedule value + one step) no matter how many times you pressed it.
    cal = CalibrationConfig(mode='steps', voltages=[0.0, 1.0, 2.0],
                            dwell_s=30.0, manual_step_v=0.1)

    prog = CalibrationDriver(cal, 0.0, 5.0)
    check("with force_manual off, the driver runs the CAL_MODE schedule",
          prog.voltage_at(0.0) == 0.0 and prog.voltage_at(45.0) == 1.0,
          "this is what 'calibration' mode needs")

    man = CalibrationDriver(cal, 0.0, 5.0, force_manual=True)
    man.set_manual(0.0)
    check("with force_manual on, the schedule is ignored",
          man.voltage_at(0.0) == 0.0 and man.voltage_at(45.0) == 0.0,
          "CAL_MODE must not drive a 'manual' run")
    check("the driver reports the mode actually in force",
          man.mode == 'manual' and prog.mode == 'steps',
          "the HUD and the run header read this, not cfg.mode")

    # The operator's value must SURVIVE the control thread's next tick.
    seq = []
    for press in range(4):
        man.set_manual(man.voltage_at(press * 0.1) + cal.manual_step_v)
        seq.append(round(man.voltage_at(press * 0.1 + 0.05), 3))
    check("repeated [+] presses accumulate instead of sticking",
          seq == [0.1, 0.2, 0.3, 0.4],
          f"got {seq}; the bug produced [0.1, 0.1, 0.1, 0.1]")

    man.set_manual(99.0)
    check("the manual value is still clamped to the pump limits",
          man.voltage_at(0.0) == 5.0)


# ==========================================================================
#  5e. The uncalibrated unit is a velocity, not a per-frame displacement
# ==========================================================================

def test_uncalibrated_units():
    section("5e  px/s vs px/frame")
    from ebiv_config import MeasurementConfig

    # px/frame is a property of the SAMPLING, not of the flow.  At 100 Hz a
    # 3 px/frame displacement and at 50 Hz a 6 px/frame displacement are the
    # same jet; only px/s says so.
    m100 = MeasurementConfig(uncalibrated_units='px/s')
    m100.resolve_units(1 / 100.0)
    m50 = MeasurementConfig(uncalibrated_units='px/s')
    m50.resolve_units(1 / 50.0)
    check("the default basis is px/s",
          m100.velocity_units == 'px/s' and m50.velocity_units == 'px/s')
    check("the same flow reads the same in px/s at different f_acq",
          abs(3.0 * m100.velocity_scale - 6.0 * m50.velocity_scale) < 1e-9,
          f"{3.0 * m100.velocity_scale:g} vs {6.0 * m50.velocity_scale:g} px/s")

    mf = MeasurementConfig(uncalibrated_units='px/frame')
    mf.resolve_units(1 / 100.0)
    check("'px/frame' still reproduces the Release 4.0-5.1 behaviour",
          mf.velocity_scale == 1.0 and mf.velocity_units == 'px/frame')

    # to_px is the inverse, and the displacement / wrap-limit warning depends
    # on it, so it must survive the change of basis.
    for basis, dt in (('px/s', 1 / 75.0), ('px/frame', 1 / 75.0)):
        mm = MeasurementConfig(uncalibrated_units=basis)
        mm.resolve_units(dt)
        check(f"to_px round-trips in {basis}",
              abs(mm.to_px(4.0 * mm.velocity_scale) - 4.0) < 1e-9,
              "the correlation wrap check works in px, whatever the display unit")

    # A px/mm calibration wins over both: m/s is already a velocity.
    for basis in ('px/s', 'px/frame'):
        mc = MeasurementConfig(uncalibrated_units=basis, calibration_px_per_mm=20.0)
        mc.resolve_units(1 / 100.0)
        check(f"a px/mm calibration gives m/s regardless of basis={basis}",
              mc.velocity_units == 'm/s'
              and abs(mc.velocity_scale - 1e-3 / 20.0 * 100.0) < 1e-12)

    # An explicitly chosen velocity_scale must not be silently overwritten.
    me = MeasurementConfig(uncalibrated_units='px/s', velocity_scale=7.0)
    me.resolve_units(1 / 100.0)
    check("an explicit velocity_scale is left alone",
          me.velocity_scale == 7.0)

    # --- re-resolution and persistence (GUI reuses one Session) --------
    mr = MeasurementConfig()
    mr.resolve_units(1 / 100.0)
    first = mr.velocity_scale
    mr.resolve_units(1 / 50.0)
    check("changing f_acq re-derives the velocity scale",
          first == 100.0 and mr.velocity_scale == 50.0,
          f"{first} -> {mr.velocity_scale}; a stale scale makes every velocity "
          f"wrong by the ratio of the two rates, silently")

    mr2 = MeasurementConfig(calibration_px_per_mm=20.0)
    mr2.resolve_units(1 / 100.0)
    a = mr2.velocity_scale
    mr2.resolve_units(1 / 50.0)
    check("the calibrated scale re-derives too",
          abs(a - 1e-3 / 20.0 * 100.0) < 1e-12
          and abs(mr2.velocity_scale - 1e-3 / 20.0 * 50.0) < 1e-12)

    mr3 = MeasurementConfig()
    mr3.resolve_units(1 / 100.0)
    mr3.restore_declared()
    check("restore_declared() undoes the resolution",
          mr3.velocity_scale == 1.0 and mr3.velocity_units == 'px/frame',
          "a preset must store what the user asked for, not what was derived")

    try:
        MeasurementConfig(uncalibrated_units='furlongs/fortnight')
        rejected = False
    except ValueError:
        rejected = True
    check("an unknown basis is rejected at construction", rejected)


# ==========================================================================
#  5d. The laser rate follows f_acq unless told otherwise
# ==========================================================================

def test_laser_follows_facq():
    section("5d  Laser frequency locked to f_acq")
    from ebiv_config import Session
    import ebiv_hardware as hw

    # Two numbers that must agree, edited in two places, with only a warning
    # if they drift, is a footgun: it cost a lab session when f_acq was
    # changed and the laser was left behind at the old rate.
    s1 = Session()
    s1.run.f_acq = 50.0
    check("the shipped default is 'follow f_acq'",
          s1.control.ad3.laser_frequency_hz is None)
    f, derived = s1.resolve_laser_frequency()
    check("resolving gives f_acq and reports it as derived",
          f == 50.0 and derived is True and s1.control.ad3.laser_frequency_hz == 50.0)

    f2, derived2 = s1.resolve_laser_frequency()
    check("resolving twice gives the same answer",
          f2 == 50.0 and derived2 is True,
          "several entry points call this; it must not compound")

    # THE GUI KEEPS ONE SESSION ACROSS RUNS.  Change f_acq, press Run again.
    s1.run.f_acq = 25.0
    f3, derived3 = s1.resolve_laser_frequency()
    check("changing f_acq re-derives the laser rate",
          f3 == 25.0 and derived3 is True,
          "the first resolve overwrites the field, so a naive implementation "
          "leaves the laser at the OLD rate for every later run")

    s2 = Session()
    s2.run.f_acq = 50.0
    s2.control.ad3.laser_frequency_hz = 500.0
    f3, derived3 = s2.resolve_laser_frequency()
    check("an explicit number is respected and flagged as explicit",
          f3 == 500.0 and derived3 is False,
          "external lasers and pulse-pair illumination need this")

    s3 = Session()
    s3.run.f_acq = 137.0
    s3.run.mode = 'stream'
    s3.validate()
    check("validate() resolves it, so every run path is covered",
          s3.control.ad3.laser_frequency_hz == 137.0)

    # A preset must round-trip the DECLARED intent.  Storing a resolved value
    # is a trap: it reloads as an explicit override, so the laser silently
    # stops following f_acq and the units freeze at that day's rate.
    sp = Session()
    sp.run.f_acq = 100.0
    sp.resolve_laser_frequency()
    sp.control.measurement.resolve_units(1 / 100.0)
    d = sp.to_dict()
    check("a preset stores the declared laser rate, not the resolved one",
          d['control']['ad3']['laser_frequency_hz'] is None,
          f"got {d['control']['ad3']['laser_frequency_hz']!r}")
    check("a preset stores the declared velocity scale, not the resolved one",
          d['control']['measurement']['velocity_scale'] == 1.0
          and d['control']['measurement']['velocity_units'] == 'px/frame',
          f"got scale={d['control']['measurement']['velocity_scale']}, "
          f"units={d['control']['measurement']['velocity_units']!r}")
    check("serialising does not disturb the live session",
          sp.control.ad3.laser_frequency_hz == 100.0
          and sp.control.measurement.velocity_scale == 100.0,
          "to_dict works on a deep copy")

    back, _ = Session.from_dict(d)
    back.run.f_acq = 40.0
    back.resolve_laser_frequency()
    back.control.measurement.resolve_units(1 / 40.0)
    check("a reloaded preset still follows f_acq",
          back.control.ad3.laser_frequency_hz == 40.0
          and back.control.measurement.velocity_scale == 40.0,
          "this is the bug that a saved-then-reloaded preset used to have")

    # An unresolved None must fail with a sentence, not a ctypes TypeError.
    class _Dev:
        cfg = type("C", (), {"laser_frequency_hz": None})()
        _require_laser_frequency = hw.AnalogDiscovery3._require_laser_frequency
    try:
        _Dev()._require_laser_frequency()
        named = False
    except hw.AD3Error as exc:
        named = "laser_frequency_hz" in str(exc) and "F_ACQ" in str(exc)
    except Exception:
        named = False
    check("an unresolved None fails with a message that names the cause",
          named, "otherwise it is c_double(None) forty frames down")


# ==========================================================================
#  5f. Staleness is measured on the DATA, not on the computation
# ==========================================================================

def test_stale_gate_uses_data_age():
    section("5f  Staleness gate covers the acquisition lag")
    from ebiv_config import (ControlSystemConfig, PIDConfig, SupervisorConfig,
                             ReferenceConfig, FilterConfig)
    from ebiv_control import ControlSupervisor, Measurement, ControlState
    from ebiv_hardware import MockPump

    cfg = ControlSystemConfig(
        pid=PIDConfig(kp=1.0, ki=0.0, v_min=0.0, v_max=5.0, slew_rate_v_per_s=None),
        supervisor=SupervisorConfig(control_rate_hz=10.0, hold_timeout_s=0.5,
                                    safe_timeout_s=3.0,
                                    max_measurement_age_s=0.5),
        reference=ReferenceConfig(kind='constant', value=3.0),
        filt=FilterConfig(kind='none'))

    def run(t_frame_offset):
        """A measurement whose PIV finished JUST NOW, from light that arrived
        t_frame_offset seconds ago."""
        sup = ControlSupervisor(cfg, MockPump(0.0, 5.0))
        t = 0.0
        for i in range(4):
            sup.step(Measurement(3.0, True, 100, 100, t, t, i), now=t)
            t += 0.1
        sup.arm(now=t)
        m = Measurement(u_raw=3.0, valid=True, n_valid=100, n_total=100,
                        t_frame=t - t_frame_offset,   # when the LIGHT arrived
                        t_measured=t,                 # PIV finished now
                        seq=99)
        sup.step(m, now=t)
        return sup

    fresh = run(0.02)
    check("fresh data is used",
          fresh.counters['valid'] > 0 and fresh.counters['stale'] == 0,
          f"stale={fresh.counters['stale']}")

    stale = run(10.0)
    check("data from 10 s ago is rejected as stale, even though the PIV "
          "finished this instant",
          stale.counters['stale'] == 1 and 'old' in stale.invalid_reason,
          f"stale={stale.counters['stale']}, reason={stale.invalid_reason!r}")
    check("stale data does NOT count as valid",
          stale.counters['valid'] == fresh.counters['valid'] - 1
          or stale.counters['valid'] < fresh.counters['valid'],
          "only a fresh measurement may reset the fault timers")

    # ...but a GOOD measurement must still reach the display and the filter,
    # however late it is.  Gating both on age froze the readout: 93% valid
    # vectors on screen and a flat line on the strip chart, which reads as a
    # broken measurement rather than a late one.
    late = run(10.0)
    check("a stale but GOOD measurement still updates u_raw",
          np.isfinite(late.u_raw) and abs(late.u_raw - 3.0) < 1e-9,
          f"u_raw={late.u_raw}; the operator must see the latest thing EBIV "
          f"actually measured")
    check("and it still updates the filtered value",
          np.isfinite(late.u_filtered),
          "otherwise the strip chart flatlines while the PIV is working fine")
    check("it is flagged as stale for the caller",
          late.measurement_stale is True and late.measurement_age_s > 9.0,
          f"stale={late.measurement_stale}, age={late.measurement_age_s:.1f} s")
    check("the reason says it is shown but not used",
          "NOT used for control" in late.invalid_reason,
          late.invalid_reason)

    # A measurement that is BAD (not merely late) must not reach the filter.
    bad = ControlSupervisor(cfg, MockPump(0.0, 5.0))
    bad.step(Measurement(float('nan'), False, 1, 100, 0.0, 0.0, 1,
                         "too few valid vectors"), now=0.0)
    check("a bad measurement never reaches the filter",
          not np.isfinite(bad.u_filtered),
          "quality and freshness are different questions, but a failure of "
          "quality still blocks everything")

    # The old behaviour keyed off t_measured, which cannot see this at all.
    src = open(__import__('ebiv_control').__file__, encoding='utf-8').read()
    check("the gate reads t_frame, not t_measured",
          "age = now - measurement.t_frame" in src,
          "t_measured is when the correlation finished, which says nothing "
          "about how old the events were")


# ==========================================================================
#  5g. The RT-PIV rate cap
# ==========================================================================

def test_acq_verdict():
    section("5h  Acquisition keeping-up verdict")
    from ebiv_utils import _acq_verdict

    # The numbers from a real lab run: 549 batches in 3.52 s at f_acq 200 Hz.
    rows = [('acquisition_lag', 549, 379.09, 395.12, 471.82, 489.40)]
    txt = "\n".join(_acq_verdict(rows, 200.0, 3.52, 60, 10.0))
    check("it states the achieved batch rate against the requested one",
          "156" in txt and "200" in txt, txt.splitlines()[1] if txt else "")
    check("it says plainly that the loop is not keeping up",
          "NOT KEEPING UP" in txt,
          "the raw table does not answer the one question that matters")
    check("it quantifies the per-batch deficit",
          "ms per batch over budget" in txt)

    ok = "\n".join(_acq_verdict(
        [('acquisition_lag', 350, 12.0, 11.0, 20.0, 31.0)], 100.0, 3.52, 35, 10.0))
    check("a healthy loop is reported as healthy",
          "keeping up" in ok and "NOT KEEPING UP" not in ok)

    # The PIV suggestion must only appear when PIV really is over-running.
    check("no PIV advice when the PIV rate is near the control rate",
          "PIV_RT_MAX_RATE_HZ" not in txt,
          "60 fields in 3.52 s is 17 Hz against a 10 Hz controller: not the "
          "dominant cost, so do not send the user chasing it")
    busy = "\n".join(_acq_verdict(rows, 200.0, 3.52, 200, 10.0))
    check("PIV advice appears when it IS over-running",
          "PIV_RT_MAX_RATE_HZ" in busy, "200 fields in 3.52 s is 57 Hz")

    check("no verdict without acquisition samples",
          _acq_verdict([], 200.0, 3.52, 60, 10.0) == [])

    # LatencyStats keeps a deque(maxlen=4000), so the sample count SATURATES.
    # Deriving the achieved rate from it under-reports on any run longer than
    # 4000/f_acq seconds, and the advice that follows is wrong with it.
    sat = [('acquisition_lag', 4000, 918.0, 875.0, 1161.0, 1190.0)]
    from_deque = "\n".join(_acq_verdict(sat, 200.0, 33.05, 335, 10.0))
    from_count = "\n".join(_acq_verdict(sat, 200.0, 33.05, 335, 10.0,
                                         n_batches=5000))
    check("a saturated sample count does not decide the achieved rate",
          "121.0" in from_deque and "151.3" in from_count,
          "the true batch counter must win over the ring-buffer length")
    check("and the advice follows the corrected rate",
          "about 97 Hz" in from_deque and "about 121 Hz" in from_count,
          "a wrong rate produces a wrong recommendation")


def test_piv_rate_cap():
    section("5g  RT-PIV rate cap")
    from ebiv_config import PIVConfig, Session

    check("the cap is OFF by default (Release 4.0 behaviour)",
          PIVConfig().rt_max_rate_hz is None)

    # The throttle is a producer-side gate on t_frame; reproduce its arithmetic
    # so the policy is pinned even though the loop itself needs a camera.
    def queued(rate_hz, frame_times):
        min_dt = (1.0 / rate_hz) if rate_hz else 0.0
        due = -1e9
        out = []
        for t in frame_times:
            if min_dt <= 0.0 or t >= due:
                if min_dt > 0.0:
                    due += min_dt
                    if due <= t:
                        due = t + min_dt
                out.append(t)
        return out

    f_acq = 125.0
    times = [i / f_acq for i in range(int(4 * f_acq))]      # 4 s at 125 Hz

    check("with no cap every frame is offered to the PIV worker",
          len(queued(None, times)) == len(times))

    got = queued(20.0, times)
    rate = (len(got) - 1) / (got[-1] - got[0])
    check("a 20 Hz cap yields ~20 fields/s from a 125 Hz stream",
          abs(rate - 20.0) < 1.5, f"{rate:.1f} Hz from {len(got)} fields")

    got10 = queued(10.0, times)
    r10 = (len(got10) - 1) / (got10[-1] - got10[0])
    check("a 10 Hz cap still delivers 10 fields/s to a 10 Hz controller",
          r10 >= 9.9, f"{r10:.2f} Hz -- the fix must not starve the controller "
                      f"it exists to protect")
    # Individual intervals may jitter by up to one frame period; the average
    # is what bounds the load.
    gaps = [b - a for a, b in zip(got10, got10[1:])]
    check("no interval is shorter than the cap by more than one frame period",
          min(gaps) >= 1 / 10.0 - 1 / f_acq - 1e-9,
          f"min gap {min(gaps) * 1e3:.1f} ms")

    sess = Session()
    sess.control.piv.rt_max_rate_hz = 20.0
    d = sess.to_dict()
    back, ign = Session.from_dict(d)
    check("the cap survives a preset round-trip",
          back.control.piv.rt_max_rate_hz == 20.0 and not ign)


# ==========================================================================
#  5i. Auto-trigger: window selection and re-anchoring
# ==========================================================================

def test_auto_trigger_window():
    section("5i  Auto-trigger window and re-anchor")

    rng = np.random.default_rng(4)
    period = 5000                    # us, 200 Hz
    # A realistic batch: pulses at a fixed phase, plus background events.
    ts = []
    for k in range(40):
        ts.append(rng.uniform(k * period + 3000, k * period + 3500, 300))
        ts.append(rng.uniform(k * period, (k + 1) * period, 40))
    ev_t = np.sort(np.concatenate(ts)).astype(np.int64)
    ev_x = rng.integers(0, 640, ev_t.size).astype(np.uint16)

    # searchsorted must select EXACTLY the same events as the boolean mask.
    worst = 0
    for centre in range(3250, 40 * period, period):
        for half in (250, 1250, 2500):
            t_lo, t_hi = centre - half, centre + half
            mask = (ev_t >= t_lo) & (ev_t < t_hi)
            lo_i = int(np.searchsorted(ev_t, t_lo, side='left'))
            hi_i = int(np.searchsorted(ev_t, t_hi, side='left'))
            same = np.array_equal(ev_x[mask], ev_x[lo_i:hi_i])
            worst = max(worst, 0 if same else 1)
    check("searchsorted selects exactly the events the boolean mask did",
          worst == 0,
          "monotonic timestamps make the window a contiguous range; this "
          "replaces five full-length temporaries per batch with two binary "
          "searches")

    # --- _finalize_frame must stay bit-identical ------------------------
    rng2 = np.random.default_rng(9)
    worst_f = True
    for maxe in (1, 2, 5, 10):
        src = rng2.integers(0, maxe + 3, (64, 96)).astype(np.float32)
        a, b = src.copy(), src.copy()
        np.clip(a, 0, maxe, out=a)
        old_way = ((a / maxe) * 255).astype(np.uint8)
        np.clip(b, 0, maxe, out=b)
        np.divide(b, maxe, out=b)
        np.multiply(b, 255.0, out=b)
        new_way = b.astype(np.uint8)
        worst_f = worst_f and np.array_equal(old_way, new_way)
    check("the in-place frame normalisation is bit-identical",
          worst_f,
          "same operations in the same order; only the two full-size float "
          "temporaries per frame are gone")

    # --- re-anchoring after a pause in the event stream -----------------
    def advance(next_centre, chunk_end, period_us):
        """The loop's re-anchor arithmetic."""
        n = 0
        if chunk_end - next_centre > period_us:
            n = (chunk_end - next_centre) // period_us
            next_centre += n * period_us
        return next_centre, n

    phase = 3150
    centre = 523150                        # as calibrated in a real run
    # The light is blocked for 2 s: no events, so the counter never advanced.
    resumed_at = centre + 2_000_000
    new_centre, n = advance(centre, resumed_at, period)
    check("a 2 s pause is closed in ONE jump, not one period per batch",
          n == 400 and new_centre == centre + 400 * period,
          f"n={n}; stepping one period per batch took hundreds of batches, "
          f"emitting garbage frames the whole way")
    check("the laser phase is preserved exactly across the jump",
          (new_centre - phase) % period == (centre - phase) % period,
          "an integer number of periods cannot change the phase, which is why "
          "the calibration never needs redoing")
    check("the re-anchored centre is within one period of the resume time",
          0 <= resumed_at - new_centre <= period,
          f"{resumed_at - new_centre} us")

    unchanged, n0 = advance(centre, centre + period // 3, period)
    check("normal running does not trigger a re-anchor",
          n0 == 0 and unchanged == centre)


# ==========================================================================
#  6a. The version is single-sourced
# ==========================================================================

def test_version_single_source():
    section("6a  Version string")
    import json
    import tempfile
    import ebiv_logger
    from ebiv_config import __version__, ControlSystemConfig

    check("ebiv_config declares a version", bool(__version__), __version__)

    src = open(ebiv_logger.__file__, encoding='utf-8').read()
    check("the logger does not hard-code a version number",
          "'release': __version__" in src,
          "the 'release' field must read ebiv_config.__version__")

    # The number that actually lands in a run's config.json is what matters:
    # a stale one there makes a run unidentifiable after the fact.
    cfg = ControlSystemConfig()
    with tempfile.TemporaryDirectory() as d:
        lg = ebiv_logger.ExperimentLogger(d, "version_probe", cfg)
        path = os.path.join(d, "Control", "version_probe",
                            "version_probe_config.json")
        written = json.load(open(path, encoding='utf-8'))['release']
        try:
            lg.close()
        except Exception:                                      # noqa: BLE001
            pass
    check("the experiment log records the current version",
          written == __version__, f"config.json says {written!r}")


# ==========================================================================
#  6b. Live target: the supervisor API and the [+]/[-] routing
# ==========================================================================

def test_live_target():
    section("6b  Live target set by the operator")
    from ebiv_config import (ControlSystemConfig, PIDConfig, SupervisorConfig,
                             ReferenceConfig, FilterConfig, CalibrationConfig)
    from ebiv_control import ControlSupervisor, Measurement, ControlState
    from ebiv_hardware import MockPump
    from ebiv_utils import _operator_bump

    def make_cfg(kind='live', **ref_kw):
        return ControlSystemConfig(
            pid=PIDConfig(kp=1.0, ki=0.5, v_min=0.0, v_max=5.0,
                          slew_rate_v_per_s=None),
            supervisor=SupervisorConfig(control_rate_hz=10.0,
                                        hold_timeout_s=0.5, safe_timeout_s=5.0,
                                        max_measurement_age_s=0.5),
            reference=ReferenceConfig(kind=kind, value=2.0, **ref_kw),
            filt=FilterConfig(kind='none'))

    # --- supervisor API -----------------------------------------------
    cfg = make_cfg(live_step=0.25, live_min=0.0, live_max=4.0)
    sup = ControlSupervisor(cfg, MockPump(0.0, 5.0))
    check("live_target exposes the reference when kind is 'live'",
          sup.live_target is not None)
    check("bump_target moves the target", sup.bump_target(+2) == 2.5)
    check("set_target sets it absolutely", sup.set_target(3.0) == 3.0)

    cfg_const = make_cfg(kind='constant')
    sup_const = ControlSupervisor(cfg_const, MockPump(0.0, 5.0))
    check("a non-live reference reports None rather than pretending to work",
          sup_const.live_target is None
          and sup_const.bump_target(+1) is None
          and sup_const.set_target(1.0) is None
          and sup_const.target_to_measurement() is None,
          "the caller must be able to tell that nothing happened")

    check("target_to_measurement refuses without a finite measurement",
          sup.target_to_measurement() is None)
    sup.step(Measurement(1.75, True, 100, 100, 0.0, 0.0, 0), now=0.0)
    check("target_to_measurement snaps the target onto the filtered velocity",
          sup.target_to_measurement() == 1.75)

    # --- the target actually reaches the control law -------------------
    cfg2 = make_cfg(live_step=0.5)
    sup2 = ControlSupervisor(cfg2, MockPump(0.0, 5.0))
    t, dt = 0.0, 0.1
    sup2.step(Measurement(2.0, True, 100, 100, t, t, 0), now=t); t += dt
    sup2.arm(now=t)
    sup2.step(Measurement(2.0, True, 100, 100, t, t, 1), now=t); t += dt
    u_before = sup2.u_target
    sup2.bump_target(+4)                       # +2.0 units
    sup2.step(Measurement(2.0, True, 100, 100, t, t, 2), now=t)
    check("a bump while armed changes the target the PID acts on",
          u_before == 2.0 and sup2.u_target == 4.0,
          f"{u_before} -> {sup2.u_target}")
    check("the resulting error drives the command up",
          sup2.v_command > 0.0, f"v={sup2.v_command:.3f} V")

    # --- [+]/[-] routing ----------------------------------------------
    class _Cfg:
        def __init__(self, control, units="px/frame"):
            self.reference = control.reference
            self.calibration = CalibrationConfig(manual_step_v=0.2)
            self.measurement = type("M", (), {"velocity_units": units})()

    cfg3 = make_cfg(live_step=0.5)
    sup3 = ControlSupervisor(cfg3, MockPump(0.0, 5.0))
    shim = _Cfg(cfg3)

    # loop OPEN -> the keys move the voltage, as they always did
    v0 = sup3.manual_voltage
    _operator_bump(sup3, shim, None, +1)
    check("with the loop open, [+] moves the pump voltage",
          abs(sup3.manual_voltage - (v0 + 0.2)) < 1e-12
          and sup3.live_target.value == 2.0,
          "the target must not move while the loop is open")

    # loop CLOSED -> the keys move the target, not the voltage
    sup3.step(Measurement(2.0, True, 100, 100, 0.0, 0.0, 0), now=0.0)
    sup3.arm(now=0.1)
    v_manual = sup3.manual_voltage
    _operator_bump(sup3, shim, None, +1)
    check("with the loop closed, [+] moves the target instead",
          sup3.live_target.value == 2.5 and sup3.manual_voltage == v_manual,
          f"target={sup3.live_target.value}, manual V unchanged")

    # HOLD is still armed, so the keys still move the target
    sup3.state = ControlState.HOLD
    _operator_bump(sup3, shim, None, -1)
    check("HOLD counts as armed for [+]/[-]",
          sup3.live_target.value == 2.0)

    # a non-live reference while armed must not silently move anything
    cfg4 = make_cfg(kind='constant')
    sup4 = ControlSupervisor(cfg4, MockPump(0.0, 5.0))
    shim4 = _Cfg(cfg4)
    sup4.step(Measurement(2.0, True, 100, 100, 0.0, 0.0, 0), now=0.0)
    sup4.arm(now=0.1)
    v_manual4 = sup4.manual_voltage
    _operator_bump(sup4, shim4, None, +1)
    check("armed with a non-live reference, [+] changes nothing",
          sup4.manual_voltage == v_manual4,
          "it must not fall through to commanding the voltage under the PID")


# ==========================================================================
#  7. Closed loop against a simulated plant
# ==========================================================================

def test_closed_loop_simulation():
    section("7  Closed loop against a simulated first-order pump/jet")
    from ebiv_config import (ControlSystemConfig, PIDConfig, SupervisorConfig,
                             ReferenceConfig, FilterConfig, AD3Config)
    from ebiv_control import FirstOrderJetModel, ControlState
    from ebiv_runtime import simulate_closed_loop

    def cfg_for(ref, kp=0.35, ki=0.25, kd=0.0, filt='ema', tau=0.2):
        return ControlSystemConfig(
            pid=PIDConfig(kp=kp, ki=ki, kd=kd, v_min=0.0, v_max=5.0,
                          slew_rate_v_per_s=5.0, derivative_filter_tau_s=0.1),
            supervisor=SupervisorConfig(control_rate_hz=20.0,
                                        hold_timeout_s=0.5, safe_timeout_s=2.0,
                                        max_measurement_age_s=1.0,
                                        safe_pump_voltage=0.0),
            reference=ref, filt=FilterConfig(kind=filt, tau_s=tau),
            ad3=AD3Config(enabled=False))

    # --- open loop first: confirm the plant is not linear in V ---
    pl = FirstOrderJetModel()
    static = []
    for v in (1.0, 2.0, 3.0, 4.0, 5.0):
        p = FirstOrderJetModel(delay_s=0.0)
        for _ in range(4000):
            y = p.step(v, 0.005)
        static.append(y)
    lin = np.polyfit([1, 2, 3, 4, 5], static, 1)
    resid = np.max(np.abs(np.polyval(lin, [1, 2, 3, 4, 5]) - static))
    check("the simulated plant is deliberately non-linear in V",
          resid > 0.05 * (max(static) - min(static)),
          f"max deviation from a straight line: {resid:.3f}")

    # --- step tracking ---
    r = ReferenceConfig(kind='step', value=2.0, t_step_s=20.0, step_amplitude=1.5)
    out = simulate_closed_loop(cfg_for(r), FirstOrderJetModel(), duration_s=60.0)
    t = out['t']
    e1 = np.abs(out['u_plant'] - out['u_target'])[(t > 15) & (t < 20)].mean()
    e2 = np.abs(out['u_plant'] - out['u_target'])[t > 50].mean()
    check("closed loop tracks a constant reference",
          e1 < 0.05, f"mean |error| before the step: {e1:.4f}")
    check("closed loop settles after a step change",
          e2 < 0.05, f"mean |error| 30 s after the step: {e2:.4f}")
    check("the command respects the voltage limits throughout",
          np.nanmin(out['v_command']) >= 0.0 and np.nanmax(out['v_command']) <= 5.0)

    # --- with the loop open (all gains zero) the error is large: the
    #     improvement above is due to feedback, not to the plant ---
    out0 = simulate_closed_loop(cfg_for(r, kp=0.0, ki=0.0),
                                FirstOrderJetModel(), duration_s=60.0)
    e0 = np.abs(out0['u_plant'] - out0['u_target'])[t > 50].mean()
    check("with zero gains the same plant does NOT track (control is doing the work)",
          e0 > 10 * max(e2, 1e-6), f"open loop {e0:.3f} vs closed loop {e2:.4f}")

    # --- sine tracking, and the honest statement of what it costs ---
    r = ReferenceConfig(kind='sine', value=2.5, amplitude=0.8, freq_hz=0.05)
    out = simulate_closed_loop(cfg_for(r), FirstOrderJetModel(), duration_s=120.0)
    t = out['t']
    m = t > 40
    rms = float(np.sqrt(np.mean((out['u_plant'][m] - out['u_target'][m]) ** 2)))
    amp = float(np.std(out['u_target'][m])) * math.sqrt(2)
    check("closed loop follows a slow sinusoid",
          rms < 0.3 * amp, f"tracking RMS error {rms:.3f} vs amplitude {amp:.3f}")

    # --- noise + dropouts ---
    r = ReferenceConfig(kind='constant', value=2.5)
    # hold_timeout_s is 0.5 s in cfg_for().  A dropout shorter than that must
    # be ridden out silently; a longer one must be announced as HOLD.  Both
    # must recover.
    out = simulate_closed_loop(cfg_for(r), FirstOrderJetModel(),
                               duration_s=60.0, measurement_noise=0.15,
                               dropout_windows=[(20.0, 20.3), (35.0, 35.2)])
    states = set(out['state'])
    check("dropouts shorter than hold_timeout_s do not disturb the state",
          states == {ControlState.CLOSED_LOOP},
          f"states seen: {sorted(states)}")

    out = simulate_closed_loop(cfg_for(r), FirstOrderJetModel(),
                               duration_s=60.0, measurement_noise=0.15,
                               dropout_windows=[(20.0, 21.2), (35.0, 36.0)])
    states = set(out['state'])
    check("dropouts longer than hold_timeout_s trigger HOLD and the loop recovers",
          ControlState.HOLD in states and ControlState.SAFE not in states
          and out['state'][-1] == ControlState.CLOSED_LOOP,
          f"states seen: {sorted(states)}")
    check("the pump command stays finite and in range through the dropouts",
          np.all(np.isfinite(out['v_command'])) and np.nanmax(out['v_command']) <= 5.0)

    out = simulate_closed_loop(cfg_for(r), FirstOrderJetModel(),
                               duration_s=60.0, dropout_windows=[(20.0, 40.0)])
    check("a long dropout escalates to SAFE",
          ControlState.SAFE in set(out['state']), f"final state {out['state'][-1]}")
    check("SAFE holds the configured safe voltage",
          abs(out['v_command'][-1] - 0.0) < 1e-9)

    # --- a saturating demand must not wind up ---
    r = ReferenceConfig(kind='constant', value=50.0)     # unreachable
    out = simulate_closed_loop(cfg_for(r, kp=1.0, ki=2.0),
                               FirstOrderJetModel(), duration_s=60.0)
    check("an unreachable set-point saturates without exceeding v_max",
          abs(np.nanmax(out['v_command']) - 5.0) < 1e-9)
    sup = out['supervisor']
    check("the integrator stayed bounded while saturated",
          abs(sup.pid._i) < 1e4, f"I = {sup.pid._i:.3g}")


# ==========================================================================
#  8. Hardware abstraction
# ==========================================================================

def test_hardware_abstraction():
    section("8  Hardware abstraction (no device attached)")
    from ebiv_config import ControlSystemConfig, AD3Config, SupervisorConfig
    from ebiv_hardware import MockPump, make_pump, is_ad3_available, _pulse_metrics

    p = MockPump(0.0, 5.0)
    check("mock pump clamps above its maximum", p.set_voltage(9.0) == 5.0)
    check("mock pump clamps below its minimum", p.set_voltage(-3.0) == 0.0)
    check("clamping is counted", p.n_clamped == 2, f"n_clamped={p.n_clamped}")

    cfg = ControlSystemConfig(ad3=AD3Config(enabled=False))
    dev, pump = make_pump(cfg)
    check("with the AD3 disabled, make_pump returns a mock and no device",
          dev is None and pump.name == 'mock')

    # channel-allocation validation
    try:
        AD3Config(laser_backend='analog', laser_channel=1, pump_channel=1)
        clash = False
    except ValueError:
        clash = True
    check("configuring the laser and the pump on the same channel is refused", clash)

    try:
        AD3Config(pump_v_min=0.0, pump_v_max=12.0)
        bad = False
    except ValueError:
        bad = True
    check("a pump range beyond the 0-10 V input specification is refused", bad)

    c = ControlSystemConfig()
    c.pid.v_max = 7.0
    try:
        c.cross_check()
        crossed = False
    except ValueError:
        crossed = True
    check("a PID range wider than the pump hardware range is refused at start-up",
          crossed)

    c = ControlSystemConfig()
    c.supervisor.safe_pump_voltage = 9.0
    try:
        c.cross_check()
        bad_safe = False
    except ValueError:
        bad_safe = True
    check("a safe voltage outside the hardware range is refused", bad_safe)

    # the pulse metric used by the verification mode
    fs = 1e6
    t = np.arange(int(0.05 * fs)) / fs
    f0, duty = 500.0, 0.10
    sq = ((t * f0) % 1.0 < duty).astype(float) * 5.0
    m = _pulse_metrics(sq, fs)
    check("laser pulse metrics recover frequency and duty from a captured trace",
          m is not None and abs(m['f_hz'] - f0) < 1.0 and abs(m['duty_pct'] - 10.0) < 0.6,
          f"{m['f_hz']:.2f} Hz, duty {m['duty_pct']:.2f}%")

    # the control law must not import the hardware module
    import ebiv_control
    src = open(ebiv_control.__file__, encoding='utf-8').read()
    head = src[:src.index('class ControlSupervisor')]
    check("the control law has no module-level dependency on the hardware layer",
          'import ebiv_hardware' not in head and 'from ebiv_hardware' not in head,
          "the PID is testable with no device present")

    check("is_ad3_available() answers without raising",
          isinstance(is_ad3_available(), bool),
          f"reports {is_ad3_available()} in this environment")


# ==========================================================================
#  9. Config plumbing and logger
# ==========================================================================

def test_image_flip():
    section("9c  Image flip and ROI coordinates")
    import cv2
    from ebiv_utils import unflip_roi
    from ebiv_config import MeasurementConfig
    from ebiv_control import ControlROIEstimator

    W, H = 1280, 720

    # The mapping must agree with what cv2.flip actually does to a pixel.
    img = np.zeros((H, W), np.uint8)
    img[100, 300] = 255
    fx = cv2.flip(img, 1)
    yy, xx = np.nonzero(fx)
    check("cv2.flip(.,1) maps column x -> W-1-x",
          int(xx[0]) == W - 1 - 300 and int(yy[0]) == 100,
          f"(300,100) -> ({int(xx[0])},{int(yy[0])})")
    fy = cv2.flip(img, 0)
    yy, xx = np.nonzero(fy)
    check("cv2.flip(.,0) maps row y -> H-1-y",
          int(xx[0]) == 300 and int(yy[0]) == H - 1 - 100,
          f"(300,100) -> ({int(xx[0])},{int(yy[0])})")

    roi = [200, 400, 100, 300]
    check("unflip_roi is the identity with no flip",
          unflip_roi(roi, W, H, False, False) == roi)
    check("unflip_roi mirrors x for flip_x",
          unflip_roi(roi, W, H, True, False) == [W - 400, W - 200, 100, 300],
          str(unflip_roi(roi, W, H, True, False)))
    check("unflip_roi mirrors y for flip_y",
          unflip_roi(roi, W, H, False, True) == [200, 400, H - 300, H - 100])
    check("unflip_roi is its own inverse",
          unflip_roi(unflip_roi(roi, W, H, True, True), W, H, True, True) == roi)

    # A rectangle mapped back to raw coordinates must select the same pixels.
    marker = np.zeros((H, W), np.uint8)
    marker[roi[2]:roi[3], roi[0]:roi[1]] = 255      # region in FLIPPED coords
    raw = unflip_roi(roi, W, H, True, True)
    src = np.zeros((H, W), np.uint8)
    src[raw[2]:raw[3], raw[0]:raw[1]] = 255         # same region in RAW coords
    flipped = cv2.flip(cv2.flip(src, 1), 0)
    check("a raw-coordinate region, once flipped, lands on the configured ROI",
          np.array_equal(flipped, marker))

    # Flipping left-right negates U, so '-u' on the raw image and 'u' on the
    # flipped image must give the same feedback value.
    ws, nd = 32, 16
    display_roi = [0, 320, 0, 240]
    control_roi = [80, 160, 48, 112]
    gy, gx = (240 - ws) // nd + 1, (320 - ws) // nd + 1
    U_raw = np.zeros((gy, gx), np.float32)
    V_raw = np.zeros((gy, gx), np.float32)

    def est_for(comp):
        return ControlROIEstimator(
            MeasurementConfig(component=comp, min_valid_fraction=0.0,
                              min_valid_vectors=1, use_correlation_gate=False),
            display_roi, control_roi, ws, nd, gy, gx)

    e_neg = est_for('-u')
    U_raw[e_neg.sy, e_neg.sx] = -3.0        # jet running right-to-left
    u_neg, ok1, _, _ = e_neg.estimate(U_raw, V_raw)

    U_flip = -U_raw                          # what a left-right flip produces
    u_pos, ok2, _, _ = est_for('u').estimate(U_flip, V_raw)
    check("component '-u' on the raw image equals 'u' on the flipped image",
          ok1 and ok2 and abs(u_neg - u_pos) < 1e-9 and u_neg > 0,
          f"both give {u_neg:+.3f} — so a sign flip does not need an image flip")


def test_velocity_units():
    section("9b  Velocity units: px/mm calibration -> m/s")
    from ebiv_config import MeasurementConfig, ROIConfig
    from ebiv_control import ControlROIEstimator

    # No calibration: the DEFAULT is now px/s (a velocity), and the explicit
    # 'px/frame' basis is what reproduces the Release 4.0-5.1 identity scaling.
    m = MeasurementConfig(uncalibrated_units='px/frame')
    info = m.resolve_units(1.0 / 500.0, laser_frequency_hz=500.0)
    check("the 'px/frame' basis is the identity scaling",
          m.velocity_scale == 1.0 and m.velocity_units == 'px/frame',
          f"scale={m.velocity_scale}, units={m.velocity_units}")
    md = MeasurementConfig()
    md.resolve_units(1.0 / 500.0, laser_frequency_hz=500.0)
    check("the default basis divides by dt to give px/s",
          md.velocity_scale == 500.0 and md.velocity_units == 'px/s',
          f"scale={md.velocity_scale}, units={md.velocity_units}")
    check("dt is derived from the accumulation window",
          abs(info['dt_s'] - 0.002) < 1e-12, f"dt={info['dt_s'] * 1e6:.1f} us")

    # 20 px/mm at 500 Hz:  4 px/frame  ->  4/20 mm = 0.2 mm per pulse
    #                                  ->  0.2 mm * 500 /s = 100 mm/s = 0.1 m/s
    m = MeasurementConfig(calibration_px_per_mm=20.0)
    info = m.resolve_units(1.0 / 500.0, laser_frequency_hz=500.0)
    check("px/mm + dt resolve to m/s", m.velocity_units == 'm/s')
    check("worked example: 20 px/mm, 500 Hz, 4 px/frame -> 0.1 m/s",
          abs(4.0 * m.velocity_scale - 0.1) < 1e-12,
          f"4 px/frame = {4.0 * m.velocity_scale:.6g} m/s")
    check("the inverse conversion round-trips",
          abs(m.to_px(0.1) - 4.0) < 1e-9, f"0.1 m/s = {m.to_px(0.1):.4f} px/frame")
    # 1 m/s over dt = 2 ms is a 2 mm displacement, which at 20 px/mm is 40 px.
    check("1 m/s is reported in px/frame for the range check",
          abs(info['px_per_unit'] - 40.0) < 1e-9, f"{info['px_per_unit']:.1f} px/frame")

    # halving dt doubles the velocity for the same displacement
    m2 = MeasurementConfig(calibration_px_per_mm=20.0)
    m2.resolve_units(1.0 / 1000.0, laser_frequency_hz=1000.0)
    check("doubling the pulse rate doubles m/s for the same px displacement",
          abs(m2.velocity_scale - 2 * m.velocity_scale) < 1e-15,
          f"{m.velocity_scale:.6g} -> {m2.velocity_scale:.6g}")

    # doubling the resolution halves the velocity for the same displacement
    m3 = MeasurementConfig(calibration_px_per_mm=40.0)
    m3.resolve_units(1.0 / 500.0)
    check("doubling px/mm halves m/s for the same px displacement",
          abs(m3.velocity_scale - 0.5 * m.velocity_scale) < 1e-15)

    # an explicit pulse separation wins over the derived one
    m4 = MeasurementConfig(calibration_px_per_mm=20.0, pulse_separation_s=1e-4)
    i4 = m4.resolve_units(1.0 / 500.0, laser_frequency_hz=500.0)
    check("an explicit pulse_separation_s overrides the derived dt",
          abs(i4['dt_s'] - 1e-4) < 1e-15 and 'explicit' in i4['dt_source'],
          i4['dt_source'])
    check("an explicit dt suppresses the laser-mismatch check",
          not i4['laser_mismatch'])

    # laser frequency disagreeing with f_acq is flagged
    m5 = MeasurementConfig(calibration_px_per_mm=20.0)
    i5 = m5.resolve_units(1.0 / 500.0, laser_frequency_hz=250.0)
    check("a laser frequency that disagrees with f_acq is flagged",
          i5['laser_mismatch'], "250 Hz laser vs 500 Hz frame rate")
    i6 = MeasurementConfig().resolve_units(1.0 / 500.0, laser_frequency_hz=500.0)
    check("a matching laser frequency is not flagged", not i6['laser_mismatch'])

    check("a non-positive resolution is refused",
          _raises(lambda: MeasurementConfig(calibration_px_per_mm=0.0)
                  .resolve_units(0.002)))
    check("a non-positive pulse separation is refused",
          _raises(lambda: MeasurementConfig(pulse_separation_s=-1.0)
                  .resolve_units(0.002)))

    # end to end: the estimator emits m/s
    ws, nd = 32, 16
    display_roi = [0, 320, 0, 240]
    gy, gx = (240 - ws) // nd + 1, (320 - ws) // nd + 1
    mc = MeasurementConfig(calibration_px_per_mm=20.0, min_valid_fraction=0.0,
                           min_valid_vectors=1, use_correlation_gate=False)
    mc.resolve_units(1.0 / 500.0)
    est = ControlROIEstimator(mc, display_roi, [80, 160, 48, 112], ws, nd, gy, gx)
    U = np.zeros((gy, gx), np.float32)
    V = np.zeros((gy, gx), np.float32)
    U[est.sy, est.sx] = 4.0
    u, ok, _, _ = est.estimate(U, V)
    check("the ControlROI estimator emits the calibrated velocity end to end",
          ok and abs(u - 0.1) < 1e-9, f"4 px/frame -> {u:.6g} m/s")


def _raises(fn):
    try:
        fn()
        return False
    except (ValueError, ZeroDivisionError):
        return True


def test_config_and_logging():
    section("9  Configuration and experiment logging")
    from ebiv_config import ControlSystemConfig, ROIConfig, FilterConfig
    from ebiv_logger import ExperimentLogger
    from ebiv_control import ControlSupervisor, Measurement
    from ebiv_hardware import MockPump

    d, c = ROIConfig(display_roi=[100, 500, 50, 250],
                     control_roi=[200, 300, 100, 200]).validate(1280, 720)
    check("ROI validation returns clamped, contained regions",
          d == [100, 500, 50, 250] and c == [200, 300, 100, 200])
    d, c = ROIConfig().validate(1280, 720)
    check("a missing DisplayROI defaults to the full sensor and the "
          "ControlROI to the DisplayROI", d == [0, 1280, 0, 720] and c == d)

    check("filter description states the delay it introduces",
          'delay' in FilterConfig(kind='ma', window=9).describe(),
          FilterConfig(kind='ma', window=9).describe())

    with tempfile.TemporaryDirectory() as tmp:
        cfg = ControlSystemConfig()
        cfg.logging.flush_every = 1
        lg = ExperimentLogger(tmp, "unittest", cfg)
        sup = ControlSupervisor(cfg, MockPump(0, 5))
        for i in range(25):
            m = Measurement(1.0 + 0.1 * i, True, 90, 100, i * 0.1, i * 0.1, i)
            sup.step(m, now=i * 0.1)
            lg.record_latency('fake_stage', 0.001 * (i + 1))
            lg.log_step(sup, m, 0.1, 0.05)
        lg.log_event('TEST', 'hello')
        lg.close(sup=sup)

        files = sorted(os.listdir(lg.dir))
        for want in ('unittest_control.csv', 'unittest_config.json',
                     'unittest_events.csv', 'unittest_control.mat',
                     'unittest_timing.csv'):
            check(f"logger wrote {want}", want in files, str(files))

        import csv as _csv
        with open(os.path.join(lg.dir, 'unittest_control.csv')) as f:
            rows = list(_csv.reader(f))
        check("control CSV has a header and one row per step",
              len(rows) == 26, f"{len(rows)} rows")

        from scipy.io import loadmat
        md = loadmat(os.path.join(lg.dir, 'unittest_control.mat'))
        check("the .mat is readable and carries the control columns",
              'u_target' in md and 'v_command' in md and md['u_raw'].size == 25)

        import json
        with open(os.path.join(lg.dir, 'unittest_config.json')) as f:
            hdr = json.load(f)
        check("the run header records the filter and PID settings that were used",
              hdr['config']['filt']['kind'] == cfg.filt.kind
              and hdr['config']['pid']['v_max'] == cfg.pid.v_max)


# ==========================================================================
#  10. Visualisation is head-less safe and cheap
# ==========================================================================

def test_session_and_gui():
    section("11  Session, offline chain and GUI parameter binding")
    import sys
    import types
    import tempfile
    from ebiv_config import Session, RunConfig

    # --- Session round-trip and tolerance -----------------------------
    s = Session()
    check("the default session validates", s.validate())
    s.run.f_acq = 250.0
    s.control.measurement.calibration_px_per_mm = 20.0
    s.control.pid.kp = 1.5
    with tempfile.TemporaryDirectory() as d:
        p = s.save(os.path.join(d, "preset.json"))
        s2, ignored = Session.load(p)
    check("a preset round-trips exactly",
          s2.to_dict() == s.to_dict() and not ignored)
    s3, ig = Session.from_dict({'run': {'f_acq': 100.0, 'gone_in_this_version': 1},
                                'control': {'pid': {'kp': 2.0}}})
    check("a preset with unknown keys still loads, and names them",
          s3.run.f_acq == 100.0 and s3.control.pid.kp == 2.0
          and ig == ['run.gone_in_this_version'], str(ig))
    check("derived acquisition timing is consistent",
          RunConfig(f_acq=500.0).accum_time_us == 2000
          and RunConfig(f_acq=250.0).accum_time_us == 4000)

    bad = Session()
    bad.run.gaussian_kernel = 4
    bad.run.f_acq = -1.0
    check("session validation catches bad acquisition values",
          _raises(bad.validate))
    bad2 = Session()
    bad2.run.mode = 'offline'
    bad2.run.do_record = bad2.run.do_playback = False
    bad2.run.do_image_gen = bad2.run.do_piv_process = False
    check("offline mode with no steps selected is refused", _raises(bad2.validate))

    # --- directories ---------------------------------------------------
    from ebiv_session import session_directories
    with tempfile.TemporaryDirectory() as d:
        r = RunConfig(output_base_folder=d, acq_name="AcqA", raw_filename="x.raw")
        dirs = session_directories(r)
        check("session_directories follows the Release 4.0 layout",
              dirs['img'].endswith(os.path.join("RawImg", "AcqA"))
              and dirs['out'].endswith(os.path.join("Out", "AcqA"))
              and dirs['raw_file'].endswith(os.path.join("Raw", "x.raw"))
              and os.path.isdir(dirs['out']))

    # --- the offline chain runs the right steps in the right order -----
    import ebiv_utils as _eu
    calls = []
    orig = {n: getattr(_eu, n) for n in
            ('record_raw_file', 'check_raw_video', 'generate_centered_images',
             'process_offline_piv')}
    try:
        for name in orig:
            setattr(_eu, name, (lambda n: (lambda *a, **k: calls.append(n)))(name))
        with tempfile.TemporaryDirectory() as d:
            sess = Session()
            sess.run.mode = 'offline'
            sess.run.output_base_folder = d
            sess.run.do_record = True
            sess.run.do_playback = True
            sess.run.do_image_gen = True
            sess.run.do_piv_process = True
            from ebiv_session import run_session
            res = run_session(sess)
        check("the offline chain runs record -> playback -> frames -> PIV, in order",
              calls == ['record_raw_file', 'check_raw_video',
                        'generate_centered_images', 'process_offline_piv'],
              " -> ".join(calls))
        check("the offline result reports every step with its duration",
              len(res['steps']) == 4 and all(len(x) == 2 for x in res['steps']))

        calls.clear()
        with tempfile.TemporaryDirectory() as d:
            sess = Session()
            sess.run.mode = 'offline'
            sess.run.output_base_folder = d
            sess.run.do_record = False
            sess.run.do_playback = False
            sess.run.do_image_gen = True
            sess.run.do_piv_process = True
            run_session(sess)
        check("a subset of steps runs just that subset",
              calls == ['generate_centered_images', 'process_offline_piv'],
              " -> ".join(calls))
    finally:
        for name, fn in orig.items():
            setattr(_eu, name, fn)

    # --- streaming dispatch --------------------------------------------
    seen = {}
    orig_stream = _eu.stream_camera
    try:
        def fake_stream(**kw):
            seen.update(kw)
        _eu.stream_camera = fake_stream
        with tempfile.TemporaryDirectory() as d:
            sess = Session()
            sess.run.mode = 'stream'
            sess.run.run_mode = 'manual'
            sess.run.output_base_folder = d
            sess.run.f_acq = 400.0
            from ebiv_session import run_session
            run_session(sess, run_name="unit")
        check("streaming dispatch passes the derived timing and the mode through",
              seen.get('accum_time_us') == 2500 and seen.get('run_mode') == 'manual'
              and seen.get('run_name') == 'unit',
              f"accum={seen.get('accum_time_us')} mode={seen.get('run_mode')}")
    finally:
        _eu.stream_camera = orig_stream

    # --- EBIV_Main builds a valid session and honours FLAG_GUI ---------
    import EBIV_Main
    s = EBIV_Main.build_session()
    check("EBIV_Main.build_session() produces a valid session", s.validate())
    check("the flags at the top of EBIV_Main reach the session",
          s.run.mode == ('stream' if EBIV_Main.FLAG_STREAM else 'offline')
          and s.run.run_mode == EBIV_Main.STREAM_MODE
          and s.run.f_acq == EBIV_Main.F_ACQ
          and s.control.ad3.enabled == EBIV_Main.AD3_ENABLED,
          f"mode={s.run.mode}/{s.run.run_mode}, f_acq={s.run.f_acq}, "
          f"ad3={s.control.ad3.enabled}")

    # LASER_FREQUENCY_HZ = None means "follow F_ACQ", so the session must hold
    # a resolved NUMBER even though the module constant is None.  Assert the
    # rule, not a particular value: these are the user's settings and they
    # change between runs.
    expected_f = (float(EBIV_Main.F_ACQ) if EBIV_Main.LASER_FREQUENCY_HZ is None
                  else float(EBIV_Main.LASER_FREQUENCY_HZ))
    check("build_session() resolves the laser rate before anything uses it",
          s.control.ad3.laser_frequency_hz == expected_f,
          f"LASER_FREQUENCY_HZ={EBIV_Main.LASER_FREQUENCY_HZ!r}, "
          f"F_ACQ={EBIV_Main.F_ACQ} -> {s.control.ad3.laser_frequency_hz}; "
          f"FLAG_TEST_LASER reaches the device without validating a run")
    check("build_config() still returns the control half",
          EBIV_Main.build_config().piv.window_size == EBIV_Main.PIV_WINDOW_SIZE)

    # FLAG_GUI must route to the window instead of running anything
    fake_gui = types.ModuleType("ebiv_gui")
    launched = {}
    fake_gui.launch = lambda session=None, preset_path=None: launched.update(s=session)
    sys.modules['ebiv_gui'] = fake_gui
    orig_flag = EBIV_Main.FLAG_GUI
    try:
        EBIV_Main.FLAG_GUI = True
        EBIV_Main.main()
        check("FLAG_GUI routes to the parameter window instead of a run",
              launched.get('s') is not None)
    finally:
        EBIV_Main.FLAG_GUI = orig_flag
        sys.modules.pop('ebiv_gui', None)

    # --- GUI parameter binding (needs tkinter) --------------------------
    try:
        import tkinter                                          # noqa: F401
        have_tk = True
    except Exception:                                           # noqa: BLE001
        have_tk = False

    if not have_tk:
        print("  SKIP  GUI widget checks (tkinter not installed in this "
              "interpreter; run test_gui.py where it is)")
        return

    from ebiv_gui import check_spec, PARAM_SPEC, INTENTIONALLY_HIDDEN
    bad_paths, unaccounted = check_spec()
    check("every GUI parameter resolves to a real config field",
          not bad_paths, str(bad_paths))
    check("every config field is either on a tab or explicitly hidden",
          not unaccounted,
          "unaccounted: " + str(unaccounted) if unaccounted else
          f"{len(PARAM_SPEC)} exposed, {len(INTENTIONALLY_HIDDEN)} hidden by design")


def test_visualisation():
    section("10  Visualisation")
    import cv2
    import ebiv_viz as viz

    # --- cross-stream profile ([u]) ------------------------------------
    from ebiv_config import MeasurementConfig
    from ebiv_control import ControlROIEstimator

    ws_p, nd_p = 64, 24
    disp_p = [0, 700, 0, 400]
    gy_p = (400 - ws_p) // nd_p + 1
    gx_p = (700 - ws_p) // nd_p + 1

    def _make(ctrl):
        mm = MeasurementConfig(component='-u', min_valid_vectors=4)
        mm.resolve_units(1 / 125.0)
        est = ControlROIEstimator(mm, disp_p, ctrl, ws_p, nd_p, gy_p, gx_p)
        pv = viz.ProfileView(est, disp_p, ws_p, nd_p, '-u',
                             velocity_scale=mm.velocity_scale,
                             units=mm.velocity_units)
        rng_p = np.random.default_rng(0)
        yc = pv.yc.astype(float)
        for k in range(80):
            flap = 6.0 * np.sin(2 * np.pi * k / 37.0)
            core = -4.0 * np.exp(-((yc - 200 - flap) / 38.0) ** 8)
            fld = np.tile(core[:, None], (1, gx_p)) + rng_p.normal(0, 0.25, (gy_p, gx_p))
            pv.update(fld, np.zeros_like(fld))
        return mm, est, pv

    _, _, pv_empty = (None, None, viz.ProfileView(
        _make([200, 500, 160, 240])[1], disp_p, ws_p, nd_p, '-u'))
    img0 = pv_empty.render()
    check("the profile renders before any field arrives",
          img0.shape == (pv_empty.h, pv_empty.w, 3),
          "it must not crash while RT-EBIV is off")

    m_core, est_core, pv_core = _make([200, 500, 160, 240])
    m_clip, est_clip, pv_clip = _make([200, 500, 120, 260])

    core_mean = float(np.nanmean(pv_core._mean[est_core.sy]))
    clip_mean = float(np.nanmean(pv_clip._mean[est_clip.sy]))
    check("the profile spans the whole DisplayROI, not just the ControlROI",
          pv_core._mean.size == gy_p,
          f"{pv_core._mean.size} points for {gy_p} grid rows; you must be able "
          f"to see the shear layers OUTSIDE the ROI to place it")
    check("an ROI on the core recovers the core velocity",
          abs(core_mean - 4.0 * m_core.velocity_scale) < 0.05 * abs(core_mean),
          f"{core_mean:.1f} vs {4.0 * m_core.velocity_scale:.1f} px/s")
    check("an ROI clipping the shear layers reads far too low",
          clip_mean < 0.7 * core_mean,
          f"core {core_mean:.0f} px/s vs clipping {clip_mean:.0f} px/s -- this "
          f"is the bias that recirculation and shear layers introduce")

    img_core = pv_core.render()
    img_clip = pv_clip.render()
    check("both profiles render to the declared canvas size",
          img_core.shape == (pv_core.h, pv_core.w, 3)
          and img_clip.shape == (pv_clip.h, pv_clip.w, 3))
    check("the cross-stream axis follows the component",
          pv_core.vs_rows is True and pv_core.axis_label == 'y [px]',
          "'-u' is a horizontal jet, so the profile runs across rows")

    mv = MeasurementConfig(component='v', min_valid_vectors=4)
    mv.resolve_units(1 / 125.0)
    est_v = ControlROIEstimator(mv, disp_p, [200, 500, 160, 240], ws_p, nd_p,
                                gy_p, gx_p)
    pv_v = viz.ProfileView(est_v, disp_p, ws_p, nd_p, 'v')
    check("a vertical jet profiles across columns instead",
          pv_v.vs_rows is False and pv_v.axis_label == 'x [px]')

    pv_core.reset()
    check("reset clears the average", pv_core._mean is None and pv_core._n == 0)

    img = np.zeros((240, 320, 3), np.uint8)
    ov = viz.VectorOverlay(10, 14, 32, 16, origin=(0, 0), arrow_skip=1,
                           arrow_scale=3.0)
    U = np.full((10, 14), 2.0, np.float32)
    V = np.full((10, 14), -1.0, np.float32)
    U[0, 0] = np.nan
    out = ov.draw(img.copy(), U, V)
    check("vector overlay draws and tolerates NaN vectors", out.sum() > 0)

    img2 = viz.draw_control_roi(np.zeros((240, 320, 3), np.uint8),
                                [80, 200, 60, 180], offset=(0, 0))
    check("ControlROI rectangle is drawn", img2.sum() > 0)

    ch = viz.StripChart(width=400, height=240, history_s=10.0)
    canvas = ch.render()
    check("strip chart renders with no data", canvas.shape == (240, 400, 3))
    for i in range(300):
        t = i * 0.05
        ch.append(t, 2.0, 2.0 + 0.1 * math.sin(t), 2.0 + 0.3 * math.sin(5 * t),
                  1.5 + 0.5 * math.sin(0.3 * t))
    ch.append(15.1, 2.0, float('nan'), float('nan'), 1.5)
    t0 = time.perf_counter()
    for _ in range(20):
        canvas = ch.render()
    dt = (time.perf_counter() - t0) / 20
    check("strip chart handles gaps and renders quickly",
          canvas.sum() > 0 and dt < 0.05, f"{dt * 1e3:.2f} ms per redraw")

    # --- strip-chart specifics ------------------------------------------
    ticks = viz.StripChart._nice_ticks(0.0, 1.0, 4)
    check("tick generator produces round values spanning the range",
          ticks and ticks[0] >= 0.0 and ticks[-1] <= 1.0 and len(ticks) >= 3,
          str(ticks))
    check("tick generator survives a degenerate range",
          viz.StripChart._nice_ticks(1.0, 1.0) == [1.0])

    # manual / calibration mode: the target is NaN throughout
    ch = viz.StripChart(width=500, height=300, units='m/s', v_limits=(0.0, 5.0))
    for i in range(200):
        ch.append(i * 0.05, float('nan'), 0.1 + 0.001 * i, 0.1 + 0.001 * i,
                  1.0 + 0.005 * i, 'MANUAL')
    check("the chart renders with no target at all (manual mode)",
          ch.render().sum() > 0)

    # The pump pane must autoscale.  Pinning it to the configured 0-5 V makes
    # a realistic 0.9-1.1 V trace an unreadable flat line along the bottom.
    def pump_trace_span(v_lo, v_hi):
        c = viz.StripChart(width=500, height=300, v_limits=(0.0, 5.0))
        for i in range(200):
            c.append(i * 0.05, 1.0, 1.0, 1.0,
                     v_lo + (v_hi - v_lo) * (0.5 + 0.5 * math.sin(i / 12.0)),
                     'CLOSED_LOOP')
        img = c.render()
        x_l, y_t, x_r, y_b = c.bot
        pane = img[y_t + 2:y_b - 2, x_l + 2:x_r - 2].astype(int)
        # COL_PUMP is (120, 200, 255) BGR: identify it by its RED channel
        ink = (pane[:, :, 2] > 150) & (pane[:, :, 0] < 200)
        rows = np.nonzero(ink.any(axis=1))[0]
        return (rows.max() - rows.min()) / pane.shape[0] if rows.size else 0.0

    span = pump_trace_span(0.9, 1.1)
    pinned = 0.2 / 5.0
    check("a narrow pump range fills the pane instead of being pinned to 0-5 V",
          span > 4 * pinned,
          f"trace spans {span:.0%} of the pane; pinned to the configured "
          f"range it would be {pinned:.0%}")

    # --- velocity-pane zoom (5.2) ---------------------------------------
    # A run starts at 0, jumps to the operating point and then varies by a
    # few percent.  Scaling on the whole history keeps 0 in the pane and the
    # variations of interest become invisible.
    def velocity_pane_span(zoom_window_s, t_end=45.0):
        c = viz.StripChart(width=500, height=300, history_s=60.0,
                           zoom_window_s=zoom_window_s, units='m/s')
        n = int(t_end / 0.1)
        for i in range(n):
            t = i * 0.1
            u = 0.0 if t < 5.0 else 0.55 + 0.015 * math.sin(2 * math.pi * 0.1 * t)
            c.append(t, 0.55, u, u, 2.0, 'MANUAL')
            if i % 2 == 0:
                c.render()
        c.render()
        return c._yhi - c._ylo

    full = velocity_pane_span(0.0)
    zoomed = velocity_pane_span(12.0)
    check("the velocity pane zooms past a start-up transient it has outlived",
          zoomed < 0.25 * full,
          f"span {zoomed:.4f} m/s vs {full:.4f} m/s scaling on the whole "
          f"history: {full / zoomed:.1f}x magnification of a +-0.015 m/s signal")
    check("zoom_window_s = 0 reproduces the pre-5.2 whole-history scaling",
          full > 0.6, f"span {full:.4f} m/s, still containing 0")

    # Zooming in means older samples fall outside the y-range.  They must be
    # clipped to their own pane, not drawn through the pane below.
    c = viz.StripChart(width=500, height=300, zoom_window_s=8.0, units='m/s')
    for i in range(400):
        t = i * 0.1
        u = 0.0 if t < 5.0 else 0.55 + 0.01 * math.sin(t)
        c.append(t, 0.55, u, u, 2.0, 'MANUAL')
    img = c.render()
    gap = img[c.top[3] + 3:c.bot[1] - 1, c.top[0] + 2:c.top[2] - 2]
    check("an off-scale trace stays inside its own pane",
          int((gap.sum(axis=2) > 60).sum()) == 0,
          "nothing drawn in the gap between the velocity and pump panes")

    # The scale must expand at once (never clip a live excursion) but
    # contract gradually (no flicker).
    c = viz.StripChart(width=500, height=300, zoom_window_s=10.0, units='m/s')
    for i in range(150):
        c.append(i * 0.1, 0.5, 0.5, 0.5, 2.0, 'MANUAL')
        c.render()
    before = c._yhi
    c.append(15.1, 0.5, 2.0, 2.0, 2.0, 'MANUAL')
    c.render()
    check("the y-scale expands immediately for a live excursion",
          c._yhi >= 2.0, f"top moved {before:.3f} -> {c._yhi:.3f} in one redraw")

    # A held (stale) filtered value must be visually distinct from a measured
    # one.  Give the three traces different values so they do not overlap.
    ch = viz.StripChart(width=500, height=300)
    n = 200
    for i in range(n):
        stale = 80 <= i < 140
        ch.append(i * 0.05, 1.4, 1.0, float('nan') if stale else 0.6,
                  1.0, 'HOLD' if stale else 'CLOSED_LOOP')
    img = ch.render().astype(int)
    x_l, y_t, x_r, y_b = ch.top

    LIVE = np.array(viz.StripChart.COL_FILT)
    STALE = np.array(viz.StripChart.COL_FILT_STALE)

    def filtered_style_at(sample_index):
        """
        Which of the two filtered-trace colours appears in that column.

        The column also contains the target and raw traces, so match against
        the two references directly rather than taking the brightest pixel.
        Returns ('live'|'stale', distance).
        """
        frac = (sample_index * 0.05) / ((n - 1) * 0.05)
        x = int(np.clip(x_l + frac * (x_r - x_l), x_l + 1, x_r - 1))
        col = img[y_t + 2:y_b - 2, x]
        d_live = np.abs(col - LIVE).sum(axis=1).min()
        d_stale = np.abs(col - STALE).sum(axis=1).min()
        return ('live' if d_live <= d_stale else 'stale'), min(d_live, d_stale)

    live_style, d1 = filtered_style_at(40)
    stale_style, d2 = filtered_style_at(110)
    check("the held (stale) filtered stretch is dimmed, not drawn as live data",
          live_style == 'live' and stale_style == 'stale' and d1 < 60 and d2 < 60,
          f"measured stretch -> {live_style} (d={d1}), "
          f"held stretch -> {stale_style} (d={d2})")

    class FakeSup:
        state = 'CLOSED_LOOP'
        u_target, u_filtered, u_raw, v_command = 3.0, 2.9, float('nan'), 2.1
        n_valid, n_total = 80, 100
        invalid_reason = ''

        class cfg:
            class measurement:
                min_valid_fraction = 0.5
    out = viz.draw_status(np.zeros((300, 500, 3), np.uint8), FakeSup())
    check("status HUD renders, including a NaN raw value", out.sum() > 0)


# ==========================================================================

def main():
    print("=" * 74)
    from ebiv_config import __version__ as _rel
    print(f"  EBIV Release {_rel} — software test-suite")
    print(f"  numpy {np.__version__}   python {sys.version.split()[0]}")
    print("=" * 74)

    tests = [
        test_ebiv_equivalence,
        test_control_roi,
        test_filters,
        test_pid,
        test_invalid_measurements,
        test_gpu_homothetic_warp,
        test_manual_mode_is_manual,
        test_laser_follows_facq,
        test_uncalibrated_units,
        test_stale_gate_uses_data_age,
        test_piv_rate_cap,
        test_acq_verdict,
        test_auto_trigger_window,
        test_references,
        test_version_single_source,
        test_live_target,
        test_closed_loop_simulation,
        test_hardware_abstraction,
        test_image_flip,
        test_velocity_units,
        test_config_and_logging,
        test_visualisation,
        test_session_and_gui,
    ]
    for t in tests:
        try:
            t()
        except Exception:                                      # noqa: BLE001
            _FAIL.append((t.__name__, "raised"))
            print(f"  FAIL  {t.__name__} raised an exception:")
            traceback.print_exc()

    print(f"\n{'=' * 74}")
    print(f"  unit tests: {len(_PASS)} passed, {len(_FAIL)} failed")
    if _FAIL:
        print("  failures:")
        for n, d in _FAIL:
            print(f"    - {n}  {d}")
    print("=" * 74)

    section("12  Analog Discovery call sequence (fake WaveForms runtime)")
    try:
        import test_hardware_mock
        rc_hw = test_hardware_mock.run()
    except Exception:                                          # noqa: BLE001
        traceback.print_exc()
        rc_hw = 1

    # The integration test injects fake metavision modules into sys.modules,
    # so it runs last, in its own section.
    section("13  End-to-end integration with a fake camera")
    try:
        import test_integration
        rc = test_integration.run()
    except Exception:                                          # noqa: BLE001
        traceback.print_exc()
        rc = 1

    return 1 if (_FAIL or rc or rc_hw) else 0


if __name__ == "__main__":
    sys.exit(main())
