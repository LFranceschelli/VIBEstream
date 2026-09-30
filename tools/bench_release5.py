"""
EBIV Release 5.0 — CPU benchmark, Release 4.0 vs Release 5.0.

The numbers quoted in AUDIT_R4_to_R5.md were measured on a slow 2-core cloud
machine and DO NOT transfer to your workstation.  Run this on the machine that
will do the experiment:

    python bench_release5.py

It reports, for the real-time correlation path:
  * per-stage cost for Release 4.0 and Release 5.0
  * whether Release 5.0 reproduces Release 4.0 BIT-EXACTLY on the same input
  * the cost of the optional sub-pixel and correlation-quality stages
  * a scan over fft_workers, so you can pick the value for YOUR core count
  * the maximum sustainable field rate for each configuration

Use it to choose fft_workers and to decide how big a DisplayROI you can afford
at your acquisition frequency.
"""

# Run from anywhere: put the parent folder (the library) on the path.
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)                      # EBIV_Main.py
_sys.path.insert(0, _os.path.join(_ROOT, 'lib'))  # the library


import os
import sys
import time
import platform

import numpy as np
from scipy.fft import rfft2, irfft2

from ebiv_piv import CPUCorrelator


# --------------------------------------------------------------------------
#  Release 4.0 real-time path, copied verbatim for a fair comparison
# --------------------------------------------------------------------------

def _extract_windows_R4(frame, ws, nd, gy, gx):
    W = np.empty((gy * gx, ws, ws), dtype=np.float32)
    i = 0
    for y in range(gy):
        for x in range(gx):
            W[i] = frame[y * nd:y * nd + ws, x * nd:x * nd + ws]
            i += 1
    return W


class _P:
    def __init__(self):
        self.t = {}

    class _T:
        def __init__(self, p, s):
            self.p, self.s = p, s

        def __enter__(self):
            self.t0 = time.perf_counter()
            return self

        def __exit__(self, *a):
            self.p.t.setdefault(self.s, []).append(time.perf_counter() - self.t0)

    def measure(self, s):
        return self._T(self, s)


def r4_correlate(frames, ws, nd, prof):
    h, w = frames[0].shape
    gy = (h - ws) // nd + 1
    gx = (w - ws) // nd + 1
    with prof.measure("window_extraction"):
        W1 = _extract_windows_R4(frames[0], ws, nd, gy, gx).astype(np.float32)
        W2 = _extract_windows_R4(frames[1], ws, nd, gy, gx).astype(np.float32)
    with prof.measure("mean_subtract"):
        W1 -= W1.mean(axis=(1, 2), keepdims=True)
        W2 -= W2.mean(axis=(1, 2), keepdims=True)
    with prof.measure("fft_forward"):
        F1, F2 = rfft2(W1), rfft2(W2)
    with prof.measure("cross_power"):
        cp = np.conj(F1) * F2
    with prof.measure("fft_inverse"):
        R = np.real(irfft2(cp))
    with prof.measure("fftshift"):
        R = np.fft.fftshift(R, axes=(1, 2))
    with prof.measure("peak_finding"):
        Rf = R.reshape(R.shape[0], -1)
        pk = np.argmax(Rf, axis=1)
        cy, cx = np.unravel_index(pk, (ws, ws))
        U = (cx - ws // 2).reshape(gy, gx).astype(np.float32)
        V = (cy - ws // 2).reshape(gy, gx).astype(np.float32)
    return U, V


def make_pair(h, w, sx, sy, n=40000, seed=0):
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


def _time(fn, frames, reps):
    fn(frames, _P())
    p = _P()
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn(frames, p)
    return (time.perf_counter() - t0) / reps, p, out


def bench_case(h, w, ws, nd, reps=10, workers_scan=(1, 2, 4, 8)):
    f1, f2 = make_pair(h, w, 5.0, -3.0, seed=1)
    frames = [f1, f2]
    gy, gx = (h - ws) // nd + 1, (w - ws) // nd + 1
    print(f"\n{'=' * 78}")
    print(f" {w} x {h} px   ws={ws}  nd={nd}   grid {gy} x {gx} = {gy * gx} vectors")
    print('=' * 78)

    t4, p4, (U4, V4) = _time(lambda fr, p: r4_correlate(fr, ws, nd, p), frames, reps)

    c5 = CPUCorrelator(ws, nd, (h, w), fft_workers=1, subpixel=False,
                       compute_quality=False)
    t5, p5, (U5, V5, _) = _time(lambda fr, p: c5.correlate(fr, p), frames, reps)
    exact = np.array_equal(U4, U5) and np.array_equal(V4, V5)

    keys = sorted(set(p4.t) | set(p5.t),
                  key=lambda k: -float(np.mean(p4.t.get(k, [0]))))
    print(f" {'stage':<22s} {'R4 [ms]':>10s} {'R5 [ms]':>10s} {'speed-up':>10s}")
    print(' ' + '-' * 54)
    for k in keys:
        a = float(np.mean(p4.t[k])) * 1e3 if k in p4.t else 0.0
        b = float(np.mean(p5.t[k])) * 1e3 if k in p5.t else 0.0
        s = f"x{a / b:.2f}" if a > 0 and b > 0 else "-"
        print(f" {k:<22s} {a:>10.3f} {b:>10.3f} {s:>10s}")
    print(' ' + '-' * 54)
    print(f" {'TOTAL':<22s} {t4 * 1e3:>10.3f} {t5 * 1e3:>10.3f} "
          f"{'x%.2f' % (t4 / t5):>10s}")
    print(f" {'max field rate [Hz]':<22s} {1 / t4:>10.1f} {1 / t5:>10.1f}")
    print(f" numerics: {'BIT-EXACT vs Release 4.0' if exact else '*** DIFFERS ***'}")

    print(f"\n {'configuration':<44s} {'ms':>9s} {'Hz':>8s}")
    print(' ' + '-' * 62)
    print(f" {'R4 baseline':<44s} {t4 * 1e3:>9.3f} {1 / t4:>8.1f}")
    for wk in workers_scan:
        if wk > (os.cpu_count() or 1) * 2:
            continue
        for sub, qual in ((False, False), (True, False), (True, True)):
            c = CPUCorrelator(ws, nd, (h, w), fft_workers=wk, subpixel=sub,
                              compute_quality=qual)
            t, _, _ = _time(lambda fr, p: c.correlate(fr, p), frames, reps)
            tag = (f"R5 workers={wk}"
                   + (" +subpixel" if sub else "")
                   + (" +quality" if qual else ""))
            print(f" {tag:<44s} {t * 1e3:>9.3f} {1 / t:>8.1f}")

    # fast_mean_removal: faster, not bit-exact — report both facts together
    cA = CPUCorrelator(ws, nd, (h, w), fft_workers=1, subpixel=False,
                       compute_quality=False, fast_mean_removal=False)
    cB = CPUCorrelator(ws, nd, (h, w), fft_workers=1, subpixel=False,
                       compute_quality=False, fast_mean_removal=True)
    tA, _, (UA, VA, _) = _time(lambda fr, p: cA.correlate(fr, p), frames, reps)
    tB, _, (UB, VB, _) = _time(lambda fr, p: cB.correlate(fr, p), frames, reps)
    nd_ = int(np.sum((UA != UB) | (VA != VB)))
    print(f"\n optional fast_mean_removal: x{tA / tB:.2f} faster, "
          f"{nd_}/{UA.size} integer peaks differ ({100 * nd_ / UA.size:.2f}%)")
    return t4, t5


def main():
    print("=" * 78)
    print("  EBIV Release 5.0 CPU benchmark")
    print(f"  {platform.processor() or platform.machine()}  |  "
          f"{os.cpu_count()} logical cores  |  numpy {np.__version__}  |  "
          f"python {sys.version.split()[0]}")
    print("=" * 78)

    cases = [
        (720, 1280, 48, 24, 10),    # full EVK4 sensor
        (400, 900, 48, 24, 20),     # a cropped jet DisplayROI
        (200, 400, 32, 16, 40),     # a small, fast ROI
    ]
    for h, w, ws, nd, reps in cases:
        bench_case(h, w, ws, nd, reps)

    print(f"\n{'=' * 78}")
    print("  Reading this:")
    print("   * The correlation is FFT-bound. Cropping the DisplayROI is the")
    print("     single biggest lever: cost scales with the number of windows.")
    print("   * fft_workers>1 helps on large grids and hurts on small ones,")
    print("     because the thread hand-off costs more than the transform.")
    print("     Pick the fastest row above for YOUR ROI.")
    print("   * The field rate here is the correlation only. The end-to-end")
    print("     measurement-to-actuation latency is reported at the end of")
    print("     every live run; use that number for control design.")
    print("=" * 78)


if __name__ == "__main__":
    main()
