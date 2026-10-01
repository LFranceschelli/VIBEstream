"""
The GPU correlation reached from the user-facing entry points:

    1. VIBE class, live:     vibe.velocity(..., gpu=True)          (fake camera)
    2. VIBE class, offline:  vibe.piv(frames, gpu=True)
                             vibe.piv_offline(dir, out, gpu=True)  (pyramidal)
    3. GUI / EBIV_Main path: run_session(...) with control.piv.use_gpu = True
                             (live stream, fake camera, headless OpenCV)

Each GPU result is compared with the same call on the CPU backend.

Runs WITHOUT a GPU: torch is required, CUDA is not.  The GPU backend
(ebiv_gpu.GPUCorrelator) is forced onto torch CPU tensors, so this checks the
code paths and the numerics, not CUDA itself, nor speed or latency.  On the
machine with the GPU, run tools/check_gpu.py as well.

    python tools/test_gpu_paths.py
"""

import os
import sys
import tempfile

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(_ROOT, 'lib'), os.path.join(_ROOT, 'tools')]

import numpy as np                  # noqa: E402

_PASS, _FAIL = [], []


def check(name, cond, detail=""):
    (_PASS if cond else _FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
    return bool(cond)


def particle_frames(n, dx, dy, h=240, w=320, n_part=2500, seed=3):
    """Gaussian particle images translated by (dx, dy) px per frame, uint8."""
    rng = np.random.default_rng(seed)
    p = rng.uniform([0, 0], [w, h], size=(n_part, 2))
    yy, xx = np.mgrid[0:h, 0:w]
    out = []
    for k in range(n):
        img = np.zeros((h, w), np.float32)
        for x, y in p + k * np.array([dx, dy]):
            xi, yi = int(x), int(y)
            if 2 <= xi < w - 3 and 2 <= yi < h - 3:
                sl = (slice(yi - 2, yi + 3), slice(xi - 2, xi + 3))
                img[sl] += np.exp(-((xx[sl] - x) ** 2 + (yy[sl] - y) ** 2) / 1.5)
        out.append(np.clip(img / img.max() * 255, 0, 255).astype(np.uint8))
    return out


def run():
    try:
        import torch                                                    # noqa: F401
    except ImportError:
        print("  torch is not installed: nothing to test (pip install torch).")
        return True
    import test_vibe as T          # first: installs the fake Metavision before ebiv_utils
    import ebiv_utils as U
    import ebiv_gpu
    import vibe as V

    print("=" * 76)
    print("  GPU backend through the VIBE class and the GUI session (torch, CPU tensors)")
    print("=" * 76)

    # --- force the GPU backend onto CPU tensors -------------------------------
    built = []
    Orig = ebiv_gpu.GPUCorrelator

    class CPUTensorGPU(Orig):
        def __init__(self, *a, **k):
            k['device'] = 'cpu'
            super().__init__(*a, **k)
            built.append(self)
    for mod in (U, V):
        if hasattr(mod, 'GPUCorrelator'):
            mod.GPUCorrelator = CPUTensorGPU
    U.GPUCorrelator = CPUTensorGPU
    U._HAS_GPU = True

    tmp = tempfile.mkdtemp(prefix="vibe_gpu_")
    from vibe import VIBE
    vibe = VIBE(output_dir=tmp, max_events_per_pixel=1)
    vibe_s = VIBE(output_dir=tmp, max_events_per_pixel=1, smooth_sigma=0.75)

    # --- 1. live, VIBE.velocity -----------------------------------------------
    print("\n[1] VIBE.velocity(gpu=True), fake camera")
    n0 = len(built)
    g = list(vibe_s.velocity(f=T.F_LASER, n=8, window=32, step=16, gpu=True))
    c = list(vibe_s.velocity(f=T.F_LASER, n=8, window=32, step=16, gpu=False))
    check("GPU backend built for the live stream", len(built) > n0)
    d = max(np.nanmax(np.abs(a.u_raw - b.u_raw)) for a, b in zip(g, c))
    check("live fields (frames smoothed 0.75 px, as in the paper): GPU = CPU",
          len(g) == len(c) == 8 and d < 1e-4, f"max |du| {d:.1e} px")
    # Binary, unsmoothed frames give integer-valued correlation planes, so a
    # window can have two EXACTLY equal highest peaks; the backends may then
    # pick different ones.  Not a bug of either: those vectors are ambiguous.
    g = list(vibe.velocity(f=T.F_LASER, n=8, window=32, step=16, gpu=True))
    c = list(vibe.velocity(f=T.F_LASER, n=8, window=32, step=16, gpu=False))
    nd = sum(int((np.abs(a.u_raw - b.u_raw) > 0.01).sum()) for a, b in zip(g, c))
    nt = sum(a.u_raw.size for a in g)
    print(f"        (binary, unsmoothed frames: {nd} of {nt} vectors differ: tied "
          f"correlation peaks, which smoothing removes)")

    # --- 2. offline, VIBE.piv and VIBE.piv_offline ------------------------------
    print("\n[2] VIBE.piv(gpu=True) and VIBE.piv_offline(gpu=True)")
    frames = particle_frames(6, 3.3, -1.7)
    from vibe import Frame
    fr = [Frame(i * 5000, im, 0.0, i) for i, im in enumerate(frames)]
    n0 = len(built)
    fg = vibe.piv(fr, window=32, step=16, gpu=True, f=200.0)
    fc = vibe.piv(fr, window=32, step=16, gpu=False, f=200.0)
    check("GPU backend built for VIBE.piv", len(built) > n0)
    ug = np.nanmedian([f.u for f in fg]); vg = np.nanmedian([f.v for f in fg])
    d = max(np.nanmax(np.abs(a.u - b.u)) for a, b in zip(fg, fc))
    check("VIBE.piv: GPU = CPU, and the imposed shift is recovered",
          d < 1e-3 and abs(ug - 3.3) < 0.1 and abs(vg + 1.7) < 0.1,
          f"max |du| {d:.1e} px; median (u, v) = ({ug:.2f}, {vg:.2f}), imposed (3.30, -1.70)")

    import cv2
    from scipy.io import loadmat
    img_dir = os.path.join(tmp, "img")
    os.makedirs(img_dir)
    for i, im in enumerate(particle_frames(8, 2.4, 1.1, seed=5)):
        cv2.imwrite(os.path.join(img_dir, f"frame_{i:05d}.tif"), im)
    res = {}
    for gpu in (True, False):
        out = os.path.join(tmp, f"out_{gpu}")
        n0 = len(built)
        vibe.piv_offline(img_dir, out, window=32, step=16, levels=2, gpu=gpu)
        if gpu:
            check("GPU backend built for the offline pyramidal PIV", len(built) > n0)
        m = loadmat(os.path.join(out, "offline_piv_average.mat"))
        res[gpu] = (m['U_avg'], m['V_avg'])
    du = np.nanmax(np.abs(res[True][0] - res[False][0]))
    ui = np.nanmedian(res[True][0]); vi = np.nanmedian(res[True][1])
    check("offline pyramidal PIV: GPU = CPU, and the imposed shift is recovered",
          du < 5e-3 and abs(abs(ui) - 2.4) < 0.15 and abs(abs(vi) - 1.1) < 0.15,
          f"max |dU| {du:.1e} px; median |U|, |V| = {abs(ui):.2f}, {abs(vi):.2f} (imposed 2.40, 1.10)")

    # --- 3. GUI / EBIV_Main path: run_session with control.piv.use_gpu -----------
    print("\n[3] run_session(): live stream with control.piv.use_gpu = True")
    import ebiv_session
    from ebiv_config import Session
    shown = []
    cv2.imshow = lambda name, img: shown.append(name)
    # 'ebiv' mode starts with the real-time PIV OFF: [p] switches it on
    cv2.waitKey = lambda _: (ord('p') if len(shown) == 2 else
                             ord('q') if len(shown) >= 40 else 0xFF)
    cv2.destroyAllWindows = lambda *a: None
    cv2.destroyWindow = lambda *a: None
    s = Session()
    r, cfg = s.run, s.control
    r.output_base_folder, r.acq_name = tmp, "gpu"
    r.mode, r.run_mode = 'stream', 'ebiv'
    r.f_acq, r.trigger_mode = T.F_LASER, 'auto'
    cfg.piv.window_size, cfg.piv.node_distance = 32, 16
    cfg.piv.use_gpu = True
    r.frame_smooth_sigma = 0.75
    calls = []
    orig_corr = CPUTensorGPU.correlate

    def spy(self, *a, **k):
        calls.append(1)
        return orig_corr(self, *a, **k)
    CPUTensorGPU.correlate = spy
    n0 = len(built)
    ebiv_session.run_session(s)
    CPUTensorGPU.correlate = orig_corr
    check("the GUI's 'GPU correlation' tick builds the GPU backend for the stream",
          len(built) > n0)
    check("live fields correlated on it", len(calls) > 5, f"{len(calls)} correlations")

    print("\n" + "=" * 76)
    print(f"  {len(_PASS)} passed, {len(_FAIL)} failed")
    if _FAIL:
        print("  FAILED: " + "; ".join(_FAIL))
    print("=" * 76)
    return not _FAIL


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    sys.exit(0 if run() else 1)
