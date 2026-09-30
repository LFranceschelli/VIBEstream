"""
vibe_train — end-to-end test: .raw -> LR + HR fields -> trained estimators.

A fake Metavision SDK (from test_vibe.py) serves a synthetic pulsed-laser
recording in which ~2500 particles are ADVECTED by a convected two-scale,
divergence-free velocity field:
    large scale  (lambda_x = 160 px): resolved by the LR windows (32 px)
    small scale  (lambda_x =  40 px): resolved only by the HR windows (16 px)
plus stochastic amplitude modulation, so the flow is time-resolved but not
periodic.  Checks:
    1. MultiFrameCorrelator('forward') == ebiv_utils.process_offline_piv
    2. both stencils recover a known uniform displacement
    3. build_training_set: shapes, time stamps one period apart
    4. train_from_set: all three methods beat cubic interpolation
    5. centred vs forward stencil (time alignment of HR with LR)
    6. check_processing flags a live setting that differs from training
    7. online: the trained model runs on VIBE.velocity() fields

    python tools/test_hr_train.py        (~2-4 min)
"""

import os
import sys
import glob
import tempfile

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(_ROOT, 'lib'), os.path.join(_ROOT, 'tools')]

import numpy as np                 # noqa: E402
import test_vibe as T              # noqa: E402  (installs the fake Metavision)

_PASS, _FAIL = [], []


def check(name, cond, detail=""):
    (_PASS if cond else _FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
    return bool(cond)


# ==========================================================================
#  synthetic advected-particle recording
# ==========================================================================
W, H = T.W, T.H                      # 320 x 240
PERIOD, PHASE = T.PERIOD, T.PHASE    # 5000 us, 1300 us (200 Hz)
U0 = 4.0                             # px / frame


def flow(x, y, t, amp):
    """
    Displacement per frame at (x, y), frame time t: uniform U0 plus two
    convected eddy rows from a streamfunction (DIVERGENCE-FREE, v = 0 at the
    walls y = 0, H), so particles stay uniformly seeded over time.
        psi_i = A_i sin(k_i (x - U0 t) + phi_i) sin(m_i pi y / H)
    """
    k1, k2 = 2 * np.pi / 160, 2 * np.pi / 40
    m1, m2 = 3, 12                       # m pi / H ~ k: roughly round eddies
    a1, a2 = amp
    u = np.full(np.broadcast(x, y).shape, U0, dtype=float)
    v = np.zeros_like(u)
    for a_, k_, m_, ph in ((a1, k1, m1, 0.0), (a2, k2, m2, 1.0)):
        ky = m_ * np.pi / H
        A = a_ / ky                      # u amplitude = a_
        xi = k_ * (x - U0 * t) + ph
        u = u + A * ky * np.sin(xi) * np.cos(ky * y)
        v = v - A * k_ * np.cos(xi) * np.sin(ky * y)
    return u, v


def make_recording(n_frames, seed=5, n_part=2500, ev_per=5):
    rng = np.random.default_rng(seed)
    p = rng.uniform([0, 0], [W, H], size=(n_part, 2))
    amp = np.array([1.2, 0.6])
    amp_state = amp.copy()
    amps = []
    xs, ys, ts = [], [], []
    for k in range(n_frames):
        pos = np.repeat(p, ev_per, axis=0) + rng.normal(0, 0.7, (n_part * ev_per, 2))
        ok = (pos[:, 0] >= 0) & (pos[:, 0] < W) & (pos[:, 1] >= 0) & (pos[:, 1] < H)
        pos = pos[ok]
        xs.append(np.rint(pos[:, 0]).clip(0, W - 1))
        ys.append(np.rint(pos[:, 1]).clip(0, H - 1))
        ts.append(k * PERIOD + PHASE + np.abs(rng.normal(0, 60, len(pos))))
        # advect with the (slowly modulated) field at this frame's time
        amp_state = amp + 0.97 * (amp_state - amp) + 0.08 * rng.standard_normal(2)
        amps.append(amp_state.copy())
        u, v = flow(p[:, 0], p[:, 1], k, amp_state)                 # midpoint (RK2) step
        u, v = flow(np.mod(p[:, 0] + u / 2, W), p[:, 1] + v / 2, k + 0.5, amp_state)
        p[:, 0] = np.mod(p[:, 0] + u, W)
        p[:, 1] = np.clip(p[:, 1] + v, 0, H - 1e-3)
    n_noise = int(n_frames * PERIOD * 1e-6 * 20_000)
    xs.append(rng.integers(0, W, n_noise)); ys.append(rng.integers(0, H, n_noise))
    ts.append(rng.uniform(0, n_frames * PERIOD, n_noise))
    x, y, t = (np.concatenate(a) for a in (xs, ys, ts))
    o = np.argsort(t, kind='stable')
    ev = np.zeros(len(t), T.EV_DTYPE)
    ev['x'], ev['y'], ev['t'], ev['p'] = x[o], y[o], t[o].astype(np.int64), 1
    return ev, np.array(amps)


def truth_box(k, hx, hy, ws, amps):
    """True displacement between frames k and k+1, box-averaged over each HR window."""
    s = (np.arange(4) + 0.5) / 4 * ws - ws / 2
    ox, oy = np.meshgrid(s, s)
    X = hx[..., None, None] + ox
    Y = hy[..., None, None] + oy
    # the particle whose displacement midpoint is at (X, Y) moved by u(mid, k + 1/2)
    u1, v1 = flow(X, Y, k + 0.5, amps[k])
    return u1.mean(axis=(-1, -2)), v1.mean(axis=(-1, -2))


def install(ev, n_frames):
    T.BLOCK = ev
    T.BLOCK_T = np.ascontiguousarray(ev['t'])
    T.BLOCK_S = n_frames * PERIOD / 1e6


# ==========================================================================
def run():
    import vibe_train as VT
    import vibe_hr as HH
    from vibe import VIBE
    import ebiv_utils as U

    print("=" * 76)
    print("  vibe_train — .raw -> LR/HR -> estimators, end-to-end (fake camera)")
    print("=" * 76)
    tmp = tempfile.mkdtemp(prefix="vibe_train_")
    global AMPS
    NF = 1000
    ev, AMPS = make_recording(NF)
    install(ev, NF)
    raw = os.path.join(tmp, 'synth.raw')
    open(raw, 'wb').write(b'FAKE')
    vibe = VIBE(output_dir=tmp, max_events_per_pixel=1, smooth_sigma=0.75)
    F = 1e6 / PERIOD

    # --- 1. multi-frame correlator ---------------------------------------------
    print("\n[1] multi-frame correlator")
    frs = list(vibe.frames_from_raw(raw, f=F, n=8, normalize='clip'))
    import cv2
    tif = os.path.join(tmp, 'tif'); os.makedirs(tif)
    for fr in frs[:5]:
        cv2.imwrite(os.path.join(tif, f"f_{fr.index:05d}.tif"), fr.image)
    outd = os.path.join(tmp, 'piv'); os.makedirs(outd)
    U.process_offline_piv(tif, outd, window_size=32, node_distance=8, apply_validation=False,
                          pyramid_levels=2)
    from scipy.io import loadmat
    ref = loadmat(sorted(glob.glob(os.path.join(outd, 'piv_snapshot_*.mat')))[0])
    mf = VT.MultiFrameCorrelator(frs[0].image.shape, 32, 8, 2, 'forward', 'planes', False)
    Uf, Vf, _ = mf.correlate([f.image.astype(np.float32) for f in frs[:3]])
    dmax = max(np.abs(Uf - ref['U']).max(), np.abs(Vf - ref['V']).max())
    check("no predictor + planes + forward == ebiv_utils.process_offline_piv", dmax < 1e-3,
          f"max |diff| {dmax:.1e} px")

    rng = np.random.default_rng(1)
    base = np.zeros((160, 240), np.float32)
    pts = rng.uniform([2, 2], [238, 158], size=(700, 2))

    def img(shift):
        im = np.zeros_like(base)
        for x, y in pts + shift:
            xi, yi = int(np.floor(x)), int(np.floor(y))
            if 0 <= xi < 239 and 0 <= yi < 159:
                fx, fy = x - xi, y - yi
                im[yi, xi] += (1 - fx) * (1 - fy); im[yi, xi + 1] += fx * (1 - fy)
                im[yi + 1, xi] += (1 - fx) * fy; im[yi + 1, xi + 1] += fx * fy
        return cv2.GaussianBlur(im, (0, 0), 1.0)
    print("        uniform shift, window 32 px, 2 separations: median |error| (px)")
    for d in (np.array([2.3, -1.4]), np.array([9.3, -4.6])):
        row = []
        for st in ('forward', 'centred'):
            for comb in ('planes', 'peaks'):
                for pred in (False, True):
                    mc = VT.MultiFrameCorrelator(base.shape, 32, 16, 2, st, comb, pred)
                    ims = [img(k * d) for k in range(mc.n_frames)]
                    u_, v_, _ = mc.correlate(ims)
                    inner = (slice(1, -1), slice(1, -1))
                    e = float(np.median(np.hypot(u_[inner] - d[0], v_[inner] - d[1])))
                    row.append((st, comb, pred, e))
        print(f"          d = ({d[0]}, {d[1]}):  " + "  ".join(
            f"{a[:4]}/{b[:4]}/{'P' if c else '-'} {e:.3f}" for a, b, c, e in row))
        if d[0] < 4:
            check("small shift: every variant within 0.2 px", max(r[3] for r in row) < 0.2)
        else:
            check("shift > window/4 at the longest separation: predictor needed, and sufficient",
                  all(r[3] < 0.2 for r in row if r[2]) and
                  any(r[3] > 1.0 for r in row if not r[2]))

    # HR accuracy against the known flow of the synthetic recording
    print("        synthetic recording, HR window 16 px step 4, 2 separations: rms error vs")
    print("        the box-averaged true displacement (px/frame), 40 fields")
    frs_all = list(vibe.frames_from_raw(raw, f=F, n=60, normalize='clip'))
    k_of = lambda fr: int(round((fr.t_us - PHASE) / PERIOD))             # noqa: E731
    acc = {}
    for st in ('forward', 'centred'):
        for comb in ('planes', 'peaks'):
            mc = VT.MultiFrameCorrelator(frs_all[0].image.shape, 16, 4, 2, st, comb, True)
            hx = np.arange(mc.gx) * 4 + 8.0
            hy = np.arange(mc.gy) * 4 + 8.0
            HX, HY = np.meshgrid(hx, hy)
            inner = (slice(2, -2), slice(2, -2))
            errs = []
            for i in range(10, 50):
                win = frs_all[i:i + mc.n_frames]
                u_, v_, _ = mc.correlate([f.image for f in win])
                u_, v_, _ = U.universal_outlier_detection(u_, v_, 2.0, 0.1)
                k = k_of(win[mc.anchor])                       # LR pair (k, k+1)
                tu, tv = truth_box(k, HX, HY, 16, AMPS)
                errs.append(np.sqrt(((u_ - tu) ** 2 + (v_ - tv) ** 2)[inner].mean()))
            acc[(st, comb)] = float(np.mean(errs))
    print("          " + "  ".join(f"{a}/{b}: {e:.3f}" for (a, b), e in acc.items()))
    best = min(acc, key=acc.get)
    check("centred stencil more accurate than forward for the LR-pair time",
          min(acc[('centred', c)] for c in ('planes', 'peaks')) <
          min(acc[('forward', c)] for c in ('planes', 'peaks')), f"best: {best}")

    # --- 2. training set --------------------------------------------------------
    print("\n[2] build_training_set")
    sets = VT.TrainingSettings(f_hz=F, lr_window=32, lr_step=32, hr_window=16, hr_step=4,
                               hr_levels=2)
    ds = VT.build_training_set(vibe, raw, sets)
    nt = len(ds['lr_u'])
    check("LR and HR fields for every frame pair", nt >= NF - 10 and len(ds['hr_u']) == nt,
          f"{nt} fields from {NF} frames; LR {ds['lr_u'].shape[1:]}, HR {ds['hr_u'].shape[1:]}")
    dt = np.diff(ds['t_us'])
    check("consecutive fields exactly one period apart", np.all(dt == PERIOD) and ds['n_gaps'] == 0)
    mean_u = float(ds['hr_u'].mean())
    check("HR mean streamwise displacement ~ U0", abs(mean_u - U0) < 0.3, f"{mean_u:.2f} px/frame")
    p = VT.save_training_set(os.path.join(tmp, 'ts.npz'), ds)
    ds2 = VT.load_training_set(p)
    check("training set save / load", np.array_equal(ds2['hr_v'], ds['hr_v'])
          and ds2['settings']['hr_window'] == 16)
    VT.export_matlab(os.path.join(tmp, 'mat'), ds)
    lm = HH.load_mat_dataset(os.path.join(tmp, 'mat', 'LR.mat'))
    check("MATLAB export readable with load_mat_dataset",
          np.allclose(lm['U'], np.transpose(ds['lr_u'], (0, 2, 1))))

    # --- 3. train + evaluate ----------------------------------------------------
    print("\n[3] train_from_set (split 450 | 50 | 150 | 50 | 200)")
    split = VT.SplitSettings(n_train=450, n_val=150, n_test=200, gap=50)
    model, rep = VT.train_from_set(ds, split)
    print(f"        {model.summary()}")
    print("        delta vs HR-LOR:  cubic {delta_cubic:.4f} | KF {delta_kf:.4f} | LSE {delta_lse:.4f}"
          " | LSE+VR {delta_lse_vr:.4f}".format(**rep))
    print("        TKE / TKE(HR-LOR): KF {tke_ratio_kf:.2f} | LSE {tke_ratio_lse:.2f} | "
          "LSE+VR {tke_ratio_lse_vr:.2f}".format(**rep))
    check("all three methods beat cubic interpolation",
          max(rep['delta_kf'], rep['delta_lse'], rep['delta_lse_vr']) < rep['delta_cubic'])
    mp = model.save(os.path.join(tmp, 'model.npz'))
    m2 = HH.HRModel.load(mp)
    check("model keeps the LR processing record",
          m2.meta['lr_processing']['window'] == 32 and m2.meta['lr_processing']['smooth_sigma'] == 0.75)

    # --- 4. stencil / time alignment ---------------------------------------------
    print("\n[4] HR stencil: centred vs forward")
    sets_f = VT.TrainingSettings(**{**sets.__dict__, 'hr_stencil': 'forward'})
    ds_f = VT.build_training_set(vibe, raw, sets_f)
    n_f = len(ds_f['lr_u'])
    for k in list(ds_f):
        if isinstance(ds_f[k], np.ndarray) and ds_f[k].shape[:1] == (n_f,):
            ds_f[k] = ds_f[k][:min(n_f, nt)]
    model_f, rep_f = VT.train_from_set(ds_f, split)
    # what matters: the HR estimate against the TRUE flow at the LR-pair time
    def err_vs_truth(m, d_, method='kf'):
        te = split.slices(len(d_['lr_u']))[2]
        X = HH.KalmanEstimator(m, method=method).run(d_['lr_u'][te], d_['lr_v'][te])
        eu, ev = m.reconstruct(X)
        inner = (slice(2, -2), slice(2, -2))
        e = []
        for j, t_us in enumerate(d_['t_us'][te]):
            k = int(round((t_us - PHASE) / PERIOD)) - 1                  # pair (k, k+1)
            tu, tv = truth_box(k, d_['hr_x'], d_['hr_y'], sets.hr_window, AMPS)
            e.append(np.sqrt(((eu[j] - tu) ** 2 + (ev[j] - tv) ** 2)[inner].mean()))
        return float(np.mean(e))
    te_ = split.slices(nt)[2]
    cu_, cv_ = HH.cubic_baseline(ds['lr_u'][te_], ds['lr_v'][te_], ds['lr_x'], ds['lr_y'],
                                 ds['hr_x'], ds['hr_y'])
    e_cub = []
    for j, t_us in enumerate(ds['t_us'][te_]):
        k = int(round((t_us - PHASE) / PERIOD)) - 1
        tu, tv = truth_box(k, ds['hr_x'], ds['hr_y'], sets.hr_window, AMPS)
        e_cub.append(np.sqrt(((cu_[j] - tu) ** 2 + (cv_[j] - tv) ** 2)[2:-2, 2:-2].mean()))
    e_c = {mm: err_vs_truth(model, ds, mm) for mm in ('kf', 'lse', 'lse_vr')}
    e_f = err_vs_truth(model_f, ds_f)
    print(f"        rms error vs TRUE flow (px/frame): KF centred {e_c['kf']:.3f} | KF forward "
          f"{e_f:.3f} | LSE {e_c['lse']:.3f} | LSE+VR {e_c['lse_vr']:.3f} | cubic {np.mean(e_cub):.3f}")
    check("trained on centred HR: all methods closer to the true flow than cubic",
          max(e_c.values()) < np.mean(e_cub))

    # --- 5. processing consistency ------------------------------------------------
    print("\n[5] check_processing")
    live = dict(f_hz=F, roi=None, window=32, step=32, **vibe.processing_settings())
    check("consistent live settings -> no mismatch", VT.check_processing(model, live) == [])
    bad = dict(live, window=48, smooth_sigma=None)
    mism = VT.check_processing(model, bad)
    check("different window / smoothing flagged", len(mism) == 2, "; ".join(mism))

    # --- 6. online on VIBE fields ----------------------------------------------------
    print("\n[6] online: model on VIBE.velocity() fields")
    est = HH.KalmanEstimator(model, method='lse_vr', steady_state=True)
    fl = list(vibe.velocity(f=F, n=60, window=32, step=32, trigger='auto'))
    for fld in fl:
        x = est.step_timed(fld.t_us, fld.u, fld.v, PERIOD)
    uh, vh = est.field()
    check("live fields accepted, HR field on the HR grid",
          uh.shape == ds['hr_u'].shape[1:] and np.all(np.isfinite(x)),
          f"{len(fl)} fields, {est.n_skipped} bridged")

    print("\n" + "=" * 76)
    print(f"  {len(_PASS)} passed, {len(_FAIL)} failed")
    if _FAIL:
        print("  FAILED: " + "; ".join(_FAIL))
    print("=" * 76)
    return not _FAIL


if __name__ == "__main__":
    import logging
    logging.getLogger().setLevel(logging.WARNING)
    sys.exit(0 if run() else 1)
