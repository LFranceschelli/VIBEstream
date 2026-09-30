"""
Resolution-enhancement parameters computed OUTSIDE VibeStream (vibe_import).

    1. geometry: MATLAB-layout, Y-up, physical-unit fields land on the live
       grid with the right orientation, signs and px/frame scale
    2. operators: a model trained in the FILE convention, written in the
       format of tools/matlab/vibe_export_model.m, imported -> the live model
       gives the file model's estimate, converted (exact linear algebra)
    3. partial operator files (KF only / LSE only); wrong grids are refused
    4. the GUI path: ebiv_session.train_hr_model with source 'fields' and
       'operators'; HRLive accepts the result; overlay at the exact nodes

    python tools/test_hr_import.py

The Octave run of the real scripts + vibe_export_model.m was checked
separately (agreement 1e-15 - 2e-14, see the release notes).
"""

import os
import sys
import tempfile

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(_ROOT, 'lib'), os.path.join(_ROOT, 'tools')]

import numpy as np                  # noqa: E402
from scipy.io import savemat        # noqa: E402
import vibe_hr as H                 # noqa: E402
import vibe_import as VI            # noqa: E402
import hr_synth as S                # noqa: E402

_PASS, _FAIL = [], []


def check(name, cond, detail=""):
    (_PASS if cond else _FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
    return bool(cond)


F_HZ = 50.0                                   # hr_synth dt = 0.02
LIVE = dict(f_hz=F_HZ, roi=[0, 512, 0, 256], window=32, step=32, flip_x=False, flip_y=False,
            phase_locked=True)
EXT = VI.ExternalSettings(y_up=True, velocity_units='unit/s', f_hz=0.0)


def write_operator_file(path, m, lr_file, hr_file, method):
    """What vibe_export_model.m writes, from a model trained on FILE-layout arrays."""
    def mat_modes(phi, shp):             # rows C-order over shp -> MATLAB column-major
        n = int(np.prod(shp))
        out = []
        for blk in (phi[:n], phi[n:]):
            A = blk.reshape(tuple(shp) + (-1,))
            out.append(A.reshape(n, -1, order='F'))
        return np.vstack(out)
    n_lr, n_hr = int(np.prod(m.lr_shape)), int(np.prod(m.hr_shape))
    S_ = dict(vibe_format=VI.FORMAT, method=method, r=m.r, r_LR=m.r_lr,
              PhiLR=mat_modes(m.phi_lr, m.lr_shape), SigmaLR=m.sigma_lr_all[:, None],
              PhiMF=mat_modes(m.phi_hr, m.hr_shape), SigmaMF=m.sigma_hr_all[:, None],
              UmLR=m.lr_mean[:n_lr].reshape(m.lr_shape), VmLR=m.lr_mean[n_lr:].reshape(m.lr_shape),
              UmMF=m.hr_mean[:n_hr].reshape(m.hr_shape), VmMF=m.hr_mean[n_hr:].reshape(m.hr_shape),
              XLR=lr_file['X'], YLR=lr_file['Y'], XMF=hr_file['X'], YMF=hr_file['Y'],
              F=m.F, Q=m.Q, n_train=m.meta['n_train'], created='test')
    if method == 'kf':
        S_.update(C=m.C, R_kf=m.R)
    elif method == 'lse':
        S_.update(M=m.M.T, R_lse=m.R_lse)
    else:
        S_.update(M=m.M.T, R_vr=m.R_vr, Gain=m.Gamma[:, None])
    savemat(path, S_)
    return path


def run():
    print("=" * 76)
    print("  External resolution-enhancement data (vibe_import)")
    print("=" * 76)
    tmp = tempfile.mkdtemp(prefix="vibe_import_")
    lr, hr, info = S.make_dataset(nt=2400, dt=1 / F_HZ)
    S.save_matlab(tmp, lr, hr)                      # MATLAB layout, Y up, unit/s
    lr_path, hr_path = os.path.join(tmp, "LR.mat"), os.path.join(tmp, "HR.Mat")

    # ---- 1. geometry --------------------------------------------------------------
    print("\n[1] geometry and conventions")
    ds, conv = VI.fields_training_set(lr_path, hr_path, LIVE, EXT)
    lx, ly = VI.live_lr_grid(LIVE['roi'], 32, 32)
    check("LR nodes = live rt-EBIV nodes (px)",
          np.allclose(ds['lr_x'], lx) and np.allclose(ds['lr_y'], ly),
          f"{conv.px_per_unit:g} px/unit, fit rms {conv.fit_rms_px:.1e} px")
    check("MATLAB (nx, ny) arrays transposed to (rows = y, cols = x)",
          ds['lr_u'].shape[1:] == lx.shape and conv.lr_map.transpose)
    # file: Y up, row 0 of the live grid must be the TOP = largest file Y
    Yf = lr['Y']                                    # hr_synth: (ny, nx), Y grows with row
    check("row 0 is the top of the image (largest file Y)",
          np.isclose(ds['lr_y'][0, 0], ly.min()) and
          np.isclose(conv.ay - conv.px_per_unit * Yf.max(), ly.min()))
    s = conv.px_per_unit / F_HZ
    k = 5
    # the same snapshot: compare u at the top-left node with the file's top-left
    iy_top = np.argmax(lr['Y'][:, 0])
    check("u: same sign, scaled by px_per_unit / f",
          np.isclose(ds['lr_u'][k, 0, 0], lr['U'][k, iy_top, 0] * s, rtol=1e-5), f"x {s:g}")
    check("v: sign flipped (Y up -> image y down)",
          np.isclose(ds['lr_v'][k, 0, 0], -lr['V'][k, iy_top, 0] * s, rtol=1e-5))
    check("HR grid placed with the same fit (inside the LR extent)",
          ds['hr_x'].min() >= 0 and ds['hr_x'].max() <= 512 and ds['hr_y'].max() <= 256
          and ds['hr_y'][0, 0] < ds['hr_y'][-1, 0])
    try:
        VI.fields_training_set(lr_path, hr_path, dict(LIVE, step=24, window=24), EXT)
        check("another window/step (other LR grid) refused", False)
    except ValueError as e:
        check("another window/step (other LR grid) refused", "SAME processing" in str(e))

    # ---- 2. operators ---------------------------------------------------------------
    print("\n[2] operators computed elsewhere (vibe_export_model.m format)")
    Lf, Hf = H.load_mat_dataset(lr_path), H.load_mat_dataset(hr_path)   # file layout
    tr, va, te = slice(0, 1200), slice(1300, 1800), slice(1900, 2400)
    mf = H.train(Lf['U'][tr], Lf['V'][tr], Hf['U'][tr], Hf['V'][tr],
                 Lf['U'][va], Lf['V'][va], Hf['U'][va], Hf['V'][va],
                 lr_mean=np.concatenate([Lf['U'].mean(0).ravel(), Lf['V'].mean(0).ravel()]),
                 hr_mean=np.concatenate([Hf['U'].mean(0).ravel(), Hf['V'].mean(0).ravel()]),
                 matlab_compat=True)
    for meth in ('lse_vr', 'kf', 'lse'):
        path = write_operator_file(os.path.join(tmp, f"model_{meth}.mat"), mf, Lf, Hf, meth)
        mi, cv = VI.import_operators(path, LIVE, EXT)
        X_file = H.KalmanEstimator(mf, method=meth).run(Lf['U'][te], Lf['V'][te])
        uf, vf = mf.reconstruct(X_file)                      # file units, file layout
        uf, vf = cv.fields(uf, vf, 'hr')                     # -> live convention
        lu, lv = cv.fields(Lf['U'][te], Lf['V'][te], 'lr')
        X_live = H.KalmanEstimator(mi, method=meth).run(lu, lv)
        ul, vl = mi.reconstruct(X_live)
        err = max(np.abs(ul - uf).max(), np.abs(vl - vf).max()) / np.abs(uf).max()
        check(f"{meth}: imported model = file model, converted", err < 1e-10 and
              mi.methods == (meth,), f"max rel diff {err:.1e}, methods {mi.methods}")
    check("latent state identical (operators untouched)", np.allclose(X_live, X_file))
    savemat(os.path.join(tmp, "bad.mat"), dict(vibe_format='other', XLR=Lf['X']))
    try:
        VI.import_operators(os.path.join(tmp, "bad.mat"), LIVE, EXT)
        check("a file of another format is refused", False)
    except ValueError as e:
        check("a file of another format is refused", "vibe_export_model" in str(e))

    # ---- 3. through the session (GUI path) -------------------------------------------
    print("\n[3] ebiv_session.train_hr_model (the dialog's worker)")
    import ebiv_session
    import vibe_hr_live
    from ebiv_config import Session
    ss = Session()
    r, c = ss.run, ss.control
    r.output_base_folder, r.acq_name = tmp, "ext"
    r.f_acq, r.trigger_mode = F_HZ, 'auto'
    c.roi.display_roi = [0, 512, 0, 256]
    c.piv.window_size = c.piv.node_distance = 32
    h = r.hr
    h.source, h.ext_lr_file, h.ext_hr_file = 'fields', lr_path, hr_path
    h.ext_y_up, h.ext_velocity_units = True, 'unit/s'
    h.n_train, h.gap, h.n_val, h.n_test = 1200, 100, 500, 400
    h.model_name = "from_fields.npz"
    plan = ebiv_session.hr_training_plan(ss)
    check("plan: inputs found, conversion previewed, no problems",
          not plan['problems'] and plan['conversion'] and 'px/frame' in plan['conversion'],
          plan['problems'] or plan['conversion'][:70])
    rep = ebiv_session.train_hr_model(ss)
    check("source 'fields': model trained here, test report", os.path.exists(rep['model_path'])
          and 'delta_lse_vr' in rep and rep['conversion'],
          f"delta KF {rep.get('delta_kf', float('nan')):.3f}, cubic "
          f"{rep.get('delta_cubic', float('nan')):.3f}")
    hl = vibe_hr_live.HRLive(rep['model_path'], 'kf',
                             live_settings=vibe_hr_live.live_settings_from_session(ss))
    check("HRLive accepts it with the live settings; overlay at the exact HR nodes",
          hl.overlay is not None and np.array_equal(
              hl.overlay.Y0, np.rint(hl.model.hr_y[::hl.overlay.skip, ::hl.overlay.skip]).ravel()))
    hl.update(ds['lr_u'][0], ds['lr_v'][0], 0)
    hl.update(ds['lr_u'][1], ds['lr_v'][1], 1)
    check("and steps on live-convention LR fields", hl.ready and hl.est.k == 2)

    h.source, h.ext_model_file, h.model_name = 'operators', os.path.join(tmp, "model_lse_vr.mat"), \
        "imported.npz"
    rep2 = ebiv_session.train_hr_model(ss)
    hdr = H.read_model_header(rep2['model_path'])
    check("source 'operators': converted model saved, only its estimator",
          rep2['methods'] == ['lse_vr'] and hdr['methods'] == ['lse_vr']
          and hdr.get('lr_processing_declared'))
    try:
        vibe_hr_live.HRLive(rep2['model_path'], 'kf',
                            live_settings=vibe_hr_live.live_settings_from_session(ss))
        check("HRLive refuses an estimator the file does not have", False)
    except ValueError as e:
        check("HRLive refuses an estimator the file does not have", "no 'kf'" in str(e))
    r.f_acq = 100.0
    try:
        vibe_hr_live.HRLive(rep2['model_path'], 'lse_vr',
                            live_settings=vibe_hr_live.live_settings_from_session(ss))
        check("a different live f_acq is refused (declared record)", False)
    except ValueError as e:
        check("a different live f_acq is refused (declared record)", "f_hz" in str(e))

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
