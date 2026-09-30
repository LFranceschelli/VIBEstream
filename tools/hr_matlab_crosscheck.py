"""
Cross-check vibe_hr against Proc_Main_FullKF_DEF.m (Method I) or
Proc_Main_EPOD_DEF.m (Methods II / III) on YOUR machine.

    SCRIPT            'fullkf' or 'epod': which script produced MATLAB_OUT.
                      For 'epod' the method (LSE or LSE+VR), MaxGain, tuneQ/R,
                      intermittent DS and alpha_smooth are read from its
                      ProcInfo.mat.

    FLAG_WRITE_SYNTH  write a synthetic LR.mat / HR.Mat (MATLAB layout) to
                      SYNTH_DIR, to run the MATLAB script on (point its load()
                      lines there and run one case).
    FLAG_COMPARE      train the Python model exactly like the script
                      (all-snapshot mean, matlab_compat=True, ranks read from
                      ProcInfo.mat) on LR_FILE / HR_FILE and compare with the
                      script's output file MATLAB_OUT.  Works on the real
                      jet/channel data too.

What agreement to expect: with the same ranks, the reconstructed fields agree
to ~1e-12 relative (checked against Octave 8.4 on synthetic data: 5e-14 for
Method I; 1e-14 to 2e-14 for LSE, LSE+VR and intermittent LSE+VR, DS = 5,
alpha = 0.2).
Differences at the 1e-3 level mean a mismatch in settings; the usual suspect
is the mean or matlab_compat.  The elbow rank (vibe_hr.pod_elbow_rank, a port
of pod_elbow_rank.m) is reported separately; the comparison itself uses the
ranks in ProcInfo.mat so that the two checks stay independent.
"""

import os
import sys
import glob

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(_ROOT, 'lib'), os.path.join(_ROOT, 'tools')]
import numpy as np            # noqa: E402
import vibe_hr as H           # noqa: E402

FLAG_WRITE_SYNTH = False
FLAG_COMPARE = True
SCRIPT = 'fullkf'           # 'fullkf' | 'epod'

SYNTH_DIR = r"C:\temp\vibe_synth"
LR_FILE = r"C:\data\jet\LR.mat"
HR_FILE = r"C:\data\jet\HR.mat"
MATLAB_OUT = r"C:\data\jet\matlab_out"   # folder with Data_*.mat + ProcInfo.mat
CASE_GLOB = "Data_Jet*_r*.mat"
N_TRAIN, N_VAL, N_TEST = (1, 4500), (5000, 6499), (7000, 8999)   # as in the script


def _read(path, names):
    """Read variables from a v7 (scipy) or v7.3 (HDF5) .mat file, MATLAB orientation."""
    if H._is_hdf5(path):
        import h5py
        out = {}
        with h5py.File(path, 'r') as f:
            for k in names:
                if k in f:
                    out[k] = np.asarray(f[k]).T
        return out
    from scipy.io import loadmat
    d = loadmat(path, variable_names=names, squeeze_me=False)
    return {k: d[k] for k in names if k in d}


def _proc(path, keys=('rank', 'rankLR')):
    """Proc fields: numbers as float, char arrays as str, missing ones -> None."""
    out = {}
    if H._is_hdf5(path):
        import h5py
        with h5py.File(path, 'r') as f:
            for k in keys:
                if k not in f['Proc']:
                    out[k] = None
                    continue
                a = np.asarray(f['Proc'][k])
                out[k] = ''.join(map(chr, a.ravel())) if a.dtype == np.uint16 \
                    else float(a.ravel()[0])
        return out
    from scipy.io import loadmat
    p = loadmat(path, squeeze_me=True)['Proc']
    for k in keys:
        if k not in p.dtype.names:
            out[k] = None
            continue
        v = p[k].item() if hasattr(p[k], 'item') else p[k]
        out[k] = v if isinstance(v, str) else float(v)
    return out


def compare():
    sl = lambda ab: slice(ab[0] - 1, ab[1])                          # noqa: E731
    LR, HR = H.load_mat_dataset(LR_FILE), H.load_mat_dataset(HR_FILE)
    pr = _proc(os.path.join(MATLAB_OUT, 'ProcInfo.mat'),
               ('rank', 'rankLR', 'FlagVarRescaling', 'MaxGain', 'FlagIntermittentKF', 'DS',
                'alpha_smooth', 'tuneQ', 'tuneR', 'ThrElbow', 'SmoothElbow'))
    r_m, rlr_m = int(pr['rank']), int(pr['rankLR'])
    out_file = sorted(glob.glob(os.path.join(MATLAB_OUT, CASE_GLOB)))[0]
    o = _read(out_file, ['UMFestK', 'UMFlor', 'UMFest'])
    print(f"MATLAB: r={r_m}, r_LR={rlr_m}  ({os.path.basename(out_file)})")
    method, ds, alpha, qs, rs, gain = 'kf', None, None, 1.0, 1.0, 10.0
    if SCRIPT == 'epod':
        method = 'lse_vr' if pr['FlagVarRescaling'] else 'lse'
        gain = pr['MaxGain'] or 10.0
        qs, rs = pr['tuneQ'] or 1.0, pr['tuneR'] or 1.0
        if pr['FlagIntermittentKF']:
            ds, alpha = int(pr['DS']), pr['alpha_smooth']
        print(f"EPOD script: method {method}, MaxGain {gain:g}, tuneQ {qs:g}, tuneR {rs:g}, "
              f"intermittent DS {ds}, alpha {alpha}")

    lr_mean = np.concatenate([LR['U'].mean(0).ravel(), LR['V'].mean(0).ravel()])
    hr_mean = np.concatenate([HR['U'].mean(0).ravel(), HR['V'].mean(0).ravel()])
    tr, va, te = sl(N_TRAIN), sl(N_VAL), sl(N_TEST)
    m = H.train_kf(LR['U'][tr], LR['V'][tr], HR['U'][tr], HR['V'][tr],
                   LR['U'][va], LR['V'][va], HR['U'][va], HR['V'][va],
                   rank=r_m, rank_lr=rlr_m, lr_mean=lr_mean, hr_mean=hr_mean,
                   matlab_compat=True, max_gain=gain)
    eo = dict(threshold=pr['ThrElbow'] or 0.999, smooth=pr['SmoothElbow'] or 'none')
    print(f"Python elbow on the same spectra ({eo}): r={H.elbow_rank(m.sigma_hr_all, **eo)}, "
          f"r_LR={H.elbow_rank(m.sigma_lr_all, **eo)}  (MATLAB: {r_m}, {rlr_m})")
    X = H.KalmanEstimator(m, method=method, steady_state=False, q_scale=qs, r_scale=rs).run(
        LR['U'][te], LR['V'][te], ds=ds, alpha_smooth=alpha)
    u_p, v_p = m.reconstruct(X, add_mean=False)
    u_l, v_l = m.lor_hr(HR['U'][te], HR['V'][te], add_mean=False)

    nx, ny = u_p.shape[1:]
    n = nx * ny

    def split(M):
        u = np.moveaxis(M[:n].reshape(nx, ny, -1, order='F'), -1, 0)
        v = np.moveaxis(M[n:].reshape(nx, ny, -1, order='F'), -1, 0)
        return u, v

    rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))   # noqa: E731
    uo, vo = split(o['UMFestK'])
    ul, vl = split(o['UMFlor'])
    print(f"HR-LOR  rel diff: u {rel(u_l, ul):.2e}  v {rel(v_l, vl):.2e}")
    print(f"{method.upper():<7s} rel diff: u {rel(u_p, uo):.2e}  v {rel(v_p, vo):.2e}   (KF output)")
    if SCRIPT == 'epod' and 'UMFest' in o:
        psi = m.project_lr(LR['U'][te], LR['V'][te])
        ue, ve = m.reconstruct(m.lse(psi, vr=(method == 'lse_vr')), add_mean=False)
        uoe, voe = split(o['UMFest'])
        print(f"LSE{'+VR' if method == 'lse_vr' else ''} estimate (before the KF) rel diff: "
              f"u {rel(ue, uoe):.2e}  v {rel(ve, voe):.2e}")


if __name__ == "__main__":
    if FLAG_WRITE_SYNTH:
        import hr_synth as S
        lr, hr, info = S.make_dataset(hr_step=1 / 12, hr_win=1 / 3)
        S.save_matlab(SYNTH_DIR, lr, hr)
        print(f"synthetic LR.mat / HR.Mat written to {SYNTH_DIR}: {info}")
    if FLAG_COMPARE:
        compare()
