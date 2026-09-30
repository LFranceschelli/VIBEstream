"""
Resolution enhancement (KF / LSE / LSE+VR) from a script: train, import, test,
run live.  The GUI does the same from Resolution enhancement > Train a model...

    1. Set the flags in section 1 and the parameters in section 2.
    2. python HR_Example.py

WHERE THE MODEL COMES FROM (one of the first three)
FLAG_TRAIN_RAW         .raw recording -> LR (live rt-EBIV settings) + HR
                       (multi-frame PIV) -> POD + operators, all here.
FLAG_IMPORT_FIELDS     LR.mat + HR.mat computed elsewhere (MATLAB pipeline)
                       -> converted to the live convention -> POD + operators here.
FLAG_IMPORT_OPERATORS  operators computed elsewhere: the .mat written by
                       tools/matlab/vibe_export_model.m -> converted, nothing
                       recomputed.
All three give a model in the LIVE convention (px/frame, rows = y down), ready
for FLAG_ONLINE and for the GUI.

ANALYSIS IN THE FILE CONVENTION (e.g. to compare with the MATLAB scripts)
FLAG_TRAIN_MAT / FLAG_TEST   train and test on LR.mat / HR.mat as they are.
                             The resulting model is NOT for live use.

FLAG_ONLINE            a live-convention model + the camera (VIBE).

See lib/vibe_hr.py (estimators), lib/vibe_train.py (training set),
lib/vibe_import.py (external data and the conversion).
"""

import os
import sys
import time
import logging

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # small matrices: 1 thread is faster
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
import numpy as np                                                  # noqa: E402
import vibe_hr as H                                                 # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# =============================================================================
#  1. WHAT TO DO
# =============================================================================
FLAG_TRAIN_RAW        = False
FLAG_IMPORT_FIELDS    = False
FLAG_IMPORT_OPERATORS = False
FLAG_TRAIN_MAT        = False
FLAG_TEST             = False
FLAG_ONLINE           = False

# =============================================================================
#  2. PARAMETERS
# =============================================================================
# --- the LIVE rt-EBIV processing the model is for (all sources) --------------------
F_ACQ    = 100.0                  # Hz
ROI      = [200, 1100, 150, 600]  # [x0, x1, y0, y1] px; None = full sensor
WINDOW   = 48                     # px   (rt-EBIV: single pass, 0 % overlap)
STEP     = 48                     # px
SMOOTH_SIGMA = 0.75               # px, frame smoothing: the SAME value live
SENSOR_WH = (1280, 720)           # only used when ROI is None

MODEL_FILE = "hr_model.npz"       # live-convention model (output of the first three flags)

# --- FLAG_TRAIN_RAW --------------------------------------------------------------
RAW_FILE = r"C:\data\jet\jet.raw"
HR_WINDOW, HR_STEP, HR_LEVELS = 32, 8, 2      # HR: multi-frame, 75 % overlap
N_FRAMES = None                               # None = whole recording

# --- external data (FLAG_IMPORT_*, FLAG_TRAIN_MAT, FLAG_TEST) --------------------------
LR_FILE = r"C:\data\jet\LR.mat"               # MATLAB layout: X, Y (nx, ny), U, V (nx, ny, Nt)
HR_FILE = r"C:\data\jet\HR.mat"
OPERATOR_FILE = r"C:\data\jet\model_lse_vr.mat"   # from tools/matlab/vibe_export_model.m
Y_UP = True                  # the files' Y axis points up (v positive up)
VELOCITY_UNITS = 'm/s'       # 'm/s' (X, Y in mm) | 'px/frame' | 'unit/s'

# --- training (all but FLAG_IMPORT_OPERATORS) -------------------------------------------
# consecutive blocks, in fields: train | gap | val | gap | test
N_TRAIN, GAP, N_VAL, N_TEST = 4500, 500, 1500, 2000
RANK        = 'elbow'     # 'elbow' or an int
TRUNCATE_LR = True        # FlagLRRankLimit: True = elbow on LR too, False = full LR rank
THRESHOLD   = 0.999
ELBOW       = dict(smooth='none', span=25)   # pod_elbow_rank options
LAMBDA_C    = 1e-12
MAX_GAIN    = 10.0        # cap of the variance-rescaling gain (Proc.MaxGain)

# --- FLAG_TRAIN_MAT / FLAG_TEST (file convention, as the MATLAB scripts) ------------
MAT_MODEL_FILE = "hr_model_fileconv.npz"
RESULTS_FILE   = "hr_results.mat"
MEAN   = 'all'            # 'all' (the MATLAB scripts) | 'train' (online-realisable)
U_REF  = 0.2              # velocity scale for delta, in the files' units
METHOD = 'kf'             # 'kf' | 'lse' | 'lse_vr'  (test and online)

# --- FLAG_ONLINE ---------------------------------------------------------------------
ONLINE_SECONDS    = 30
STEADY_STATE      = True  # converged gain: same estimate after the transient, much cheaper
RECONSTRUCT_FIELD = True  # False: latent state only


# =============================================================================
#  3. RUN
# =============================================================================
LIVE = dict(f_hz=F_ACQ, roi=ROI, window=WINDOW, step=STEP, flip_x=False, flip_y=False,
            phase_locked=True)


def train_and_save(ds):
    import vibe_train as VT
    split = VT.SplitSettings(n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST, gap=GAP)
    model, report = VT.train_from_set(ds, split, rank=RANK, truncate_lr=TRUNCATE_LR,
                                      threshold=THRESHOLD, elbow=ELBOW, lambda_c=LAMBDA_C,
                                      max_gain=MAX_GAIN)
    model.save(MODEL_FILE)
    print(model.summary())
    print({k: round(v, 4) for k, v in report.items() if isinstance(v, float)})


if FLAG_TRAIN_RAW:
    import vibe_train as VT
    from vibe import VIBE
    vibe = VIBE(roi=ROI, max_events_per_pixel=1, smooth_sigma=SMOOTH_SIGMA)
    st = VT.TrainingSettings(f_hz=F_ACQ, roi=ROI, lr_window=WINDOW, lr_step=STEP,
                             hr_window=HR_WINDOW, hr_step=HR_STEP, hr_levels=HR_LEVELS,
                             n_frames=N_FRAMES)
    ds = VT.build_training_set(vibe, RAW_FILE, st, progress=lambda i, n, eta: print(
        f"  {i}/{n} fields, ~{eta / 60:.1f} min left" if n > 0 else f"  {i} fields"))
    VT.save_training_set(os.path.splitext(MODEL_FILE)[0] + "_trainingset.npz", ds)
    train_and_save(ds)

if FLAG_IMPORT_FIELDS or FLAG_IMPORT_OPERATORS:
    import vibe_import as VI
    ext = VI.ExternalSettings(y_up=Y_UP, velocity_units=VELOCITY_UNITS, f_hz=F_ACQ)
    if FLAG_IMPORT_FIELDS:
        ds, conv = VI.fields_training_set(LR_FILE, HR_FILE, LIVE, ext, SENSOR_WH)
        print(conv.summary())
        train_and_save(ds)
    else:
        model, conv = VI.import_operators(OPERATOR_FILE, LIVE, ext, SENSOR_WH)
        print(conv.summary())
        model.save(MODEL_FILE)
        print(model.summary())

if FLAG_TRAIN_MAT or FLAG_TEST:
    t0 = time.time()
    LR = H.load_mat_dataset(LR_FILE)
    HR = H.load_mat_dataset(HR_FILE)
    logging.info("Loaded LR %s and HR %s in %.0f s", LR['U'].shape, HR['U'].shape,
                 time.time() - t0)
    a, b = N_TRAIN + GAP, N_TRAIN + GAP + N_VAL
    tr, va, te = slice(0, N_TRAIN), slice(a, b), slice(b + GAP, b + GAP + N_TEST)

if FLAG_TRAIN_MAT:
    means = {}
    if MEAN == 'all':
        means = dict(lr_mean=np.concatenate([LR['U'].mean(0).ravel(), LR['V'].mean(0).ravel()]),
                     hr_mean=np.concatenate([HR['U'].mean(0).ravel(), HR['V'].mean(0).ravel()]))
    model = H.train(LR['U'][tr], LR['V'][tr], HR['U'][tr], HR['V'][tr],
                    LR['U'][va], LR['V'][va], HR['U'][va], HR['V'][va],
                    rank=RANK, truncate_lr=TRUNCATE_LR, threshold=THRESHOLD, elbow=ELBOW,
                    lambda_c=LAMBDA_C, max_gain=MAX_GAIN, matlab_compat=(MEAN == 'all'),
                    grids=dict(lr_x=LR['X'], lr_y=LR['Y'], hr_x=HR['X'], hr_y=HR['Y']),
                    **means)
    model.meta.update(f_acq_hz=F_ACQ, source_lr=LR_FILE, source_hr=HR_FILE,
                      units='file convention (not for live use)')
    model.save(MAT_MODEL_FILE)
    print(model.summary())

if FLAG_TEST:
    from scipy.io import savemat
    model = H.HRModel.load(MAT_MODEL_FILE)
    est = H.KalmanEstimator(model, method=METHOD)
    X = est.run(LR['U'][te], LR['V'][te])
    u_e, v_e = model.reconstruct(X)
    u_lor, v_lor = model.lor_hr(HR['U'][te], HR['V'][te])
    u_cu, v_cu = H.cubic_baseline(LR['U'][te], LR['V'][te], LR['X'], LR['Y'], HR['X'], HR['Y'])
    d_e = H.delta_error(u_e, v_e, u_lor, v_lor, U_REF)
    d_cu = H.delta_error(u_cu, v_cu, u_lor, v_lor, U_REF)
    print(f"delta (Eq. 34, vs HR-LOR): {METHOD} {d_e:.4f} | cubic {d_cu:.4f}")
    to_m = lambda a: np.moveaxis(a, 0, -1)                         # noqa: E731  (Nt,..)->(..,Nt)
    savemat(RESULTS_FILE, dict(U_est=to_m(u_e), V_est=to_m(v_e), U_lor=to_m(u_lor),
                               V_lor=to_m(v_lor), U_cubic=to_m(u_cu), V_cubic=to_m(v_cu),
                               Psi_est=X, Xhr=HR['X'], Yhr=HR['Y'], delta_est=d_e,
                               delta_cubic=d_cu, r=model.r, r_lr=model.r_lr))
    print(f"results written to {RESULTS_FILE}")

if FLAG_ONLINE:
    from vibe import VIBE
    model = H.HRModel.load(MODEL_FILE)
    if model.meta.get('units') != 'px/frame':
        raise SystemExit(f"{MODEL_FILE} is not in the live convention: make it with "
                         "FLAG_TRAIN_RAW, FLAG_IMPORT_FIELDS or FLAG_IMPORT_OPERATORS.")
    bad = [m for m in H.check_processing(model, LIVE) if H.is_critical(m)]
    if bad:
        raise SystemExit("model trained for other live settings: " + "; ".join(bad))
    est = H.KalmanEstimator(model, method=METHOD, steady_state=STEADY_STATE)
    period_us = 1e6 / F_ACQ
    with VIBE(roi=ROI, smooth_sigma=SMOOTH_SIGMA) as vibe:
        # No max_rate_hz: F is a one-step model, so every field the correlator
        # drops costs a blind prediction (step_timed bridges it).  Watch
        # est.n_skipped: if it grows fast, the rt-EBIV is not keeping up with F_ACQ.
        vibe.start(f=F_ACQ, window=WINDOW, step=STEP, trigger='auto')
        t_end = time.time() + ONLINE_SECONDS
        fld = None
        while time.time() < t_end:
            fld = vibe.wait(timeout=1.0, newer_than=fld)
            if fld is None:
                continue
            t0 = time.perf_counter()
            x = est.step_timed(fld.t_us, fld.u, fld.v, period_us)   # latent HR state (r,)
            if RECONSTRUCT_FIELD:
                u_hr, v_hr = est.field()
            dt_ms = (time.perf_counter() - t0) * 1e3
            print(f"t={fld.t_us / 1e6:8.3f} s  psi_1..3 = {x[:3].round(4)}  "
                  f"{dt_ms:.2f} ms  (bridged {est.n_skipped} missing periods so far)")
        vibe.stop()
