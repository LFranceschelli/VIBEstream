"""
vibe_train — build the LR/HR training set from a .raw recording and train the
HR estimators, entirely in Python (no MATLAB step).

    raw ──> phase-locked pseudo-frames (VIBE, same settings as the live stream)
             ├─ LR: single-pass correlation of consecutive frames, with the
             │      SAME correlator, ROI, window, step and validation the live
             │      stream uses                                      (Sec. 3.4)
             └─ HR: multi-frame pyramidal correlation (VibeStream's offline
                    method: correlation planes of frame separations s scaled
                    by 1/s and summed), smaller windows, overlap      (Sec. 3.4)
    fields ──> split train / val / test ──> vibe_hr.train() ──> HRModel

TIME ALIGNMENT (why the default HR stencil is 'centred')
    An LR field comes from the pair (i, i+1): it describes t_{i+1/2}.
    VibeStream's offline PIV uses a FORWARD stencil anchored at frame i
    (pairs (i, i+s), s = 1..L), whose effective time is later than t_{i+1/2}.
    For a convecting flow the HR field would then be displaced with respect
    to the LR one by ~U*dt*(L-1)/2, which biases C and M.  The 'centred'
    stencil uses the pairs (i-s+1, i+s), separations 1, 3, 5, ... dt, all
    centred on t_{i+1/2}; its first member IS the LR pair.  'forward'
    reproduces process_offline_piv exactly.

This is NOT the paper's HR pipeline (in-house multi-pass, window deformation):
there is no window deformation and a single pass.  Expect a less refined HR
reference; the estimator can only learn what the HR processing resolves.
"""

import os
import time
import json
import logging
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

import ebiv_utils as _U
from ebiv_piv import batch_subpixel_flat
import vibe_hr as H

log = logging.getLogger("vibe.train")


# ==========================================================================
#  settings
# ==========================================================================

@dataclass
class TrainingSettings:
    """Everything that defines how a training set is produced."""
    f_hz: float                       # laser / acquisition frequency
    roi: Optional[list] = None        # [x0, x1, y0, y1] in the (flipped) image
    duty_cycle: float = 0.8
    # LR = the live rt-EBIV
    lr_window: int = 48
    lr_step: int = 48
    lr_validate: bool = True
    lr_subpixel: bool = True
    val_threshold: float = 2.0
    val_epsilon: float = 0.1
    # HR = offline multi-frame
    hr_window: int = 32
    hr_step: int = 8
    hr_levels: int = 2                # number of frame separations per field
    hr_stencil: str = 'centred'       # 'centred' | 'forward'
    hr_combine: str = 'peaks'         # 'peaks' | 'planes' (process_offline_piv)
    hr_predictor: bool = True         # window-offset predictor pass
    hr_predictor_window: Optional[int] = None   # default 2 * hr_window
    hr_validate: bool = True
    # frames
    start_frame: int = 0
    n_frames: Optional[int] = None    # None = whole recording
    phase_us: Optional[float] = None  # None = detect from the first second

    def lr_processing(self, vibe):
        """What the live stream must reproduce for a model trained on this set."""
        return dict(f_hz=self.f_hz, roi=self.roi, window=self.lr_window, step=self.lr_step,
                    validate=self.lr_validate, subpixel=self.lr_subpixel,
                    val_threshold=self.val_threshold, val_epsilon=self.val_epsilon,
                    duty_cycle=self.duty_cycle, phase_locked=True,
                    **vibe.processing_settings())


# ==========================================================================
#  multi-frame correlator (in memory; numerics of process_offline_piv)
# ==========================================================================

class MultiFrameCorrelator:
    """
    Multi-frame FFT cross-correlation (VibeStream's offline method, extended
    with a window-offset predictor pass).

    For every frame pair (a, b) of the stencil with separation s (in frame
    periods):

      predictor   (predictor=True) a first pass on the central pair with
                  larger windows (predictor_window, default 2*window) gives an
                  integer displacement d0 per node; the two windows of pair
                  (a, b) are then shifted symmetrically by s*d0, so the
                  residual displacement s*(d - d0) stays small for every s.
                  Without it, every s*d must stay well below window/2 or the
                  circular correlation aliases.
      combine     'planes': each residual plane is contracted by 1/s about its
                  centre (the 'homothetic warp' of process_offline_piv: with
                  integer centre and s it samples R_s at c + s (q - c)) and
                  the planes are summed; one sub-pixel peak on the sum.
                  'peaks': one sub-pixel peak per pair, d_s = (s*d0 + r_s)/s,
                  combined with weights s^2 (the error of d_s scales as 1/s).

    stencil='forward'  frames [i .. i+L]:        pairs (i, i+s),       s = 1..L
    stencil='centred'  frames [i-L+1 .. i+L]:    pairs (i-s+1, i+s),   sep 2s-1

    predictor=False, combine='planes', stencil='forward' reproduces
    ebiv_utils.process_offline_piv (checked in tools/test_hr_train.py).
    Displacements are per ONE frame period; v positive down.
    """

    def __init__(self, shape, window=32, step=8, levels=2, stencil='centred',
                 combine='peaks', predictor=True, predictor_window=None):
        if stencil not in ('centred', 'forward'):
            raise ValueError("stencil must be 'centred' or 'forward'")
        if combine not in ('planes', 'peaks'):
            raise ValueError("combine must be 'planes' or 'peaks'")
        h, w = shape
        self.shape = (h, w)
        self.ws, self.nd, self.L = int(window), int(step), int(levels)
        self.stencil, self.combine, self.predictor = stencil, combine, bool(predictor)
        self.pw = int(predictor_window or 2 * self.ws)
        if h < self.ws or w < self.ws:
            raise ValueError(f"image {w}x{h} smaller than the HR window {self.ws}")
        self.gy = (h - self.ws) // self.nd + 1
        self.gx = (w - self.ws) // self.nd + 1
        self.n = self.gy * self.gx
        self.c = self.ws // 2
        if stencil == 'forward':
            self.pairs = [(0, s, s) for s in range(1, self.L + 1)]
            self.n_frames = self.L + 1
            self.anchor = 0                   # pair (anchor, anchor+1) = the LR pair
        else:
            self.pairs = [(self.L - s, self.L - 1 + s, 2 * s - 1) for s in range(1, self.L + 1)]
            self.n_frames = 2 * self.L
            self.anchor = self.L - 1
        gy_, gx_ = np.meshgrid(np.arange(self.gy), np.arange(self.gx), indexing='ij')
        self._y0 = (gy_ * self.nd).ravel()        # window top-left, image coords
        self._x0 = (gx_ * self.nd).ravel()
        q = np.arange(self.ws)
        self._idx = {}
        for _, _, sep in self.pairs:
            src = self.c + sep * (q - self.c)
            ok = (src >= 0) & (src < self.ws)
            self._idx[sep] = (np.nonzero(ok)[0], src[ok])

    # ------------------------------------------------------------------
    @staticmethod
    def _windows(padded, pad, y0, x0, ws):
        """Windows with per-node top-left (y0, x0) (image coords) from a padded frame."""
        from numpy.lib.stride_tricks import sliding_window_view
        view = sliding_window_view(padded, (ws, ws))
        W = view[y0 + pad, x0 + pad].astype(np.float32)          # (N, ws, ws) copy
        W -= W.mean(axis=(1, 2), keepdims=True)
        return W

    @staticmethod
    def _corr(Wa, Wb):
        from scipy.fft import rfft2, irfft2
        ws = Wa.shape[-1]
        R = irfft2(np.conj(rfft2(Wa)) * rfft2(Wb), s=(ws, ws))
        return np.fft.fftshift(R, axes=(1, 2)), np.einsum('nij,nij->n', Wa, Wa), \
            np.einsum('nij,nij->n', Wb, Wb)

    @staticmethod
    def _peak(R):
        n, ws, _ = R.shape
        Rf = R.reshape(n, -1)
        pk = np.argmax(Rf, axis=1)
        cy, cx = np.divmod(pk, ws)
        dx, dy = batch_subpixel_flat(Rf, pk, ws)
        c = ws // 2
        return cx - c + dx, cy - c + dy, Rf[np.arange(n), pk]

    def predict(self, fa, fb):
        """Integer displacement per node from the central pair, larger windows."""
        pw = self.pw
        pad = pw
        Pa = np.pad(fa.astype(np.float32), pad)
        Pb = np.pad(fb.astype(np.float32), pad)
        off = self.c - pw // 2                                    # same window centres
        y0, x0 = self._y0 + off, self._x0 + off
        R, _, _ = self._corr(self._windows(Pa, pad, y0, x0, pw),
                             self._windows(Pb, pad, y0, x0, pw))
        u, v, _ = self._peak(R)
        U, V = u.reshape(self.gy, self.gx), v.reshape(self.gy, self.gx)
        U, V, _ = _U.universal_outlier_detection(U, V, 2.0, 0.1)
        return np.rint(U).astype(int), np.rint(V).astype(int)

    def correlate(self, frames):
        """frames: list of self.n_frames 2-D arrays (same crop).  -> U, V, CC (gy, gx)."""
        ws, n = self.ws, self.n
        if self.predictor:
            a0 = self.anchor
            du0, dv0 = self.predict(frames[a0], frames[a0 + 1])
            du0, dv0 = du0.ravel(), dv0.ravel()
        else:
            du0 = dv0 = np.zeros(n, int)
        max_sep = max(sep for _, _, sep in self.pairs)
        pad = int(max_sep * max(np.abs(du0).max(initial=0), np.abs(dv0).max(initial=0))) + 1
        padded = {}

        def P(k):
            if k not in padded:
                padded[k] = np.pad(frames[k].astype(np.float32), pad)
            return padded[k]

        R_sum = np.zeros((n, ws, ws))
        num_u = np.zeros(n)
        num_v = np.zeros(n)
        wsum = 0.0
        norm = np.zeros(n)
        cc_sum = np.zeros(n)
        for a, b, sep in self.pairs:
            Tx, Ty = sep * du0, sep * dv0               # integer total offset
            ax, ay = -(Tx // 2), -(Ty // 2)             # symmetric split
            bx, by = Tx + ax, Ty + ay
            Wa = self._windows(P(a), pad, self._y0 + ay, self._x0 + ax, ws)
            Wb = self._windows(P(b), pad, self._y0 + by, self._x0 + bx, ws)
            R, ea, eb = self._corr(Wa, Wb)
            norm += np.sqrt(ea * eb)
            if self.combine == 'planes':
                dst, src = self._idx[sep]
                if sep == 1:
                    R_sum += R
                else:
                    R_sum[:, dst[:, None], dst[None, :]] += R[:, src[:, None], src[None, :]]
            else:
                ru, rv, pk = self._peak(R)
                w_ = float(sep) ** 2
                num_u += w_ * (Tx + ru) / sep
                num_v += w_ * (Ty + rv) / sep
                wsum += w_
                cc_sum += pk
        if self.combine == 'planes':
            ru, rv, pk = self._peak(R_sum)
            U, V, CC = du0 + ru, dv0 + rv, pk / (norm + 1e-20)
        else:
            U, V, CC = num_u / wsum, num_v / wsum, cc_sum / (norm + 1e-20)
        g = (self.gy, self.gx)
        return (U.reshape(g).astype(np.float32), V.reshape(g).astype(np.float32),
                CC.reshape(g).astype(np.float32))


# ==========================================================================
#  building the training set
# ==========================================================================

def build_training_set(vibe, raw_path, settings: TrainingSettings, progress=None,
                       stop_event=None):
    """
    Run both pipelines over a .raw recording.  Returns a dict:
        lr_u, lr_v (Nt, gyL, gxL)  hr_u, hr_v (Nt, gyH, gxH)   float32, px/frame
        lr_x, lr_y, hr_x, hr_y     vector positions (px, flipped full image)
        t_us (Nt,)                 camera time of the NEWER frame of the LR pair
        settings, lr_processing    dicts
    progress(i, n_expected, eta_s) is called every 25 fields.
    """
    from vibe import _PIVProc, _clip_roi
    st = settings
    t0 = time.perf_counter()
    raw_path = vibe._path(raw_path)
    n_req = None if st.n_frames is None else st.start_frame + st.n_frames
    frames = vibe.frames_from_raw(raw_path, f=st.f_hz, n=n_req, duty_cycle=st.duty_cycle,
                                  normalize='clip', phase_us=st.phase_us)
    lr_proc = hr = None
    buf = []
    out = dict(lr_u=[], lr_v=[], hr_u=[], hr_v=[], t_us=[])
    n_done = 0
    for fr in frames:
        if stop_event is not None and stop_event.is_set():
            log.warning("Training-set build stopped by the user after %d fields.", n_done)
            break
        if fr.index < st.start_frame:
            continue
        if lr_proc is None:
            H_, W_ = fr.image.shape
            x0, x1, y0, y1 = _clip_roi(st.roi if st.roi is not None else [0, W_, 0, H_], W_, H_)
            lr_proc = _PIVProc(fr.image.shape, [x0, x1, y0, y1], st.lr_window, st.lr_step,
                               False, st.lr_subpixel, False, st.lr_validate,
                               st.val_threshold, st.val_epsilon, st.f_hz)
            hr = MultiFrameCorrelator((y1 - y0, x1 - x0), st.hr_window, st.hr_step,
                                      st.hr_levels, st.hr_stencil, st.hr_combine,
                                      st.hr_predictor, st.hr_predictor_window)
            roi = (x0, x1, y0, y1)
            hx = x0 + np.arange(hr.gx) * hr.nd + hr.ws // 2
            hy = y0 + np.arange(hr.gy) * hr.nd + hr.ws // 2
            HX, HY = np.meshgrid(hx.astype(np.float32), hy.astype(np.float32))
            n_exp = (st.n_frames or 0) - hr.n_frames + 1
            log.info("Training set: LR grid %s (window %d, step %d), HR grid %s (window %d, "
                     "step %d, %d separations, %s stencil)", lr_proc.X.shape, st.lr_window,
                     st.lr_step, HX.shape, st.hr_window, st.hr_step, st.hr_levels, st.hr_stencil)
        buf.append(fr)
        if len(buf) > hr.n_frames:
            buf.pop(0)
        if len(buf) < hr.n_frames:
            continue
        a = hr.anchor
        pair = [buf[a].image, buf[a + 1].image]
        lf = lr_proc.run(pair, buf[a + 1].t_us, n_done + 1)
        x0, x1, y0, y1 = roi
        U, V, _ = hr.correlate([b.image[y0:y1, x0:x1] for b in buf])
        if st.hr_validate:
            U, V, _ = _U.universal_outlier_detection(U, V, st.val_threshold, st.val_epsilon)
        out['lr_u'].append(lf.u.astype(np.float32))
        out['lr_v'].append(lf.v.astype(np.float32))
        out['hr_u'].append(np.asarray(U, np.float32))
        out['hr_v'].append(np.asarray(V, np.float32))
        out['t_us'].append(buf[a + 1].t_us)
        n_done += 1
        if progress is not None and n_done % 25 == 0:
            el = time.perf_counter() - t0
            eta = el / n_done * max(n_exp - n_done, 0) if n_exp > 0 else float('nan')
            progress(n_done, n_exp, eta)
    if n_done == 0:
        raise ValueError("no field could be computed: recording too short or no events")
    res = {k: np.asarray(v) for k, v in out.items()}
    res.update(lr_x=lr_proc.X, lr_y=lr_proc.Y, hr_x=HX, hr_y=HY,
               settings=asdict(st), lr_processing=st.lr_processing(vibe),
               seconds=time.perf_counter() - t0)
    # consecutive fields must be ONE period apart, or F is meaningless
    d = np.diff(res['t_us'])
    period = 1e6 / st.f_hz
    bad = np.abs(d - period) > 0.25 * period
    res['n_gaps'] = int(bad.sum())
    if bad.any():
        log.warning("%d gaps in the frame sequence (stream paused?). Splits should avoid "
                    "them: F assumes consecutive snapshots.", int(bad.sum()))
    log.info("Training set built: %d fields in %.0f s (%.1f ms/field).", n_done,
             res['seconds'], res['seconds'] / n_done * 1e3)
    return res


def save_training_set(path, ds):
    """.npz (Python) of a training set."""
    arrs = {k: v for k, v in ds.items() if isinstance(v, np.ndarray)}
    meta = {k: v for k, v in ds.items() if not isinstance(v, np.ndarray)}
    np.savez_compressed(path, meta=json.dumps(meta, default=str), **arrs)
    return path


def load_training_set(path):
    d = np.load(path, allow_pickle=False)
    out = {k: d[k] for k in d.files if k != 'meta'}
    out.update(json.loads(str(d['meta'])))
    return out


def export_matlab(folder, ds):
    """
    LR.mat / HR.mat in the layout of the MATLAB pipeline (X, Y (nx, ny);
    U, V (nx, ny, Nt)), for comparison with Proc_Main_*.m.  Units: px/frame,
    v positive DOWN (image rows).
    """
    from scipy.io import savemat
    os.makedirs(folder, exist_ok=True)
    for name, p in (('LR.mat', 'lr'), ('HR.mat', 'hr')):
        savemat(os.path.join(folder, name), {
            'X': ds[f'{p}_x'].T, 'Y': ds[f'{p}_y'].T,
            'U': np.transpose(ds[f'{p}_u'], (2, 1, 0)),
            'V': np.transpose(ds[f'{p}_v'], (2, 1, 0))})
    return folder


# ==========================================================================
#  split + train + evaluate
# ==========================================================================

@dataclass
class SplitSettings:
    """Consecutive blocks: train | gap | val | gap | test (counts, in fields)."""
    n_train: int = 4500
    n_val: int = 1500
    n_test: int = 2000
    gap: int = 500

    def slices(self, n_total):
        need = self.n_train + self.n_val + self.n_test + 2 * self.gap
        if need > n_total:
            raise ValueError(f"split needs {need} fields, the training set has {n_total}")
        a = self.n_train
        b = a + self.gap
        c = b + self.n_val
        d = c + self.gap
        return slice(0, a), slice(b, c), slice(d, d + self.n_test)


def train_from_set(ds, split: SplitSettings, u_ref=None, rank='elbow', truncate_lr=True,
                   threshold=0.999, elbow=None, lambda_c=1e-12, evaluate=True,
                   max_gain=10.0, q_scale=1.0, r_scale=1.0):
    """
    Train the three estimators on a training set and (optionally) evaluate them
    on the test block against HR-LOR, with the cubic-interpolation baseline.
    Returns (model, report dict).  The model carries the LR processing
    settings in model.meta['lr_processing'].
    """
    tr, va, te = split.slices(len(ds['lr_u']))
    for name, sl in (('training', tr), ('validation', va)):
        d = np.diff(ds['t_us'][sl])
        per = 1e6 / ds['settings']['f_hz']
        if np.any(np.abs(d - per) > 0.25 * per):
            log.warning("The %s block contains gaps in time; F/Q assume consecutive "
                        "snapshots. Choose a split that avoids them.", name)
    m = H.train(ds['lr_u'][tr], ds['lr_v'][tr], ds['hr_u'][tr], ds['hr_v'][tr],
                ds['lr_u'][va], ds['lr_v'][va], ds['hr_u'][va], ds['hr_v'][va],
                rank=rank, truncate_lr=truncate_lr, threshold=threshold, elbow=elbow,
                lambda_c=lambda_c, max_gain=max_gain,
                grids=dict(lr_x=ds['lr_x'], lr_y=ds['lr_y'],
                                              hr_x=ds['hr_x'], hr_y=ds['hr_y']))
    m.meta.update(lr_processing=ds['lr_processing'], training_settings=ds['settings'],
                  split=asdict(split), f_acq_hz=ds['settings']['f_hz'], units='px/frame')
    m.meta['test_tuning'] = dict(q_scale=float(q_scale), r_scale=float(r_scale))
    report = dict(r=m.r, r_lr=m.r_lr, hr_energy=m.meta['hr_energy'])
    if not evaluate or split.n_test == 0:
        return m, report
    lu, lv = ds['lr_u'][te], ds['lr_v'][te]
    u_lor, v_lor = m.lor_hr(ds['hr_u'][te], ds['hr_v'][te])
    if u_ref is None:                   # default: mean speed of the HR mean field
        mu, mv = m.reconstruct(np.zeros(m.r))
        u_ref = float(np.sqrt(mu ** 2 + mv ** 2).mean())
    report['u_ref'] = u_ref
    try:
        cu, cv = H.cubic_baseline(lu, lv, ds['lr_x'], ds['lr_y'], ds['hr_x'], ds['hr_y'])
        report['delta_cubic'] = H.delta_error(cu, cv, u_lor, v_lor, u_ref)
    except Exception as exc:                                     # noqa: BLE001
        log.warning("cubic baseline failed: %s", exc)
    for meth in m.methods:
        X = H.KalmanEstimator(m, method=meth, q_scale=q_scale, r_scale=r_scale).run(lu, lv)
        eu, ev = m.reconstruct(X)
        report[f'delta_{meth}'] = H.delta_error(eu, ev, u_lor, v_lor, u_ref)
        k_ref = ((u_lor - u_lor.mean(0)) ** 2 + (v_lor - v_lor.mean(0)) ** 2).mean()
        k_est = ((eu - eu.mean(0)) ** 2 + (ev - ev.mean(0)) ** 2).mean()
        report[f'tke_ratio_{meth}'] = float(k_est / k_ref)
    m.meta['test_report'] = report
    log.info("Test block: " + ", ".join(f"{k}={v:.4g}" for k, v in report.items()
                                          if isinstance(v, float)))
    return m, report


def check_processing(model, live: dict):
    """See vibe_hr.check_processing (kept here for backward compatibility)."""
    return H.check_processing(model, live)
