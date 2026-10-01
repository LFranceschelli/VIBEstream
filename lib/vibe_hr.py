"""
vibe_hr — real-time high-resolution (HR) estimation from low-resolution (LR)
rt-EBIV fields, in POD space, with the three estimators of

    Franceschelli et al., "Real-Time Estimation of High-Resolution Flow Fields
    and Reduced-Order Coordinates from Event-Based Imaging Velocimetry",
    Sections 2.1-2.3:
        'kf'      Method I    direct Kalman filter          (Sec. 2.2.2)
        'lse'     Method II   LSE + Kalman filter           (Sec. 2.2.3)
        'lse_vr'  Method III  variance-rescaled LSE + KF    (Sec. 2.2.4)

  OFFLINE  train(...) -> HRModel   (vibe_train.py builds the LR/HR training
                                    set from a .raw recording, no MATLAB)
        1. fluctuations about the mean, [u; v] stacked per snapshot
        2. economy SVD  Z = Phi Sigma Psi^T  of the TRAINING LR and HR sets
        3. HR rank r by the elbow criterion (pod_elbow_rank, ported from
           MATLAB); LR: elbow (truncate_lr=True) or full rank
        4. coefficients psi = Sigma^-1 Phi^T z  (training bases for every set)
        5. F (Eq. 9), Q (Eq. 14)                     common
           C (Eq. 12), R (Eq. 16)                     Method I
           M (Eq. 20), R_lse (Eq. 26)                 Method II
           Gamma (Eq. 27), R_vr (Eq. 29)              Method III

  ONLINE  KalmanEstimator(model, method).step(u_lr, v_lr) -> latent HR state
      project the LR snapshot (Eq. 5), predict with F/Q, correct with the
      method's measurement, optionally reconstruct Phi_r Sigma_r x + mean.

ARRAY LAYOUT
    Snapshot sequences are (Nt, *grid): u[k] is one field.  A single field is
    (*grid).  Fields are flattened in C order internally; the SVD is invariant
    to how grid points are ordered, so results do not depend on it as long as
    training and online data use the same grid.  load_mat_dataset() converts
    MATLAB (nx, ny, Nt) arrays to (Nt, nx, ny).

VERIFIED
    Method I and pod_elbow_rank against the MATLAB code run in Octave 8.4 on
    synthetic data: machine precision with matlab_compat=True and the
    all-snapshot mean.  Methods II/III against Proc_Main_EPOD_DEF.m the same
    way (Gamma as std ratio with the MaxGain cap, tuneQ/tuneR, the
    intermittent-mode output smoothing).

WHERE THIS DIFFERS FROM THE MATLAB SCRIPT (on purpose, switchable)
    * mean      : training mean by default (the only one available online).
                  The script uses the mean over ALL snapshots, test included.
                  Pass lr_mean=/hr_mean= to reproduce it.
    * LR rank   : with truncate_lr=False the full NUMERICAL rank is kept; the
                  script keeps all min(n, Nt) modes and zeroes the last one
                  (SigmaLR_inv(end,end) = 0).  matlab_compat=True reproduces it.
    * LSE init  : x0 = first LSE (+VR) estimate, P0 = I, as Proc_Main_EPOD_DEF.m.
"""

import os
import json
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

__version__ = "1.0.0"
log = logging.getLogger("vibe.hr")

_EPS = np.finfo(np.float64).eps


# ==========================================================================
#  linear algebra helpers (MATLAB-compatible)
# ==========================================================================

def pinv_matlab(A):
    """pinv with MATLAB's default tolerance max(size(A)) * eps(norm(A))."""
    return np.linalg.pinv(A, rcond=max(A.shape) * _EPS)


def pod(Z):
    """
    Economy SVD of a snapshot matrix Z (n_points x Nt).

    Returns Phi (n x m), sigma (m,), Psi (Nt x m) with Z = Phi diag(sigma) Psi^T.
    Signs of the singular vectors are arbitrary (as in MATLAB); every quantity
    downstream is invariant to them.
    """
    Phi, s, Vt = np.linalg.svd(Z, full_matrices=False)
    return Phi, s, Vt.T


def _movmean_shrink_omitnan(x, span):
    """MATLAB movmean(x, span, 'omitnan', 'Endpoints', 'shrink')."""
    n = len(x)
    if span % 2:
        kb = kf = (span - 1) // 2
    else:                                   # even: current + previous centred
        kb, kf = span // 2, span // 2 - 1
    out = np.full(n, np.nan)
    for i in range(n):
        w = x[max(0, i - kb):min(n, i + kf + 1)]
        w = w[np.isfinite(w)]
        if w.size:
            out[i] = w.mean()
    return out


def _fillmissing_linear_nearest(x):
    """MATLAB fillmissing(x, 'linear', 'EndValues', 'nearest')."""
    x = np.array(x, dtype=np.float64)
    ok = np.isfinite(x)
    if ok.all() or not ok.any():
        return x
    idx = np.arange(len(x))
    x[~ok] = np.interp(idx[~ok], idx[ok], x[ok])     # np.interp holds end values
    return x


def pod_elbow_rank(sigma, threshold=0.999, kmin=2, kmax=np.inf, smooth='none',
                   span=25, fir_order=24, fir_cutoff=0.0313, fir_filtfilt=True):
    """
    Port of pod_elbow_rank.m (Raiola et al. elbow criterion).

    lambda = sigma^2, residual energy delta2(k) = sum_{j>k} lambda_j, and for
    k = 2..r-1
        F(k) = (delta2(k+1) - delta2(k)) / (delta2(k) - delta2(k-1))
             = lambda_{k+1} / lambda_k
    optionally smoothed ('none' | 'movmean' | 'fir').  k_opt is the first k in
    [kmin, min(kmax, r-1)] with F(k) >= threshold, else the k with F closest
    to the threshold.  Returns a dict with the same fields as the MATLAB struct
    (k indices are MATLAB's: number of retained modes).
    """
    s = np.asarray(sigma, dtype=np.float64)
    if s.ndim == 2 and s.shape[0] == s.shape[1]:
        s = np.diag(s)
    s = s.ravel()
    s = s[np.isfinite(s)]
    r = s.size
    if r < 4:
        raise ValueError("Need at least 4 singular values to compute F(k) for k=2..r-1.")
    lam = s ** 2
    delta2 = np.zeros(r + 1)
    delta2[0] = lam.sum()
    tail = np.cumsum(lam[::-1])                   # tail[m-1] = sum of last m
    for k in range(1, r + 1):
        m = r - k
        delta2[k] = tail[m - 1] if m > 0 else 0.0
    k_grid = np.arange(2, r)                      # 2..r-1
    num = delta2[k_grid + 1] - delta2[k_grid]
    den = delta2[k_grid] - delta2[k_grid - 1]
    with np.errstate(divide='ignore', invalid='ignore'):
        F_raw = np.where(np.abs(den) < _EPS, np.nan, num / den)

    smooth = str(smooth).lower()
    if smooth == 'none':
        F_s = F_raw.copy()
    elif smooth == 'movmean':
        F_s = F_raw.copy() if span <= 1 else _movmean_shrink_omitnan(F_raw, int(round(span)))
    elif smooth == 'fir':
        from scipy.signal import firwin, filtfilt, lfilter
        x = _fillmissing_linear_nearest(F_raw)
        b = firwin(int(fir_order) + 1, fir_cutoff)        # = fir1(order, cutoff): Hamming, DC gain 1
        if fir_filtfilt:
            F_s = filtfilt(b, [1.0], x, padtype='odd', padlen=3 * (len(b) - 1))  # MATLAB padding
        else:
            F_s = lfilter(b, [1.0], x)
    else:
        raise ValueError("smooth must be 'none', 'movmean' or 'fir'")

    kmax_eff = min(kmax, r - 1)
    mask = (k_grid >= kmin) & (k_grid <= kmax_eff) & np.isfinite(F_s)
    if not mask.any():
        raise ValueError("No valid F(k) values in the requested k range.")
    kg, Fm = k_grid[mask], F_s[mask]
    hit = np.nonzero(Fm >= threshold)[0]
    k_best = int(kg[np.argmin(np.abs(Fm - threshold))])
    k_opt = int(kg[hit[0]]) if hit.size else k_best
    return dict(k_opt=k_opt, k_best=k_best, threshold=threshold, smoothType=smooth,
                k_grid=k_grid, F_raw=F_raw, F_smooth=F_s, delta2=delta2, lambda_=lam, r=r)


def elbow_rank(sigma, threshold=0.999, **opts):
    """Retained rank from pod_elbow_rank() (options: kmin, kmax, smooth, span, fir_*)."""
    return pod_elbow_rank(sigma, threshold, **opts)['k_opt']


def numerical_rank(sigma, shape):
    tol = max(shape) * _EPS * float(sigma[0])
    return int(np.sum(sigma > tol))


def _stack(u, v):
    """(Nt, *grid) u, v  ->  (2*n_grid, Nt) snapshot matrix [u; v]."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if u.shape != v.shape:
        raise ValueError(f"u {u.shape} and v {v.shape} differ")
    nt = u.shape[0]
    Z = np.concatenate([u.reshape(nt, -1), v.reshape(nt, -1)], axis=1).T
    if not np.all(np.isfinite(Z)):
        raise ValueError("NaN/Inf in the velocity data: mask or fill them before "
                         "training (the SVD would propagate them everywhere).")
    return Z


def _stack1(u, v):
    """one field (*grid) -> (2*n_grid,)"""
    return np.concatenate([np.ravel(u), np.ravel(v)]).astype(np.float64)


# ==========================================================================
#  the trained model
# ==========================================================================

@dataclass
class HRModel:
    """Everything identified offline.  Saved to / loaded from a single .npz."""
    # grids
    lr_shape: Tuple[int, ...]
    hr_shape: Tuple[int, ...]
    lr_mean: np.ndarray            # (2*n_lr,)
    hr_mean: np.ndarray            # (2*n_hr,)
    # bases (truncated)
    phi_lr: np.ndarray             # (2*n_lr, r_lr)
    sigma_lr: np.ndarray           # (r_lr,)
    phi_hr: np.ndarray             # (2*n_hr, r)
    sigma_hr: np.ndarray           # (r,)
    # state-space operators (C, R: Method I; None in a model imported with
    # only the LSE operators, see vibe_import)
    C: Optional[np.ndarray]        # (r_lr, r)
    F: np.ndarray                  # (r, r)
    Q: np.ndarray                  # (r, r)
    R: Optional[np.ndarray]        # (r_lr, r_lr)
    # LSE-based methods (Sec. 2.2.3-2.2.4); None in models from older versions
    M: Optional[np.ndarray] = None       # (r, r_lr)   LSE operator, Eq. 20
    R_lse: Optional[np.ndarray] = None   # (r, r)      Eq. 26
    Gamma: Optional[np.ndarray] = None   # (r,)        variance rescaling, Eq. 27
    R_vr: Optional[np.ndarray] = None    # (r, r)      Eq. 29
    # full spectra, for inspection / plots
    sigma_lr_all: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sigma_hr_all: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # grids coordinates (optional, for plots/interpolation)
    lr_x: Optional[np.ndarray] = None
    lr_y: Optional[np.ndarray] = None
    hr_x: Optional[np.ndarray] = None
    hr_y: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)

    @property
    def r(self):
        return int(self.phi_hr.shape[1])

    @property
    def r_lr(self):
        return int(self.phi_lr.shape[1])

    # -- projections / reconstructions -----------------------------------
    def inv_sigma(self, which='lr'):
        """1/sigma, with the last entry zeroed when trained with matlab_compat."""
        s = self.sigma_lr if which == 'lr' else self.sigma_hr
        inv = 1.0 / s
        if self.meta.get(f'{which}_zero_last'):
            inv[-1] = 0.0
        return inv

    def project_lr(self, u, v):
        """LR field(s) -> LR coefficients (Eq. 5).  (Nt,*grid)->(Nt, r_lr) or (*grid)->(r_lr,)"""
        inv = self.inv_sigma('lr')
        if np.ndim(u) == len(self.lr_shape):
            z = _stack1(u, v) - self.lr_mean
            return (self.phi_lr.T @ z) * inv
        Z = _stack(u, v) - self.lr_mean[:, None]
        return ((self.phi_lr.T @ Z) * inv[:, None]).T

    def project_hr(self, u, v):
        """HR field(s) -> HR coefficients on the truncated basis."""
        inv = self.inv_sigma('hr')
        if np.ndim(u) == len(self.hr_shape):
            z = _stack1(u, v) - self.hr_mean
            return (self.phi_hr.T @ z) * inv
        Z = _stack(u, v) - self.hr_mean[:, None]
        return ((self.phi_hr.T @ Z) * inv[:, None]).T

    def reconstruct(self, x, add_mean=True):
        """
        HR coefficients -> HR field(s) (Eq. 30).
        x (r,) -> (u, v) each (*hr_shape);  x (Nt, r) -> (Nt, *hr_shape) each.
        """
        x = np.asarray(x, dtype=np.float64)
        B = self._recon_basis()                             # (2n, r)
        n = int(np.prod(self.hr_shape))
        if x.ndim == 1:
            z = B @ x
            if add_mean:
                z = z + self.hr_mean
            return z[:n].reshape(self.hr_shape), z[n:].reshape(self.hr_shape)
        Z = x @ B.T
        if add_mean:
            Z = Z + self.hr_mean
        nt = x.shape[0]
        return (Z[:, :n].reshape((nt,) + tuple(self.hr_shape)),
                Z[:, n:].reshape((nt,) + tuple(self.hr_shape)))

    def _recon_basis(self):
        """Phi_r Sigma_r, computed once (the online reconstruction is one mat-vec)."""
        B = self.__dict__.get('_B')
        if B is None or B.shape != self.phi_hr.shape:
            B = np.ascontiguousarray(self.phi_hr * self.sigma_hr)
            self.__dict__['_B'] = B
        return B

    def lse(self, psi_lr, vr=False):
        """Instantaneous LSE estimate of the HR coefficients (Eq. 21 / Eq. 28)."""
        if self.M is None:
            raise RuntimeError("this model has no LSE operator (retrain with this version)")
        x = np.asarray(psi_lr) @ self.M.T if np.ndim(psi_lr) == 2 else self.M @ psi_lr
        return x * self.Gamma if vr else x

    @property
    def methods(self):
        """The estimators whose operators this model carries."""
        m = []
        if self.C is not None and self.R is not None:
            m.append('kf')
        if self.M is not None and self.R_lse is not None:
            m.append('lse')
        if self.M is not None and self.Gamma is not None and self.R_vr is not None:
            m.append('lse_vr')
        return tuple(m)

    def lor_hr(self, u, v, add_mean=True):
        """HR low-order reconstruction (the paper's HR-LOR reference)."""
        return self.reconstruct(self.project_hr(u, v), add_mean)

    # -- persistence ---------------------------------------------------------
    def save(self, path):
        arrs = {k: getattr(self, k) for k in (
            'lr_mean', 'hr_mean', 'phi_lr', 'sigma_lr', 'phi_hr', 'sigma_hr',
            'F', 'Q', 'sigma_lr_all', 'sigma_hr_all')}
        for k in ('C', 'R', 'lr_x', 'lr_y', 'hr_x', 'hr_y', 'M', 'R_lse', 'Gamma', 'R_vr'):
            if getattr(self, k) is not None:
                arrs[k] = getattr(self, k)
        header = dict(self.meta, lr_shape=list(self.lr_shape),
                      hr_shape=list(self.hr_shape), version=__version__,
                      r_hr=self.r, r_lr_modes=self.r_lr, methods=list(self.methods))
        np.savez_compressed(path, header=json.dumps(header, default=str), **arrs)
        log.info("HR model saved to %s (r=%d, r_lr=%d)", path, self.r, self.r_lr)
        return path

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        h = json.loads(str(d['header']))
        for k in ('r_hr', 'r_lr_modes'):
            h.pop(k, None)
        kw = {k: d[k] for k in d.files if k != 'header'}
        kw.setdefault('C', None)
        kw.setdefault('R', None)
        return cls(lr_shape=tuple(h.pop('lr_shape')), hr_shape=tuple(h.pop('hr_shape')),
                   meta=h, **kw)

    def summary(self):
        m = self.meta
        return (f"HRModel [{', '.join(self.methods)}]: r={self.r} (HR, "
                f"{m.get('hr_energy', float('nan')) * 100:.1f}% energy), r_lr={self.r_lr} "
                f"({'elbow' if m.get('truncate_lr', True) else 'full'}), "
                f"LR grid {self.lr_shape}, HR grid {self.hr_shape}, "
                f"trained on {m.get('n_train')} snapshots, R from {m.get('n_val')}")


# ==========================================================================
#  OFFLINE: identification
# ==========================================================================

def train(lr_u, lr_v, hr_u, hr_v, lr_val_u, lr_val_v, hr_val_u, hr_val_v,
          rank='elbow', truncate_lr=True, rank_lr=None, threshold=0.999, elbow=None,
          lambda_c=1e-12, q=None, r=None, lr_mean=None, hr_mean=None,
          grids=None, matlab_compat=False, max_gain=10.0) -> HRModel:
    """
    Identify the operators of all three estimators from paired LR/HR snapshots:
        common      POD bases, F, Q                  (Sec. 2.1, 2.2.1)
        'kf'        C, R                             (Method I,  Eq. 12, 15-16)
        'lse'       M, R_lse                         (Method II, Eq. 20, 26)
        'lse_vr'    Gamma, R_vr                      (Method III, Eq. 27, 29)

    lr_u, lr_v, hr_u, hr_v      TRAINING sequences, (Nt, *grid).  CONSECUTIVE
                                snapshots at the acquisition rate: F is a
                                one-step model.
    lr_val_*, hr_val_*          VALIDATION sequences (for R, R_lse, R_vr).
    rank                        'elbow' or int: HR truncation r.
    truncate_lr                 True: elbow on the LR spectrum too (the script's
                                FlagLRRankLimit = 1).  False: full LR rank
                                (FlagLRRankLimit = 0; the paper's Sec. 2.1).
    rank_lr                     int: force the LR rank (overrides truncate_lr).
    threshold, elbow            pod_elbow_rank() threshold and options
                                (dict: kmin, kmax, smooth, span, fir_*).
    lambda_c                    ridge parameter for C and M (Eq. 12, 20).
    q, r                        if both given: Q = q I, R = r I (FlagKFuser=1,
                                KF only; LSE covariances are still estimated).
    lr_mean, hr_mean            stacked [u; v] means; default: training mean.
    max_gain                    cap on the variance-rescaling gain Gamma
                                (Proc.MaxGain of Proc_Main_EPOD_DEF.m).
    matlab_compat               reproduce SigmaLR_inv(end,end) = 0: with the
                                full LR rank the LAST LR mode is zeroed, and
                                'full' means all min(n, Nt) modes as in the
                                script (size(SigmaLR, 1)) rather than the
                                numerical rank.
    """
    t0 = time.perf_counter()
    elbow = dict(elbow or {})
    lr_shape, hr_shape = tuple(np.shape(lr_u)[1:]), tuple(np.shape(hr_u)[1:])
    Zlr = _stack(lr_u, lr_v)
    Zhr = _stack(hr_u, hr_v)
    nt = Zlr.shape[1]
    if Zhr.shape[1] != nt:
        raise ValueError("LR and HR training sets have different lengths")
    mean_src = 'given' if lr_mean is not None else 'train'
    if lr_mean is None:
        lr_mean = Zlr.mean(axis=1)
    if hr_mean is None:
        hr_mean = Zhr.mean(axis=1)
    lr_mean = np.asarray(lr_mean, dtype=np.float64).ravel()
    hr_mean = np.asarray(hr_mean, dtype=np.float64).ravel()
    Zlr -= lr_mean[:, None]
    Zhr -= hr_mean[:, None]

    # --- POD ----------------------------------------------------------------
    phi_hr, s_hr, _ = pod(Zhr)
    phi_lr, s_lr, _ = pod(Zlr)

    r_ = elbow_rank(s_hr, threshold, **elbow) if rank == 'elbow' else int(rank)
    if rank_lr is not None:
        r_lr, lr_mode = int(rank_lr), 'fixed'
    elif truncate_lr:
        r_lr, lr_mode = elbow_rank(s_lr, threshold, **elbow), 'elbow'
    else:
        r_lr = len(s_lr) if matlab_compat else numerical_rank(s_lr, Zlr.shape)
        lr_mode = 'full'
    r_ = min(r_, len(s_hr))
    r_lr = min(r_lr, len(s_lr))
    log.info("HR rank r=%d (%s), LR rank r_lr=%d (%s)", r_, rank, r_lr, lr_mode)

    phi_hr_r, s_hr_r = phi_hr[:, :r_], s_hr[:r_]
    phi_lr_r, s_lr_r = phi_lr[:, :r_lr], s_lr[:r_lr]
    with np.errstate(divide='ignore'):
        inv_hr = np.where(s_hr_r > 0, 1.0 / s_hr_r, 0.0)
        inv_lr = np.where(s_lr_r > 0, 1.0 / s_lr_r, 0.0)
    if matlab_compat:
        if r_ == len(s_hr):
            inv_hr[-1] = 0.0
        if r_lr == len(s_lr):
            inv_lr[-1] = 0.0
            log.info("matlab_compat: last LR mode (%d) zeroed, as SigmaLR_inv(end,end)=0.", r_lr)

    # --- coefficients (projection on the training bases) ----------------------
    A = (phi_hr_r.T @ Zhr) * inv_hr[:, None]          # (r, Nt)     A^HR  = Psi_MF^T
    Y = (phi_lr_r.T @ Zlr) * inv_lr[:, None]          # (r_lr, Nt)  A^LR  = Psi_LR^T
    Zlr_v = _stack(lr_val_u, lr_val_v) - lr_mean[:, None]
    Zhr_v = _stack(hr_val_u, hr_val_v) - hr_mean[:, None]
    n_val = Zlr_v.shape[1]
    A_val = (phi_hr_r.T @ Zhr_v) * inv_hr[:, None]
    Y_val = (phi_lr_r.T @ Zlr_v) * inv_lr[:, None]

    def cov(E, n):
        c = (E @ E.T) / max(E.shape[1] - 1, 1)
        return 0.5 * (c + c.T) + 1e-12 * np.eye(n)

    # --- common: F (Eq. 9), Q (Eq. 13-14) ------------------------------------
    Am, Ap = A[:, :-1], A[:, 1:]
    F = Ap @ pinv_matlab(Am)
    # --- Method I: C (Eq. 12), R (Eq. 15-16) ---------------------------------
    C = np.linalg.solve((A @ A.T + lambda_c * np.eye(r_)).T, (Y @ A.T).T).T
    Q = cov(Ap - F @ Am, r_)
    R = cov(Y_val - C @ A_val, r_lr)
    # user-defined q, r (FlagKFuser = 1) are stored in meta and used by the
    # KF method at run time; the model always keeps the estimated Q and R.
    # --- Method II: M (Eq. 20), R_lse (Eq. 26) -------------------------------
    M = np.linalg.solve((Y @ Y.T + lambda_c * np.eye(r_lr)).T, (A @ Y.T).T).T   # (r, r_lr)
    R_lse = cov(M @ Y_val - A_val, r_)
    # --- Method III: Gamma (Eq. 27), R_vr (Eq. 29) ----------------------------
    # As Proc_Main_EPOD_DEF.m: Gain = std(A)/std(M A_LR) per mode (centred,
    # N-1), capped at max_gain, NaN -> 1.  With zero-mean coefficients this is
    # exactly Eq. 27; the std form also holds when the mean is not the
    # training mean.
    A_tilde = M @ Y
    with np.errstate(divide='ignore', invalid='ignore'):
        Gamma = A.std(axis=1, ddof=1) / A_tilde.std(axis=1, ddof=1)
    n_clip = int(np.sum(Gamma > max_gain))
    Gamma = np.where(Gamma > max_gain, max_gain, Gamma)       # inf -> max_gain too
    Gamma = np.where(np.isnan(Gamma), 1.0, Gamma)
    R_vr = cov(Gamma[:, None] * (M @ Y_val) - A_val, r_)

    g = grids or {}
    meta = dict(methods=['kf', 'lse', 'lse_vr'], rank_mode=str(rank),
                truncate_lr=bool(truncate_lr), rank_lr_mode=lr_mode,
                threshold=threshold, elbow=elbow, lambda_c=lambda_c, q=q, r=r,
                matlab_compat=bool(matlab_compat),
                lr_zero_last=bool(inv_lr[-1] == 0.0), hr_zero_last=bool(inv_hr[-1] == 0.0),
                n_train=nt, n_val=n_val, mean_source=mean_src,
                hr_energy=float((s_hr[:r_] ** 2).sum() / (s_hr ** 2).sum()),
                lr_energy=float((s_lr[:r_lr] ** 2).sum() / (s_lr ** 2).sum()),
                gamma_range=[float(Gamma.min()), float(Gamma.max())],
                max_gain=float(max_gain), gamma_n_capped=n_clip,
                train_seconds=round(time.perf_counter() - t0, 2),
                created=time.strftime("%Y-%m-%dT%H:%M:%S"))
    model = HRModel(lr_shape, hr_shape, lr_mean, hr_mean, phi_lr_r, s_lr_r, phi_hr_r,
                    s_hr_r, C, F, Q, R, M, R_lse, Gamma, R_vr, s_lr, s_hr,
                    g.get('lr_x'), g.get('lr_y'), g.get('hr_x'), g.get('hr_y'), meta)
    log.info(model.summary())
    return model


def train_kf(*args, rank_lr='elbow', elbow_error=None, **kw):
    """Backward-compatible wrapper of train() (rank_lr='elbow'|'full'|int)."""
    if rank_lr == 'elbow':
        return train(*args, truncate_lr=True, **kw)
    if rank_lr == 'full':
        return train(*args, truncate_lr=False, **kw)
    return train(*args, rank_lr=int(rank_lr), **kw)


# ==========================================================================
#  ONLINE: steady-state gain, BLAS threads
# ==========================================================================

def steady_state_gain(F, H, Q, R, P0=None, tol=1e-10, max_iter=20000):
    """
    Iterate the Riccati recursion of the filter (Eq. 17-18, from P0 = I) until
    the a-priori covariance stops changing.  Returns (K_inf, P_pred_inf, n_iter).

    For a time-invariant system the time-varying filter converges to this
    gain, so after its transient (~n_iter steps) the steady-state filter gives
    the same estimate at O(r^2 + r*n_meas) per step instead of O(r^3).
    NOT what the MATLAB scripts do (they update P every step); an
    implementation choice for real-time use.
    """
    r = F.shape[0]
    P = np.eye(r) if P0 is None else np.asarray(P0, dtype=np.float64)
    I = np.eye(r)
    P_pred = F @ P @ F.T + Q
    rel = np.inf
    for it in range(1, max_iter + 1):
        S = H @ P_pred @ H.T + R
        K = np.linalg.solve(S.T, (P_pred @ H.T).T).T
        P = (I - K @ H) @ P_pred
        P_next = F @ P @ F.T + Q
        rel = np.linalg.norm(P_next - P_pred) / max(np.linalg.norm(P_pred), 1e-300)
        P_pred = P_next
        if rel < tol:
            break
    else:
        log.warning("steady_state_gain: not converged after %d iterations (rel %.2e)",
                    max_iter, rel)
    S = H @ P_pred @ H.T + R
    K = np.linalg.solve(S.T, (P_pred @ H.T).T).T
    return K, P_pred, it


def limit_blas_threads(n=1):
    """
    Limit BLAS threads (returns a context manager, or None).  The KF works on
    small matrices; letting OpenBLAS/MKL spread them over all cores, while
    the acquisition and PIV threads also want the CPU, measured 3.4 ms/step
    single-threaded vs 50 ms/step with 2 threads on a loaded 2-core machine.
    Needs threadpoolctl; otherwise set OPENBLAS_NUM_THREADS=1 (or
    MKL_NUM_THREADS=1) in the environment BEFORE importing numpy.
    """
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(n)
    except ImportError:
        log.warning("threadpoolctl not installed: set OPENBLAS_NUM_THREADS=%d "
                    "before starting Python instead.", n)
        return None


# ==========================================================================
#  ONLINE: the estimator
# ==========================================================================

class KalmanEstimator:
    """
    Online estimator, one class for the three methods of the paper:
        method='kf'      Method I   direct KF, measurement psi_lr = C x + eta
        method='lse'     Method II  y = M psi_lr (LSE), measurement y = x + eta
        method='lse_vr'  Method III y = Gamma M psi_lr, measurement y = x + eta
    Same F, Q; same recursion; initial state pinv(C) psi_1 for 'kf' (script),
    the first LSE estimate for 'lse'/'lse_vr'.  One call per LR snapshot:

        est = KalmanEstimator(model)
        for fld in vibe.velocity(...):               # or start()/latest()
            x = est.step(fld.u, fld.v)               # latent HR state (r,)
            u_hr, v_hr = est.field()                 # only if the field is needed

    The first step initialises x = pinv(C) psi_lr, P = P0 (script: I).
    step(None) (no measurement available) propagates the model only, like
    the script's intermittent mode.

    steady_state=True uses the converged gain (steady_state_gain()) instead of
    updating P every step: same estimate after the filter's transient, much
    cheaper per step.  Default False = the script's time-varying filter.

    The LR field must come from the SAME processing as the training LR data:
    same ROI, window, step, validation, same array layout, units and sign
    conventions.  Nothing can check units or signs for you: a mismatch gives
    wrong estimates, not an error.
        lr_scale      multiplies the LR field (e.g. px/frame -> m/s)
        lr_transform  callable (u, v) -> (u, v) applied first, e.g. for a
                      model trained on MATLAB (nx, ny) arrays with v positive
                      up, fed from VIBE (rows = y, v positive down):
                          lr_transform=lambda u, v: (u.T, -v.T)
                      (check the row order too: flip if y runs the other way)
    """

    METHODS = ('kf', 'lse', 'lse_vr')

    def __init__(self, model: HRModel, method='kf', P0=None, lr_scale=1.0,
                 steady_state=False, lr_transform=None, q_scale=1.0, r_scale=1.0):
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}")
        if method not in model.methods:
            raise ValueError(f"this model has no operators for method {method!r} "
                             f"(it has: {', '.join(model.methods) or 'none'})")
        self.m = model
        self.method = method
        self.lr_scale = float(lr_scale)
        self.lr_transform = lr_transform
        r = model.r
        self.P0 = np.eye(r) if P0 is None else np.asarray(P0, dtype=np.float64)
        self._proj = (model.phi_lr * model.inv_sigma('lr')).T   # (r_lr, 2n_lr)
        self._I = np.eye(r)
        # measurement:  y = G psi_lr,  y = H x + noise(R)
        Q = model.Q
        if method == 'kf':
            G = None                                  # y = psi_lr
            H, R = model.C, model.R
        elif method == 'lse':
            G, H, R = model.M, self._I, model.R_lse                  # Eq. 21-24
        else:
            G, H, R = model.Gamma[:, None] * model.M, self._I, model.R_vr   # Eq. 28
        q, rr = model.meta.get('q'), model.meta.get('r')
        if q is not None and rr is not None:          # FlagKFuser = 1 (both scripts)
            Q, R = q * np.eye(r), rr * np.eye(H.shape[0])
        Q, R = Q * float(q_scale), R * float(r_scale)  # tuneQ, tuneR (EPOD script)
        self._G, self._H, self._Q, self._R = G, H, Q, R
        self._H_pinv = pinv_matlab(H) if method == 'kf' else self._I
        self.steady_state = bool(steady_state)
        if self.steady_state:
            K, self.P_inf, self.ss_iters = steady_state_gain(model.F, H, Q, R, self.P0)
            self._K = K
            self._A = (self._I - K @ H) @ model.F           # x+ = A x + K y
            log.info("Steady-state gain (%s): converged in %d iterations.",
                     method, self.ss_iters)
        self.reset()

    def reset(self):
        self.x = None
        self.P = None
        self.k = 0
        self.n_updates = 0
        self.t_last_us = None
        self.n_skipped = 0
        self._n_reinit = getattr(self, '_n_reinit', 0)

    @property
    def initialised(self):
        return self.x is not None

    def measure(self, u, v):
        """LR field -> LR coefficient vector psi_lr (Eq. 5)."""
        if self.lr_transform is not None:
            u, v = self.lr_transform(u, v)
        if np.shape(u) != tuple(self.m.lr_shape):
            raise ValueError(f"LR field is {np.shape(u)}, the model was trained on "
                             f"{tuple(self.m.lr_shape)}: same ROI/window/step needed.")
        z = _stack1(u, v) * self.lr_scale - self.m.lr_mean
        return self._proj @ z

    def step(self, u=None, v=None, psi=None):
        """
        Advance one time step.  Give either the LR field (u, v), or its
        coefficients psi, or nothing (no measurement: prediction only).
        Returns the updated latent HR state x (r,).
        """
        if psi is None and u is not None:
            psi = self.measure(u, v)
        y = None if psi is None else (psi if self._G is None else self._G @ psi)
        F, H = self.m.F, self._H
        if self.x is None:
            if y is None:
                raise ValueError("the first step needs a measurement")
            self.x = self._H_pinv @ y          # KF: pinv(C) psi (script); LSE: the LSE estimate
            self.P = self.P0.copy()
            self.k = 1
            self.n_updates = 1
            return self.x
        if self.steady_state:
            self.x = (F @ self.x) if y is None else (self._A @ self.x + self._K @ y)
            self.k += 1
            self.n_updates += y is not None
            return self.x
        # prediction (Eq. 18)
        x_pred = F @ self.x
        P_pred = F @ self.P @ F.T + self._Q
        if y is None:
            self.x, self.P = x_pred, P_pred
        else:                                  # correction (Eq. 17, 19 / 24, 25)
            innov = y - H @ x_pred
            S = H @ P_pred @ H.T + self._R
            K = np.linalg.solve(S.T, (P_pred @ H.T).T).T
            self.x = x_pred + K @ innov
            self.P = (self._I - K @ H) @ P_pred
            self.n_updates += 1
        self.k += 1
        return self.x

    def step_timed(self, t_us, u, v, period_us, max_gap=50):
        """
        step() for a live stream that may skip fields (start()/latest() drops
        a field when the correlator is busy).  F is a ONE-STEP model at the
        acquisition rate, so each missing period is bridged with a
        prediction-only step before the measurement is used.

        t_us       camera time of the field (VelocityField.t_us)
        period_us  1e6 / f, the acquisition period the model was trained at
        max_gap    beyond this many missing periods the state is re-initialised
                   from the measurement instead of predicted blind.
        """
        if self.x is not None and self.t_last_us is not None:
            gap = int(round((t_us - self.t_last_us) / period_us)) - 1
            if gap < 0:
                return self.x                      # same or older field: ignore
            if gap > max_gap:
                n_re = getattr(self, '_n_reinit', 0) + 1
                self._n_reinit = n_re
                if n_re <= 3 or n_re % 100 == 0:
                    log.warning("KF: %d periods without a field; re-initialising "
                                "[%d so far]. Is the rt-EBIV keeping up with f?", gap, n_re)
                self.reset()
                self._n_reinit = n_re
            else:
                for _ in range(gap):
                    self.step()
                self.n_skipped += gap
        self.t_last_us = t_us
        return self.step(u, v)

    def field(self, add_mean=True):
        """Current HR estimate as (u, v) on the HR grid."""
        if self.x is None:
            raise RuntimeError("no state yet")
        return self.m.reconstruct(self.x, add_mean)

    def run(self, lr_u, lr_v, ds=None, alpha_smooth=None):
        """
        Batch run over a sequence (Nt, *grid), as the scripts' KF loops.
        ds            intermittent mode: measurement used only every ds-th
                      step (mod(j-1, DS) == 0).
        alpha_smooth  output smoothing of the EPOD script (intermittent mode
                      only there): out_j = (1-a) x_j + a out_{j-1}; the
                      filter state itself is not smoothed.
        Returns X (Nt, r).
        """
        if self.lr_scale == 1.0 and self.lr_transform is None:
            psi = self.m.project_lr(lr_u, lr_v)
        else:
            psi = np.array([self.measure(a, b) for a, b in zip(lr_u, lr_v)])
        self.reset()
        X = np.empty((psi.shape[0], self.m.r))
        for j in range(psi.shape[0]):
            use = (j == 0) or (ds is None) or (j % ds == 0)
            X[j] = self.step(psi=psi[j] if use else None)
            if alpha_smooth is not None and j > 0:
                X[j] = (1.0 - alpha_smooth) * X[j] + alpha_smooth * X[j - 1]
        return X


# ==========================================================================
#  evaluation helpers
# ==========================================================================

def delta_error(u_est, v_est, u_ref, v_ref, u_ref_scale=1.0, mask=None):
    """
    Paper Eq. 33-34: per snapshot, RMS over grid points of |U_est - U_ref|,
    averaged over snapshots and divided by U_ref.  Arrays (Nt, *grid).
    mask (*grid) bool restricts the spatial average (delta_alpha).
    """
    e2 = (np.asarray(u_est) - u_ref) ** 2 + (np.asarray(v_est) - v_ref) ** 2
    nt = e2.shape[0]
    e2 = e2.reshape(nt, -1)
    if mask is not None:
        e2 = e2[:, np.ravel(mask)]
    return float(np.mean(np.sqrt(e2.mean(axis=1))) / u_ref_scale)


def cubic_baseline(lr_u, lr_v, lr_x, lr_y, hr_x, hr_y):
    """
    Direct cubic interpolation of LR fields onto the HR grid (the paper's
    baseline).  Grids must be rectilinear; lr_x/lr_y are 2-D (*lr grid) arrays.
    Points of the HR grid outside the LR hull are extrapolated linearly by
    the spline (RegularGridInterpolator, method='cubic', no fill value).
    """
    from scipy.interpolate import RegularGridInterpolator
    xs, ys, transpose = _axes(lr_x, lr_y)
    pts = np.stack([np.ravel(hr_y), np.ravel(hr_x)], axis=-1) if not transpose else \
        np.stack([np.ravel(hr_x), np.ravel(hr_y)], axis=-1)
    out_u = np.empty((len(lr_u),) + np.shape(hr_x))
    out_v = np.empty_like(out_u)
    for k in range(len(lr_u)):
        for src, dst in ((lr_u[k], out_u), (lr_v[k], out_v)):
            f = RegularGridInterpolator((ys, xs) if not transpose else (xs, ys), src,
                                        method='cubic', bounds_error=False, fill_value=None)
            dst[k] = f(pts).reshape(np.shape(hr_x))
    return out_u, out_v


def _axes(X, Y):
    """1-D axes of a rectilinear meshgrid, and whether x runs along axis 0."""
    X, Y = np.asarray(X), np.asarray(Y)
    if np.allclose(X[0, :], X[0, 0]):          # x constant along axis 1 -> x varies on axis 0
        return X[:, 0], Y[0, :], True
    return X[0, :], Y[:, 0], False


# ==========================================================================
#  I/O
# ==========================================================================

def load_mat_dataset(path):
    """
    Load LR.mat / HR.mat as saved by the MATLAB pipeline: X, Y (nx, ny) and
    U, V (nx, ny, Nt).  Returns dict(X, Y, U, V) with U, V as (Nt, nx, ny).
    Handles v7 (scipy) and v7.3/HDF5 (h5py) files.
    """
    if _is_hdf5(path):                            # v7.3
        import h5py
        with h5py.File(path, 'r') as f:
            # HDF5 stores MATLAB arrays transposed
            out = {k: np.asarray(f[k], dtype=np.float64).T for k in ('X', 'Y', 'U', 'V')}
    else:
        from scipy.io import loadmat
        d = loadmat(path, variable_names=['X', 'Y', 'U', 'V'])
        out = {k: np.asarray(d[k], dtype=np.float64) for k in ('X', 'Y', 'U', 'V')}
    out['U'] = np.moveaxis(out['U'], -1, 0)
    out['V'] = np.moveaxis(out['V'], -1, 0)
    return out


def _is_hdf5(path):
    """True for MATLAB v7.3 (HDF5, with or without the 512-byte MATLAB header)."""
    with open(path, 'rb') as fh:
        head = fh.read(1024)
    sig = b'\x89HDF\r\n\x1a\n'
    return head.startswith(sig) or head[512:520] == sig


# ==========================================================================
#  model header (fast) and processing consistency
# ==========================================================================

_HEADER_CACHE = {}


def read_model_header(path):
    """
    The metadata of a saved model WITHOUT loading its POD modes: np.load on an
    .npz is lazy, so only 'header' (and, for old models, the small sigma
    arrays) is read.  Cached by (path, mtime).  Returns a dict with at least
    r_hr, r_lr_modes, lr_shape, hr_shape, methods, plus everything in meta.
    """
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_mtime_ns)
    if key in _HEADER_CACHE:
        return _HEADER_CACHE[key]
    with np.load(path, allow_pickle=False) as d:
        h = json.loads(str(d['header']))
        if 'r_hr' not in h:
            h['r_hr'] = int(d['sigma_hr'].shape[0])
            h['r_lr_modes'] = int(d['sigma_lr'].shape[0])
            h['methods'] = ['kf'] + (['lse', 'lse_vr'] if 'M' in d.files else [])
    h['file_mtime'] = time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime))
    _HEADER_CACHE[key] = h
    return h


CRITICAL_SETTINGS = ('f_hz', 'roi', 'window', 'step', 'flip_x', 'flip_y', 'pulse_frames')
# 'pulse_frames': True = one pseudo-image per laser pulse (trigger auto/external),
# False = fixed-dt accumulation (trigger 'none').  Models saved before this name
# was introduced carry it as 'phase_locked'; processing_record() renames it.
_LEGACY_KEYS = {'phase_locked': 'pulse_frames'}


def processing_record(model_or_meta):
    """The model's LR-processing record, with legacy key names updated."""
    meta = model_or_meta.meta if hasattr(model_or_meta, 'meta') else model_or_meta
    ref = meta.get('lr_processing') or {}
    return {_LEGACY_KEYS.get(k, k): v for k, v in ref.items()}


def check_processing(model_or_meta, live: dict):
    """
    Compare the live LR processing with what a model was trained on.
    model_or_meta: HRModel, its meta dict, or read_model_header() output.
    Returns a list of 'key: live X vs trained Y' strings (empty = consistent).
    Keys you do not know should be omitted from `live`.
    """
    ref = processing_record(model_or_meta)
    live = {_LEGACY_KEYS.get(k, k): v for k, v in live.items()}
    if not ref:
        return ["the model carries no LR-processing record (trained outside vibe_train): "
                "consistency cannot be checked"]

    def same(a, b):
        if a is None or b is None or isinstance(a, (bool, str)) or isinstance(b, (bool, str)):
            return a == b
        try:
            return bool(np.allclose(np.asarray(a, float), np.asarray(b, float)))
        except (TypeError, ValueError):
            return a == b

    return [f"{k}: live {live[k]!r} vs trained {v!r}"
            for k, v in ref.items() if k in live and not same(live[k], v)]


def is_critical(mismatch):
    return mismatch.split(':')[0] in CRITICAL_SETTINGS
