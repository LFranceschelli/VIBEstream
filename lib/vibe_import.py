"""
vibe_import — resolution-enhancement data computed OUTSIDE VibeStream.

Two entry points, both used by the GUI (Resolution enhancement > Train a
model..., sources 2 and 3) and by HR_Example.py:

    fields_training_set(...)   LR.mat + HR.mat (e.g. from the MATLAB pipeline)
                               -> a training set in the live convention; the
                               POD and the operators are then computed here
                               (vibe_train.train_from_set), as for a .raw.
    import_operators(...)      a .mat written by tools/matlab/vibe_export_model.m
                               at the end of Proc_Main_FullKF_DEF.m or
                               Proc_Main_EPOD_DEF.m -> HRModel, nothing
                               recomputed (POD bases, F, Q, C/R or M/Gamma/R
                               are taken as they are).

The live convention
    arrays (gy, gx), row 0 = TOP of the image, columns left -> right,
    u positive right, v positive DOWN, velocities in px/frame, node positions
    in pixels of the (flipped) full sensor frame.

How external data are brought into it (nothing is guessed from the velocity)
    The LR nodes of the file must be the live LR nodes (same ROI, window, step:
    the model is only valid for the processing it was trained on).  Matching
    the two grids gives:
      - the array layout: MATLAB (nx, ny) arrays are transposed; the axis along
        which X varies is detected from X itself;
      - the orientation: columns sorted by increasing X; rows by the declared
        direction of the file's Y axis (y_up=True: Y grows upwards, as in most
        PIV software; then v changes sign);
      - the length scale b [px per file length unit], from a least-squares fit
        of the node positions (checked: regular grid, same scale in x and y);
      - the velocity scale to px/frame:  'px/frame' -> 1;  'm/s' with X, Y in
        mm -> 1000 b / f;  'unit/s' (the length unit of X, Y per second) ->
        b / f.
    The HR grid is placed with the same fit.

    For imported operators this is exact linear algebra: with z_live = s T z
    (T a signed permutation, s the velocity scale) the model in live units has
    Phi' = T Phi, Sigma' = s Sigma, mean' = s T mean, and every operator
    (F, Q, C, R, M, Gamma, R_lse, R_vr) is unchanged, because the POD
    coefficients themselves are unchanged.

What cannot be checked
    That the external LR fields really come from the live rt-EBIV processing
    (validation, smoothing, sub-pixel, event accumulation...).  The model is
    stamped with the live settings as DECLARED at import (meta
    'lr_processing_declared'), so that later changes of frequency, ROI,
    window, step or flips are still refused.
"""

import os
import time
import logging
from dataclasses import dataclass, asdict

import numpy as np

import vibe_hr as H

log = logging.getLogger("vibe.import")

UNITS = ('px/frame', 'm/s', 'unit/s')
FORMAT = 'vibe_hr_model_v1'


@dataclass
class ExternalSettings:
    y_up: bool = True                 # the file's Y axis points up (v positive up)
    velocity_units: str = 'm/s'       # 'px/frame' | 'm/s' (X, Y in mm) | 'unit/s'
    f_hz: float = 0.0                 # rate of the external data; 0 = live f_acq


# ==========================================================================
#  live LR grid
# ==========================================================================

def live_lr_grid(roi, window, step, sensor_wh=(1280, 720)):
    """Node centres (px) of the live rt-EBIV grid, as vibe._PIVProc: X, Y (gy, gx)."""
    W, Hh = int(sensor_wh[0]), int(sensor_wh[1])
    if roi is None:
        x0, x1, y0, y1 = 0, W, 0, Hh
    else:
        x0, x1, y0, y1 = [int(v) for v in roi]
        x0, x1, y0, y1 = max(0, x0), min(W, x1), max(0, y0), min(Hh, y1)
    gx = (x1 - x0 - window) // step + 1
    gy = (y1 - y0 - window) // step + 1
    if gx < 1 or gy < 1:
        raise ValueError(f"ROI {roi} is smaller than one {window}-px window")
    xc = x0 + np.arange(gx) * step + window // 2
    yc = y0 + np.arange(gy) * step + window // 2
    return np.meshgrid(xc.astype(np.float64), yc.astype(np.float64))


# ==========================================================================
#  array orientation and the grid fit
# ==========================================================================

@dataclass
class ArrayMap:
    """How a file array (a, b) becomes a live array (rows = y down, cols = x right)."""
    transpose: bool
    flip_rows: bool
    flip_cols: bool

    def apply(self, A):
        """(..., a, b) -> (..., gy, gx)"""
        A = np.asarray(A)
        if self.transpose:
            A = np.swapaxes(A, -1, -2)
        if self.flip_rows:
            A = A[..., ::-1, :]
        if self.flip_cols:
            A = A[..., :, ::-1]
        return np.ascontiguousarray(A)

    def modes_matlab(self, phi, shape):
        """Rows of phi = MATLAB column-major flattening of `shape` -> C order, live."""
        k = phi.shape[1]
        A = np.moveaxis(np.asarray(phi).reshape(tuple(shape) + (k,), order='F'), -1, 0)
        return self.apply(A).reshape(k, -1).T


def array_map(X, Y, y_up):
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    if X.ndim != 2 or X.shape != Y.shape:
        raise ValueError(f"grid coordinates must be 2-D and alike, got X {X.shape}, Y {Y.shape}")
    d0 = np.abs(np.diff(X, axis=0)).mean() if X.shape[0] > 1 else 0.0
    d1 = np.abs(np.diff(X, axis=1)).mean() if X.shape[1] > 1 else 0.0
    transpose = d0 > d1                        # X varies along axis 0: (nx, ny), MATLAB ndgrid
    Xt, Yt = (X.T, Y.T) if transpose else (X, Y)
    flip_cols = Xt.shape[1] > 1 and Xt[0, -1] < Xt[0, 0]
    rising = Yt.shape[0] > 1 and Yt[-1, 0] > Yt[0, 0]     # Y grows with the row index
    flip_rows = rising if y_up else (Yt.shape[0] > 1 and not rising)
    return ArrayMap(bool(transpose), bool(flip_rows), bool(flip_cols))


@dataclass
class Conversion:
    """External (file) convention -> live convention.  See the module docstring."""
    lr_map: ArrayMap
    hr_map: ArrayMap
    px_per_unit: float
    ax: float                  # px = ax + b X
    ay: float                  # py = ay + sy b Y
    sy: float                  # -1 if y_up else +1
    vel_scale: float           # file velocity -> px/frame
    fit_rms_px: float
    lr_shape_file: tuple
    lr_shape: tuple
    hr_shape: tuple
    settings: dict

    @property
    def v_sign(self):
        return self.sy                       # v_live = sy * v_file * scale

    def fields(self, U, V, which='lr', scale=None):
        m = self.lr_map if which == 'lr' else self.hr_map
        s = self.vel_scale if scale is None else scale
        return m.apply(U) * s, m.apply(V) * (s * self.sy)

    def grid_px(self, X, Y, which='lr'):
        m = self.lr_map if which == 'lr' else self.hr_map
        return (self.ax + self.px_per_unit * m.apply(X),
                self.ay + self.sy * self.px_per_unit * m.apply(Y))

    def record(self):
        d = asdict(self)
        return dict(px_per_unit=self.px_per_unit, vel_scale=self.vel_scale,
                    fit_rms_px=self.fit_rms_px, lr_map=d['lr_map'], hr_map=d['hr_map'],
                    lr_shape_file=list(self.lr_shape_file), **self.settings)

    def summary(self):
        lm = self.lr_map
        orient = ("x-first (MATLAB) arrays, transposed" if lm.transpose else "rows = y arrays")
        return (f"file LR grid {tuple(self.lr_shape_file)} ({orient}) -> live {tuple(self.lr_shape)}; "
                f"{self.px_per_unit:.4g} px per file length unit (fit rms {self.fit_rms_px:.2g} px); "
                f"velocity x {self.vel_scale:.4g} -> px/frame; v sign "
                f"{'flipped (Y up)' if self.sy < 0 else 'kept (Y down)'}; HR grid "
                f"{tuple(self.hr_shape)}")


def fit_conversion(lr_X, lr_Y, hr_X, hr_Y, live_X, live_Y, ext: ExternalSettings, f_hz):
    """Match the file's LR nodes to the live LR nodes.  Raises ValueError with the reason."""
    if ext.velocity_units not in UNITS:
        raise ValueError(f"velocity_units must be one of {UNITS}")
    lm = array_map(lr_X, lr_Y, ext.y_up)
    Xl, Yl = lm.apply(np.asarray(lr_X, float)), lm.apply(np.asarray(lr_Y, float))
    if Xl.shape != live_X.shape:
        raise ValueError(
            f"the LR grid of the external data is {Xl.shape[0]} x {Xl.shape[1]} (rows x cols), "
            f"the live rt-EBIV grid (ROI, window, step, sensor) is {live_X.shape[0]} x "
            f"{live_X.shape[1]}.  The LR fields must come from the SAME processing as the live "
            f"stream: set the ROI / window / step they were computed with.")
    sy = -1.0 if ext.y_up else 1.0
    n = Xl.size
    # px = ax + b X ;  py = ay + sy b Y   (one scale: square pixels, isotropic units)
    A = np.zeros((2 * n, 3))
    A[:n, 0], A[:n, 2] = 1.0, Xl.ravel()
    A[n:, 1], A[n:, 2] = 1.0, sy * Yl.ravel()
    rhs = np.concatenate([live_X.ravel(), live_Y.ravel()])
    sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    ax, ay, b = (float(v) for v in sol)
    rms = float(np.sqrt(np.mean((A @ sol - rhs) ** 2)))
    step = float(np.median(np.diff(live_X[0]))) if live_X.shape[1] > 1 else \
        float(np.median(np.diff(live_Y[:, 0])))
    if not b > 0:
        raise ValueError("the file's X axis runs opposite to the image x axis, or the grid is "
                         "degenerate: cannot map it")
    if rms > 0.1 * step:
        # separate scales tell the user WHY
        bx = np.polyfit(Xl.ravel(), live_X.ravel(), 1)[0] if Xl.shape[1] > 1 else b
        by = np.polyfit(sy * Yl.ravel(), live_Y.ravel(), 1)[0] if Yl.shape[0] > 1 else b
        raise ValueError(
            f"the file's LR nodes do not map onto the live nodes (rms {rms:.2g} px, step "
            f"{step:g} px; scale x {bx:.4g}, y {by:.4g} px/unit).  Wrong 'Y axis up' setting, "
            f"non-uniform grid, or a different window/step.")
    f = float(ext.f_hz or f_hz)
    if f <= 0:
        raise ValueError("the acquisition rate of the external data is needed")
    vs = {'px/frame': 1.0, 'm/s': 1000.0 * b / f, 'unit/s': b / f}[ext.velocity_units]
    hm = array_map(hr_X, hr_Y, ext.y_up) if hr_X is not None else None
    hr_shape = hm.apply(np.asarray(hr_X)).shape if hm is not None else ()
    return Conversion(lm, hm, b, ax, ay, sy, vs, rms, tuple(np.shape(lr_X)), Xl.shape,
                      tuple(hr_shape), dict(asdict(ext), f_hz=f))


def _hr_display_settings(Xp, Yp):
    """Approximate HR window/step (px) for the vector overlay; positions are exact."""
    d = []
    if Xp.shape[1] > 1:
        d.append(np.median(np.diff(Xp[0])))
    if Xp.shape[0] > 1:
        d.append(np.median(np.diff(Yp[:, 0])))
    nd = max(1, int(round(float(np.mean(d))))) if d else 8
    return dict(hr_window=2 * nd, hr_step=nd)


def declared_processing(live_settings, f_hz):
    """The LR-processing record an imported model is stamped with."""
    keys = ('roi', 'window', 'step', 'flip_x', 'flip_y', 'phase_locked')
    rec = {k: live_settings[k] for k in keys if k in live_settings}
    rec['f_hz'] = float(f_hz)
    return rec


# ==========================================================================
#  A. external LR / HR fields -> training set
# ==========================================================================

def load_fields(path):
    """LR.mat / HR.mat (MATLAB layout, v7 or v7.3) -> dict(X, Y, U, V), U (Nt, a, b)."""
    return H.load_mat_dataset(path)


def fields_training_set(lr_path, hr_path, live_settings, ext: ExternalSettings,
                        sensor_wh=(1280, 720), progress=None):
    """
    External LR / HR fields -> training set dict for vibe_train.train_from_set,
    in the live convention.  live_settings: vibe_hr_live.live_settings_from_session().
    A VibeStream training set (.npz from vibe_train.save_training_set) is
    loaded as it is (it is already in the live convention).
    """
    if str(lr_path).lower().endswith('.npz'):
        import vibe_train as VT
        ds = VT.load_training_set(lr_path)
        ds.setdefault('source', dict(kind='training set', file=os.path.abspath(lr_path)))
        return ds, None
    t0 = time.perf_counter()
    lr = load_fields(lr_path)
    if progress:
        progress('loaded LR')
    hr = load_fields(hr_path)
    if lr['U'].shape[0] != hr['U'].shape[0]:
        raise ValueError(f"LR has {lr['U'].shape[0]} snapshots, HR {hr['U'].shape[0]}: they "
                         "must be paired one to one")
    lx, ly = live_lr_grid(live_settings.get('roi'), live_settings['window'],
                          live_settings['step'], sensor_wh)
    conv = fit_conversion(lr['X'], lr['Y'], hr['X'], hr['Y'], lx, ly, ext,
                          live_settings['f_hz'])
    lu, lv = conv.fields(lr['U'], lr['V'], 'lr')
    hu, hv = conv.fields(hr['U'], hr['V'], 'hr')
    lxp, lyp = conv.grid_px(lr['X'], lr['Y'], 'lr')
    hxp, hyp = conv.grid_px(hr['X'], hr['Y'], 'hr')
    f = conv.settings['f_hz']
    nt = lu.shape[0]
    settings = dict(f_hz=f, **_hr_display_settings(hxp, hyp), external=True,
                    roi=live_settings.get('roi'), lr_window=live_settings['window'],
                    lr_step=live_settings['step'])
    ds = dict(lr_u=lu.astype(np.float32), lr_v=lv.astype(np.float32),
              hr_u=hu.astype(np.float32), hr_v=hv.astype(np.float32),
              t_us=np.arange(nt) * 1e6 / f,
              lr_x=lxp.astype(np.float32), lr_y=lyp.astype(np.float32),
              hr_x=hxp.astype(np.float32), hr_y=hyp.astype(np.float32),
              settings=settings, lr_processing=declared_processing(live_settings, f),
              seconds=time.perf_counter() - t0,
              source=dict(kind='external fields', lr=os.path.abspath(lr_path),
                          hr=os.path.abspath(hr_path), conversion=conv.record()))
    log.info("External fields: %d snapshots; %s", nt, conv.summary())
    return ds, conv


# ==========================================================================
#  B. operators computed elsewhere -> HRModel
# ==========================================================================

def read_operator_file(path):
    """The .mat of vibe_export_model.m (v7 or v7.3) -> dict of arrays / strings."""
    if H._is_hdf5(path):
        import h5py
        out = {}
        with h5py.File(path, 'r') as f:
            for k in f.keys():
                if k.startswith('#'):
                    continue
                a = np.asarray(f[k])
                if a.dtype == np.uint16:                      # MATLAB char
                    out[k] = ''.join(map(chr, a.ravel()))
                else:
                    out[k] = a.T
        return out
    from scipy.io import loadmat
    d = loadmat(path)
    out = {}
    for k, v in d.items():
        if k.startswith('__'):
            continue
        out[k] = str(v[0]) if v.dtype.kind == 'U' else v
    return out


def _vec(a):
    return np.asarray(a, dtype=np.float64).ravel()


def import_operators(path, live_settings, ext: ExternalSettings, sensor_wh=(1280, 720)):
    """
    Operators computed elsewhere (vibe_export_model.m) -> HRModel in the live
    convention.  See the module docstring for the conversion.
    """
    d = read_operator_file(path)
    fmt = str(d.get('vibe_format', ''))
    if fmt != FORMAT:
        raise ValueError(f"{os.path.basename(path)} is not an operator file written by "
                         f"vibe_export_model.m (vibe_format = {fmt!r}, expected {FORMAT!r})")
    need = ('PhiLR', 'SigmaLR', 'PhiMF', 'SigmaMF', 'UmLR', 'VmLR', 'UmMF', 'VmMF',
            'XLR', 'YLR', 'XMF', 'YMF', 'F', 'Q')
    miss = [k for k in need if k not in d]
    if miss:
        raise ValueError(f"operator file: missing {miss}")
    r, r_lr = int(_vec(d['r'])[0]), int(_vec(d['r_LR'])[0])
    lx, ly = live_lr_grid(live_settings.get('roi'), live_settings['window'],
                          live_settings['step'], sensor_wh)
    conv = fit_conversion(d['XLR'], d['YLR'], d['XMF'], d['YMF'], lx, ly, ext,
                          live_settings['f_hz'])
    s, sy = conv.vel_scale, conv.sy
    shp_lr, shp_hr = np.shape(d['XLR']), np.shape(d['XMF'])

    def basis(phi, shp, m):
        phi = np.asarray(phi, np.float64)
        n = int(np.prod(shp))
        if phi.shape[0] != 2 * n:
            raise ValueError(f"modes have {phi.shape[0]} rows, the grid {shp} needs {2 * n}")
        return np.vstack([m.modes_matlab(phi[:n], shp), sy * m.modes_matlab(phi[n:], shp)])

    def mean(Um, Vm, m):
        return np.concatenate([(m.apply(np.asarray(Um, float)) * s).ravel(),
                               (m.apply(np.asarray(Vm, float)) * s * sy).ravel()])

    sLR, sMF = _vec(d['SigmaLR']), _vec(d['SigmaMF'])
    phi_lr = basis(np.asarray(d['PhiLR'])[:, :r_lr], shp_lr, conv.lr_map)
    phi_hr = basis(np.asarray(d['PhiMF'])[:, :r], shp_hr, conv.hr_map)
    get = lambda k: np.asarray(d[k], np.float64) if k in d else None     # noqa: E731
    M = get('M')
    if M is not None:
        M = M.T                               # script: A ~ B M (r_LR x r); here x = M psi
    Gamma = _vec(d['Gain']) if 'Gain' in d else None
    lxp, lyp = conv.grid_px(d['XLR'], d['YLR'], 'lr')
    hxp, hyp = conv.grid_px(d['XMF'], d['YMF'], 'hr')
    f = conv.settings['f_hz']
    meta = dict(source='imported operators', source_file=os.path.abspath(path),
                source_method=str(d.get('method', '?')), created=str(d.get('created', '')),
                n_train=int(_vec(d['n_train'])[0]) if 'n_train' in d else None,
                rank_mode='imported', truncate_lr=r_lr < len(sLR),
                lr_zero_last=r_lr == len(sLR), hr_zero_last=r == len(sMF),
                hr_energy=float((sMF[:r] ** 2).sum() / (sMF ** 2).sum()),
                lr_energy=float((sLR[:r_lr] ** 2).sum() / (sLR ** 2).sum()),
                f_acq_hz=f, units='px/frame', conversion=conv.record(),
                lr_processing=declared_processing(live_settings, f),
                lr_processing_declared=True,
                training_settings=dict(f_hz=f, **_hr_display_settings(hxp, hyp)))
    if Gamma is not None:
        meta.update(gamma_range=[float(Gamma.min()), float(Gamma.max())])
    model = H.HRModel(conv.lr_shape, conv.hr_shape, mean(d['UmLR'], d['VmLR'], conv.lr_map),
                      mean(d['UmMF'], d['VmMF'], conv.hr_map), phi_lr, sLR[:r_lr] * s,
                      phi_hr, sMF[:r] * s, get('C'), get('F'), get('Q'), get('R_kf'),
                      M, get('R_lse'), Gamma, get('R_vr'), sLR * s, sMF * s,
                      lxp, lyp, hxp, hyp, meta)
    if not model.methods:
        raise ValueError("the operator file has no complete estimator (C + R_kf, M + R_lse, "
                         "or M + Gain + R_vr)")
    log.info("Imported %s: methods %s, r=%d, r_lr=%d; %s", os.path.basename(path),
             model.methods, r, r_lr, conv.summary())
    return model, conv


# ==========================================================================
#  preview: the conversion from the grids only (cheap; for the GUI)
# ==========================================================================

def read_grids(path, names):
    """Only the named (small) variables of a .mat file."""
    if H._is_hdf5(path):
        import h5py
        with h5py.File(path, 'r') as f:
            return {k: np.asarray(f[k]).T for k in names if k in f}
    from scipy.io import loadmat
    d = loadmat(path, variable_names=list(names))
    return {k: d[k] for k in names if k in d}


_PREVIEW = {}


def preview(source, files, live_settings, ext: ExternalSettings, sensor_wh=(1280, 720)):
    """
    Check external inputs without loading the velocity data.
    source 'fields': files = (lr_path, hr_path); 'operators': files = (path,).
    Returns a one-line summary of the conversion; raises ValueError / OSError.
    Cached on (paths, mtimes, settings).
    """
    key = (source, tuple((p, os.path.getmtime(p)) for p in files if p),
           tuple(sorted((k, str(v)) for k, v in live_settings.items())),
           tuple(sorted(asdict(ext).items())), tuple(sensor_wh))
    if key in _PREVIEW:
        res = _PREVIEW[key]
        if isinstance(res, Exception):
            raise res
        return res
    try:
        if source == 'fields' and str(files[0]).lower().endswith('.npz'):
            with np.load(files[0], allow_pickle=False) as d:
                import json
                meta = json.loads(str(d['meta']))
                shp = d['lr_u'].shape
            res = (f"VibeStream training set: {shp[0]} fields, LR grid {shp[1:]}, recorded with "
                   f"the processing it carries (checked like a .raw)")
            rec = meta.get('lr_processing') or {}
            bad = H.check_processing({'lr_processing': rec}, live_settings)
            crit = [b for b in bad if H.is_critical(b)]
            if crit:
                raise ValueError("the training set was computed with other live settings: "
                                 + "; ".join(crit))
        else:
            if source == 'fields':
                g1 = read_grids(files[0], ('X', 'Y'))
                g2 = read_grids(files[1], ('X', 'Y'))
                lX, lY, hX, hY = g1.get('X'), g1.get('Y'), g2.get('X'), g2.get('Y')
            else:
                g = read_grids(files[0], ('XLR', 'YLR', 'XMF', 'YMF', 'vibe_format'))
                if 'XLR' not in g:
                    raise ValueError("not an operator file of vibe_export_model.m (no XLR)")
                lX, lY, hX, hY = g['XLR'], g['YLR'], g.get('XMF'), g.get('YMF')
            if lX is None or hX is None:
                raise ValueError("the file(s) have no X / Y grid variables")
            lx, ly = live_lr_grid(live_settings.get('roi'), live_settings['window'],
                                  live_settings['step'], sensor_wh)
            res = fit_conversion(lX, lY, hX, hY, lx, ly, ext, live_settings['f_hz']).summary()
    except (ValueError, OSError, KeyError) as exc:
        _PREVIEW[key] = exc
        raise
    _PREVIEW[key] = res
    return res
