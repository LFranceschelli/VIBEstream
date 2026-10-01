"""
vibe_hr_live — HR estimation inside the VibeStream live stream.

One HRLive object is created by ebiv_session when 'HR estimation' is enabled
and handed to ebiv_utils.stream_camera:

    PIV worker thread   hr.update(U, V, frame_index)   after every LR field
    UI thread           hr.field()                     newest HR field (u, v)
                        hr.overlay.draw(...)           HR vectors, key [h]
                        hr.hud()                       one status line

The estimator step is cheap (steady-state gain: ~0.01 ms); the HR field is
reconstructed only when the UI draws it, at the display rate, never in the
PIV worker.  Frame indices, not wall-clock times, are used to bridge fields the
correlator dropped, so a missed field costs one prediction-only step.
"""

import os
import time
import logging
import threading

import numpy as np

import vibe_hr as H

log = logging.getLogger("vibe.hr.live")

# settings that define the LR grid or its geometry/sign: a mismatch refuses the model
CRITICAL = H.CRITICAL_SETTINGS


class HRLive:
    def __init__(self, model_path, method='kf', steady_state=True, f_acq=None,
                 live_settings=None, show=True, arrow_skip=2, arrow_scale=4.0,
                 q_scale=1.0, r_scale=1.0):
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"HR model not found: {model_path!r}")
        self.model = H.HRModel.load(model_path)
        self.model_path = model_path
        if method not in self.model.methods:
            raise ValueError(f"this HR model has no {method!r} estimator (it has: "
                             f"{', '.join(self.model.methods)}); choose another estimator "
                             "or retrain")
        self.method = method
        self.f_acq = float(f_acq or self.model.meta.get('f_acq_hz') or 0)
        if self.f_acq <= 0:
            raise ValueError("acquisition frequency unknown")
        self.period_us = 1e6 / self.f_acq

        # --- consistency with the live processing --------------------------
        self.mismatches = []
        if live_settings is not None:
            self.mismatches = H.check_processing(self.model, live_settings)
            crit = [m for m in self.mismatches if H.is_critical(m)]
            for m in self.mismatches:
                log.warning("HR model vs live settings: %s", m)
            if crit:
                raise ValueError("the HR model was trained with different live settings:\n  "
                                 + "\n  ".join(crit) +
                                 "\nRetrain it (offline step 'train HR model') or restore "
                                 "those settings.")
        if self.model.meta.get('units') not in (None, 'px/frame'):
            log.warning("HR model units are %r; the live stream gives px/frame.",
                        self.model.meta.get('units'))

        self.est = H.KalmanEstimator(self.model, method=method, steady_state=steady_state,
                                     q_scale=q_scale, r_scale=r_scale)
        self._lock = threading.Lock()
        self._x = None
        self._seq = 0
        self._cache = (None, None)
        self.show = bool(show)
        self.enabled = True
        self._t_step = []
        self._t_rec = []
        self._err_count = 0
        self.overlay = self._make_overlay(arrow_skip, arrow_scale)
        log.info("HR estimation ON: %s, method=%s%s, r=%d, r_lr=%d, HR grid %s. "
                 "Press [h] to switch the display between LR and HR vectors.",
                 os.path.basename(model_path), method,
                 " (steady-state gain)" if steady_state else "", self.model.r,
                 self.model.r_lr, tuple(self.model.hr_shape))

    # ------------------------------------------------------------------
    def _make_overlay(self, skip, scale):
        """VectorOverlay at the model's HR node positions (px of the full frame)."""
        m = self.model
        ts = m.meta.get('training_settings')
        if m.hr_x is None or not ts or m.meta.get('units') != 'px/frame':
            log.warning("This HR model has no pixel grid record (trained on data in another "
                        "convention): HR vectors cannot be drawn; estimation still runs.")
            return None
        import ebiv_viz as viz
        ws, nd = int(ts['hr_window']), int(ts['hr_step'])
        gy, gx = m.hr_x.shape
        ox = float(m.hr_x[0, 0]) - ws // 2
        oy = float(m.hr_y[0, 0]) - ws // 2
        ov = viz.VectorOverlay(gy, gx, ws, nd, origin=(ox, oy), arrow_skip=skip,
                               arrow_scale=scale)
        # exact positions: an imported grid need not be an integer-pixel lattice
        k = ov.skip
        ov.X0 = np.rint(m.hr_x[::k, ::k]).astype(np.int32).ravel()
        ov.Y0 = np.rint(m.hr_y[::k, ::k]).astype(np.int32).ravel()
        return ov

    @property
    def ready(self):
        return self._x is not None

    # ------------------------------------------------------------------
    #  PIV worker thread
    # ------------------------------------------------------------------
    def update(self, U, V, frame_index):
        if not self.enabled:
            return
        t0 = time.perf_counter()
        try:
            x = self.est.step_timed(frame_index * self.period_us, U, V, self.period_us)
        except Exception as exc:                                 # noqa: BLE001
            self._err_count += 1
            if self._err_count <= 3:
                log.error("HR estimation failed (%s: %s).", type(exc).__name__, exc)
            if self._err_count >= 10:
                log.error("HR estimation DISABLED after repeated failures.")
                self.enabled = False
            return
        with self._lock:
            self._x = x.copy()
            self._seq += 1
        dt = time.perf_counter() - t0
        self._t_step.append(dt)
        if len(self._t_step) > 4000:
            del self._t_step[:2000]

    # ------------------------------------------------------------------
    #  UI thread
    # ------------------------------------------------------------------
    def latent(self):
        with self._lock:
            return (None if self._x is None else self._x.copy()), self._seq

    def field(self):
        """Newest HR field (u, v) on the HR grid, reconstructed once per new state."""
        x, seq = self.latent()
        if x is None:
            return None, None
        if self._cache[0] != seq:
            t0 = time.perf_counter()
            self._cache = (seq, self.model.reconstruct(x))
            self._t_rec.append(time.perf_counter() - t0)
            if len(self._t_rec) > 2000:
                del self._t_rec[:1000]
        return self._cache[1]

    LABELS = {'kf': 'KF', 'lse': 'LSE', 'lse_vr': 'LSE+VR'}

    def banner(self):
        """(text, BGR colour) for the stream window's top-right status banner."""
        name = self.LABELS.get(self.method, self.method)
        if not self.enabled:
            return f"HR ESTIMATION FAILED ({name}) - see log", (0, 0, 255)
        if not self.ready:
            return f"HR ON: {name} - waiting for rt-EBIV fields", (0, 200, 255)
        if self.show:
            return f"HR ON: {name} - showing HR vectors  [h]", (0, 220, 0)
        return f"HR ON: {name} - showing LR vectors  [h]", (0, 220, 220)

    def hud(self):
        if not self.enabled:
            return "HR estimation: DISABLED (see log)"
        st = np.median(self._t_step) * 1e3 if self._t_step else float('nan')
        return (f"HR {self.method.upper()} r={self.model.r}: step {st:.2f} ms | "
                f"bridged {self.est.n_skipped} | {'HR' if self.show else 'LR'} vectors [h]")

    def close(self):
        if self._t_step:
            log.info("HR estimation: %d steps, %d missing periods bridged, step median "
                     "%.3f ms (p99 %.3f ms), HR reconstruction median %.3f ms.",
                     self.est.k, self.est.n_skipped, np.median(self._t_step) * 1e3,
                     np.percentile(self._t_step, 99) * 1e3,
                     (np.median(self._t_rec) * 1e3) if self._t_rec else float('nan'))


def live_settings_from_session(session):
    """The LR-processing record of the live stream, in vibe_train's terms."""
    run, cfg = session.run, session.control
    roi = cfg.roi.display_roi
    return dict(f_hz=float(run.f_acq), roi=list(roi) if roi is not None else None,
                window=int(cfg.piv.window_size), step=int(cfg.piv.node_distance),
                validate=bool(cfg.piv.validation), subpixel=bool(cfg.piv.subpixel),
                val_threshold=float(cfg.piv.val_threshold),
                val_epsilon=float(cfg.piv.val_epsilon),
                duty_cycle=float(run.trigger_duty_cycle),
                flip_x=bool(run.flip_x), flip_y=bool(run.flip_y),
                max_events_per_pixel=int(run.max_events_per_pixel),
                smooth_sigma=(float(run.frame_smooth_sigma) if run.frame_smooth_sigma else None),
                pulse_frames=run.trigger_mode != 'none')
