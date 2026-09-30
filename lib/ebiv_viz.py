"""
EBIV Release 5.2 — Real-time visualisation.

Two independently-throttled outputs:

    the main image window  — pseudo-frame, vectors over the DisplayROI, the
                             ControlROI rectangle, and a status HUD;
    a strip chart          — U_target, U_control_filtered, optionally
                             U_control_raw, and the pump command V_pump.

Both are drawn with OpenCV primitives into preallocated buffers.  There is no
matplotlib anywhere in the live path: a matplotlib redraw costs tens of
milliseconds and would dominate the loop.

Neither output is allowed to influence control timing.  The controller runs
on its own thread at its own rate; if rendering stalls, the control loop is
unaffected and the strip chart simply skips samples.
"""

import time
from collections import deque

import cv2
import numpy as np


# ==========================================================================
#  Vector overlay
# ==========================================================================

class VectorOverlay:
    """
    Draws the PIV vector field.  All grid geometry is precomputed once; only
    the arrow endpoints change per frame.
    """

    def __init__(self, grid_y, grid_x, window_size, node_distance,
                 origin=(0, 0), arrow_skip=1, arrow_scale=4.0,
                 colormap=cv2.COLORMAP_JET):
        """
        origin : (x, y) pixel offset of the DisplayROI within the image that
                 is actually shown.  Use the DisplayROI origin when the full
                 sensor frame is displayed, or (0, 0) when the frame has
                 already been cropped to the DisplayROI.
        """
        self.skip = max(1, int(arrow_skip))
        self.scale = float(arrow_scale)
        self.colormap = colormap

        gy = np.arange(0, grid_y, self.skip)
        gx = np.arange(0, grid_x, self.skip)
        GX, GY = np.meshgrid(gx, gy)
        half = window_size // 2
        self.X0 = (GX * node_distance + half + int(origin[0])).astype(np.int32).ravel()
        self.Y0 = (GY * node_distance + half + int(origin[1])).astype(np.int32).ravel()
        self.shape = GX.shape

    def draw(self, img, U, V, mag_max=None):
        """Draw arrows onto a BGR image.  U, V are the full grid."""
        Ud = U[::self.skip, ::self.skip].ravel()
        Vd = V[::self.skip, ::self.skip].ravel()
        M = np.hypot(Ud, Vd)

        finite = np.isfinite(M)
        if not np.any(finite):
            return img

        mmax = float(mag_max) if mag_max else float(np.nanmax(M[finite]))
        if not np.isfinite(mmax) or mmax <= 0:
            mmax = 1.0
        norm = np.zeros(M.size, dtype=np.uint8)
        norm[finite] = np.clip(M[finite] / mmax * 255.0, 0, 255).astype(np.uint8)
        colors = cv2.applyColorMap(norm.reshape(-1, 1), self.colormap).reshape(-1, 3)

        X1 = self.X0 + np.nan_to_num(Ud * self.scale).astype(np.int32)
        Y1 = self.Y0 + np.nan_to_num(Vd * self.scale).astype(np.int32)

        # Only draw arrows that are finite. Iterating zipped 1-D arrays avoids
        # Release 4.0's per-arrow divmod() and .flat[] indexing.
        sel = np.nonzero(finite)[0]
        for i in sel:
            c = colors[i]
            cv2.arrowedLine(img, (int(self.X0[i]), int(self.Y0[i])),
                            (int(X1[i]), int(Y1[i])),
                            (int(c[0]), int(c[1]), int(c[2])), 1, tipLength=0.3)
        return img


def draw_control_roi(img, control_roi, offset=(0, 0), label="ControlROI",
                     color=(0, 255, 255), thickness=2):
    """
    Draw the ControlROI rectangle.

    control_roi is in full-sensor coordinates.  `offset` is subtracted, so
    pass (0, 0) when the full sensor frame is displayed and the DisplayROI
    origin when the image has been cropped to the DisplayROI.
    """
    x0 = int(control_roi[0] - offset[0])
    x1 = int(control_roi[1] - offset[0])
    y0 = int(control_roi[2] - offset[1])
    y1 = int(control_roi[3] - offset[1])
    cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), color, thickness)
    # corner ticks make the region unmistakable even over a busy field
    L = max(8, min(24, (x1 - x0) // 6))
    for (cx, cy, dx, dy) in ((x0, y0, 1, 1), (x1 - 1, y0, -1, 1),
                             (x0, y1 - 1, 1, -1), (x1 - 1, y1 - 1, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * L, cy), color, thickness + 1)
        cv2.line(img, (cx, cy), (cx, cy + dy * L), color, thickness + 1)
    # Label at the BOTTOM-RIGHT of the rectangle: the top-left corner is where
    # the status panel lives, and the two used to overlap.
    tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
    lx = min(max(0, x1 - tw - 2), img.shape[1] - tw - 2)
    ly = min(y1 + 15, img.shape[0] - 4)
    cv2.putText(img, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),
                3, cv2.LINE_AA)
    cv2.putText(img, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                cv2.LINE_AA)
    return img


# ==========================================================================
#  Status HUD
# ==========================================================================

_STATE_COLORS = {
    'MANUAL':      (200, 200, 200),
    'CLOSED_LOOP': (0, 255, 0),
    'HOLD':        (0, 200, 255),
    'SAFE':        (0, 0, 255),
}


def draw_status(img, sup, units='px/frame', extra_lines=(), origin=(16, 26),
                line_h=26, scale=0.6):
    """
    Compact status block: controller state, the three velocities, the pump
    command and the valid-vector count.

    U_control is labelled explicitly so that nobody mistakes a ControlROI
    statistic for a calibrated bulk jet velocity.
    """
    state = sup.state
    col = _STATE_COLORS.get(state, (255, 255, 255))

    def fmt(v):
        return f"{v:8.3f}" if np.isfinite(v) else "     ---"

    frac = (sup.n_valid / sup.n_total) if sup.n_total else 0.0
    vc = (0, 255, 0) if frac >= sup.cfg.measurement.min_valid_fraction else (0, 165, 255)

    rows = [(f"[{state}]", col, True),
            (f"U_target   {fmt(sup.u_target)} {units}", (120, 255, 120), False),
            (f"U_ctrl_flt {fmt(sup.u_filtered)} {units}", (255, 200, 120), False),
            (f"U_ctrl_raw {fmt(sup.u_raw)} {units}", (200, 160, 100), False),
            (f"V_pump     {sup.v_command:8.3f} V", (120, 200, 255), False),
            (f"valid      {sup.n_valid:4d}/{sup.n_total:<4d} ({frac:5.1%})", vc, False)]
    if sup.invalid_reason:
        rows.append((f"! {sup.invalid_reason[:44]}", (0, 200, 255), False))
    rows += [(line, (200, 200, 200), False) for line in extra_lines]

    # A translucent panel behind the text.  Without it the block is drawn
    # straight over the vector field and over the ROI labels, and neither is
    # readable — the numbers you most need at a glance are the ones sitting on
    # top of the busiest part of the image.
    x, y0 = origin
    w_px = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale,
                               2 if b else 1)[0][0] for t, _, b in rows)
    pad = 8
    x0p, y0p = max(0, x - pad), max(0, y0 - line_h + 4)
    x1p = min(img.shape[1], x + w_px + pad)
    y1p = min(img.shape[0], y0 + line_h * (len(rows) - 1) + pad)
    if x1p > x0p and y1p > y0p:
        sub = img[y0p:y1p, x0p:x1p]
        cv2.addWeighted(sub, 0.25, np.zeros_like(sub), 0.75, 0, dst=sub)
        cv2.rectangle(img, (x0p, y0p), (x1p - 1, y1p - 1), (70, 70, 70), 1)

    y = y0
    for text, c, bold in rows:
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                    3 if bold else 2, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, c,
                    2 if bold else 1, cv2.LINE_AA)
        y += line_h
    return img


# ==========================================================================
#  Strip chart
# ==========================================================================

class StripChart:
    """
    Rolling time-history plot rendered with OpenCV.

    Two stacked panes sharing a time axis:
        top     U_target, U_control_filtered, U_control_raw
        bottom  V_pump

    Samples are appended from the control thread (cheap: a deque append) and
    the canvas is redrawn only when render() is called, at plot_rate_hz.
    """

    COL_TARGET = (120, 255, 120)
    COL_FILT = (255, 200, 120)
    COL_FILT_STALE = (110, 90, 60)   # filtered value held during a dropout
    COL_RAW = (110, 110, 200)
    COL_PUMP = (120, 200, 255)
    COL_AXIS = (80, 80, 80)
    COL_GRID = (44, 44, 44)
    COL_TEXT = (200, 200, 200)

    def __init__(self, width=760, height=340, history_s=60.0, max_points=4000,
                 show_raw=True, units='px/frame', v_limits=(0.0, 5.0),
                 zoom_window_s=12.0,
                 window_name="EBIV Control — time history"):
        self.w, self.h = int(width), int(height)
        self.history_s = float(history_s)
        self.show_raw = bool(show_raw)
        self.units = units
        self.v_limits = v_limits
        self.window_name = window_name
        # The y-scale is set from the last `zoom_window_s` of data, not from
        # the whole plotted history.  Otherwise a start-up transient from 0 to
        # the operating point pins the pane to [0, U] for a full minute and
        # the few-percent variations that actually matter are invisible.
        # <= 0 restores the old behaviour (scale on everything on screen).
        self.zoom_window_s = float(zoom_window_s)
        self._ylo = None                  # smoothed pane limits
        self._yhi = None

        self.t = deque(maxlen=max_points)
        self.target = deque(maxlen=max_points)
        self.filt = deque(maxlen=max_points)
        self.raw = deque(maxlen=max_points)
        self.pump = deque(maxlen=max_points)
        self.state = deque(maxlen=max_points)

        self.canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        m = 44                      # left margin: room for the tick labels
        r = 62                      # right margin: room for the value readout
        self.top = (m, 22, self.w - r, int(self.h * 0.60))
        self.bot = (m, int(self.h * 0.66), self.w - r, self.h - 22)
        self._last_render = 0.0

    # ------------------------------------------------------------------

    def append(self, t, u_target, u_filt, u_raw, v_pump, state=''):
        self.t.append(float(t))
        self.target.append(float(u_target))
        self.filt.append(float(u_filt))
        self.raw.append(float(u_raw))
        self.pump.append(float(v_pump))
        self.state.append(state)

    # ------------------------------------------------------------------

    @staticmethod
    def _poly(canvas, t, y, box, t0, t1, y0, y1, color, thickness=1):
        x_l, y_t, x_r, y_b = box
        good = np.isfinite(y)
        if np.count_nonzero(good) < 2 or t1 <= t0 or y1 <= y0:
            return
        px = x_l + (t - t0) / (t1 - t0) * (x_r - x_l)
        py = y_b - (y - y0) / (y1 - y0) * (y_b - y_t)
        # NaN stretches are split out below, but they must not reach the int32
        # cast: numpy raises a RuntimeWarning and produces undefined values.
        # Clip to the PANE, not to a loose bound: since 5.2 the y-scale is set
        # from a recent slice, so older samples can sit far outside it and
        # would otherwise be drawn straight through the pane below and the
        # legend.  Riding the edge is the usual scope convention and reads
        # correctly as "off scale".
        py = np.clip(np.nan_to_num(py, nan=y_b), y_t + 1, y_b - 1)
        pts = np.column_stack([px, py]).astype(np.int32)
        # split on gaps so invalid stretches are not bridged by a straight line
        idx = np.nonzero(good)[0]
        for seg in np.split(idx, np.nonzero(np.diff(idx) != 1)[0] + 1):
            if seg.size >= 2:
                cv2.polylines(canvas, [pts[seg]], False, color, thickness,
                              cv2.LINE_AA)

    # ------------------------------------------------------------------

    @staticmethod
    def _nice_ticks(lo, hi, n=4):
        """A few round tick values spanning [lo, hi]."""
        import math
        span = hi - lo
        if not np.isfinite(span) or span <= 0:
            return [lo]
        raw = span / max(1, n)
        mag = 10.0 ** math.floor(math.log10(raw))
        for m in (1.0, 2.0, 2.5, 5.0, 10.0):
            if raw <= m * mag:
                step = m * mag
                break
        else:
            step = 10.0 * mag
        first = math.ceil(lo / step) * step
        ticks, v = [], first
        while v <= hi + 1e-12 and len(ticks) < 12:
            ticks.append(v)
            v += step
        return ticks

    def _frame(self, box, title, lo, hi, fmt="%.3g"):
        """Axes, gridlines and labelled ticks for one pane."""
        x_l, y_t, x_r, y_b = box
        cv2.rectangle(self.canvas, (x_l, y_t), (x_r, y_b), self.COL_AXIS, 1)
        cv2.putText(self.canvas, title, (x_l + 4, y_t + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.COL_TEXT, 1, cv2.LINE_AA)
        for v in self._nice_ticks(lo, hi):
            y = int(y_b - (v - lo) / (hi - lo) * (y_b - y_t))
            if not (y_t + 2 <= y <= y_b - 2):
                continue
            cv2.line(self.canvas, (x_l + 1, y), (x_r - 1, y), self.COL_GRID, 1)
            cv2.putText(self.canvas, fmt % v, (2, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, self.COL_TEXT, 1, cv2.LINE_AA)

    def _legend(self):
        x = self.top[0]
        items = [("target", self.COL_TARGET), ("filtered", self.COL_FILT)]
        if self.show_raw:
            items.append(("raw", self.COL_RAW))
        items.append(("V_pump", self.COL_PUMP))
        for lab, c in items:
            cv2.line(self.canvas, (x, 10), (x + 14, 10), c, 2)
            cv2.putText(self.canvas, lab, (x + 18, 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, c, 1, cv2.LINE_AA)
            x += 26 + int(7.2 * len(lab))

    def _last_value(self, box, y_lo, y_hi, value, color, fmt="%.3f"):
        """Print the latest value at the right edge, level with the trace."""
        if not np.isfinite(value):
            return
        x_l, y_t, x_r, y_b = box
        y = int(np.clip(y_b - (value - y_lo) / (y_hi - y_lo) * (y_b - y_t),
                        y_t + 6, y_b - 2))
        txt = fmt % value
        cv2.putText(self.canvas, txt, (x_r + 4, y + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.40, color, 1, cv2.LINE_AA)

    def _state_bands(self, box, t, states, t0, t1):
        """Tint the stretches where the controller was not in closed loop."""
        x_l, y_t, x_r, y_b = box
        if t1 <= t0:
            return
        colors = {'HOLD': (0, 90, 130), 'SAFE': (0, 0, 130), 'MANUAL': (60, 60, 60)}
        h = 5
        for i, st in enumerate(states):
            c = colors.get(st)
            if c is None:
                continue
            x = int(x_l + (t[i] - t0) / (t1 - t0) * (x_r - x_l))
            cv2.line(self.canvas, (x, y_b - h), (x, y_b - 1), c, 1)

    def render(self):
        """Redraw the canvas.  Returns the BGR image (does not call imshow)."""
        self.canvas[:] = 18
        if len(self.t) < 2:
            cv2.putText(self.canvas, "waiting for control data...",
                        (56, self.h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        self.COL_TEXT, 1, cv2.LINE_AA)
            return self.canvas

        t = np.fromiter(self.t, dtype=np.float64)
        t1 = t[-1]
        t0 = max(t[0], t1 - self.history_s)
        keep = t >= t0
        t = t[keep]
        tgt = np.fromiter(self.target, dtype=np.float64)[keep]
        flt = np.fromiter(self.filt, dtype=np.float64)[keep]
        raw = np.fromiter(self.raw, dtype=np.float64)[keep]
        pmp = np.fromiter(self.pump, dtype=np.float64)[keep]
        states = [s for s, k in zip(self.state, keep) if k]

        # --- velocity pane limits ---------------------------------------
        # Scaled on a RECENT slice, not on everything drawn.  A run that
        # starts at 0 and settles at 0.5 m/s would otherwise keep 0 in the
        # pane for the whole history window, squashing the few-percent
        # variations that the control work is actually about.
        if self.zoom_window_s > 0:
            zsel = t >= (t1 - self.zoom_window_s)
        else:
            zsel = np.ones(t.shape, dtype=bool)

        # Percentiles on the RAW trace, so one bad frame cannot compress the
        # whole pane.  The filtered trace is taken in full over the zoom
        # slice; the target only needs its CURRENT value in frame -- an old
        # set-point that has since been left behind must not hold the scale.
        parts = []
        for a, robust in ((flt[zsel], False), (raw[zsel], self.show_raw)):
            f = a[np.isfinite(a)]
            if f.size == 0:
                continue
            parts.append((np.percentile(f, 1), np.percentile(f, 99)) if robust
                         else (f.min(), f.max()))
        f = tgt[np.isfinite(tgt)]
        if f.size:
            parts.append((f[-1], f[-1]))
        if parts:
            lo = min(p[0] for p in parts)
            hi = max(p[1] for p in parts)
        else:
            lo, hi = 0.0, 1.0

        # A floor on the span, or pure measurement noise fills the pane and
        # every run looks equally unsteady.  1% of the mean level is roughly
        # the resolution worth showing; the absolute term covers levels near 0.
        centre = 0.5 * (lo + hi)
        min_span = max(1e-9, 0.01 * abs(centre))
        if hi - lo < min_span:
            lo, hi = centre - 0.5 * min_span, centre + 0.5 * min_span
        pad = 0.10 * (hi - lo)
        lo, hi = lo - pad, hi + pad

        # Expand at once so a transient is never clipped; contract with a
        # short time constant so the pane zooms back in within about a second
        # instead of waiting for the excursion to leave the history window.
        if self._ylo is None or not np.isfinite(self._ylo):
            self._ylo, self._yhi = lo, hi
        else:
            a = 0.35                       # per render; ~0.5 s at 5 Hz
            self._ylo = min(lo, self._ylo + a * (lo - self._ylo))
            self._yhi = max(hi, self._yhi + a * (hi - self._yhi))
        lo, hi = self._ylo, self._yhi

        self._frame(self.top, f"velocity [{self.units}]", lo, hi)
        self._state_bands(self.top, t, states, t0, t1)
        if self.show_raw:
            self._poly(self.canvas, t, raw, self.top, t0, t1, lo, hi, self.COL_RAW, 1)

        # The filtered trace HOLDS its last value while the measurement is
        # invalid, which is correct behaviour but reads as live flow.  Draw
        # the stale stretches dimmed so a glance at the chart cannot mistake a
        # frozen number for a measured one.  Staleness is "no raw sample this
        # step", not "not in closed loop": in manual and calibration mode the
        # filtered value is perfectly live, it is simply not being acted on.
        stale = ~np.isfinite(raw)
        if stale.any():
            self._poly(self.canvas, t, np.where(stale, flt, np.nan), self.top,
                       t0, t1, lo, hi, self.COL_FILT_STALE, 2)
            live = np.where(stale, np.nan, flt)
        else:
            live = flt
        self._poly(self.canvas, t, live, self.top, t0, t1, lo, hi, self.COL_FILT, 2)
        self._poly(self.canvas, t, tgt, self.top, t0, t1, lo, hi, self.COL_TARGET, 2)
        self._last_value(self.top, lo, hi, flt[-1], self.COL_FILT)
        self._last_value(self.top, lo, hi, tgt[-1], self.COL_TARGET)

        # --- pump pane limits -------------------------------------------
        # Autoscaled to what the pump is actually doing.  Pinning it to the
        # full configured range makes a 0-1.5 V trace an unreadable squiggle
        # along the bottom, which is the usual case early in an experiment.
        # The configured ceiling is still drawn, so the headroom stays visible.
        vlo_cfg, vhi_cfg = self.v_limits
        f = pmp[np.isfinite(pmp)]
        if f.size:
            vlo, vhi = float(f.min()), float(f.max())
        else:
            vlo, vhi = vlo_cfg, vhi_cfg
        min_span = max(0.1 * (vhi_cfg - vlo_cfg), 0.2)
        if vhi - vlo < min_span:
            mid = 0.5 * (vlo + vhi)
            vlo, vhi = mid - min_span / 2, mid + min_span / 2
        vpad = 0.12 * (vhi - vlo)
        vlo = max(vlo_cfg, vlo - vpad)
        vhi = min(vhi_cfg, vhi + vpad)
        if vhi - vlo < 1e-6:
            vlo, vhi = vlo_cfg, vhi_cfg

        self._frame(self.bot, f"pump command [V]   (limits {vlo_cfg:g}-{vhi_cfg:g})",
                    vlo, vhi, fmt="%.2f")
        if vhi >= vhi_cfg - 1e-9:
            y = self.bot[1] + 1
            for x in range(self.bot[0] + 2, self.bot[2] - 2, 8):
                cv2.line(self.canvas, (x, y), (x + 4, y), (60, 60, 110), 1)
        self._poly(self.canvas, t, pmp, self.bot, t0, t1, vlo, vhi, self.COL_PUMP, 2)
        self._last_value(self.bot, vlo, vhi, pmp[-1], self.COL_PUMP, fmt="%.3f V")

        self._legend()
        st_now = states[-1] if states else ''
        cv2.putText(self.canvas, f"{t1 - t0:.0f} s window    t = {t1:.1f} s"
                                 + (f"    [{st_now}]" if st_now else ""),
                    (self.bot[0], self.h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    _STATE_COLORS.get(st_now, self.COL_TEXT), 1, cv2.LINE_AA)
        return self.canvas

    def maybe_show(self, rate_hz):
        """Render and imshow at most rate_hz times per second."""
        now = time.perf_counter()
        if now - self._last_render < 1.0 / max(rate_hz, 1e-6):
            return False
        self._last_render = now
        cv2.imshow(self.window_name, self.render())
        return True

    def save(self, path):
        cv2.imwrite(path, self.render())


# ==========================================================================
#  Cross-stream profile  (the [u] key)
# ==========================================================================

class ProfileView:
    """
    Live cross-stream profile of the feedback component, over the WHOLE
    DisplayROI, with the ControlROI marked on it.

    This exists to answer one question while a jet is being set up: is the
    ControlROI sitting on the potential core, or is it clipping the shear
    layers?  A mean taken across a shear layer moves when the jet flaps and
    when the ROI is a few pixels off, which is exactly the kind of feedback
    signal that makes a controller look badly tuned when the measurement is
    the problem.

    It deliberately spans the whole DisplayROI rather than only the
    ControlROI: the point is to CHOOSE where the ControlROI goes, so you need
    to see the shear layers on either side of it.

    The profile is averaged over the streamwise direction and then smoothed
    over frames with an EMA, because a single pseudo-frame from a turbulent
    jet does not show a plateau — a few seconds of averaging does.  The faint
    trace is the instantaneous profile, the bright one is the average.
    """

    COL_MEAN = (255, 200, 120)
    COL_INST = (90, 80, 60)
    COL_BAND = (60, 55, 40)
    COL_ROI = (120, 255, 120)
    COL_AXIS = (80, 80, 80)
    COL_GRID = (44, 44, 44)
    COL_TEXT = (200, 200, 200)
    COL_WARN = (90, 160, 255)

    def __init__(self, estimator, display_roi, window_size, node_distance,
                 component, velocity_scale=1.0, units='px/s',
                 width=820, height=430, ema_frames=40.0,
                 window_name="EBIV - cross-stream profile"):
        self.est = estimator
        self.window_size = int(window_size)
        self.node_distance = int(node_distance)
        self.component = component
        self.scale = float(velocity_scale)
        self.units = units
        self.w, self.h = int(width), int(height)
        self.window_name = window_name
        self.alpha = 1.0 / max(ema_frames, 1.0)
        self._last_render = 0.0
        self._mean = None
        self._m2 = None
        self._inst = None
        self._n = 0

        half = self.window_size // 2
        gy, gx = estimator.grid_shape
        self.xc = display_roi[0] + np.arange(gx) * self.node_distance + half
        self.yc = display_roi[2] + np.arange(gy) * self.node_distance + half

        # For a streamwise-horizontal jet ('u'/'-u') the cross-stream
        # direction is vertical, so we average along x and plot against y.
        self.vs_rows = component in ('u', '-u')
        if self.vs_rows:
            self.axis_px = self.yc
            self.axis_label = "y [px]"
            self.roi_lo, self.roi_hi = self.yc[self.est.sy][0], self.yc[self.est.sy][-1]
        else:
            self.axis_px = self.xc
            self.axis_label = "x [px]"
            self.roi_lo, self.roi_hi = self.xc[self.est.sx][0], self.xc[self.est.sx][-1]

    # ------------------------------------------------------------------

    def reset(self):
        self._mean = self._m2 = self._inst = None
        self._n = 0

    def update(self, U, V):
        """Accumulate one field.  Cheap: two reductions and an EMA."""
        if U is None or V is None:
            return
        comp = self.est._component(np.asarray(U, float), np.asarray(V, float))
        comp = comp * self.scale
        axis = 1 if self.vs_rows else 0        # average ALONG the stream
        with np.errstate(invalid='ignore'):
            prof = np.nanmean(comp, axis=axis)
        if prof.size != self.axis_px.size or not np.any(np.isfinite(prof)):
            return
        self._inst = prof
        if self._mean is None:
            self._mean = prof.copy()
            self._m2 = np.zeros_like(prof)
        else:
            a = self.alpha
            d = prof - self._mean
            self._mean += a * d
            self._m2 += a * (d * d - self._m2)
        self._n += 1

    # ------------------------------------------------------------------

    def render(self):
        img = np.zeros((self.h, self.w, 3), np.uint8)
        L, R, T, B = 78, self.w - 14, 34, self.h - 56

        cv2.putText(img, f"cross-stream profile of {self.component}"
                         f"   [{self.units}]   n={self._n}",
                    (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.46, self.COL_TEXT, 1)

        if self._mean is None:
            cv2.putText(img, "waiting for a velocity field...  press [p] to"
                             " enable RT-EBIV", (L, (T + B) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COL_WARN, 1)
            return img

        sd = np.sqrt(np.maximum(self._m2, 0.0))
        lo = float(np.nanmin(self._mean - sd))
        hi = float(np.nanmax(self._mean + sd))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
            lo, hi = lo - 1.0, hi + 1.0
        pad = 0.08 * (hi - lo)
        lo, hi = lo - pad, hi + pad

        ax0, ax1 = float(self.axis_px[0]), float(self.axis_px[-1])
        if ax1 <= ax0:
            ax1 = ax0 + 1.0

        def px(v):      # cross-stream coordinate -> screen x
            return int(L + (v - ax0) / (ax1 - ax0) * (R - L))

        def py(v):      # value -> screen y
            return int(B - (v - lo) / (hi - lo) * (B - T))

        # --- the ControlROI band, drawn first so the curves sit on top ---
        x_lo, x_hi = px(self.roi_lo), px(self.roi_hi)
        band = img[T:B, min(x_lo, x_hi):max(x_lo, x_hi) + 1]
        if band.size:
            band[:] = (band * 0.45 + np.array((18, 40, 18)) * 0.55).astype(np.uint8)
        for xx in (x_lo, x_hi):
            cv2.line(img, (xx, T), (xx, B), self.COL_ROI, 1)
        cv2.putText(img, "ControlROI", (min(x_lo, x_hi) + 4, T + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COL_ROI, 1)

        # --- axes ---
        cv2.rectangle(img, (L, T), (R, B), self.COL_AXIS, 1)
        for v in StripChart._nice_ticks(lo, hi):
            y = py(v)
            if not (T + 2 <= y <= B - 2):
                continue
            cv2.line(img, (L + 1, y), (R - 1, y), self.COL_GRID, 1)
            cv2.putText(img, f"{v:g}", (4, y + 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, self.COL_TEXT, 1)
        for v in StripChart._nice_ticks(ax0, ax1):
            x = px(v)
            if not (L + 2 <= x <= R - 2):
                continue
            cv2.line(img, (x, T + 1), (x, B - 1), self.COL_GRID, 1)
            cv2.putText(img, f"{v:g}", (x - 12, B + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.COL_TEXT, 1)
        cv2.putText(img, self.axis_label, (R - 46, B + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.COL_TEXT, 1)

        # --- +-1 sd band around the averaged profile ---
        up = [(px(a), py(m + s)) for a, m, s in zip(self.axis_px, self._mean, sd)]
        dn = [(px(a), py(m - s)) for a, m, s in zip(self.axis_px, self._mean, sd)]
        poly = np.array(up + dn[::-1], np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [poly], self.COL_BAND)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

        if self._inst is not None:
            pts = np.array([(px(a), py(v)) for a, v in
                            zip(self.axis_px, self._inst)], np.int32)
            cv2.polylines(img, [pts], False, self.COL_INST, 1, cv2.LINE_AA)
        pts = np.array([(px(a), py(v)) for a, v in
                        zip(self.axis_px, self._mean)], np.int32)
        cv2.polylines(img, [pts], False, self.COL_MEAN, 2, cv2.LINE_AA)
        for x, y in pts:
            cv2.circle(img, (int(x), int(y)), 2, self.COL_MEAN, -1)

        # --- what the controller currently averages over ----------------
        sel = self.est.sy if self.vs_rows else self.est.sx
        inside = self._mean[sel]
        if inside.size:
            m = float(np.nanmean(inside))
            cv2.line(img, (min(x_lo, x_hi), py(m)), (max(x_lo, x_hi), py(m)),
                     self.COL_ROI, 1, cv2.LINE_AA)
            flat = (float(np.nanmax(inside) - np.nanmin(inside))
                    / max(abs(m), 1e-9) * 100.0)
            msg = (f"ControlROI mean {m:.4g} {self.units}   "
                   f"spread across it {flat:.0f}% of the mean   "
                   f"{inside.size} profile points, {self.est.n_total} nodes")
            col = self.COL_TEXT if flat < 15.0 else self.COL_WARN
            cv2.putText(img, msg, (10, self.h - 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, col, 1)
            if flat >= 15.0:
                cv2.putText(img,
                            "the ControlROI is not on a plateau: this mean "
                            "moves when the jet flaps",
                            (10, self.h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            self.COL_WARN, 1)
        return img

    def maybe_show(self, rate_hz):
        now = time.perf_counter()
        if now - self._last_render < 1.0 / max(rate_hz, 1e-6):
            return False
        self._last_render = now
        cv2.imshow(self.window_name, self.render())
        return True

    def save(self, path):
        cv2.imwrite(path, self.render())
