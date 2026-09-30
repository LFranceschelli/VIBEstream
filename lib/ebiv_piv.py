"""
EBIV Release 5.2 — CPU correlation engine.

This is Release 4.0's `batch_cross_correlate` with the same numerics and
fewer allocations.  With the default settings

    subpixel=False, fast_mean_removal=False, fft_workers=1

it reproduces Release 4.0's real-time output BIT-FOR-BIT.  test_release5.py
asserts that on random and synthetic data.

What changed, and why (all measured — see AUDIT_R4_to_R5.md §3):

  ACCEPTED, bit-exact
    * window extraction by strided view instead of a Python double loop,
      writing into a preallocated buffer.  Release 4.0 allocated two or three
      (N, ws, ws) float32 arrays per field and then called .astype(np.float32)
      on each one, which copies AGAIN even though _extract_windows already
      returned float32.  For a 1280x720 sensor at ws=48 that is ~14 MB per
      array per field, twice over.  Measured 3.0-4.1x on this stage.
    * in-place cross-power (np.conj(F1, out=F1); F1 *= F2) instead of building
      two new complex arrays.  Measured 2.3-2.5x on this stage.
    * irfft2(..., overwrite_x=True), which reuses the cross-power buffer.
    * optional multi-threaded batch FFT via scipy.fft workers.  Splitting a
      batch across threads does not change any individual transform.

  REJECTED, not bit-exact
    * replacing np.fft.fftshift with an index remap.  It looks free, and it
      is a pure permutation, but argmax breaks ties by flat position, and the
      flat order differs between the shifted and unshifted planes.  On sparse
      EBIV-like images the correlation planes are integer-valued and ties are
      common: measured ~2% of vectors changing on ordinary fields, and 100%
      on degenerate ones.  fftshift costs 6-8% of the correlation and is kept.

  OPTIONAL, off by default, NOT bit-exact
    * fast_mean_removal: zero the DC bin of each window's FFT instead of
      subtracting the spatial mean.  Mathematically the same operation, but
      float32 rounding differs: measured max relative difference on the
      correlation plane ~1.5e-7, up to ~0.3% of integer peaks flipping on
      tie-prone data, for a measured 1.2-1.4x on the whole correlation.

  NEW
    * sub-pixel interpolation in the real-time path.  Release 4.0's real-time
      output is INTEGER pixels on both the CPU and the GPU backend; only the
      offline pyramidal path interpolates.  A 1 px quantisation on a typical
      5 px displacement is a 20% quantisation of the control signal, which is
      not acceptable as a feedback quantity.  The estimator is the same
      3-point Gaussian used offline.
    * per-vector normalised correlation coefficient, used by the ControlROI
      validity gate.
"""

import logging

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.fft import rfft2, irfft2, set_workers

from ebiv_profiler import PipelineProfiler

_NULL_PROF = PipelineProfiler(enabled=False)


# ==========================================================================

def extract_windows(frame, window_size, node_distance, grid_y, grid_x, out=None):
    """
    All interrogation windows of one frame as (N, ws, ws) float32.

    Uses a strided view followed by a single C-level copy.  The elements are
    exactly those Release 4.0's Python loop produced.
    """
    view = sliding_window_view(frame, (window_size, window_size))[::node_distance,
                                                                 ::node_distance]
    view = view[:grid_y, :grid_x]
    if out is None:
        out = np.empty((grid_y * grid_x, window_size, window_size), dtype=np.float32)
    np.copyto(out.reshape(grid_y, grid_x, window_size, window_size), view)
    return out


def batch_subpixel_flat(R_flat, peak_idx, window_size):
    """
    Vectorised 3-point Gaussian sub-pixel interpolation, gathering the four
    neighbours by flat index on the contiguous (N, ws*ws) correlation view.

    Identical formulation to Release 4.0's batch_subpixel_peak (Gaussian where
    all three samples are positive, parabolic fallback otherwise, clamped to
    +-1 px), verified bit-exact against it in test_release5.py.
    """
    n, m = R_flat.shape
    idx = np.arange(n)
    cy, cx = np.divmod(peak_idx, window_size)
    centre = R_flat[idx, peak_idx]

    def _solve(a, b, c, ok):
        out = np.zeros(n, dtype=np.float64)
        g = ok & (a > 0) & (b > 0) & (c > 0)
        if np.any(g):
            la, lb, lc = np.log(a[g]), np.log(b[g]), np.log(c[g])
            den = 2.0 * (la - 2.0 * lb + lc)
            s = np.abs(den) > 1e-12
            t = np.zeros(int(g.sum()))
            t[s] = np.clip((la[s] - lc[s]) / den[s], -1.0, 1.0)
            out[g] = t
        p = ok & ~g
        if np.any(p):
            ap, bp, cp = a[p], b[p], c[p]
            den = 2.0 * (ap - 2.0 * bp + cp)
            s = np.abs(den) > 1e-12
            t = np.zeros(int(p.sum()))
            t[s] = np.clip((ap[s] - cp[s]) / den[s], -1.0, 1.0)
            out[p] = t
        return out

    ok_x = (cx > 0) & (cx < window_size - 1)
    ok_y = (cy > 0) & (cy < window_size - 1)
    lo = np.clip(peak_idx - 1, 0, m - 1)
    hi = np.clip(peak_idx + 1, 0, m - 1)
    up = np.clip(peak_idx - window_size, 0, m - 1)
    dn = np.clip(peak_idx + window_size, 0, m - 1)

    dx = _solve(R_flat[idx, lo], centre, R_flat[idx, hi], ok_x)
    dy = _solve(R_flat[idx, up], centre, R_flat[idx, dn], ok_y)
    return dx, dy


# ==========================================================================

class CPUCorrelator:
    """
    Batched FFT cross-correlation with persistent buffers.

    Mirrors the GPUCorrelator interface (numpy in, numpy out) so the two are
    interchangeable, but keeps everything on the CPU: Release 5.0 is
    explicitly CPU-only.

    The frame shape is fixed at construction.  Feeding a differently shaped
    frame raises rather than silently reallocating in the real-time loop.
    """

    def __init__(self, window_size, node_distance, frame_shape,
                 triple_corr=False, fft_workers=1, subpixel=True,
                 compute_quality=True, fast_mean_removal=False):
        self.ws = int(window_size)
        self.nd = int(node_distance)
        self.shape = tuple(frame_shape)
        self.triple = bool(triple_corr)
        self.workers = max(1, int(fft_workers))
        self.subpixel = bool(subpixel)
        self.quality = bool(compute_quality)
        self.fast_mean = bool(fast_mean_removal)

        h, w = self.shape
        if h < self.ws or w < self.ws:
            raise ValueError(
                f"DisplayROI is {w}x{h} px but the interrogation window is "
                f"{self.ws} px. The ROI must be at least window_size in both "
                f"directions.")
        self.grid_y = (h - self.ws) // self.nd + 1
        self.grid_x = (w - self.ws) // self.nd + 1
        self.n = self.grid_y * self.grid_x

        nbuf = 3 if self.triple else 2
        self._W = [np.empty((self.n, self.ws, self.ws), dtype=np.float32)
                   for _ in range(nbuf)]
        self._centre = self.ws // 2

        mb = nbuf * self._W[0].nbytes / 1e6
        logging.info(
            "CPUCorrelator: %dx%d px, ws=%d nd=%d -> grid %dx%d = %d vectors; "
            "%.1f MB of persistent window buffers; workers=%d, subpixel=%s, "
            "quality=%s, fast_mean_removal=%s",
            w, h, self.ws, self.nd, self.grid_y, self.grid_x, self.n, mb,
            self.workers, self.subpixel, self.quality, self.fast_mean)

    # ------------------------------------------------------------------

    def correlate(self, frames, prof=_NULL_PROF):
        """
        frames : list of 2 (or 3 if triple_corr) 2-D uint8/float32 arrays.
        Returns (U, V, CC) with CC=None when compute_quality is False.
        U is the column displacement, V the row displacement, positive DOWN —
        the same image convention Release 4.0 saves to .mat.
        """
        ws, nd, gy, gx, n = self.ws, self.nd, self.grid_y, self.grid_x, self.n
        if frames[0].shape != self.shape:
            raise ValueError(f"frame shape {frames[0].shape} != {self.shape}")
        k = 3 if self.triple else 2
        if len(frames) < k:
            raise ValueError(f"need {k} frames, got {len(frames)}")

        with prof.measure("window_extraction"):
            W = [extract_windows(frames[i], ws, nd, gy, gx, out=self._W[i])
                 for i in range(k)]

        if not self.fast_mean:
            with prof.measure("mean_subtract"):
                for w_ in W:
                    w_ -= w_.mean(axis=(1, 2), keepdims=True)

        energies = None
        if self.quality:
            with prof.measure("peak_quality_energy"):
                if self.fast_mean:
                    # energy of the mean-removed window, without materialising it
                    energies = [float_energy_after_mean_removal(w_) for w_ in W]
                else:
                    energies = [np.einsum('nij,nij->n', w_, w_) for w_ in W]

        with set_workers(self.workers):
            with prof.measure("fft_forward"):
                F = [rfft2(w_) for w_ in W]
            if self.fast_mean:
                with prof.measure("mean_subtract"):
                    for f_ in F:
                        f_[:, 0, 0] = 0.0
            with prof.measure("cross_power"):
                if self.triple:
                    c0 = np.conj(F[0]) * F[1]
                    c0 += np.conj(F[1]) * F[2]
                    cross = c0
                else:
                    np.conj(F[0], out=F[0])
                    F[0] *= F[1]
                    cross = F[0]
            with prof.measure("fft_inverse"):
                R = irfft2(cross, s=(ws, ws), overwrite_x=True)

        with prof.measure("fftshift"):
            R = np.fft.fftshift(R, axes=(1, 2))

        with prof.measure("peak_finding"):
            R_flat = R.reshape(n, -1)
            peak = np.argmax(R_flat, axis=1)
            cy, cx = np.unravel_index(peak, (ws, ws))
            U = (cx - self._centre).reshape(gy, gx).astype(np.float32)
            V = (cy - self._centre).reshape(gy, gx).astype(np.float32)

        CC = None
        if self.quality:
            with prof.measure("peak_quality_cc"):
                peak_val = R_flat[np.arange(n), peak]
                if self.triple:
                    norm = (np.sqrt(energies[0] * energies[1]) +
                            np.sqrt(energies[1] * energies[2]))
                else:
                    norm = np.sqrt(energies[0] * energies[1])
                CC = (peak_val / (norm + 1e-20)).reshape(gy, gx).astype(np.float32)

        if self.subpixel:
            with prof.measure("subpixel"):
                dx, dy = batch_subpixel_flat(R_flat, peak, ws)
                U += dx.reshape(gy, gx).astype(np.float32)
                V += dy.reshape(gy, gx).astype(np.float32)

        return U, V, CC


def float_energy_after_mean_removal(w):
    """
    sum((w - mean(w))^2) per window, computed as sum(w^2) - n*mean^2, so the
    mean-removed array never has to be materialised when fast_mean_removal is
    on.  Mathematically exact; float32 rounding differs from the explicit
    form, which is one of the reasons fast_mean_removal is off by default.
    """
    n_px = w.shape[1] * w.shape[2]
    s = w.sum(axis=(1, 2), dtype=np.float64)
    ss = np.einsum('nij,nij->n', w, w).astype(np.float64)
    return np.maximum(ss - s * s / n_px, 0.0)
