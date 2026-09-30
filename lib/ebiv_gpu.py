"""
EBIV GPU Backend — Release 5.2  (Release 4.0's module, corrected and extended)

PyTorch-based GPU acceleration for the FFT cross-correlation engine.

Entry points:

    correlate(frames)            REAL-TIME.  Same signature and semantics as
                                 CPUCorrelator.correlate: returns (U, V, CC)
                                 with sub-pixel interpolation.  This is what
                                 the live stream and the closed loop use.
                                 NEW in 5.2.

    pyramidal_correlate(...)     OFFLINE multi-dt pyramid.  The homothetic
                                 rescaling was wrong in Release 4.0 (audit
                                 B7/B8) and is fixed here.

    batch_cross_correlate(...)   Release 4.0's real-time kernel, kept
                                 unchanged for reference: INTEGER peak, no
                                 CC.  Nothing in Release 5 calls it, because
                                 a controller cannot use either omission.

Design principles:
  - numpy in, numpy out: callers don't need to know about PyTorch.
  - Persistent GPU buffers via GPUCorrelator class to avoid per-frame
    allocation and transfer overhead.
  - Automatic CPU fallback if CUDA is not available.
  - Profiler-compatible: accepts optional PipelineProfiler.

Usage:
    from ebiv_gpu import GPUCorrelator

    gpu = GPUCorrelator(window_size=32, node_distance=32, device='cuda')
    U, V = gpu.batch_cross_correlate(frames)           # RT path
    U, V = gpu.pyramidal_correlate(frames, levels=3)   # offline path
"""

import logging
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Local import for the null profiler
from ebiv_profiler import PipelineProfiler
_NULL_PROF = PipelineProfiler(enabled=False)


def is_gpu_available():
    """Check if PyTorch + CUDA are available."""
    return HAS_TORCH and torch.cuda.is_available()


class GPUCorrelator:
    """
    GPU-accelerated FFT cross-correlation engine.

    Keeps a persistent device handle and pre-allocated buffers to minimize
    CPU↔GPU transfer overhead in the real-time loop.
    """

    def __init__(self, window_size, node_distance, device=None,
                 triple_corr=False, subpixel=True, compute_quality=True,
                 frame_shape=None):
        """
        Parameters
        ----------
        window_size : int
            Side length of the square interrogation window (pixels).
        node_distance : int
            Grid spacing between interrogation points (pixels).
        device : str or None
            'cuda', 'cuda:0', 'cpu', or None (auto-select).
        triple_corr, subpixel, compute_quality : bool
            Release 5.2.  Used by correlate(), which is the real-time entry
            point and mirrors CPUCorrelator.  They do NOT affect the legacy
            batch_cross_correlate(), which is kept exactly as Release 4.0 left
            it: integer peak, no CC.
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is not installed. Install with: "
                               "pip install torch --index-url https://download.pytorch.org/whl/cu121")

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.ws = window_size
        self.nd = node_distance
        self.triple = bool(triple_corr)
        self.subpixel = bool(subpixel)
        self.quality = bool(compute_quality)

        # grid_y / grid_x are read by stream_camera, the experiment-log header
        # and the .mat writer, so the GPU backend must expose them exactly as
        # CPUCorrelator does.  Without frame_shape they stay None until the
        # first correlate() call fills them in.
        self.grid_y = self.grid_x = None
        if frame_shape is not None:
            h, w = frame_shape
            self.grid_y = (h - window_size) // node_distance + 1
            self.grid_x = (w - window_size) // node_distance + 1

        # Cache for pre-computed homothetic affine grids (pyramid levels)
        self._warp_grids = {}

        # Track last grid shape to detect when we need to rebuild grids
        self._last_grid_shape = None

        logging.info(f"GPUCorrelator initialized on device: {self.device}")
        if self.device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(self.device)
            gpu_mem = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
            logging.info(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # ------------------------------------------------------------------
    #  Window extraction via unfold (strided view, no Python loop)
    # ------------------------------------------------------------------

    def _extract_windows_gpu(self, frame_t):
        """
        Extract interrogation windows from a 2D tensor using unfold.

        Parameters
        ----------
        frame_t : torch.Tensor, shape (H, W), float32, on self.device

        Returns
        -------
        windows : torch.Tensor, shape (N, ws, ws), float32
        grid_y, grid_x : int
        """
        H, W = frame_t.shape
        ws = self.ws
        nd = self.nd

        grid_y = (H - ws) // nd + 1
        grid_x = (W - ws) // nd + 1

        # unfold along H, then along W → (grid_y, grid_x, ws, ws)
        windows = frame_t.unfold(0, ws, nd).unfold(1, ws, nd)
        # Reshape to (N, ws, ws) with contiguous memory for FFT
        windows = windows.contiguous().reshape(-1, ws, ws)

        return windows, grid_y, grid_x

    # ------------------------------------------------------------------
    #  Sub-pixel peak interpolation (fully vectorized on GPU)
    # ------------------------------------------------------------------

    def _batch_subpixel_gpu(self, R, cy, cx):
        """
        Vectorized 3-point Gaussian sub-pixel interpolation on GPU.

        Parameters
        ----------
        R : torch.Tensor, shape (N, ws, ws)
        cy, cx : torch.Tensor, shape (N,), long — integer peak positions

        Returns
        -------
        dx, dy : torch.Tensor, shape (N,), float32
        """
        N = R.shape[0]
        ws = R.shape[1]
        idx = torch.arange(N, device=R.device)

        dx = torch.zeros(N, dtype=torch.float32, device=R.device)
        dy = torch.zeros(N, dtype=torch.float32, device=R.device)

        # --- X sub-pixel ---
        valid_x = (cx > 0) & (cx < ws - 1)
        if valid_x.any():
            vi = idx[valid_x]
            c_left = R[vi, cy[vi], cx[vi] - 1]
            c_cent = R[vi, cy[vi], cx[vi]]
            c_right = R[vi, cy[vi], cx[vi] + 1]

            # Gaussian where all positive
            gauss_ok = (c_left > 0) & (c_cent > 0) & (c_right > 0)
            gi = vi[gauss_ok]
            if gi.numel() > 0:
                cl = torch.log(c_left[gauss_ok])
                cc = torch.log(c_cent[gauss_ok])
                cr = torch.log(c_right[gauss_ok])
                denom = 2.0 * (cl - 2 * cc + cr)
                safe = denom.abs() > 1e-12
                dx[gi[safe]] = ((cl[safe] - cr[safe]) / denom[safe]).clamp(-1, 1)

            # Parabolic fallback
            pi = vi[~gauss_ok]
            if pi.numel() > 0:
                cl2 = c_left[~gauss_ok]
                cc2 = c_cent[~gauss_ok]
                cr2 = c_right[~gauss_ok]
                denom2 = 2.0 * (cl2 - 2 * cc2 + cr2)
                safe2 = denom2.abs() > 1e-12
                dx[pi[safe2]] = ((cl2[safe2] - cr2[safe2]) / denom2[safe2]).clamp(-1, 1)

        # --- Y sub-pixel ---
        valid_y = (cy > 0) & (cy < ws - 1)
        if valid_y.any():
            vi = idx[valid_y]
            c_up = R[vi, cy[vi] - 1, cx[vi]]
            c_cent = R[vi, cy[vi], cx[vi]]
            c_down = R[vi, cy[vi] + 1, cx[vi]]

            gauss_ok = (c_up > 0) & (c_cent > 0) & (c_down > 0)
            gi = vi[gauss_ok]
            if gi.numel() > 0:
                cu = torch.log(c_up[gauss_ok])
                cc = torch.log(c_cent[gauss_ok])
                cd = torch.log(c_down[gauss_ok])
                denom = 2.0 * (cu - 2 * cc + cd)
                safe = denom.abs() > 1e-12
                dy[gi[safe]] = ((cu[safe] - cd[safe]) / denom[safe]).clamp(-1, 1)

            pi = vi[~gauss_ok]
            if pi.numel() > 0:
                cu2 = c_up[~gauss_ok]
                cc2 = c_cent[~gauss_ok]
                cd2 = c_down[~gauss_ok]
                denom2 = 2.0 * (cu2 - 2 * cc2 + cd2)
                safe2 = denom2.abs() > 1e-12
                dy[pi[safe2]] = ((cu2[safe2] - cd2[safe2]) / denom2[safe2]).clamp(-1, 1)

        return dx, dy

    # ------------------------------------------------------------------
    #  Homothetic scaling via grid_sample (replaces per-window warpAffine)
    # ------------------------------------------------------------------

    def _get_warp_grid(self, k, N):
        """
        Build or retrieve a cached affine grid that shrinks the level-k
        correlation plane by 1/k about the CPU path's centre.

        Returns a grid tensor of shape (N, ws, ws, 2).

        RELEASE 5.2 FIX (audit B7/B8).  The Release 4.0 version of this method
        was wrong twice, and both errors are corrected here.

        1. DIRECTION.  It built theta = [[1/k, 0, 0], [0, 1/k, 0]].  But
           `affine_grid` produces a SAMPLING grid: `grid_sample` evaluates
           out(x) = in(theta . x), so theta maps OUTPUT coordinates back to
           INPUT coordinates and therefore acts as the inverse of the
           transformation you want.  Asking for 1/k magnified the plane by k
           instead of shrinking it — the opposite of the CPU path, which uses
           cv2.warpAffine with a FORWARD matrix.  To make a peak at k*d land
           at d, theta must scale by k.

        2. CENTRE.  With align_corners=False, normalized coordinate 0 sits at
           pixel (ws-1)/2, whereas the CPU path scales about
           center_c = ws // 2 and later measures the displacement from that
           same pixel.  On an even window those differ by half a pixel, and
           the mismatch shows up as a systematic sub-pixel bias of
           0.5*(1 - 1/k) px per level: -0.25 px at k=2, -0.33 at k=3,
           -0.375 at k=4.  The translation term below removes it.

        Derivation of the translation.  With align_corners=False, pixel p maps
        to normalized x = (2p + 1)/ws - 1.  Requiring
        p_in = c + k*(p_out - c) and substituting gives

            x_in = k * x_out + t,     t = (2c + k*(ws - 1 - 2c) + 1)/ws - 1

        which is zero when c = (ws-1)/2, i.e. it reduces to the natural centre
        exactly as it should.

        Verified against PyTorch 2.14 on CPU tensors (the arithmetic is the
        same on CUDA; the machine that wrote this has no GPU).  With both
        corrections, for k = 2, 3, 4, the warped plane agrees with
        cv2.warpAffine to about 1e-6 relative and the recovered sub-pixel
        displacement to about 1e-6 px.  That is float32 rounding between two
        different bilinear resamplers, NOT bit-equality — cv2 and grid_sample
        are separate implementations of the same geometry.  For scale: the
        centre error this replaced was 0.25-0.4 px, five orders of magnitude
        larger.

        Run tools/check_gpu.py on a CUDA machine to confirm end to end.
        """
        cache_key = (k, N)
        if cache_key not in self._warp_grids:
            ws = self.ws
            c = ws // 2                      # the CPU path's centre_c
            s = float(k)                     # see note 1: k, NOT 1/k
            t = (2 * c + s * (ws - 1 - 2 * c) + 1) / ws - 1      # see note 2
            theta = torch.tensor([[s, 0, t],
                                  [0, s, t]], dtype=torch.float32, device=self.device)
            theta = theta.unsqueeze(0).expand(N, -1, -1)  # (N, 2, 3)
            grid = F.affine_grid(theta, (N, 1, ws, ws), align_corners=False)
            self._warp_grids[cache_key] = grid
        return self._warp_grids[cache_key]

    def _batch_homothetic_warp(self, R_k, k):
        """
        Apply homothetic scaling by 1/k to all correlation planes at once.

        Parameters
        ----------
        R_k : torch.Tensor, shape (N, ws, ws)
        k : int, pyramid level

        Returns
        -------
        R_warped : torch.Tensor, shape (N, ws, ws)
        """
        N = R_k.shape[0]
        grid = self._get_warp_grid(k, N)
        # grid_sample needs (N, C, H, W) input
        R_4d = R_k.unsqueeze(1)  # (N, 1, ws, ws)
        R_warped = F.grid_sample(R_4d, grid, mode='bilinear',
                                 padding_mode='zeros', align_corners=False)
        return R_warped.squeeze(1)  # (N, ws, ws)

    # ------------------------------------------------------------------
    #  Core: batch FFT cross-correlation (real-time path)
    # ------------------------------------------------------------------

    def batch_cross_correlate(self, frames, prof=_NULL_PROF):
        """
        GPU-accelerated batch FFT cross-correlation.

        Drop-in replacement for the CPU batch_cross_correlate.
        numpy in → numpy out.

        Parameters
        ----------
        frames : list of 2 or 3 numpy arrays (H, W), uint8 or float32
        prof : PipelineProfiler (optional)

        Returns
        -------
        U, V : numpy arrays, shape (grid_y, grid_x), float32
        """
        with prof.measure("gpu_transfer_to"):
            frame_tensors = [torch.from_numpy(f.astype(np.float32)).to(self.device)
                             for f in frames]

        if len(frames) == 3:
            with prof.measure("gpu_window_extraction"):
                W0, grid_y, grid_x = self._extract_windows_gpu(frame_tensors[0])
                W1, _, _ = self._extract_windows_gpu(frame_tensors[1])
                W2, _, _ = self._extract_windows_gpu(frame_tensors[2])

            with prof.measure("gpu_mean_subtract"):
                W0 = W0 - W0.mean(dim=(-2, -1), keepdim=True)
                W1 = W1 - W1.mean(dim=(-2, -1), keepdim=True)
                W2 = W2 - W2.mean(dim=(-2, -1), keepdim=True)

            with prof.measure("gpu_fft_forward"):
                F0 = torch.fft.rfft2(W0)
                F1 = torch.fft.rfft2(W1)
                F2 = torch.fft.rfft2(W2)

            with prof.measure("gpu_cross_power"):
                cross_power = (torch.conj(F0) * F1) + (torch.conj(F1) * F2)
        else:
            with prof.measure("gpu_window_extraction"):
                W1, grid_y, grid_x = self._extract_windows_gpu(frame_tensors[0])
                W2, _, _ = self._extract_windows_gpu(frame_tensors[1])

            with prof.measure("gpu_mean_subtract"):
                W1 = W1 - W1.mean(dim=(-2, -1), keepdim=True)
                W2 = W2 - W2.mean(dim=(-2, -1), keepdim=True)

            with prof.measure("gpu_fft_forward"):
                F1 = torch.fft.rfft2(W1)
                F2 = torch.fft.rfft2(W2)

            with prof.measure("gpu_cross_power"):
                cross_power = torch.conj(F1) * F2

        with prof.measure("gpu_fft_inverse"):
            R = torch.fft.irfft2(cross_power, s=(self.ws, self.ws))

        with prof.measure("gpu_fftshift"):
            R = torch.fft.fftshift(R, dim=(-2, -1))

        with prof.measure("gpu_peak_finding"):
            R_flat = R.reshape(R.shape[0], -1)
            peak_indices = torch.argmax(R_flat, dim=1)
            cy = peak_indices // self.ws
            cx = peak_indices % self.ws

            U_t = (cx - (self.ws // 2)).float()
            # Image convention: V is the row displacement, positive DOWNWARD
            # (row index increases downward). Matches the CPU path and the .mat data.
            V_t = (cy - (self.ws // 2)).float()

        with prof.measure("gpu_transfer_from"):
            U = U_t.reshape(grid_y, grid_x).cpu().numpy()
            V = V_t.reshape(grid_y, grid_x).cpu().numpy()

        return U, V

    # ------------------------------------------------------------------
    #  Release 5.2 real-time entry point
    # ------------------------------------------------------------------

    def correlate(self, frames, prof=_NULL_PROF):
        """
        Real-time correlation with the SAME semantics as CPUCorrelator.

        Returns (U, V, CC) — float32 (grid_y, grid_x) each, CC None when
        compute_quality is off.  Interchangeable with CPUCorrelator.correlate,
        which is what makes it usable by rt_piv_worker and therefore by the
        closed loop.

        WHY THIS EXISTS AND batch_cross_correlate DOES NOT SUFFICE.
        Release 4.0's GPU real-time kernel returns the INTEGER peak and no
        correlation coefficient, because Release 4.0's CPU real-time path did
        the same and there was no controller to feed.  Release 5 changed both:

          * sub-pixel interpolation is on by default, because a 1 px
            quantisation on a typical 5 px displacement is a 20% quantisation
            of the feedback signal;
          * CC is what the ControlROI validity gate uses to decide whether a
            measurement may reach the PID at all.

        Running the closed loop on the Release 4.0 kernel would silently undo
        both.  This method adds them, so the GPU is a drop-in for the CPU
        rather than a quiet downgrade.

        CONSEQUENCE FOR SPEED, stated plainly: the Release 4.0 GPU report
        measured a kernel that did neither of these things.  Sub-pixel and CC
        are real extra work, so that 14.4x on the correlation stage does NOT
        carry over to this method.  Measure it; do not assume it.
        """
        ws = self.ws
        n_expected = 3 if self.triple else 2
        if len(frames) < n_expected:
            raise ValueError(f"triple_corr={self.triple} needs {n_expected} "
                             f"frames, got {len(frames)}")

        with prof.measure("gpu_transfer_to"):
            ft = [torch.from_numpy(np.ascontiguousarray(f, dtype=np.float32)).to(self.device)
                  for f in frames[:n_expected]]

        with prof.measure("gpu_window_extraction"):
            W = []
            grid_y = grid_x = None
            for f in ft:
                w, gy, gx = self._extract_windows_gpu(f)
                grid_y, grid_x = gy, gx
                W.append(w)
            self.grid_y, self.grid_x = grid_y, grid_x

        with prof.measure("gpu_mean_subtract"):
            W = [w - w.mean(dim=(-2, -1), keepdim=True) for w in W]

        energies = None
        if self.quality:
            with prof.measure("gpu_peak_quality_energy"):
                # sum(w^2) over each mean-removed window, matching the CPU's
                # einsum('nij,nij->n', w, w).
                energies = [(w * w).sum(dim=(-2, -1)) for w in W]

        with prof.measure("gpu_fft_forward"):
            Fs = [torch.fft.rfft2(w) for w in W]

        with prof.measure("gpu_cross_power"):
            if self.triple:
                cross = torch.conj(Fs[0]) * Fs[1] + torch.conj(Fs[1]) * Fs[2]
            else:
                cross = torch.conj(Fs[0]) * Fs[1]

        with prof.measure("gpu_fft_inverse"):
            R = torch.fft.irfft2(cross, s=(ws, ws))

        with prof.measure("gpu_fftshift"):
            R = torch.fft.fftshift(R, dim=(-2, -1))

        with prof.measure("gpu_peak_finding"):
            n = R.shape[0]
            R_flat = R.reshape(n, -1)
            peak = torch.argmax(R_flat, dim=1)
            cy = torch.div(peak, ws, rounding_mode='floor')
            cx = peak - cy * ws
            centre = ws // 2
            U_t = (cx - centre).float()
            V_t = (cy - centre).float()

        CC_t = None
        if self.quality:
            with prof.measure("gpu_peak_quality_cc"):
                peak_val = R_flat.gather(1, peak.unsqueeze(1)).squeeze(1)
                if self.triple:
                    norm = (torch.sqrt(energies[0] * energies[1])
                            + torch.sqrt(energies[1] * energies[2]))
                else:
                    norm = torch.sqrt(energies[0] * energies[1])
                CC_t = peak_val / (norm + 1e-20)

        if self.subpixel:
            with prof.measure("gpu_subpixel"):
                dx, dy = self._batch_subpixel_gpu(R, cy, cx)
                U_t = U_t + dx
                V_t = V_t + dy

        with prof.measure("gpu_transfer_from"):
            U = U_t.reshape(grid_y, grid_x).cpu().numpy().astype(np.float32)
            V = V_t.reshape(grid_y, grid_x).cpu().numpy().astype(np.float32)
            CC = (CC_t.reshape(grid_y, grid_x).cpu().numpy().astype(np.float32)
                  if CC_t is not None else None)

        return U, V, CC

    # ------------------------------------------------------------------
    #  Core: pyramidal FFT correlation (offline path)
    # ------------------------------------------------------------------

    def pyramidal_correlate(self, frames_np, pyramid_levels, prof=_NULL_PROF):
        """
        GPU-accelerated pyramidal multi-dt FFT correlation with sub-pixel
        interpolation and batched homothetic scaling.

        Parameters
        ----------
        frames_np : list of (pyramid_levels + 1) numpy arrays (H, W), float32
            Sequential frames: [frame_t, frame_t+1, ..., frame_t+K].
        pyramid_levels : int
        prof : PipelineProfiler (optional)

        Returns
        -------
        U, V : numpy arrays, shape (grid_y, grid_x), float64
        """
        with prof.measure("gpu_transfer_to"):
            frames_t = [torch.from_numpy(f.astype(np.float32)).to(self.device)
                        for f in frames_np]

        with prof.measure("gpu_window_extraction"):
            W0, grid_y, grid_x = self._extract_windows_gpu(frames_t[0])
            W0 = W0 - W0.mean(dim=(-2, -1), keepdim=True)

        N_windows = grid_y * grid_x

        with prof.measure("gpu_fft_forward"):
            F0 = torch.fft.rfft2(W0)

        R_sum = torch.zeros(N_windows, self.ws, self.ws,
                            dtype=torch.float32, device=self.device)

        for k in range(1, pyramid_levels + 1):
            with prof.measure("gpu_window_extraction"):
                Wk, _, _ = self._extract_windows_gpu(frames_t[k])
                Wk = Wk - Wk.mean(dim=(-2, -1), keepdim=True)

            with prof.measure("gpu_fft_forward"):
                Fk = torch.fft.rfft2(Wk)

            with prof.measure("gpu_cross_power"):
                cross_power = torch.conj(F0) * Fk

            with prof.measure("gpu_fft_inverse"):
                R_k = torch.fft.irfft2(cross_power, s=(self.ws, self.ws))

            with prof.measure("gpu_fftshift"):
                R_k = torch.fft.fftshift(R_k, dim=(-2, -1))

            if k == 1:
                R_sum += R_k
            else:
                with prof.measure("gpu_homothetic_warp"):
                    R_warped = self._batch_homothetic_warp(R_k, k)
                    R_sum += R_warped

        with prof.measure("gpu_peak_finding"):
            R_flat = R_sum.reshape(N_windows, -1)
            peak_indices = torch.argmax(R_flat, dim=1)
            cy = peak_indices // self.ws
            cx = peak_indices % self.ws

        with prof.measure("gpu_subpixel"):
            dx, dy = self._batch_subpixel_gpu(R_sum, cy, cx)

        center_c = self.ws // 2

        with prof.measure("gpu_transfer_from"):
            U = ((cx.float() + dx) - center_c).reshape(grid_y, grid_x).cpu().numpy().astype(np.float64)
            # Image convention: V is the row displacement, positive DOWNWARD
            # (row index increases downward). Matches the CPU path and the .mat data.
            V = ((cy.float() + dy) - center_c).reshape(grid_y, grid_x).cpu().numpy().astype(np.float64)

        return U, V

    # ------------------------------------------------------------------
    #  Cleanup
    # ------------------------------------------------------------------

    def release(self):
        """Free cached GPU tensors."""
        self._warp_grids.clear()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
