"""
Synthetic, time-resolved LR/HR velocity dataset for testing vibe_hr.

The "true" flow on [0, Lx] x [-Ly/2, Ly/2]:
    U(y) jet-like mean profile (top-hat with tanh shear layers at y = +-0.5)
  + divergence-free fluctuations from a streamfunction
        psi = sum_m g_m(y) Re{ a_m(t) exp(i k_m (x - c t)) }
    g_m   Gaussian envelope on ONE of the two shear layers
    k_m   log-spaced from large to small scales (broadband)
    a_m   complex Ornstein-Uhlenbeck amplitude (finite correlation time),
          rms ~ k^-0.9, so energy spreads over many POD modes
    c     convection speed
    u = d psi/dy,  v = -d psi/dx  (analytic)

The HR and LR fields are BOX AVERAGES of the true field over square
interrogation windows (HR: small windows, 75 % overlap; LR: 6x coarser spacing,
0 % overlap), plus independent white noise, larger on LR.  The box averages
are computed exactly per wave (they are linear and separable), so no fine
field is ever stored.

This is a best case for a LINEAR state-space model: the dynamics are linear
convection plus linear stochastic forcing.  Real turbulence is not; do not
read the absolute numbers as a prediction for experimental data.
"""

import numpy as np


def _box_avg_1d(fun, centers, width, n_sub=64):
    """Average of fun(s) over [c - w/2, c + w/2] for each centre (numerical)."""
    s = (np.arange(n_sub) + 0.5) / n_sub - 0.5
    pts = centers[:, None] + width * s[None, :]
    return fun(pts).mean(axis=1)


def make_dataset(nt=9000, dt=0.02, seed=0, lx=4.0, ly=2.0, c=0.6,
                 n_k=50, k_min=1.2, k_max=40.0,
                 hr_win=1 / 6, hr_step=1 / 24, lr_win=0.25,
                 noise_hr=0.01, noise_lr=0.05, amp=0.06):
    rng = np.random.default_rng(seed)

    # --- grids (window centres) -------------------------------------------
    def grid(win, step):
        xs = np.arange(win / 2, lx - win / 2 + 1e-9, step)
        ys = np.arange(-ly / 2 + win / 2, ly / 2 - win / 2 + 1e-9, step)
        return xs, ys

    hx, hy = grid(hr_win, hr_step)
    lx_, ly_ = grid(lr_win, lr_win)

    # --- waves ----------------------------------------------------------------
    ks = np.geomspace(k_min, k_max, n_k)
    K = np.concatenate([ks, ks])                   # upper layer, lower layer
    y0 = np.concatenate([np.full(n_k, 0.5), np.full(n_k, -0.5)])
    wid = 0.08 + 1.2 / K                           # large scales -> wider envelope
    rms = amp * (K / k_min) ** -0.9
    tau = 1.5 / (K * c) + 0.3                      # correlation time
    M = K.size

    def g(y, m):
        return np.exp(-((y - y0[m]) / wid[m]) ** 2)

    def dg(y, m):
        return -2 * (y - y0[m]) / wid[m] ** 2 * g(y, m)

    def mean_u(y):
        return 0.5 * (np.tanh((0.5 - y) / 0.08) + np.tanh((0.5 + y) / 0.08))

    def basis(xs, ys, win):
        Gu = np.stack([_box_avg_1d(lambda y: dg(y, m), ys, win) for m in range(M)], 1)
        Gv = np.stack([_box_avg_1d(lambda y: g(y, m), ys, win) for m in range(M)], 1)
        Ex = np.stack([_box_avg_1d(lambda x: np.exp(1j * K[m] * x), xs, win)
                       for m in range(M)], 0)
        Um = _box_avg_1d(mean_u, ys, win)
        return Gu, Gv, Ex, Um

    Bh = basis(hx, hy, hr_win)
    Bl = basis(lx_, ly_, lr_win)

    # --- amplitudes: complex OU, then convection phase ---------------------------
    phi = np.exp(-dt / tau)
    a = np.empty((nt, M), dtype=complex)
    a[0] = rms * (rng.standard_normal(M) + 1j * rng.standard_normal(M)) / np.sqrt(2)
    s_in = rms * np.sqrt(1 - phi ** 2) / np.sqrt(2)
    for t in range(1, nt):
        a[t] = phi * a[t - 1] + s_in * (rng.standard_normal(M) + 1j * rng.standard_normal(M))
    a *= np.exp(-1j * np.outer(np.arange(nt) * dt, K * c))

    def fields(B, noise):
        Gu, Gv, Ex, Um = B
        # u = d psi/dy, v = -d psi/dx ; shapes (nt, ny, nx)
        U = np.real(np.einsum('ym,tm,mx->tyx', Gu, a, Ex)) + Um[None, :, None]
        V = np.real(np.einsum('ym,tm,mx->tyx', Gv, -1j * K[None, :] * a, Ex))
        U += noise * rng.standard_normal(U.shape)
        V += noise * rng.standard_normal(V.shape)
        return U, V

    Uh, Vh = fields(Bh, noise_hr)
    Ul, Vl = fields(Bl, noise_lr)
    HX, HY = np.meshgrid(hx, hy)
    LX, LY = np.meshgrid(lx_, ly_)
    info = dict(nt=nt, dt=dt, c=c, n_waves=M, hr_grid=HX.shape, lr_grid=LX.shape,
                noise_hr=noise_hr, noise_lr=noise_lr, u_ref=1.0)
    return (dict(X=LX, Y=LY, U=Ul, V=Vl), dict(X=HX, Y=HY, U=Uh, V=Vh), info)


def save_matlab(folder, lr, hr):
    """Write LR.mat / HR.Mat in the MATLAB layout: X, Y (nx, ny); U, V (nx, ny, Nt)."""
    import os
    from scipy.io import savemat
    os.makedirs(folder, exist_ok=True)
    for name, d in (('LR.mat', lr), ('HR.Mat', hr)):
        savemat(os.path.join(folder, name), {
            'X': d['X'].T, 'Y': d['Y'].T,
            'U': np.transpose(d['U'], (2, 1, 0)), 'V': np.transpose(d['V'], (2, 1, 0))})
