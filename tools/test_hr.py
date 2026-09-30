"""
vibe_hr — synthetic end-to-end test of the direct-KF HR estimator.

Builds the synthetic time-resolved LR/HR dataset (tools/hr_synth.py), splits
it like Proc_Main_FullKF_DEF.m (train 1:4500, val 5000:6499, test 7000:8999),
trains with the TRAINING mean (what is available online), and checks:

  * per-snapshot step(u, v) reproduces the batch run exactly
  * steady-state gain == time-varying filter after the transient
  * save / load round trip
  * intermittent mode (measurement every DS steps)
  * delta (paper Eq. 34) of KF vs direct cubic interpolation, against HR-LOR
  * cost per step, single BLAS thread

The synthetic flow is linear convection with stochastic forcing: a best case
for a linear state-space model.  The delta values say the pipeline works;
they are NOT a prediction of the experimental errors in the paper.

    python tools/test_hr.py            (about 3-5 min, mostly the SVD)
"""

import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # see vibe_hr.limit_blas_threads
os.environ.setdefault("MKL_NUM_THREADS", "1")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(_ROOT, 'lib'), os.path.join(_ROOT, 'tools')]

import tempfile                  # noqa: E402
import numpy as np               # noqa: E402
import vibe_hr as H              # noqa: E402
import hr_synth as S             # noqa: E402

_PASS, _FAIL = [], []


def check(name, cond, detail=""):
    (_PASS if cond else _FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
    return bool(cond)


def run():
    print("=" * 76)
    print("  vibe_hr — direct Kalman filter, synthetic end-to-end test")
    print("=" * 76)
    t0 = time.time()
    lr, hr, info = S.make_dataset()
    print(f"  synthetic data: {info['nt']} snapshots, LR grid {info['lr_grid']}, "
          f"HR grid {info['hr_grid']}  ({time.time() - t0:.0f} s)")
    tr, va, te = slice(0, 4500), slice(4999, 6499), slice(6999, 8999)

    t0 = time.time()
    m = H.train_kf(lr['U'][tr], lr['V'][tr], hr['U'][tr], hr['V'][tr],
                   lr['U'][va], lr['V'][va], hr['U'][va], hr['V'][va],
                   grids=dict(lr_x=lr['X'], lr_y=lr['Y'], hr_x=hr['X'], hr_y=hr['Y']))
    print(f"  trained in {time.time() - t0:.0f} s: {m.summary()}")

    # --- operators ---------------------------------------------------------------
    print("\n[1] identified operators")
    check("shapes", m.C.shape == (m.r_lr, m.r) and m.F.shape == (m.r, m.r)
          and m.Q.shape == (m.r, m.r) and m.R.shape == (m.r_lr, m.r_lr))
    check("Q, R symmetric positive definite",
          np.allclose(m.Q, m.Q.T) and np.allclose(m.R, m.R.T)
          and np.linalg.eigvalsh(m.Q).min() > 0 and np.linalg.eigvalsh(m.R).min() > 0)
    rho = float(np.max(np.abs(np.linalg.eigvals(m.F))))
    check("F stable (spectral radius < 1)", rho < 1.0, f"{rho:.4f}")

    # --- batch vs step ---------------------------------------------------------
    print("\n[2] online API")
    est = H.KalmanEstimator(m)
    X = est.run(lr['U'][te], lr['V'][te])
    est2 = H.KalmanEstimator(m)
    Xs = np.array([est2.step(lr['U'][te][k], lr['V'][te][k]) for k in range(50)])
    check("step(u, v) per snapshot == batch run", np.allclose(Xs, X[:50], rtol=1e-10, atol=1e-14))
    u1, v1 = est2.field()
    u2, v2 = m.reconstruct(Xs[-1])
    check("field() == reconstruct(x)", np.allclose(u1, u2) and np.allclose(v1, v2))
    try:
        est2.step(lr['U'][te][0][:-1], lr['V'][te][0][:-1])
        check("wrong LR grid refused", False)
    except ValueError:
        check("wrong LR grid refused", True)

    # dropped fields: step_timed bridges them with prediction-only steps
    per = 10_000
    keep = [k for k in range(60) if k % 4 != 3 and k not in (20, 21, 22)]
    a = H.KalmanEstimator(m)
    for k in keep:
        xa = a.step_timed(k * per, lr['U'][te][k], lr['V'][te][k], per)
    b = H.KalmanEstimator(m)
    for k in range(keep[-1] + 1):
        xb = b.step(lr['U'][te][k], lr['V'][te][k]) if k in keep else b.step()
    check("step_timed(): dropped fields bridged by prediction-only steps",
          np.allclose(xa, xb, rtol=1e-12, atol=1e-15) and a.n_skipped == keep[-1] + 1 - len(keep),
          f"{a.n_skipped} periods bridged")
    tf = H.KalmanEstimator(m, lr_transform=lambda u_, v_: (u_.T, v_.T))
    xt = tf.step(lr['U'][te][0].T, lr['V'][te][0].T)
    check("lr_transform applied before projection", np.allclose(xt, X[0]))

    ss = H.KalmanEstimator(m, steady_state=True)
    Xss = ss.run(lr['U'][te], lr['V'][te])
    n_tr = ss.ss_iters
    tail = slice(min(len(X) - 100, 3 * n_tr), None)
    rel = np.abs(Xss[tail] - X[tail]).max() / np.abs(X[tail]).max()
    check("steady-state gain == time-varying after the transient", rel < 1e-6,
          f"Riccati converged in {n_tr} steps; max rel diff after {tail.start} steps: {rel:.1e}")
    early = np.abs(Xss[:20] - X[:20]).max() / np.abs(X[:20]).max()
    print(f"        (first 20 steps differ by up to {early:.1e}: the time-varying transient)")

    with tempfile.TemporaryDirectory() as td:
        p = m.save(os.path.join(td, 'model.npz'))
        m2 = H.HRModel.load(p)
        X2 = H.KalmanEstimator(m2).run(lr['U'][te][:20], lr['V'][te][:20])
        check("save / load round trip", np.allclose(X2, X[:20], rtol=1e-12, atol=1e-15)
              and m2.meta['n_train'] == 4500 and m2.hr_x is not None)

    # --- accuracy -----------------------------------------------------------------
    print("\n[3] accuracy on the test set (reference: HR-LOR, paper Sec. 4)")
    u_lor, v_lor = m.lor_hr(hr['U'][te], hr['V'][te])
    u_kf, v_kf = m.reconstruct(X)
    u_ss, v_ss = m.reconstruct(Xss)
    t0 = time.time()
    u_cu, v_cu = H.cubic_baseline(lr['U'][te], lr['V'][te], lr['X'], lr['Y'], hr['X'], hr['Y'])
    t_cu = time.time() - t0
    Ds = 3
    Xi = H.KalmanEstimator(m).run(lr['U'][te], lr['V'][te], ds=Ds)
    u_in, v_in = m.reconstruct(Xi)
    uref = info['u_ref']
    d = {k: H.delta_error(a, b, u_lor, v_lor, uref) for k, (a, b) in dict(
        cubic=(u_cu, v_cu), kf=(u_kf, v_kf), kf_ss=(u_ss, v_ss), kf_ds=(u_in, v_in)).items()}
    # the HR-LOR reference vs the raw HR (truncation + HR noise), for scale
    d_lor = H.delta_error(u_lor, v_lor, hr['U'][te], hr['V'][te], uref)
    tke = 0.5 * ((u_lor - u_lor.mean(0)) ** 2 + (v_lor - v_lor.mean(0)) ** 2).mean(0)
    for a in (0.25, 0.5):
        msk = tke > a * tke.max()
        d[f'cubic_a{a}'] = H.delta_error(u_cu, v_cu, u_lor, v_lor, uref, mask=msk)
        d[f'kf_a{a}'] = H.delta_error(u_kf, v_kf, u_lor, v_lor, uref, mask=msk)
        d[f'cov_a{a}'] = msk.mean()
    print(f"        delta   cubic {d['cubic']:.4f} | KF {d['kf']:.4f} | KF steady {d['kf_ss']:.4f}"
          f" | KF, LR every {Ds} steps {d['kf_ds']:.4f}   (HR-LOR vs raw HR: {d_lor:.4f})")
    for a in (0.25, 0.5):
        print(f"        delta_{a}: cubic {d[f'cubic_a{a}']:.4f} | KF {d[f'kf_a{a}']:.4f}"
              f"   ({d[f'cov_a{a}'] * 100:.0f}% of points)")
    check("KF beats cubic interpolation (full domain)", d['kf'] < d['cubic'],
          f"{d['kf']:.4f} vs {d['cubic']:.4f}")
    check("intermittent mode runs and degrades gracefully",
          d['kf'] <= d['kf_ds'] < 2.0 * d['cubic'], f"{d['kf_ds']:.4f}")

    # second-order statistics, as in Sec. 4.3
    def tke_of(u, v):
        return 0.5 * ((u - u.mean(0)) ** 2 + (v - v.mean(0)) ** 2).mean()
    k_ref, k_kf, k_cu = tke_of(u_lor, v_lor), tke_of(u_kf, v_kf), tke_of(u_cu, v_cu)
    print(f"        domain-mean TKE: HR-LOR {k_ref:.3e} | KF {k_kf:.3e} ({k_kf / k_ref:.2f}x)"
          f" | cubic {k_cu:.3e} ({k_cu / k_ref:.2f}x)")

    # --- Methods II and III ----------------------------------------------------------
    print("\n[4] LSE (Method II) and LSE+VR (Method III)")
    Ytr = m.project_lr(lr['U'][tr], lr['V'][tr]).T          # (r_lr, Nt)
    Atr = m.project_hr(hr['U'][tr], hr['V'][tr]).T          # (r, Nt)
    M_ls = Atr @ np.linalg.pinv(Ytr)
    check("M = ridge least-squares LR->HR map (Eq. 20)",
          np.abs(m.M - M_ls).max() / np.abs(M_ls).max() < 1e-6)
    At = m.M @ Ytr
    var_ratio = (m.Gamma[:, None] * At).var(axis=1) / Atr.var(axis=1)
    cap = m.meta['max_gain']
    free = m.Gamma < cap
    check("Gamma restores the training variance of every uncapped mode (Eq. 27)",
          free.any() and np.allclose(var_ratio[free], 1.0, rtol=1e-6),
          f"Gamma in [{m.Gamma.min():.2f}, {m.Gamma.max():.2f}], {int((~free).sum())} capped")
    check("capped modes: Gamma == MaxGain, variance still below the target",
          np.all(m.Gamma[~free] == cap) and np.all(var_ratio[~free] <= 1 + 1e-9)
          and m.meta['gamma_n_capped'] == int((~free).sum()))
    check("R_lse, R_vr symmetric positive definite",
          all(np.allclose(R_, R_.T) and np.linalg.eigvalsh(R_).min() > 0 for R_ in (m.R_lse, m.R_vr)))
    res = {}
    for meth in ('lse', 'lse_vr'):
        Xm = H.KalmanEstimator(m, method=meth).run(lr['U'][te], lr['V'][te])
        Xms = H.KalmanEstimator(m, method=meth, steady_state=True).run(lr['U'][te], lr['V'][te])
        tail = slice(400, None)
        rel_m = np.abs(Xms[tail] - Xm[tail]).max() / np.abs(Xm[tail]).max()
        check(f"{meth}: steady-state == time-varying after the transient", rel_m < 1e-6, f"{rel_m:.1e}")
        eu, ev = m.reconstruct(Xm)
        res[meth] = (H.delta_error(eu, ev, u_lor, v_lor, uref), tke_of(eu, ev) / k_ref)
        # instantaneous (pre-filter) LSE estimate, for reference
    psi_te = m.project_lr(lr['U'][te], lr['V'][te])
    iu, iv = m.reconstruct(m.lse(psi_te))
    d_inst = H.delta_error(iu, iv, u_lor, v_lor, uref)
    t_inst = tke_of(iu, iv) / k_ref
    print(f"        delta: KF {d['kf']:.4f} | LSE {res['lse'][0]:.4f} | LSE+VR {res['lse_vr'][0]:.4f}"
          f" | LSE without KF {d_inst:.4f} | cubic {d['cubic']:.4f}")
    print(f"        TKE / TKE(HR-LOR): KF {k_kf / k_ref:.3f} | LSE {res['lse'][1]:.3f} | "
          f"LSE+VR {res['lse_vr'][1]:.3f} | LSE without KF {t_inst:.3f}")
    check("LSE attenuates fluctuation energy, VR restores part of it (paper Sec. 4.3)",
          res['lse'][1] < res['lse_vr'][1] and res['lse'][1] < 1.0)
    check("all three methods beat cubic interpolation",
          max(d['kf'], res['lse'][0], res['lse_vr'][0]) < d['cubic'])

    # --- cost ---------------------------------------------------------------------
    print("\n[5] cost per snapshot (1 BLAS thread)")
    psi = m.project_lr(lr['U'][te][:201], lr['V'][te][:201])
    for label, e in (("time-varying", H.KalmanEstimator(m)),
                     ("steady-state", H.KalmanEstimator(m, steady_state=True))):
        e.step(psi=psi[0])
        t = time.perf_counter()
        for p_ in psi[1:]:
            e.step(psi=p_)
        dt_kf = (time.perf_counter() - t) / 200
        t = time.perf_counter()
        for k in range(50):
            e.measure(lr['U'][te][k], lr['V'][te][k])
        dt_pr = (time.perf_counter() - t) / 50
        t = time.perf_counter()
        for _ in range(50):
            e.field()
        dt_rc = (time.perf_counter() - t) / 50
        print(f"        {label:13s}: projection {dt_pr * 1e3:.3f} ms | KF step {dt_kf * 1e3:.3f} ms"
              f" | HR reconstruction {dt_rc * 1e3:.3f} ms   (r={m.r}, r_lr={m.r_lr}, "
              f"HR dof={m.phi_hr.shape[0]})")
    print(f"        cubic interpolation baseline: {t_cu / len(u_cu) * 1e3:.1f} ms/snapshot "
          f"(scipy RegularGridInterpolator, not optimised)")

    print("\n" + "=" * 76)
    print(f"  {len(_PASS)} passed, {len(_FAIL)} failed")
    if _FAIL:
        print("  FAILED: " + "; ".join(_FAIL))
    print("=" * 76)
    return not _FAIL


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
