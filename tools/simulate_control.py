"""
EBIV Release 5.1 — closed-loop simulation (no camera, no Analog Discovery).

Runs the REAL ControlSupervisor, PID, filter and reference generator against a
crude first-order pump/jet model, so gains, reference trajectories and fault
handling can be exercised before any hardware is touched.

    python simulate_control.py

IMPORTANT: FirstOrderJetModel is a stand-in, not a model of the EcoDrift 4.3.
Its gain, time constant, dead-band, transport delay and square-root static
curve were chosen only to be a plausible non-linear first-order plant.  Gains
that work here are a STARTING POINT for tuning, not final values.  The real
static curve and time constant must come from an open-loop calibration run
(RUN_MODE='calibration'), and the gains must be re-tuned against it.
"""

# Run from anywhere: put the parent folder (the library) on the path.
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)                      # EBIV_Main.py
_sys.path.insert(0, _os.path.join(_ROOT, 'lib'))  # the library


import os
import logging

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ebiv_config import ReferenceConfig
from ebiv_control import FirstOrderJetModel, ControlState
from ebiv_runtime import simulate_closed_loop


def _plot(ax_u, ax_v, out, title, units):
    t = out['t']
    ax_u.plot(t, out['u_target'], color='tab:green', lw=1.8, label='U_target')
    ax_u.plot(t, out['u_plant'], color='0.55', lw=1.0, label='plant (truth)')
    ax_u.plot(t, out['u_filtered'], color='tab:orange', lw=1.4,
              label='U_control_filtered')
    ax_u.set_ylabel(f"velocity [{units}]")
    ax_u.set_title(title, fontsize=10)
    ax_u.grid(alpha=0.3)
    ax_u.legend(fontsize=7, loc='best')

    # shade the non-CLOSED_LOOP stretches
    st = np.array([s for s in out['state']])
    for state, col in (('HOLD', 'orange'), ('SAFE', 'red'), ('MANUAL', 'grey')):
        m = st == state
        if m.any():
            ax_u.fill_between(t, *ax_u.get_ylim(), where=m, color=col,
                              alpha=0.15, step='mid', lw=0)

    ax_v.plot(t, out['v_command'], color='tab:blue', lw=1.4)
    ax_v.set_ylabel("V_pump [V]")
    ax_v.set_xlabel("t [s]")
    ax_v.grid(alpha=0.3)


def main(cfg=None, outdir=None):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    if cfg is None:
        from EBIV_Main import build_config
        cfg = build_config()
    outdir = outdir or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "simulation_output")
    os.makedirs(outdir, exist_ok=True)
    units = cfg.measurement.velocity_units

    # Gains for the SIMULATED plant only.  If EBIV_Main leaves them at zero
    # (the shipped default, because there is no calibration yet) substitute a
    # set that is stable against FirstOrderJetModel so the simulation is
    # informative, and say so.
    import copy
    cfg = copy.deepcopy(cfg)
    cfg.ad3.enabled = False
    if cfg.pid.kp == 0.0 and cfg.pid.ki == 0.0:
        cfg.pid.kp, cfg.pid.ki, cfg.pid.kd = 0.35, 0.25, 0.0
        logging.warning(
            "PID gains in EBIV_Main are zero (no calibration yet). Using "
            "Kp=%.2f Ki=%.2f Kd=%.2f for the SIMULATION only. These are tuned "
            "for the toy plant and mean nothing for the real facility.",
            cfg.pid.kp, cfg.pid.ki, cfg.pid.kd)

    scenarios = [
        ("step",
         ReferenceConfig(kind='step', value=2.0, t_step_s=20.0, step_amplitude=1.5),
         dict(duration_s=60.0)),
        ("sine",
         ReferenceConfig(kind='sine', value=2.5, amplitude=0.8, freq_hz=0.05),
         dict(duration_s=120.0)),
        ("band-limited random",
         ReferenceConfig(kind='random', value=2.5, random_std=0.6,
                         random_cutoff_hz=0.03, random_seed=1),
         dict(duration_s=150.0)),
        ("noise + short dropouts",
         ReferenceConfig(kind='constant', value=2.5),
         dict(duration_s=60.0, measurement_noise=0.15,
              dropout_windows=[(20.0, 20.4), (35.0, 35.5), (45.0, 45.2)])),
        ("long dropout -> SAFE",
         ReferenceConfig(kind='constant', value=2.5),
         dict(duration_s=60.0, dropout_windows=[(20.0, 45.0)])),
    ]

    fig, axes = plt.subplots(2, len(scenarios), figsize=(4.2 * len(scenarios), 6),
                             sharex='col')
    summary = []
    for i, (name, ref, kw) in enumerate(scenarios):
        c = copy.deepcopy(cfg)
        c.reference = ref
        out = simulate_closed_loop(c, FirstOrderJetModel(), **kw)
        _plot(axes[0, i], axes[1, i], out, name, units)

        t = out['t']
        m = (t > t[-1] * 0.4) & (np.array([s for s in out['state']]) == ControlState.CLOSED_LOOP)
        err = out['u_plant'][m] - out['u_target'][m]
        summary.append((name, float(np.sqrt(np.mean(err ** 2))) if err.size else float('nan'),
                        float(np.nanmax(out['v_command'])),
                        sorted(set(out['state']))))

    fig.suptitle("Release 5.1 closed loop against a SIMULATED first-order "
                 "pump/jet — not a model of the EcoDrift 4.3", fontsize=11)
    fig.tight_layout()
    path = os.path.join(outdir, "closed_loop_simulation.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)

    print(f"\n{'=' * 74}\n  SIMULATION SUMMARY (toy plant)\n{'=' * 74}")
    print(f"  {'scenario':<26s} {'RMS err':>9s} {'max V':>7s}  states")
    for n, rms, vmax, states in summary:
        print(f"  {n:<26s} {rms:>9.4f} {vmax:>7.3f}  {','.join(states)}")
    print(f"\n  figure -> {path}")
    print("  These numbers describe the toy plant, not the facility.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
