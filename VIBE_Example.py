"""
VIBE example — drive the camera, the laser and the pump from your own script.

    1. Set ONE (or more) flags in section 1.
    2. Set the parameters in section 2.
    3. python VIBE_Example.py

Everything here is done with the VIBE class (lib/vibe.py).  Copy the block you
need into your own code; the only lines you always need are the two imports
and `VIBE(...)`.
"""

import os
import sys
import time
import logging

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
from vibe import VIBE                                               # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# =============================================================================
#  1. WHAT TO DO
# =============================================================================
FLAG_LASER_TEST  = False   # start the laser on the Analog Discovery, measure it, stop
FLAG_RECORD      = False   # record T seconds of raw events
FLAG_LIVE        = True    # live velocity fields in a for-loop (print + window)
FLAG_BACKGROUND  = False   # rt-EBIV in the background, your loop reads latest()
FLAG_CLOSED_LOOP = False   # PID on the pump from a ROI-averaged velocity
FLAG_OFFLINE     = False   # .raw -> pseudo-images (one per laser pulse) -> PIV

AD3_MOCK = True            # True: no Analog Discovery touched (dry run). False: real device.

# =============================================================================
#  2. PARAMETERS
# =============================================================================
BIASES = {'bias_diff_on': 40, 'bias_diff_off': 150, 'bias_hpf': 70,
          'bias_fo': 0, 'bias_refr': 90, 'bias_diff': 0}
F = 200.0                        # Hz: laser pulse rate = frame rate
ROI = [200, 1100, 150, 600]      # [x0, x1, y0, y1] px, or None for the full sensor
WINDOW, STEP = 48, 24            # interrogation window / node distance, px
TRIGGER = 'auto'                 # 'auto' | 'external' | 'none'
PX_PER_MM = None                 # optical resolution; None -> results in px/frame
SHOW_EVERY = 5                   # draw every Nth live field in a window; 0 = no window
OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "vibe_runs")

RECORD_T = 2.0                   # s, camera clock
RAW_FILE = "run01.raw"

AD3 = dict(device_index=-1, laser_channel=0, laser_amplitude_v=2.5, laser_offset_v=2.5,
           laser_duty_percent=10.0, pump_channel=1, pump_v_min=0.0, pump_v_max=5.0)

# closed loop (gains in V per measured unit; tune from an open-loop calibration first)
CONTROL_ROI = [500, 700, 300, 450]
CONTROL_COMPONENT = 'u'          # 'u', '-u', 'v', '-v', 'mag'
TARGET = 3.0                     # same unit as the measurement (px/frame or m/s)
KP, KI, KD = 0.0, 0.0, 0.0       # all zero = the loop only logs; nothing moves
CONTROL_RATE_HZ = 20.0
CONTROL_DURATION_S = 60.0
START_VOLTAGE = 0.0


def measured(fld):
    """ROI-averaged control quantity in the chosen units, plus a validity flag."""
    val, ok, _ = VIBE.roi_mean(fld, CONTROL_ROI, CONTROL_COMPONENT, min_valid=0.5)
    if PX_PER_MM:
        val = val * fld.f_hz / PX_PER_MM / 1000.0          # px/frame -> m/s
    return val, ok


# =============================================================================
#  3. RUN
# =============================================================================
with VIBE(biases=BIASES, roi=ROI, output_dir=OUTPUT_DIR, max_events_per_pixel=1,
          ad3=AD3, ad3_mock=AD3_MOCK) as vibe:

    if FLAG_LASER_TEST:
        vibe.laser_on(f=F)
        if not AD3_MOCK:
            print("Laser measured on the AD3 scope (wire W1 -> 1+):", vibe.laser_check())
        input("Laser running. ENTER to stop... ")
        vibe.laser_off()

    if FLAG_RECORD:
        vibe.laser_on(f=F)                       # skip if the laser is driven externally
        rec = vibe.record(T=RECORD_T, f=F, path=RAW_FILE)
        print(f"{rec.path}: {rec.duration_s:.3f} s, {rec.n_events} events, "
              f"complete={rec.complete}")

    if FLAG_LIVE:
        vibe.laser_on(f=F)
        for fld in vibe.velocity(f=F, T=10, window=WINDOW, step=STEP, trigger=TRIGGER):
            u, ok = VIBE.roi_mean(fld)[:2]
            print(f"t={fld.t_us / 1e6:8.3f} s  <u>={u:+.2f} px/frame  lag={fld.lag_s:.3f} s")
            if SHOW_EVERY and fld.seq % SHOW_EVERY == 0 and vibe.show(fld) == ord('q'):
                break

    if FLAG_BACKGROUND:
        vibe.laser_on(f=F)
        vibe.start(f=F, window=WINDOW, step=STEP, trigger=TRIGGER, max_rate_hz=50)
        try:
            t_end = time.time() + 10
            while time.time() < t_end:
                fld = vibe.wait(timeout=1.0)       # blocks until a NEW field
                if fld is not None:
                    print(f"field {fld.seq}: <|U|>={fld.magnitude.mean():.2f} px/frame")
        finally:
            vibe.stop()

    if FLAG_CLOSED_LOOP:
        vibe.laser_on(f=F)
        pid = VIBE.pid(kp=KP, ki=KI, kd=KD, v_min=AD3['pump_v_min'],
                       v_max=AD3['pump_v_max'], slew_v_per_s=2.0)
        v_cmd = vibe.set_voltage(START_VOLTAGE)
        vibe.start(f=F, window=WINDOW, step=STEP, trigger=TRIGGER,
                   max_rate_hz=3 * CONTROL_RATE_HZ, record="closed_loop.raw")
        try:
            dt = 1.0 / CONTROL_RATE_HZ
            t0 = t_last = time.perf_counter()
            primed = False
            while time.perf_counter() - t0 < CONTROL_DURATION_S:
                time.sleep(max(0.0, t_last + dt - time.perf_counter()))
                now = time.perf_counter()
                step_dt, t_last = now - t_last, now
                fld = vibe.latest()
                stale = fld is None or (now - fld.t_wall) > 3 * dt
                y, ok = (float('nan'), False) if stale else measured(fld)
                if stale or not ok:
                    pid.hold(v_cmd)                   # HOLD: keep the voltage, no integration
                    print(f"HOLD ({'stale' if stale else 'invalid'})  V={v_cmd:.3f}")
                    continue
                if not primed:                        # bumpless start
                    pid.set_bumpless(v_cmd, y, TARGET)
                    primed = True
                v_cmd = vibe.set_voltage(pid.update(TARGET, y, step_dt).u_command)
                print(f"y={y:+.3f}  target={TARGET:+.3f}  V={v_cmd:.3f}")
        finally:
            vibe.stop()
            vibe.set_voltage(0.0)                     # close() parks it at 0 V anyway

    if FLAG_OFFLINE:
        frames = list(vibe.frames_from_raw(RAW_FILE, n=100, roi=ROI))   # f from the .json
        fields = vibe.piv(frames, window=WINDOW, step=STEP)
        print(f"{len(fields)} fields; mean u = "
              f"{sum(float(f_.u.mean()) for f_ in fields) / len(fields):+.2f} px/frame")
        # or the classic file pipeline:
        # vibe.save_frames(RAW_FILE, "RawImg", n=100, roi=ROI)
        # vibe.piv_offline("RawImg", "Out", window=64, step=32, levels=3)

# leaving the `with` block: streams stopped, camera released, laser and pump at 0 V
