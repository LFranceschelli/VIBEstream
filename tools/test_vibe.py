"""
VIBE — hardware-free test.

Replaces the Metavision SDK with a fake that streams a SYNTHETIC pulsed-laser
event stream (known displacement per pulse, known laser phase, background
noise), and the WaveForms runtime with the call-recording fake from
test_hardware_mock.py.  Then runs the REAL vibe.py and VibeStream library
against them.

What it proves: the event->frame logic, the trigger modes, the correlation
wiring, the recording bookkeeping, the threading/cleanup, and the exact
Analog Discovery call sequence.
What it cannot prove: that a real camera, SDK version or AD3 behaves like the
fakes.  In particular the 'external' trigger path is written against
EventsIterator.get_ext_trigger_events(), which the fake provides; check it on
the real camera before relying on it.

    python tools/test_vibe.py
"""

import os
import sys
import time
import types
import tempfile
import logging

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'lib'))
sys.path.insert(0, os.path.join(_ROOT, 'tools'))

import numpy as np

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(name)s | %(message)s")

_PASS, _FAIL = [], []


def check(name, cond, detail=""):
    (_PASS if cond else _FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
    return bool(cond)


# ==========================================================================
#  Synthetic pulsed-laser event stream
# ==========================================================================
W, H = 320, 240
F_LASER = 200.0
PERIOD = int(1e6 / F_LASER)          # 5000 us
PHASE = 1300                          # us, pulse time within each period
DX, DY = 3.3, -1.7                    # px per pulse (v negative = upward in the image)
BLOCK_S = 2.5                         # the stream repeats every BLOCK_S (a multiple of PERIOD)
EV_DTYPE = np.dtype([('x', '<u2'), ('y', '<u2'), ('p', '<i2'), ('t', '<i8')])


def _make_block(seed=1):
    rng = np.random.default_rng(seed)
    n_part, ev_per = 900, 6
    p0 = rng.uniform([0, 0], [W, H], size=(n_part, 2))
    n_pulses = int(BLOCK_S * 1e6 / PERIOD)
    xs, ys, ts = [], [], []
    for k in range(n_pulses):
        pos = p0 + k * np.array([DX, DY])
        pos = np.repeat(pos, ev_per, axis=0) + rng.normal(0, 0.7, size=(n_part * ev_per, 2))
        xs.append(np.mod(np.rint(pos[:, 0]), W))
        ys.append(np.mod(np.rint(pos[:, 1]), H))
        ts.append(k * PERIOD + PHASE + np.abs(rng.normal(0, 60, n_part * ev_per)))
    n_noise = int(BLOCK_S * 20_000)
    xs.append(rng.integers(0, W, n_noise))
    ys.append(rng.integers(0, H, n_noise))
    ts.append(rng.uniform(0, BLOCK_S * 1e6, n_noise))
    x = np.concatenate(xs); y = np.concatenate(ys); t = np.concatenate(ts)
    o = np.argsort(t, kind='stable')
    ev = np.zeros(len(t), EV_DTYPE)
    ev['x'], ev['y'], ev['t'], ev['p'] = x[o], y[o], t[o].astype(np.int64), 1
    return ev


BLOCK = _make_block()
BLOCK_T = np.ascontiguousarray(BLOCK['t'])   # searchsorted on a strided field copies it every call


class FakeEventsIterator:
    SPEED = 20.0          # live: camera time runs SPEED x faster than wall time

    def __init__(self, input_path=None, delta_t=10_000, _live=False, _device=None):
        self.delta_t = int(delta_t)
        self.live = _live
        self.device = _device
        self._t = 0
        self._trig = []
        if input_path is not None and not os.path.exists(input_path):
            raise FileNotFoundError(input_path)

    @classmethod
    def from_device(cls, device, delta_t=10_000):
        device.n_iterators += 1
        return cls(delta_t=delta_t, _live=True, _device=device)

    def get_size(self):
        return H, W

    def get_current_time(self):
        return self._t

    def get_ext_trigger_events(self):
        a = np.zeros(len(self._trig), dtype=[('p', '<i2'), ('t', '<i8'), ('id', '<i2')])
        if self._trig:
            a['t'] = self._trig
            a['p'] = 1
        return a

    def clear_ext_trigger_events(self):
        self._trig = []

    def __iter__(self):
        blk_us = int(BLOCK_S * 1e6)
        wall0 = time.perf_counter()
        rep = 0
        while True:
            off = rep * blk_us
            t_blk = BLOCK_T
            for a in range(0, blk_us, self.delta_t):
                i0 = np.searchsorted(t_blk, a, 'left')
                i1 = np.searchsorted(t_blk, a + self.delta_t, 'left')
                ev = BLOCK[i0:i1].copy()
                ev['t'] += off
                self._t = off + a + self.delta_t
                if self.live:
                    # pulses in this batch as external trigger events
                    k0 = int(np.ceil((off + a - PHASE) / PERIOD))
                    k1 = int(np.floor((off + a + self.delta_t - 1 - PHASE) / PERIOD))
                    self._trig += [k * PERIOD + PHASE for k in range(k0, k1 + 1)]
                    lag = (self._t / 1e6) / self.SPEED - (time.perf_counter() - wall0)
                    if lag > 0:
                        time.sleep(lag)
                if self.device is not None and self.device.stream.logging_path:
                    self.device.stream.n_logged += len(ev)
                yield ev
            if not self.live:
                return                   # a file ends
            rep += 1


class FakeStream:
    def __init__(self):
        self.logging_path = None
        self.n_logged = 0
        self.stopped = False

    def log_raw_data(self, path):
        self.logging_path = path
        with open(path, 'wb') as fh:
            fh.write(b'FAKE-RAW')

    def stop_log_raw_data(self):
        self.logging_path = None

    def stop(self):
        self.stopped = True


class FakeBiases:
    def __init__(self):
        self.values = {'bias_diff_on': 0, 'bias_diff_off': 0}

    def set(self, k, v):
        self.values[k] = v
        return True

    def get(self, k):
        return self.values[k]

    def get_all_biases(self):
        return dict(self.values)


class FakeTrigIn:
    def __init__(self):
        self.enabled = []

    def enable(self, ch):
        self.enabled.append(ch)


class FakeDevice:
    open_count = 0

    def __init__(self, serial):
        FakeDevice.open_count += 1
        self.serial = serial
        self.stream = FakeStream()
        self.biases = FakeBiases()
        self.trig = FakeTrigIn()
        self.n_iterators = 0

    def get_i_events_stream(self):
        return self.stream

    def get_i_ll_biases(self):
        return self.biases

    def get_i_trigger_in(self):
        return self.trig


def install_fake_metavision(serials=('FAKE-0001',)):
    hal = types.ModuleType('metavision_hal')

    class DeviceDiscovery:
        devices = list(serials)
        last = None

        @staticmethod
        def list():
            return list(DeviceDiscovery.devices)

        @staticmethod
        def open(s):
            DeviceDiscovery.last = FakeDevice(s)
            return DeviceDiscovery.last

    class _Ch:
        MAIN, AUX = 'MAIN', 'AUX'

    class I_TriggerIn:
        Channel = _Ch

    hal.DeviceDiscovery = DeviceDiscovery
    hal.I_TriggerIn = I_TriggerIn
    core = types.ModuleType('metavision_core')
    eio = types.ModuleType('metavision_core.event_io')
    eio.EventsIterator = FakeEventsIterator
    core.event_io = eio
    sys.modules['metavision_hal'] = hal
    sys.modules['metavision_core'] = core
    sys.modules['metavision_core.event_io'] = eio
    return DeviceDiscovery


DD = install_fake_metavision()

import ebiv_utils as U        # noqa: E402  (after the fakes)
import vibe as V              # noqa: E402
from vibe import VIBE, _FrameBuilder   # noqa: E402


def interior_mean(fld, margin=40):
    m = ((fld.x > margin) & (fld.x < W - margin) & (fld.y > margin) & (fld.y < H - margin))
    if fld.valid is not None:
        m &= fld.valid
    return float(fld.u_raw[m].mean()), float(fld.v_raw[m].mean()), int(m.sum())


# ==========================================================================
def run():
    print("=" * 76)
    print("  VIBE — hardware-free test (fake Metavision + fake WaveForms)")
    print("=" * 76)
    tmp = tempfile.mkdtemp(prefix="vibe_test_")
    check("fake Metavision active", U._HAS_METAVISION and U.EventsIterator is FakeEventsIterator)

    # --- 1. FrameBuilder, auto trigger --------------------------------------
    print("\n[1] frame builder")
    fb = _FrameBuilder(W, H, F_LASER, 'auto', 0.8)
    out = []
    for ev in FakeEventsIterator(input_path=None, delta_t=PERIOD):
        out += fb.push(ev['t'], ev['x'], ev['y'])
        if len(out) >= 20:
            break
    check("auto phase within 100 us of the true pulse time",
          fb.phase_us is not None and abs(fb.phase_us - PHASE) < 100, f"{fb.phase_us}")
    offs = {(tc - int(fb.phase_us)) % PERIOD for tc, _ in out}
    check("every window centred on the detected phase", offs == {0}, str(offs))
    check("consecutive windows one period apart",
          set(np.diff([tc for tc, _ in out])) == {PERIOD})
    check("no frame emitted during the 0.5 s calibration",
          out[0][0] >= 500_000, f"first centre {out[0][0]}")

    fb2 = _FrameBuilder(W, H, F_LASER, 'auto', 0.8)
    big = []
    for ev in FakeEventsIterator(input_path=None, delta_t=4 * PERIOD):
        big += fb2.push(ev['t'], ev['x'], ev['y'])
        if len(big) >= 12:
            break
    check("batch longer than a period: every window emitted (no frames lost)",
          set(np.diff([tc for tc, _ in big])) == {PERIOD})

    # pause in the stream -> re-anchor by whole periods
    fb3 = _FrameBuilder(W, H, F_LASER, 'auto', 0.8, phase_us=PHASE)
    ev = BLOCK
    fb3.push(ev["t"][:50_000], ev["x"][:50_000], ev["y"][:50_000])
    late = ev[ev['t'] > 2_000_000]
    after = fb3.push(late['t'], late['x'], late['y'])
    check("stream gap: schedule re-anchored, phase preserved",
          fb3.n_resync >= 1 and all((tc - PHASE) % PERIOD == 0 for tc, _ in after))

    # --- 2. live velocity, three trigger modes -----------------------------
    print("\n[2] live velocity()")
    vibe = VIBE(biases={'bias_diff_on': 40, 'bias_diff_off': 150}, output_dir=tmp,
                max_events_per_pixel=4)
    for trig in ('auto', 'none', 'external'):
        flds = list(vibe.velocity(f=F_LASER, n=10, window=32, step=16, trigger=trig))
        u, v, nv = interior_mean(flds[-1])
        check(f"trigger={trig}: 10 fields", len(flds) == 10)
        check(f"trigger={trig}: displacement ({DX}, {DY}) recovered to 0.15 px",
              abs(u - DX) < 0.15 and abs(v - DY) < 0.15, f"u={u:.3f} v={v:.3f} n={nv}")
    check("biases applied at open", DD.last.biases.values['bias_diff_on'] == 40)
    check("camera released after the generator ends", not vibe.busy)

    # break early
    for fld in vibe.velocity(f=F_LASER, window=32, step=16):
        break
    check("break out of velocity(): camera released immediately", not vibe.busy)

    # T limit on the camera clock
    flds = list(vibe.velocity(f=F_LASER, T=0.1, window=32, step=16))
    span = flds[-1].t_us - flds[0].t_us
    check("velocity(T=0.1) stops after 0.1 s of camera time",
          0.09e6 <= span <= 0.11e6, f"span {span} us, {len(flds)} fields")

    # ROI + flip
    vf = VIBE(output_dir=tmp, max_events_per_pixel=4, flip_x=True, roi=[40, 280, 40, 200])
    fld = next(iter(vf.velocity(f=F_LASER, n=3, window=32, step=16)))
    u, v, _ = interior_mean(fld)
    check("flip_x mirrors u", abs(u + DX) < 0.15 and abs(v - DY) < 0.15, f"u={u:.3f}")
    check("ROI respected (vector centres inside it)",
          fld.x.min() >= 40 and fld.x.max() < 280 and fld.y.min() >= 40 and fld.y.max() < 200)
    ms_u, _ = fld.to_ms(px_per_mm=20.0)
    check("to_ms uses dt = 1/f", np.allclose(ms_u, fld.u * F_LASER / 20.0 / 1000.0))

    val, ok, n_ok = VIBE.roi_mean(fld, [60, 260, 60, 180], 'u')
    check("roi_mean on raw valid vectors", ok and abs(val + DX) < 0.15, f"{val:.3f} ({n_ok})")

    # --- 3. record ------------------------------------------------------------
    print("\n[3] record()")
    r = vibe.record(T=0.5, f=F_LASER, path='rec1.raw')
    check("record: file written", os.path.exists(r.path))
    check("record: duration from the CAMERA clock >= T",
          0.5 <= r.duration_s < 0.52, f"{r.duration_s:.4f} s")
    check("record: complete", r.complete)
    meta = vibe.read_meta('rec1.raw')
    check("record: JSON sidecar with f and measured duration",
          meta.get('f_hz') == F_LASER and abs(meta.get('duration_s', 0) - r.duration_s) < 1e-9)
    check("record: raw logging stopped and stream stopped",
          DD.last.stream.logging_path is None and DD.last.stream.stopped)

    vibe.record(T=None, path='rec2.raw', block=False)
    check("background record: camera busy", vibe.busy)
    try:
        list(vibe.velocity(f=F_LASER, n=1))
        check("second operation refused while recording", False)
    except RuntimeError:
        check("second operation refused while recording", True)
    time.sleep(0.2)
    r2 = vibe.stop()
    check("background record: stop() returns a complete result",
          r2 is not None and r2.complete and r2.duration_s > 0, f"{r2 and r2.duration_s:.3f} s")
    check("background record: camera free afterwards", not vibe.busy)

    try:
        vibe.record(T=None)
        check("record(T=None, block=True) refused", False)
    except ValueError:
        check("record(T=None, block=True) refused", True)

    # --- 4. offline -------------------------------------------------------------
    print("\n[4] offline")
    ph = vibe.detect_phase('rec1.raw', F_LASER)
    check("detect_phase on a file", abs(ph - PHASE) < 100, f"{ph:.0f}")
    frs = list(vibe.frames_from_raw('rec1.raw', n=6))        # f from the sidecar
    check("frames_from_raw: n frames, f taken from metadata", len(frs) == 6)
    check("frames_from_raw: centred on the pulses",
          all((fr.t_us - int(round(ph))) % PERIOD == 0 for fr in frs))
    res = vibe.piv(frs, window=32, step=16)
    u, v, _ = interior_mean(res[-1])
    check("piv() on offline frames", abs(u - DX) < 0.15 and abs(v - DY) < 0.15,
          f"u={u:.3f} v={v:.3f}")
    check("piv(): f inferred from frame spacing", abs(res[0].f_hz - F_LASER) < 1e-6)
    paths = vibe.save_frames('rec1.raw', 'frames', n=4)
    check("save_frames writes .tif", len(paths) == 4 and all(os.path.exists(p) for p in paths))

    # --- 5. background stream --------------------------------------------------
    print("\n[5] start() / latest() / wait()")
    got = []
    vibe.start(f=F_LASER, window=32, step=16, on_field=lambda f_: got.append(f_.seq))
    f1 = vibe.wait(timeout=5.0)
    f2 = vibe.wait(timeout=5.0, newer_than=f1)
    check("wait() returns successive fields", f1 is not None and f2 is not None and f2.seq > f1.seq)
    u, v, _ = interior_mean(f2)
    check("background field correct", abs(u - DX) < 0.15 and abs(v - DY) < 0.15, f"u={u:.3f}")
    check("latest() is the newest", vibe.latest().seq >= f2.seq)
    check("callback called", len(got) >= 2)
    try:
        vibe.record(T=0.1)
        check("record refused while streaming", False)
    except RuntimeError:
        check("record refused while streaming", True)
    vibe.stop()
    check("stop(): threads ended, camera free", not vibe.running and not vibe.busy)
    vibe.start(f=F_LASER, window=32, step=16, max_rate_hz=10.0)
    time.sleep(1.0)
    n10 = vibe.latest().seq
    vibe.stop()
    check("max_rate_hz caps the field rate", 5 <= n10 <= 13, f"{n10} fields in 1 s at 10 Hz")

    # --- 6. errors ----------------------------------------------------------------
    print("\n[6] error paths")
    DD.devices = []
    try:
        list(vibe.velocity(f=F_LASER, n=1))
        check("no camera -> CameraSetupError", False)
    except V.CameraSetupError:
        check("no camera -> CameraSetupError", True)
    check("camera free after the failure", not vibe.busy)
    DD.devices = ['FAKE-0001']
    vs = VIBE(serial='NOPE', output_dir=tmp)
    try:
        vs.record(T=0.1)
        check("unknown serial -> CameraSetupError", False)
    except V.CameraSetupError:
        check("unknown serial -> CameraSetupError", True)
    U._HAS_METAVISION = False
    try:
        vibe.record(T=0.1)
        check("no Metavision -> CameraSetupError", False)
    except V.CameraSetupError as e:
        check("no Metavision -> CameraSetupError", "check_env" in str(e))
    U._HAS_METAVISION = True

    # --- 7. Analog Discovery ---------------------------------------------------
    print("\n[7] Analog Discovery (fake WaveForms runtime)")
    from test_hardware_mock import install_fake
    fake = install_fake()
    va = VIBE(output_dir=tmp, ad3={'laser_duty_percent': 10.0})
    va.laser_on(f=200)
    fr_set = fake.of('FDwfAnalogOutNodeFrequencySet')
    check("laser_on: square wave on ch0 at 200 Hz",
          fr_set and fr_set[-1][1] == 0 and abs(fr_set[-1][3] - 200) < 1e-9)
    n_before = len(fake.calls)
    va.set_voltage(2.0)
    va.set_voltage(3.1)
    pump_calls = fake.calls[n_before:]
    ch0 = [c for c in pump_calls if len(c[1]) > 1 and c[1][1] == 0]
    check("set_voltage never addresses the laser channel", not ch0, str(ch0[:2]))
    offs = [c[1] for c in pump_calls if c[0] == 'FDwfAnalogOutNodeOffsetSet']
    check("set_voltage: OffsetSet on ch1 with the value",
          offs and offs[-1][1] == 1 and abs(offs[-1][3] - 3.1) < 1e-9)
    conf = [c[1] for c in pump_calls if c[0] == 'FDwfAnalogOutConfigure']
    check("set_voltage: Configure(ch1, APPLY=3)", conf and conf[-1][1:] == (1, 3))
    check("set_voltage clamps to pump_v_max", va.set_voltage(7.0) == 5.0 and va.voltage == 5.0)
    try:
        va.set_voltage(float('nan'))
        check("non-finite voltage refused", False)
    except Exception:
        check("non-finite voltage refused", True)

    n_before = len(fake.calls)
    va.laser_on(f=500)
    retune = fake.calls[n_before:]
    check("retune: only ch0 addressed",
          all((len(c[1]) < 2) or c[1][1] == 0 for c in retune))
    check("retune: new frequency", fake.of('FDwfAnalogOutNodeFrequencySet')[-1][3] == 500)
    check("laser_frequency property", va.laser_frequency == 500)

    # an acquisition at a different f retunes a laser VIBE is driving
    list(va.velocity(f=F_LASER, n=1, window=32, step=16))
    check("velocity(f) retunes a running laser to f", va.laser_frequency == F_LASER)
    va.laser_off()
    list(va.velocity(f=250, n=1, window=32, step=16))
    check("a laser that is off is NOT switched on by an acquisition", not va.laser_running)

    try:
        va.ad3_open(laser_channel=1)
        check("config change refused on an open device", False)
    except RuntimeError:
        check("config change refused on an open device", True)

    n_before = len(fake.calls)
    va.close()
    closing = fake.calls[n_before:]
    parked = {c[1][1] for c in closing if c[0] == 'FDwfAnalogOutNodeOffsetSet' and c[1][3] == 0.0}
    check("close(): both channels parked at 0 V", {0, 1} <= parked, str(parked))
    check("close(): device closed", any(c[0] == 'FDwfDeviceClose' for c in closing))
    check("no call ever used channel -1",
          not any(len(c[1]) > 1 and c[1][1] == -1 and c[0].startswith('FDwfAnalogOut')
                  for c in fake.calls))

    # mock mode
    vm = VIBE(output_dir=tmp, ad3_mock=True)
    vm.laser_on(200)
    check("mock: set_voltage works without hardware", vm.set_voltage(1.5) == 1.5)
    n_before = len(fake.calls)
    vm.set_voltage(2.5)
    vm.close()
    check("mock: nothing sent to the runtime", len(fake.calls) == n_before)

    # --- 8. control helpers -----------------------------------------------------
    print("\n[8] control helpers")
    pid = VIBE.pid(kp=0.5, ki=0.2, v_min=0, v_max=5, slew_v_per_s=None)
    u_cmd = pid.update(target=4.0, measurement=3.0, dt=0.05).u_command
    check("pid(): VibeStream PIDController", abs(u_cmd - (0.5 * 1.0 + 0.2 * 1.0 * 0.05)) < 1e-9,
          f"{u_cmd}")

    print("\n" + "=" * 76)
    print(f"  {len(_PASS)} passed, {len(_FAIL)} failed")
    if _FAIL:
        print("  FAILED: " + "; ".join(_FAIL))
    print("=" * 76)
    return not _FAIL


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
