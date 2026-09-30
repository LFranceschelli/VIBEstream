"""
EBIV Release 5.1 — Analog Discovery call-sequence verification.

The AD3 layer is the part of Release 5.x that cannot be exercised without the
device on the bench, and it is also the part where a mistake is expensive: a
stray call can stop the laser in the middle of a run, or push a voltage at the
pump that the control law never asked for.

This file removes that blind spot without hardware.  It substitutes a fake
WaveForms runtime that records every ctypes call with its arguments, runs the
REAL AnalogDiscovery3, AnalogOutPump and verification code against it, and
then asserts the properties the design claims:

  * the device is opened exactly once and never reset or reopened;
  * FDwfDeviceAutoConfigureSet(0) is issued before anything is configured;
  * the laser channel is configured once and NEVER addressed again — in
    particular not by a pump update;
  * idxChannel is ALWAYS an explicit channel; -1 ("all channels") never
    appears in any call;
  * a pump voltage update is exactly NodeOffsetSet followed by
    AnalogOutConfigure(pump_channel, fStart=3);
  * the pump channel is funcDC with zero amplitude, so the voltage is
    entirely the node offset;
  * commands outside the hardware limits are clamped before they reach the
    device;
  * the DIO pattern backend touches DigitalOut only, never AnalogOut;
  * a failing SDK call raises AD3Error rather than passing silently.

It does NOT prove the hardware behaves — only that the software asks for what
it is supposed to ask for.  The oscilloscope procedure in ReadMe.txt section 8
is still the authority on whether the laser is actually undisturbed.

    python test_hardware_mock.py
"""

# Run from anywhere: put the parent folder (the library) on the path.
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)                      # EBIV_Main.py
_sys.path.insert(0, _os.path.join(_ROOT, 'lib'))  # the library


import sys
import types
import ctypes

_PASS, _FAIL = [], []


def check(name, cond, detail=""):
    (_PASS if cond else _FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
    return bool(cond)


# ==========================================================================
#  Fake WaveForms runtime
# ==========================================================================

class FakeFn:
    """One recorded SDK entry point."""

    def __init__(self, lib, name):
        self.lib, self.name = lib, name
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        def unwrap(a):
            if isinstance(a, ctypes._SimpleCData):
                return a.value
            if isinstance(a, (bytes, bytearray)):
                return a
            return a
        rec = (self.name, tuple(unwrap(a) for a in args))
        self.lib.calls.append(rec)
        return self.lib.returns.get(self.name, 1)


class FakeDwf:
    """Stands in for dwf.dll / libdwf.so."""

    def __init__(self, n_devices=1):
        self.calls = []
        self.returns = {}
        self.n_devices = n_devices
        self._fns = {}
        self.missing = set()

    def __getattr__(self, name):
        if not name.startswith("FDwf"):
            raise AttributeError(name)
        if name in self.missing:
            raise AttributeError(name)
        fn = self._fns.get(name)
        if fn is None:
            fn = self._fns[name] = FakeFn(self, name)
        return fn

    # --- convenience views over the recording ---
    def names(self):
        return [c[0] for c in self.calls]

    def of(self, name):
        return [c[1] for c in self.calls if c[0] == name]

    def index_of(self, name):
        for i, c in enumerate(self.calls):
            if c[0] == name:
                return i
        return -1


def install_fake(n_devices=1, missing=(), returns=None):
    """Point ebiv_hardware at a fresh fake runtime and return it."""
    import ebiv_hardware as hw
    fake = FakeDwf(n_devices)
    fake.missing = set(missing)
    fake.returns = dict(returns or {})

    # FDwfEnum writes the device count into its out-parameter.
    def enum(_filter, p_count):
        fake.calls.append(("FDwfEnum", (_filter.value if hasattr(_filter, 'value')
                                        else _filter,)))
        p_count._obj.value = fake.n_devices
        return 1

    # FDwfDeviceOpen writes a non-zero handle.
    def open_dev(idx, p_h):
        fake.calls.append(("FDwfDeviceOpen",
                           (idx.value if hasattr(idx, 'value') else idx,)))
        if fake.n_devices <= 0:
            return 0
        p_h._obj.value = 0x1234
        return 1

    def version(buf):
        fake.calls.append(("FDwfGetVersion", ()))
        buf.value = b"3.99.0"
        return 1

    def clockinfo(_h, p_hz):
        fake.calls.append(("FDwfDigitalOutInternalClockInfo", ()))
        p_hz._obj.value = 100e6
        return 1

    class Wrapper(FakeDwf):
        pass

    fake.FDwfEnum_impl = enum
    hw._dwf = fake
    hw._dwf_load_error = None

    # Patch the three calls that use out-parameters.
    fake._fns['FDwfEnum'] = enum
    fake._fns['FDwfDeviceOpen'] = open_dev
    fake._fns['FDwfGetVersion'] = version
    fake._fns['FDwfDigitalOutInternalClockInfo'] = clockinfo
    return fake


# ==========================================================================

def run():
    print("=" * 76)
    print("  EBIV Release 5.1 — Analog Discovery call-sequence verification")
    print("  (fake WaveForms runtime; no hardware involved)")
    print("=" * 76)

    import ebiv_hardware as hw
    from ebiv_config import ControlSystemConfig, AD3Config, SupervisorConfig

    AO_APPLY, AO_START = 3, 1

    # ------------------------------------------------------------------
    #  1. Analog laser on W1, pump on W2
    # ------------------------------------------------------------------
    print("\n--- analog laser (ch0/W1) + pump (ch1/W2) ---")
    fake = install_fake()
    cfg = ControlSystemConfig(
        ad3=AD3Config(enabled=True, laser_backend='analog', laser_channel=0,
                      pump_channel=1, laser_frequency_hz=500.0,
                      laser_amplitude_v=2.5, laser_offset_v=2.5,
                      laser_duty_percent=10.0, pump_v_min=0.0, pump_v_max=5.0),
        supervisor=SupervisorConfig(startup_pump_voltage=1.0))

    dev, pump = hw.make_pump(cfg)
    check("the device opens and a real pump backend is returned",
          dev is not None and pump.name == 'analog_discovery')

    check("the device is opened exactly once",
          len(fake.of("FDwfDeviceOpen")) == 1)
    check("AutoConfigure is disabled before anything is configured",
          fake.of("FDwfDeviceAutoConfigureSet") == [(0x1234, 0)]
          and fake.index_of("FDwfDeviceAutoConfigureSet")
          < fake.index_of("FDwfAnalogOutNodeEnableSet"))
    check("no reset call is ever made",
          not any('Reset' in n for n in fake.names()),
          str([n for n in fake.names() if 'Reset' in n]))

    # laser configuration
    laser_freq = fake.of("FDwfAnalogOutNodeFrequencySet")
    laser_amp = fake.of("FDwfAnalogOutNodeAmplitudeSet")
    laser_sym = fake.of("FDwfAnalogOutNodeSymmetrySet")
    check("the laser is a square wave with the configured frequency and duty",
          laser_freq == [(0x1234, 0, 0, 500.0)] and laser_sym == [(0x1234, 0, 0, 10.0)]
          and (0x1234, 0, 0, 2.5) in laser_amp,
          f"freq={laser_freq} duty={laser_sym}")
    func_calls = fake.of("FDwfAnalogOutNodeFunctionSet")
    check("the laser channel is funcSquare and the pump channel is funcDC",
          (0x1234, 0, 0, 2) in func_calls and (0x1234, 1, 0, 0) in func_calls,
          str(func_calls))
    check("the laser runs indefinitely (RunSet 0, RepeatSet 0)",
          (0x1234, 0, 0.0) in fake.of("FDwfAnalogOutRunSet")
          and (0x1234, 0, 0) in fake.of("FDwfAnalogOutRepeatSet"))
    check("the pump amplitude is pinned at zero, so the voltage is the offset",
          (0x1234, 1, 0, 0.0) in laser_amp, str(laser_amp))

    # the crucial invariant
    n_laser_calls_at_start = sum(1 for n, a in fake.calls
                                 if n.startswith("FDwfAnalogOut") and len(a) > 1
                                 and a[1] == 0)
    check("every AnalogOut call names an explicit channel; -1 is never used",
          not any(n.startswith("FDwfAnalogOut") and len(a) > 1 and a[1] == -1
                  for n, a in fake.calls))

    # ------------------------------------------------------------------
    #  2. Pump updates must not touch the laser
    # ------------------------------------------------------------------
    print("\n--- 200 pump updates ---")
    mark = len(fake.calls)
    for i in range(200):
        pump.set_voltage(0.02 * i)          # 0 .. 3.98 V
    during = fake.calls[mark:]

    touched_laser = [c for c in during
                     if c[0].startswith("FDwfAnalogOut") and len(c[1]) > 1
                     and c[1][1] == 0]
    check("NO call addressed the laser channel during 200 pump updates",
          not touched_laser, str(touched_laser[:3]))
    check("no DigitalOut call was made either",
          not any(c[0].startswith("FDwfDigitalOut") for c in during))

    kinds = sorted({c[0] for c in during})
    check("a pump update consists only of NodeOffsetSet + AnalogOutConfigure",
          kinds == ["FDwfAnalogOutConfigure", "FDwfAnalogOutNodeOffsetSet"],
          str(kinds))
    check("every update is exactly two calls, in that order",
          len(during) == 400
          and all(during[2 * i][0] == "FDwfAnalogOutNodeOffsetSet"
                  and during[2 * i + 1][0] == "FDwfAnalogOutConfigure"
                  for i in range(200)))
    cfgs = [c[1] for c in during if c[0] == "FDwfAnalogOutConfigure"]
    check("every Configure targets the pump channel with fStart=3 (dynamic apply)",
          all(a == (0x1234, 1, AO_APPLY) for a in cfgs),
          str(sorted(set(cfgs))[:3]))
    offs = [c[1][3] for c in during if c[0] == "FDwfAnalogOutNodeOffsetSet"]
    check("the offsets sent are the voltages requested",
          offs[0] == 0.0 and abs(offs[-1] - 3.98) < 1e-12, f"{offs[0]} .. {offs[-1]}")

    # ------------------------------------------------------------------
    #  3. Limits
    # ------------------------------------------------------------------
    print("\n--- voltage limits ---")
    mark = len(fake.calls)
    check("a command above the maximum is clamped", pump.set_voltage(9.0) == 5.0)
    check("a command below the minimum is clamped", pump.set_voltage(-4.0) == 0.0)
    sent = [c[1][3] for c in fake.calls[mark:] if c[0] == "FDwfAnalogOutNodeOffsetSet"]
    check("the out-of-range values never reached the device",
          sent == [5.0, 0.0], str(sent))
    check("clamping is counted for the run summary", pump.n_clamped == 2)

    mark = len(fake.calls)
    for bad in (float('nan'), float('inf'), -float('inf')):
        try:
            pump.set_voltage(bad)
            refused = False
        except hw.AD3Error:
            refused = True
        if not refused:
            break
    after = fake.calls[mark:]
    check("a non-finite command is REFUSED and never reaches the DAC",
          refused and not after,
          f"{len(after)} calls made after nan/inf commands")

    # ------------------------------------------------------------------
    #  4. Supervisor -> pump, end to end
    # ------------------------------------------------------------------
    print("\n--- supervisor driving the real pump backend ---")
    from ebiv_control import ControlSupervisor, Measurement, ControlState
    from ebiv_config import PIDConfig, ReferenceConfig, FilterConfig
    cfg2 = ControlSystemConfig(
        ad3=cfg.ad3,
        pid=PIDConfig(kp=0.5, ki=0.8, v_min=0.0, v_max=5.0, slew_rate_v_per_s=2.0),
        reference=ReferenceConfig(kind='constant', value=3.0),
        filt=FilterConfig(kind='none'),
        supervisor=SupervisorConfig(control_rate_hz=10.0, hold_timeout_s=0.5,
                                    safe_timeout_s=2.0, safe_pump_voltage=0.8,
                                    startup_pump_voltage=1.0))
    sup = ControlSupervisor(cfg2, pump)
    mark = len(fake.calls)
    t = 0.0
    for i in range(5):
        sup.step(Measurement(2.0, True, 100, 100, t, t, i), now=t)
        t += 0.1
    sup.arm(now=t)
    for i in range(5, 40):
        sup.step(Measurement(2.0 + 0.02 * i, True, 100, 100, t, t, i), now=t)
        t += 0.1
    volts = [c[1][3] for c in fake.calls[mark:]
             if c[0] == "FDwfAnalogOutNodeOffsetSet"]
    check("the closed loop drove the real backend and moved the voltage",
          len(volts) >= 40 and max(volts) > min(volts),
          f"{len(volts)} commands, {min(volts):.3f}..{max(volts):.3f} V")
    check("every commanded voltage stayed inside the hardware limits",
          all(0.0 <= v <= 5.0 for v in volts))
    laser_during_loop = [c for c in fake.calls[mark:]
                         if c[0].startswith("FDwfAnalogOut") and len(c[1]) > 1
                         and c[1][1] == 0]
    check("the closed loop never addressed the laser channel",
          not laser_during_loop, str(laser_during_loop[:3]))

    sup.go_safe("test")
    check("go_safe sends the configured safe voltage to the device",
          fake.of("FDwfAnalogOutNodeOffsetSet")[-1][3] == 0.8,
          str(fake.of("FDwfAnalogOutNodeOffsetSet")[-1]))

    dev.close()
    check("close() closes the device exactly once",
          len(fake.of("FDwfDeviceClose")) == 1)

    # ------------------------------------------------------------------
    #  4b. THE LASER MUST START IN EVERY RUN MODE
    #
    #  Regression test for the Release 5.0 bug that made the whole hardware
    #  block conditional on `control_enabled = run_mode != 'ebiv'`: enabling
    #  the Analog Discovery and configuring the laser did nothing at all in
    #  plain measurement mode, which is the mode most runs use, and nothing in
    #  the log said so because nothing was attempted.
    # ------------------------------------------------------------------
    print("\n--- the laser starts in every run mode ---")
    for mode in ('ebiv', 'manual', 'calibration', 'closed_loop'):
        fk = install_fake()
        c = ControlSystemConfig(
            ad3=AD3Config(enabled=True, laser_backend='analog',
                          laser_channel=0, pump_channel=1,
                          laser_frequency_hz=200.0, laser_duty_percent=10.0))
        d = hw.open_device_and_start_laser(c)
        started = (0x1234, 0, AO_START) in fk.of("FDwfAnalogOutConfigure")
        freq_ok = fk.of("FDwfAnalogOutNodeFrequencySet") == [(0x1234, 0, 0, 200.0)]
        check(f"run_mode='{mode}': the laser is configured and started",
              d is not None and started and freq_ok)
        if d is not None:
            d.close()

    fk = install_fake()
    c = ControlSystemConfig(ad3=AD3Config(enabled=False))
    check("with the Analog Discovery disabled, no device call is made at all",
          hw.open_device_and_start_laser(c) is None and not fk.calls,
          f"{len(fk.calls)} calls")

    # attaching the pump to an already-open device must not re-open it
    fk = install_fake()
    c = ControlSystemConfig(ad3=AD3Config(enabled=True, laser_frequency_hz=200.0))
    d = hw.open_device_and_start_laser(c)
    n_open_before = len(fk.of("FDwfDeviceOpen"))
    d2, p2 = hw.make_pump(c, device=d)
    check("adding the pump reuses the open device instead of opening a second",
          d2 is d and len(fk.of("FDwfDeviceOpen")) == n_open_before == 1)
    check("adding the pump does not reconfigure the laser channel",
          len(fk.of("FDwfAnalogOutNodeFrequencySet")) == 1)
    d.close()

    # ------------------------------------------------------------------
    #  5. DIO pattern laser
    # ------------------------------------------------------------------
    print("\n--- pattern (DIO) laser + pump on ch0 ---")
    fake2 = install_fake()
    cfgp = ControlSystemConfig(
        ad3=AD3Config(enabled=True, laser_backend='pattern', laser_dio_channel=2,
                      pump_channel=0, laser_frequency_hz=500.0,
                      laser_duty_percent=20.0))
    dev2, pump2 = hw.make_pump(cfgp)
    check("the pattern backend configures DigitalOut",
          any(n == "FDwfDigitalOutCounterSet" for n in fake2.names()))
    counter = fake2.of("FDwfDigitalOutCounterSet")[0]
    low, high = counter[2], counter[3]
    f_actual = 100e6 / 1 / (low + high)
    duty = 100.0 * high / (low + high)
    check("the DIO counters give the requested frequency and duty",
          abs(f_actual - 500.0) < 0.5 and abs(duty - 20.0) < 0.5,
          f"{f_actual:.2f} Hz, {duty:.2f}%")
    check("the pattern laser uses NO AnalogOut channel for the laser",
          not any(n == "FDwfAnalogOutNodeFunctionSet" and a[1] == 2
                  for n, a in fake2.calls))

    mark = len(fake2.calls)
    for i in range(20):
        pump2.set_voltage(0.1 * i)
    during2 = fake2.calls[mark:]
    check("pump updates on ch0 make no DigitalOut call at all",
          not any(c[0].startswith("FDwfDigitalOut") for c in during2))
    check("pump updates address channel 0 with fStart=3",
          all(c[1][:2] == (0x1234, 0) for c in during2
              if c[0] == "FDwfAnalogOutConfigure")
          and all(c[1][2] == AO_APPLY for c in during2
                  if c[0] == "FDwfAnalogOutConfigure"))
    dev2.close()

    # ------------------------------------------------------------------
    #  6. Failure handling
    # ------------------------------------------------------------------
    print("\n--- failure handling ---")
    fake3 = install_fake(returns={"FDwfAnalogOutNodeFrequencySet": 0})
    raised = False
    try:
        d = hw.AnalogDiscovery3(cfgp.ad3.__class__(enabled=True))
        d.start_laser()
    except hw.AD3Error:
        raised = True
    check("a failing SDK call raises AD3Error instead of continuing silently",
          raised)

    fake4 = install_fake(n_devices=0)
    raised = False
    try:
        hw.AnalogDiscovery3(AD3Config(enabled=True, laser_frequency_hz=200.0))
    except hw.AD3Error as exc:
        raised = "WaveForms" in str(exc)
    check("no device present gives an AD3Error that names the usual cause",
          raised)

    fake5 = install_fake()
    dev5, pump5 = hw.make_pump(ControlSystemConfig(
        ad3=AD3Config(enabled=True, laser_frequency_hz=200.0)))
    fake5.returns["FDwfAnalogOutConfigure"] = 0
    raised = False
    try:
        pump5.set_voltage(2.0)
    except hw.AD3Error:
        raised = True
    check("a failed apply raises rather than reporting a voltage that never went out",
          raised and pump5.n_failures == 1)
    fake5.returns.pop("FDwfAnalogOutConfigure")
    dev5.close()

    # a missing SDK symbol must not escape as AttributeError
    fake6 = install_fake(missing={"FDwfAnalogOutNodeSymmetrySet"})
    dev6, pump6 = hw.make_pump(ControlSystemConfig(ad3=AD3Config(enabled=True, laser_frequency_hz=200.0)))
    check("a WaveForms build missing an entry point falls back to the mock "
          "instead of crashing",
          pump6.name == 'mock', f"got {pump6.name}")

    # ------------------------------------------------------------------
    #  6b. Shutdown must leave the laser OFF        (Release 5.2 safety fix)
    #
    #  Observed on the bench: closing the device sent the laser to FULL
    #  POWER.  Cause: laser_idle_low selected DwfAnalogOutIdleInitial, which
    #  parks the pin at the waveform's FIRST SAMPLE, and a WaveForms square
    #  starts HIGH -> offset + amplitude = 5 V, held continuously.
    # ------------------------------------------------------------------
    print("\n--- shutdown leaves the laser off ---")
    fake = install_fake()
    cfg = ControlSystemConfig(ad3=AD3Config(
        enabled=True, laser_backend='analog', laser_channel=0, pump_channel=1,
        laser_frequency_hz=200.0,
        laser_amplitude_v=2.5, laser_offset_v=2.5, laser_idle_low=True))
    dev = hw.open_device_and_start_laser(cfg)

    idle = fake.of("FDwfAnalogOutIdleSet")
    check("laser_idle_low selects IdleDisable, not IdleInitial",
          any(a[1] == 0 and a[2] == 0 for a in idle),
          f"IdleSet calls {idle}; Initial(2) parks the pin at offset+amplitude")

    mark = len(fake.calls)
    dev.close()
    shutdown = fake.calls[mark:]

    def _seq(name):
        return [a for n, a in shutdown if n == name]

    fn = _seq("FDwfAnalogOutNodeFunctionSet")
    amp = _seq("FDwfAnalogOutNodeAmplitudeSet")
    off = _seq("FDwfAnalogOutNodeOffsetSet")
    cfgs = _seq("FDwfAnalogOutConfigure")

    check("shutdown drives the laser channel to DC",
          any(a[1] == 0 and a[3] == 0 for a in fn),
          "funcDC=0 on the laser channel")
    check("shutdown zeroes the laser amplitude and offset",
          any(a[1] == 0 and a[3] == 0.0 for a in amp)
          and any(a[1] == 0 and a[3] == 0.0 for a in off),
          "a stopped channel must not sit at 2.5 V")
    check("shutdown APPLIES the 0 V before stopping",
          [a[2] for a in cfgs if a[1] == 0] == [3, 0],
          f"laser-channel Configure sequence {[a[2] for a in cfgs if a[1] == 0]}; "
          f"expected APPLY(3) then STOP(0)")
    check("the pump channel is parked too",
          any(a[1] == 1 and a[3] == 0.0 for a in off),
          "both outputs go to 0 V on the way out")
    check("the device is closed last",
          [n for n, _ in shutdown][-1] == "FDwfDeviceClose",
          str([n for n, _ in shutdown][-3:]))

    # A parking failure must not stop the device being closed, and must be
    # loud rather than silent.
    fake2 = install_fake()
    dev2 = hw.open_device_and_start_laser(ControlSystemConfig(ad3=AD3Config(
        enabled=True, laser_backend='analog', laser_channel=0, pump_channel=1,
        laser_frequency_hz=200.0)))
    fake2.missing.add("FDwfAnalogOutNodeOffsetSet")
    try:
        dev2.close()
        closed_anyway = "FDwfDeviceClose" in fake2.names()
    except Exception:
        closed_anyway = False
    check("a failure while parking still closes the device",
          closed_anyway,
          "an exception here would leak the USB handle AND leave the laser on")

    # ------------------------------------------------------------------
    #  7. Pulse metrics used by the scope verification
    # ------------------------------------------------------------------
    print("\n--- verification-mode pulse metrics ---")
    import numpy as np
    fs = 2e6
    for f0, duty in ((500.0, 0.10), (1000.0, 0.50), (200.0, 0.02),
                     (500.0, 0.01), (250.0, 0.05)):
        t = np.arange(int(0.06 * fs)) / fs
        sq = ((t * f0) % 1.0 < duty).astype(float) * 5.0
        m = hw._pulse_metrics(sq, fs)
        ok = (m is not None and abs(m['f_hz'] - f0) < max(1.0, 0.002 * f0)
              and abs(m['duty_pct'] - 100 * duty) < 0.6)
        check(f"metrics recover {f0:.0f} Hz at {100*duty:.0f}% duty", ok,
              f"{m['f_hz']:.2f} Hz, {m['duty_pct']:.2f}%, "
              f"jitter {m['period_std_us']:.4f} us" if m else "no square wave seen")
    check("a flat trace is reported as 'no square wave' rather than nonsense",
          hw._pulse_metrics(np.zeros(1000), fs) is None)
    # noise and overshoot must not move the recovered levels
    rng = np.random.default_rng(0)
    t = np.arange(int(0.06 * fs)) / fs
    sq = ((t * 500.0) % 1.0 < 0.10).astype(float) * 5.0
    noisy = sq + rng.normal(0, 0.05, sq.size)
    noisy[np.nonzero(np.diff(sq) > 0)[0] + 1] += 1.2      # ringing on the edges
    m = hw._pulse_metrics(noisy, fs)
    check("noise and edge overshoot do not break the level detection",
          m is not None and abs(m['f_hz'] - 500.0) < 2.0
          and abs(m['duty_pct'] - 10.0) < 1.0,
          f"{m['f_hz']:.2f} Hz, {m['duty_pct']:.2f}%" if m else "not detected")

    hw._dwf = None
    hw._dwf_load_error = None

    print(f"\n{'=' * 76}\n  {len(_PASS)} passed, {len(_FAIL)} failed")
    for n in _FAIL:
        print(f"    - {n}")
    print("=" * 76)
    return 1 if _FAIL else 0


if __name__ == "__main__":
    sys.exit(run())
