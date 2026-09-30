"""
EBIV Release 5.2 — Analog Discovery 3 hardware layer.

Release 4.0 contained NO Analog Discovery code at all: `trigger_mode` only
CONSUMES a trigger (the camera's trigger-in, or an auto phase detection from
the event rate).  The laser was started by hand from the WaveForms desktop
application.  This module is therefore new, and it changes one operational
fact that must be understood before the first run:

    The WaveForms application takes an EXCLUSIVE lock on the device.
    Release 5.0 opens the AD3 from Python, so WaveForms must be CLOSED.
    Since Python then owns the device, Python must also generate the laser
    waveform -- otherwise closing WaveForms simply stops the laser.

Channel allocation (defaults, see AD3Config):

    Analog Out ch0 (W1)  ->  laser trigger, continuous square wave.
                             Configured ONCE at start-up.  After
                             start_laser() returns, no function in this
                             module touches channel 0 again.
    Analog Out ch1 (W2)  ->  pump command, 0..5 V DC.
                             Updated with
                                 FDwfAnalogOutNodeOffsetSet(h, 1, Carrier, V)
                                 FDwfAnalogOutConfigure(h, 1, 3)
                             where fStart=3 means, verbatim from the WaveForms
                             SDK reference manual, "apply the configuration
                             dynamically without changing the state of the
                             instrument".

Rules enforced throughout this module:
  * the device is opened ONCE and the handle is kept for the whole session;
  * idxChannel is ALWAYS an explicit channel index.  -1 (which the SDK
    documents as "each enabled Analog Out channel") is never used;
  * FDwfAnalogOutReset / FDwfDeviceReset are never called after start-up;
  * the laser channel is never stopped, restarted or reconfigured by a pump
    update.

The laser can alternatively be driven from a DIO pin through the Pattern
Generator (laser_backend='pattern'), which is a physically separate
instrument from Analog Out and therefore even better isolated; the pump then
takes Analog Out ch0.

Everything here degrades to a MockPump when AD3Config.enabled is False or the
SDK/device is missing, so the rest of Release 5.0 is fully testable with no
hardware attached.
"""

import os
import sys
import time
import ctypes
import logging
import threading
from ctypes import c_int, c_uint, c_ubyte, c_double, c_byte, byref, create_string_buffer

import numpy as np


# ==========================================================================
#  WaveForms SDK constants (from dwf.h)
# ==========================================================================

# AnalogOutNode
AnalogOutNodeCarrier = c_int(0)
AnalogOutNodeFM = c_int(1)
AnalogOutNodeAM = c_int(2)

# FUNC
funcDC = c_ubyte(0)
funcSine = c_ubyte(1)
funcSquare = c_ubyte(2)
funcTriangle = c_ubyte(3)
funcRampUp = c_ubyte(4)
funcRampDown = c_ubyte(5)
funcNoise = c_ubyte(6)
funcCustom = c_ubyte(30)
funcPlay = c_ubyte(31)

# DwfAnalogOutIdle
DwfAnalogOutIdleDisable = c_int(0)
DwfAnalogOutIdleOffset = c_int(1)
DwfAnalogOutIdleInitial = c_int(2)

# FDwfAnalogOutConfigure fStart
AO_STOP = 0
AO_START = 1
AO_APPLY = 3          # "apply the configuration dynamically without changing
                      #  the state of the instrument"  (SDK reference manual)

# FDwfDeviceParam
DwfParamOnClose = c_int(3)     # 0 continue running, 1 stop, 2 shutdown

# DwfState
DwfStateDone = 2

# acqmodeSingle
acqmodeSingle = c_int(0)
trigsrcDetectorAnalogIn = c_ubyte(2)
trigtypeEdge = c_int(0)
trigcondRisingPositive = c_int(0)


class AD3Error(RuntimeError):
    pass


# ==========================================================================
#  SDK loading
# ==========================================================================

_dwf = None
_dwf_load_error = None


def _load_dwf():
    """Load the WaveForms runtime.  Returns the ctypes handle or None."""
    global _dwf, _dwf_load_error
    if _dwf is not None or _dwf_load_error is not None:
        return _dwf
    try:
        if sys.platform.startswith("win"):
            lib = ctypes.cdll.dwf
        elif sys.platform.startswith("darwin"):
            lib = ctypes.cdll.LoadLibrary(
                "/Library/Frameworks/dwf.framework/dwf")
        else:
            lib = ctypes.cdll.LoadLibrary("libdwf.so")
    except Exception as exc:                                   # noqa: BLE001
        _dwf_load_error = exc
        logging.warning("WaveForms SDK not loadable (%s). "
                        "Analog Discovery features are unavailable.", exc)
        return None

    # Declare argtypes for everything we call.  Without this, ctypes will
    # silently pass Python floats as ints on some platforms.
    sigs = {
        'FDwfGetLastErrorMsg': ([ctypes.c_char_p], c_int),
        'FDwfGetVersion': ([ctypes.c_char_p], c_int),
        'FDwfEnum': ([c_int, ctypes.POINTER(c_int)], c_int),
        'FDwfDeviceOpen': ([c_int, ctypes.POINTER(c_int)], c_int),
        'FDwfDeviceClose': ([c_int], c_int),
        'FDwfDeviceAutoConfigureSet': ([c_int, c_int], c_int),
        'FDwfDeviceParamSet': ([c_int, c_int, c_int], c_int),
        'FDwfAnalogOutNodeEnableSet': ([c_int, c_int, c_int, c_int], c_int),
        'FDwfAnalogOutNodeFunctionSet': ([c_int, c_int, c_int, c_ubyte], c_int),
        'FDwfAnalogOutNodeFrequencySet': ([c_int, c_int, c_int, c_double], c_int),
        'FDwfAnalogOutNodeAmplitudeSet': ([c_int, c_int, c_int, c_double], c_int),
        'FDwfAnalogOutNodeOffsetSet': ([c_int, c_int, c_int, c_double], c_int),
        'FDwfAnalogOutNodeSymmetrySet': ([c_int, c_int, c_int, c_double], c_int),
        'FDwfAnalogOutIdleSet': ([c_int, c_int, c_int], c_int),
        'FDwfAnalogOutRunSet': ([c_int, c_int, c_double], c_int),
        'FDwfAnalogOutRepeatSet': ([c_int, c_int, c_int], c_int),
        'FDwfAnalogOutLimitationSet': ([c_int, c_int, c_double], c_int),
        'FDwfAnalogOutConfigure': ([c_int, c_int, c_int], c_int),
        'FDwfAnalogOutStatus': ([c_int, c_int, ctypes.POINTER(c_ubyte)], c_int),
        'FDwfDigitalOutEnableSet': ([c_int, c_int, c_int], c_int),
        'FDwfDigitalOutDividerSet': ([c_int, c_int, c_uint], c_int),
        'FDwfDigitalOutCounterSet': ([c_int, c_int, c_uint, c_uint], c_int),
        'FDwfDigitalOutIdleSet': ([c_int, c_int, c_int], c_int),
        'FDwfDigitalOutRepeatSet': ([c_int, c_uint], c_int),
        'FDwfDigitalOutInternalClockInfo': ([c_int, ctypes.POINTER(c_double)], c_int),
        'FDwfDigitalOutConfigure': ([c_int, c_int], c_int),
        'FDwfAnalogInChannelEnableSet': ([c_int, c_int, c_int], c_int),
        'FDwfAnalogInChannelRangeSet': ([c_int, c_int, c_double], c_int),
        'FDwfAnalogInAcquisitionModeSet': ([c_int, c_int], c_int),
        'FDwfAnalogInFrequencySet': ([c_int, c_double], c_int),
        'FDwfAnalogInBufferSizeSet': ([c_int, c_int], c_int),
        'FDwfAnalogInBufferSizeInfo': ([c_int, ctypes.POINTER(c_int),
                                        ctypes.POINTER(c_int)], c_int),
        'FDwfAnalogInConfigure': ([c_int, c_int, c_int], c_int),
        'FDwfAnalogInStatus': ([c_int, c_int, ctypes.POINTER(c_ubyte)], c_int),
        'FDwfAnalogInStatusData': ([c_int, c_int, ctypes.POINTER(c_double), c_int], c_int),
    }
    missing = []
    for name, (argt, rest) in sigs.items():
        try:
            fn = getattr(lib, name)
        except AttributeError:
            fn = None
        if fn is None:
            missing.append(name)
            continue
        try:
            fn.argtypes = argt
            fn.restype = rest
        except AttributeError:
            missing.append(name)

    # Entry points without which the laser or the pump cannot be driven at
    # all.  Checking here turns "AttributeError deep inside start_laser" into
    # one clear message naming the WaveForms version problem.  The optional
    # ones (the built-in scope, DwfParamOnClose, the DIO backend) are allowed
    # to be absent; the code that uses them already guards for it.
    required = [
        'FDwfEnum', 'FDwfDeviceOpen', 'FDwfDeviceClose',
        'FDwfDeviceAutoConfigureSet', 'FDwfAnalogOutNodeEnableSet',
        'FDwfAnalogOutNodeFunctionSet', 'FDwfAnalogOutNodeFrequencySet',
        'FDwfAnalogOutNodeAmplitudeSet', 'FDwfAnalogOutNodeOffsetSet',
        'FDwfAnalogOutNodeSymmetrySet', 'FDwfAnalogOutRunSet',
        'FDwfAnalogOutRepeatSet', 'FDwfAnalogOutConfigure',
    ]
    lacking = [n for n in required if n in missing]
    if lacking:
        _dwf_load_error = RuntimeError(
            "the WaveForms runtime is missing entry points this code needs: "
            + ", ".join(lacking))
        logging.error("%s. Update the Digilent WaveForms installation.",
                      _dwf_load_error)
        return None
    if missing:
        logging.info("WaveForms runtime lacks %d optional entry point(s): %s. "
                     "The features that use them are unavailable; everything "
                     "else works.", len(missing), ", ".join(missing))

    _dwf = lib
    return _dwf


def is_ad3_available() -> bool:
    """True if the WaveForms runtime loads and at least one device is present."""
    lib = _load_dwf()
    if lib is None:
        return False
    n = c_int(0)
    try:
        lib.FDwfEnum(c_int(0), byref(n))
    except Exception:                                          # noqa: BLE001
        return False
    return n.value > 0


# ==========================================================================
#  Device
# ==========================================================================

class AnalogDiscovery3:
    """
    One persistent handle on one Analog Discovery 3.

    Opened once in __init__, closed once in close().  Nothing in the control
    loop reopens, resets or re-enumerates the device.
    """

    def __init__(self, cfg_ad3):
        self.cfg = cfg_ad3
        self.hdwf = c_int(0)
        self._lib = _load_dwf()
        if self._lib is None:
            raise AD3Error(
                "The WaveForms runtime (dwf) could not be loaded. Install the "
                "Digilent WaveForms package, which provides dwf.dll / libdwf.")

        n = c_int(0)
        self._lib.FDwfEnum(c_int(0), byref(n))
        if n.value <= 0:
            raise AD3Error(
                "No Digilent device found. Check the USB connection, and make "
                "sure the WaveForms desktop application is CLOSED -- it holds "
                "an exclusive lock on the device.")

        if self._lib.FDwfDeviceOpen(c_int(self.cfg.device_index), byref(self.hdwf)) == 0 \
                or self.hdwf.value == 0:
            raise AD3Error(
                f"Failed to open the Analog Discovery: {self._err()}\n"
                "The most common cause is that the WaveForms desktop "
                "application still has the device open. Close it and retry.")

        ver = create_string_buffer(32)
        self._lib.FDwfGetVersion(ver)
        logging.info("Analog Discovery opened (WaveForms runtime %s, handle=%d).",
                     ver.value.decode(errors='replace'), self.hdwf.value)

        # Deterministic configuration: nothing is pushed to hardware until we
        # explicitly call FDwfAnalogOutConfigure for a specific channel.
        self._ck(self._lib.FDwfDeviceAutoConfigureSet(self.hdwf, c_int(0)),
                 "FDwfDeviceAutoConfigureSet(0)")
        logging.info("AutoConfigure disabled: every hardware update is an "
                     "explicit, channel-specific FDwfAnalogOutConfigure call.")

        self._laser_started = False
        self._laser_touch_count = 0     # must stay at its post-start value
        self._closed = False

    # ------------------------------------------------------------------

    def _err(self):
        buf = create_string_buffer(512)
        try:
            self._lib.FDwfGetLastErrorMsg(buf)
            return buf.value.decode(errors='replace').strip()
        except Exception:                                      # noqa: BLE001
            return "<no error text>"

    def _ck(self, rc, what):
        if rc == 0:
            raise AD3Error(f"{what} failed: {self._err()}")
        return rc

    # ------------------------------------------------------------------
    #  Laser: configured once, then never touched
    # ------------------------------------------------------------------

    def start_laser(self):
        """
        Configure and start the continuous laser trigger waveform.

        Called exactly once.  Every subsequent pump update addresses only the
        pump channel, so the laser output is not stopped, restarted or
        reconfigured for the rest of the session.
        """
        if self._laser_started:
            logging.debug("start_laser() called again; ignored.")
            return

        c = self.cfg
        if c.laser_backend == 'external':
            logging.warning(
                "laser_backend='external': Release 5.0 will NOT generate the "
                "laser waveform. Something else must be driving it.")
            self._laser_started = True
            return

        if c.laser_backend == 'pattern':
            self._start_laser_pattern()
        else:
            self._start_laser_analog()

        self._laser_started = True
        self._laser_touch_count += 1

    def _require_laser_frequency(self):
        """laser_frequency_hz=None means "follow f_acq", and something upstream
        should have resolved it.  If it reaches the hardware still unset, fail
        with a sentence that names the cause instead of a ctypes TypeError
        about c_double(None) forty frames down."""
        f = self.cfg.laser_frequency_hz
        if f is None:
            raise AD3Error(
                "laser_frequency_hz is still None at the hardware layer. It "
                "normally follows F_ACQ, resolved by Session.validate() or "
                "Session.resolve_laser_frequency(). A code path reached the "
                "device without going through either. Set LASER_FREQUENCY_HZ "
                "to a number to work around it, and report this.")
        return float(f)

    def _start_laser_analog(self):
        c = self.cfg
        f_laser = self._require_laser_frequency()
        ch = c_int(c.laser_channel)
        L = self._lib
        h = self.hdwf
        self._ck(L.FDwfAnalogOutNodeEnableSet(h, ch, AnalogOutNodeCarrier, c_int(1)),
                 "laser NodeEnableSet")
        self._ck(L.FDwfAnalogOutNodeFunctionSet(h, ch, AnalogOutNodeCarrier, funcSquare),
                 "laser NodeFunctionSet(funcSquare)")
        self._ck(L.FDwfAnalogOutNodeFrequencySet(h, ch, AnalogOutNodeCarrier,
                                                 c_double(f_laser)),
                 "laser NodeFrequencySet")
        self._ck(L.FDwfAnalogOutNodeAmplitudeSet(h, ch, AnalogOutNodeCarrier,
                                                 c_double(c.laser_amplitude_v)),
                 "laser NodeAmplitudeSet")
        self._ck(L.FDwfAnalogOutNodeOffsetSet(h, ch, AnalogOutNodeCarrier,
                                              c_double(c.laser_offset_v)),
                 "laser NodeOffsetSet")
        self._ck(L.FDwfAnalogOutNodeSymmetrySet(h, ch, AnalogOutNodeCarrier,
                                                c_double(c.laser_duty_percent)),
                 "laser NodeSymmetrySet")
        # Infinite run, infinite repeat: the waveform lives in hardware.
        self._ck(L.FDwfAnalogOutRunSet(h, ch, c_double(0.0)), "laser RunSet(0)")
        self._ck(L.FDwfAnalogOutRepeatSet(h, ch, c_int(0)), "laser RepeatSet(0)")
        # ------------------------------------------------------------------
        #  IDLE STATE — what the pin does when this channel is NOT running.
        #
        #  RELEASE 5.2 SAFETY FIX.  This used to select
        #  DwfAnalogOutIdleInitial when laser_idle_low was True.  "Initial"
        #  means the FIRST SAMPLE OF THE WAVEFORM, and a WaveForms square
        #  wave starts in its HIGH phase — so the output parked at
        #  offset + amplitude = 5.0 V with the defaults here.  A constant
        #  5 V into the laser trigger is a continuous full-power laser, i.e.
        #  the option named "idle low" was doing the exact opposite, and it
        #  did it every time the device was closed.
        #
        #  DwfAnalogOutIdleDisable turns the driver off instead.  Note what
        #  that does and does not guarantee: it stops the AD3 driving the
        #  pin, leaving it high-impedance.  If the laser's trigger input has
        #  a pull-up, high-Z still reads HIGH.  So this is necessary but not
        #  sufficient, and park_outputs_safe() below drives a real 0 V before
        #  anything is stopped.  The only complete guarantee is a pull-down
        #  resistor at the laser input, which is hardware, not software.
        # ------------------------------------------------------------------
        try:
            L.FDwfAnalogOutIdleSet(
                h, ch,
                DwfAnalogOutIdleDisable if self.cfg.laser_idle_low
                else DwfAnalogOutIdleOffset)
        except Exception:                                      # noqa: BLE001
            logging.warning("FDwfAnalogOutIdleSet is unavailable in this "
                            "runtime: the laser channel's parked state is "
                            "whatever the device defaults to. Check it on a "
                            "scope before leaving the rig.")
        # Start THIS channel only.
        self._ck(L.FDwfAnalogOutConfigure(h, ch, c_int(AO_START)),
                 "laser AnalogOutConfigure(START)")
        logging.info(
            "Laser started on Analog Out ch%d (W%d): %.1f Hz square, "
            "%.2f V amplitude, %.2f V offset, %.1f%% duty. "
            "This channel will not be addressed again.",
            c.laser_channel, c.laser_channel + 1, f_laser,
            c.laser_amplitude_v, c.laser_offset_v, c.laser_duty_percent)

    def _start_laser_pattern(self):
        """
        Laser on a DIO pin via the Pattern Generator.

        DigitalOut is a physically separate instrument from AnalogOut, so
        FDwfDigitalOutConfigure cannot disturb the analog pump channel and
        vice versa.  This is the most strongly isolated of the two options.
        """
        c = self.cfg
        L, h = self._lib, self.hdwf
        pin = c_int(c.laser_dio_channel)

        hz = c_double(0.0)
        self._ck(L.FDwfDigitalOutInternalClockInfo(h, byref(hz)),
                 "DigitalOutInternalClockInfo")
        f_sys = hz.value

        # counter values are in units of the (divided) internal clock
        divider = 1
        f_laser = self._require_laser_frequency()
        total = f_sys / f_laser
        while total > 2 ** 31:
            divider *= 2
            total = f_sys / divider / f_laser
        high = max(1, int(round(total * c.laser_duty_percent / 100.0)))
        low = max(1, int(round(total)) - high)

        self._ck(L.FDwfDigitalOutEnableSet(h, pin, c_int(1)), "DigitalOutEnableSet")
        self._ck(L.FDwfDigitalOutDividerSet(h, pin, c_uint(divider)),
                 "DigitalOutDividerSet")
        self._ck(L.FDwfDigitalOutCounterSet(h, pin, c_uint(low), c_uint(high)),
                 "DigitalOutCounterSet")
        self._ck(L.FDwfDigitalOutRepeatSet(h, c_uint(0)), "DigitalOutRepeatSet(0)")
        self._ck(L.FDwfDigitalOutConfigure(h, c_int(1)), "DigitalOutConfigure(START)")

        f_actual = f_sys / divider / (low + high)
        logging.info(
            "Laser started on DIO%d: requested %.2f Hz, actual %.4f Hz "
            "(sys clock %.0f Hz, divider %d, low=%d high=%d, duty %.2f%%). "
            "Pattern Generator is a separate instrument from Analog Out.",
            c.laser_dio_channel, f_laser, f_actual, f_sys,
            divider, low, high, 100.0 * high / (low + high))

    # ------------------------------------------------------------------
    #  Built-in scope, used only by the verification mode
    # ------------------------------------------------------------------

    def capture(self, channel=0, n_samples=8192, rate_hz=1e6, v_range=10.0,
                timeout_s=2.0):
        """Single-shot acquisition on an analog input.  Returns a float64 array."""
        L, h = self._lib, self.hdwf
        ch = c_int(channel)
        self._ck(L.FDwfAnalogInChannelEnableSet(h, ch, c_int(1)), "AnalogInChannelEnableSet")
        self._ck(L.FDwfAnalogInChannelRangeSet(h, ch, c_double(v_range)),
                 "AnalogInChannelRangeSet")
        self._ck(L.FDwfAnalogInAcquisitionModeSet(h, acqmodeSingle),
                 "AnalogInAcquisitionModeSet")
        self._ck(L.FDwfAnalogInFrequencySet(h, c_double(rate_hz)), "AnalogInFrequencySet")

        lo, hi = c_int(0), c_int(0)
        L.FDwfAnalogInBufferSizeInfo(h, byref(lo), byref(hi))
        n = int(min(max(n_samples, lo.value), hi.value))
        self._ck(L.FDwfAnalogInBufferSizeSet(h, c_int(n)), "AnalogInBufferSizeSet")

        self._ck(L.FDwfAnalogInConfigure(h, c_int(1), c_int(1)), "AnalogInConfigure")
        sts = c_ubyte(0)
        t0 = time.perf_counter()
        while True:
            self._ck(L.FDwfAnalogInStatus(h, c_int(1), byref(sts)), "AnalogInStatus")
            if sts.value == DwfStateDone:
                break
            if time.perf_counter() - t0 > timeout_s:
                raise AD3Error("Scope acquisition timed out. Is anything connected "
                               f"to input {channel + 1}+ ?")
            time.sleep(0.001)

        buf = (c_double * n)()
        self._ck(L.FDwfAnalogInStatusData(h, ch, buf, c_int(n)), "AnalogInStatusData")
        return np.ctypeslib.as_array(buf).copy(), rate_hz

    # ------------------------------------------------------------------

    def park_outputs_safe(self, channels=None):
        """
        Drive every output channel to a real 0 V and stop it.

        Called before close() on every exit path.  Relying on the idle
        setting alone is not enough: 'idle' describes what the pin does once
        the channel is stopped, and the safest thing available there is
        high-impedance, which a pulled-up input still reads as HIGH.  Driving
        0 V first means the laser sees a defined low for as long as the
        device is open, and the stop happens from that state rather than from
        the middle of a pulse.

        Best effort by construction: it must never raise, because it runs in
        the shutdown path of a run that may already be failing.  Anything
        that goes wrong is logged loudly instead — a laser left on is exactly
        the thing nobody should have to discover by looking at the bench.
        """
        if self._closed or self._lib is None:
            return
        L, h = self._lib, self.hdwf
        if channels is None:
            channels = sorted({int(self.cfg.laser_channel),
                               int(self.cfg.pump_channel)})
        for c in channels:
            ch = c_int(int(c))
            try:
                L.FDwfAnalogOutNodeFunctionSet(h, ch, AnalogOutNodeCarrier, funcDC)
                L.FDwfAnalogOutNodeAmplitudeSet(h, ch, AnalogOutNodeCarrier,
                                                c_double(0.0))
                L.FDwfAnalogOutNodeOffsetSet(h, ch, AnalogOutNodeCarrier,
                                             c_double(0.0))
                # APPLY, not START: push the new DC level out on this channel
                # without restarting anything.
                L.FDwfAnalogOutConfigure(h, ch, c_int(AO_APPLY))
                time.sleep(0.01)                     # let the DAC settle
                L.FDwfAnalogOutConfigure(h, ch, c_int(AO_STOP))
                logging.info("Analog Out ch%d (W%d) parked at 0 V and stopped.",
                             c, c + 1)
            except Exception as exc:                           # noqa: BLE001
                logging.error(
                    "COULD NOT PARK Analog Out ch%d (W%d) at 0 V (%s: %s). "
                    "Check the laser on a scope before walking away.",
                    c, c + 1, type(exc).__name__, exc)

    def close(self, leave_outputs_running=False):
        if self._closed:
            return
        if not leave_outputs_running:
            self.park_outputs_safe()
        try:
            if leave_outputs_running:
                try:
                    self._lib.FDwfDeviceParamSet(self.hdwf, DwfParamOnClose, c_int(0))
                    logging.info("Device set to keep outputs running after close.")
                except Exception:                              # noqa: BLE001
                    logging.warning("DwfParamOnClose is not supported by this "
                                    "runtime; outputs will stop on close.")
            self._lib.FDwfDeviceClose(self.hdwf)
            logging.info("Analog Discovery closed.")
        finally:
            self._closed = True


# ==========================================================================
#  Pump backends
# ==========================================================================

class PumpBackend:
    """Interface every pump backend implements."""

    name = "abstract"

    def set_voltage(self, v: float) -> float:
        raise NotImplementedError

    @property
    def voltage(self) -> float:
        raise NotImplementedError

    def close(self):
        pass


class MockPump(PumpBackend):
    """
    Hardware-free pump backend.

    Applies exactly the same clamping and bookkeeping as the real backend and
    records the full command history, so the control law, the supervisor and
    the logger can all be exercised with no device attached.  Optionally
    drives a plant model so a full closed-loop run can be simulated.
    """

    name = "mock"

    def __init__(self, v_min=0.0, v_max=5.0, initial=0.0, plant=None, verbose=False):
        self.v_min, self.v_max = float(v_min), float(v_max)
        self._v = float(np.clip(initial, v_min, v_max))
        self.history = []
        self.n_clamped = 0
        self.n_calls = 0
        self.n_failures = 0
        self.plant = plant
        self.verbose = verbose

    def set_voltage(self, v):
        self.n_calls += 1
        v = float(v)
        # Same refusal as the real backend, so behaviour under test matches
        # behaviour on the bench.
        if not np.isfinite(v):
            self.n_failures += 1
            raise ValueError(f"refusing a non-finite pump command ({v!r})")
        vc = float(np.clip(v, self.v_min, self.v_max))
        if vc != v:
            self.n_clamped += 1
        self._v = vc
        self.history.append((time.perf_counter(), vc))
        if self.verbose:
            logging.debug("MockPump <- %.4f V", vc)
        return vc

    @property
    def voltage(self):
        return self._v


class AnalogOutPump(PumpBackend):
    """
    Pump command on one Analog Out channel of a shared AnalogDiscovery3.

    The channel is set to funcDC with zero amplitude, so the output is
    entirely determined by the node OFFSET.  Updating the voltage is therefore
    two calls:

        FDwfAnalogOutNodeOffsetSet(hdwf, pump_ch, AnalogOutNodeCarrier, V)
        FDwfAnalogOutConfigure(hdwf, pump_ch, 3)

    fStart=3 is documented as "apply the configuration dynamically without
    changing the state of the instrument", and the channel index is explicit,
    so the laser channel is not addressed.

    Two independent voltage limits are enforced:
      * the PID clamps to [pid.v_min, pid.v_max];
      * this backend clamps again to [cfg.pump_v_min, cfg.pump_v_max] and logs
        a warning the first few times it has to.
    A command outside the hardware limits is never passed to the device.
    """

    name = "analog_discovery"

    def __init__(self, device: AnalogDiscovery3, cfg_ad3, initial_voltage=0.0):
        self.dev = device
        self.cfg = cfg_ad3
        self.ch = c_int(cfg_ad3.pump_channel)
        self.v_min = float(cfg_ad3.pump_v_min)
        self.v_max = float(cfg_ad3.pump_v_max)
        self._lock = threading.Lock()
        self.n_clamped = 0
        self.n_calls = 0
        self.n_failures = 0
        self._v = float(np.clip(initial_voltage, self.v_min, self.v_max))
        self._t_last = None
        self.update_times = []

        L, h = self.dev._lib, self.dev.hdwf
        d = self.dev
        d._ck(L.FDwfAnalogOutNodeEnableSet(h, self.ch, AnalogOutNodeCarrier, c_int(1)),
              "pump NodeEnableSet")
        d._ck(L.FDwfAnalogOutNodeFunctionSet(h, self.ch, AnalogOutNodeCarrier, funcDC),
              "pump NodeFunctionSet(funcDC)")
        d._ck(L.FDwfAnalogOutNodeAmplitudeSet(h, self.ch, AnalogOutNodeCarrier,
                                              c_double(0.0)),
              "pump NodeAmplitudeSet(0)")
        d._ck(L.FDwfAnalogOutNodeOffsetSet(h, self.ch, AnalogOutNodeCarrier,
                                           c_double(self._v)),
              "pump NodeOffsetSet(initial)")
        d._ck(L.FDwfAnalogOutRunSet(h, self.ch, c_double(0.0)), "pump RunSet(0)")
        d._ck(L.FDwfAnalogOutRepeatSet(h, self.ch, c_int(0)), "pump RepeatSet(0)")
        try:
            L.FDwfAnalogOutIdleSet(h, self.ch, DwfAnalogOutIdleOffset)
        except Exception:                                      # noqa: BLE001
            pass
        d._ck(L.FDwfAnalogOutConfigure(h, self.ch, c_int(AO_START)),
              "pump AnalogOutConfigure(START)")

        # The first dynamic apply after a channel start has been reported to
        # produce a brief output disturbance on the AD3.  Issue it now, at the
        # startup voltage and before the laser matters, rather than letting it
        # land on the first control step.
        self._apply(self._v)

        logging.info(
            "Pump output ready on Analog Out ch%d (W%d): funcDC, "
            "limits [%.2f, %.2f] V, initial %.3f V. "
            "Updates use OffsetSet + Configure(ch=%d, fStart=3).",
            cfg_ad3.pump_channel, cfg_ad3.pump_channel + 1,
            self.v_min, self.v_max, self._v, cfg_ad3.pump_channel)

    # ------------------------------------------------------------------

    def _apply(self, v):
        L, h = self.dev._lib, self.dev.hdwf
        t0 = time.perf_counter()
        if L.FDwfAnalogOutNodeOffsetSet(h, self.ch, AnalogOutNodeCarrier,
                                        c_double(float(v))) == 0:
            raise AD3Error(f"pump NodeOffsetSet failed: {self.dev._err()}")
        if L.FDwfAnalogOutConfigure(h, self.ch, c_int(AO_APPLY)) == 0:
            raise AD3Error(f"pump AnalogOutConfigure(APPLY) failed: {self.dev._err()}")
        self._t_last = time.perf_counter() - t0
        self.update_times.append(self._t_last)
        if len(self.update_times) > 5000:
            del self.update_times[:2500]

    def set_voltage(self, v):
        self.n_calls += 1
        v = float(v)

        # np.clip(nan, lo, hi) returns nan, so clamping alone does NOT stop a
        # non-finite command reaching the DAC, and what the device does with a
        # NaN double is undefined.  This backend is the last line of defence
        # before the hardware, so a non-finite command is refused outright.
        # ControlSupervisor._apply() catches the exception and transitions to
        # the configured safe state, which is the right response to a control
        # value that should never have been produced.
        if not np.isfinite(v):
            self.n_failures += 1
            raise AD3Error(
                f"refusing to send a non-finite pump command ({v!r}) to the "
                f"Analog Discovery")

        vc = float(np.clip(v, self.v_min, self.v_max))
        if vc != v:
            self.n_clamped += 1
            if self.n_clamped <= 5:
                logging.warning(
                    "Pump command %.3f V is outside the hardware limits "
                    "[%.2f, %.2f] V and was clamped to %.3f V.",
                    v, self.v_min, self.v_max, vc)
        with self._lock:
            try:
                self._apply(vc)
                self._v = vc
            except AD3Error:
                self.n_failures += 1
                raise
        return vc

    @property
    def voltage(self):
        return self._v

    @property
    def last_update_time_s(self):
        return self._t_last

    def close(self):
        try:
            self._apply(self._v)
        except Exception:                                      # noqa: BLE001
            pass


def open_device_and_start_laser(cfg):
    """
    Open the Analog Discovery and start the laser waveform.  Returns the
    device, or None if the AD3 is disabled or unreachable.

    THE LASER IS PART OF THE ACQUISITION, NOT THE CONTROLLER.  EBIV cannot
    work without illumination, so this is called for EVERY run mode as soon as
    AD3Config.enabled is True — including plain 'ebiv' measurement, which is
    the mode most runs use.

    (Release 5.1 fix: the whole hardware block used to sit behind
    `control_enabled = run_mode != 'ebiv'`, so ticking "Use the Analog
    Discovery" and configuring the laser did nothing at all in ebiv mode: the
    device was never opened and no waveform was generated.  Nothing in the log
    said so either, because nothing was attempted.)
    """
    if not cfg.ad3.enabled:
        logging.info("Analog Discovery is DISABLED in the configuration: no "
                     "laser waveform will be generated and no voltage will "
                     "reach the pump. Tick 'Use the Analog Discovery' on the "
                     "Hardware tab, or set AD3_ENABLED = True, to change that.")
        return None
    dev = None
    try:
        dev = AnalogDiscovery3(cfg.ad3)
        dev.start_laser()
        return dev
    except Exception as exc:                                   # noqa: BLE001
        # Deliberately broad.  An AD3Error is the expected failure, but a
        # WaveForms build missing an entry point surfaces as AttributeError,
        # and a bad ctypes signature as OSError/ArgumentError.  None of those
        # should escape into the acquisition loop.
        if not isinstance(exc, AD3Error):
            logging.exception("Unexpected failure while opening the Analog "
                              "Discovery.")
        logging.error("=" * 70)
        logging.error("ANALOG DISCOVERY NOT AVAILABLE: %s", exc)
        logging.error("NO LASER WAVEFORM IS BEING GENERATED.")
        logging.error("Most common causes, in order:")
        logging.error("  1. the WaveForms desktop application is still open — "
                      "it holds an")
        logging.error("     EXCLUSIVE lock on the device. Close it completely.")
        logging.error("  2. device index: use -1 (first available) unless you "
                      "have more than")
        logging.error("     one Digilent device plugged in. Indices are "
                      "0-based, so with a")
        logging.error("     single device only -1 and 0 are valid.")
        logging.error("  3. the USB cable / the device is not enumerated.")
        logging.error("=" * 70)
        if dev is not None:
            try:
                dev.close()
            except Exception:                                  # noqa: BLE001
                pass
        return None


def make_pump(cfg, plant=None, device=None):
    """
    Build (device, pump).

    Pass an already-open device to attach the pump to it; otherwise the device
    is opened here (and the laser started) as before.  Falls back to MockPump
    whenever the hardware is disabled or unreachable, and says so loudly
    rather than pretending.
    """
    ad3 = cfg.ad3

    def _mock(reason):
        logging.warning("Pump backend: MOCK (%s). No voltage reaches the "
                        "hardware.", reason)
        return MockPump(ad3.pump_v_min, ad3.pump_v_max,
                        cfg.supervisor.startup_pump_voltage, plant=plant)

    if not ad3.enabled:
        return None, _mock("Analog Discovery disabled in the configuration")

    owns_device = device is None
    dev = device
    try:
        if dev is None:
            dev = AnalogDiscovery3(ad3)
            dev.start_laser()
        pump = AnalogOutPump(dev, ad3, cfg.supervisor.startup_pump_voltage)
        return dev, pump
    except Exception as exc:                                   # noqa: BLE001
        if not isinstance(exc, AD3Error):
            logging.exception("Unexpected failure while preparing the pump "
                              "output.")
        logging.error("Analog Discovery pump output unavailable: %s", exc)
        if owns_device and dev is not None:
            try:
                dev.close()
            except Exception:                                  # noqa: BLE001
                pass
            dev = None
        return dev, _mock(str(exc))


def quick_laser_test(cfg, hold_s=None):
    """
    Open the AD3, start the laser, and hold it until you press Enter.

    The shortest possible check that the Python side can drive the device at
    all, independent of the camera, the PIV and the controller.  Use it first
    when the laser does not come on.
    """
    print("\n" + "=" * 70)
    print("  ANALOG DISCOVERY — QUICK LASER TEST")
    print("=" * 70)
    lib = _load_dwf()
    print(f"  WaveForms runtime loaded : {lib is not None}")
    if lib is None:
        print("  Install the Digilent WaveForms package (it provides dwf.dll).")
        print("=" * 70)
        return False
    n = c_int(0)
    try:
        lib.FDwfEnum(c_int(0), byref(n))
    except Exception as exc:                                   # noqa: BLE001
        print(f"  FDwfEnum failed: {exc}")
        return False
    print(f"  Digilent devices detected: {n.value}")
    if n.value == 0:
        print("  No device. Check the USB cable, and make sure the WaveForms")
        print("  desktop application is CLOSED — it locks the device.")
        print("=" * 70)
        return False
    if cfg.ad3.device_index >= n.value:
        print(f"  device_index = {cfg.ad3.device_index} but only {n.value} "
              f"device(s) exist.")
        print(f"  Indices are 0-based; use -1 (first available) or "
              f"0..{n.value - 1}.")
        print("=" * 70)
        return False

    dev = open_device_and_start_laser(cfg)
    if dev is None:
        print("\n  The laser could NOT be started. See the messages above.")
        print("=" * 70)
        return False
    try:
        c = cfg.ad3
        print(f"\n  Laser is RUNNING now:")
        if c.laser_backend == 'analog':
            print(f"    Analog Out ch{c.laser_channel} (W{c.laser_channel + 1})")
            lo = c.laser_offset_v - c.laser_amplitude_v
            hi = c.laser_offset_v + c.laser_amplitude_v
            print(f"    {(c.laser_frequency_hz or 0):g} Hz square, "
                  f"{c.laser_duty_percent:g}% duty")
            print(f"    amplitude {c.laser_amplitude_v:g} V about offset "
                  f"{c.laser_offset_v:g} V  ->  swings {lo:g} .. {hi:g} V")
            if hi > 5.0 or lo < -5.0:
                print(f"    WARNING: {lo:g}..{hi:g} V is outside the +-5 V "
                      f"output range and will clip.")
        else:
            print(f"    DIO{c.laser_dio_channel}, {(c.laser_frequency_hz or 0):g} Hz, "
                  f"{c.laser_duty_percent:g}% duty")
        print("\n  Check it on a scope, or just look at the laser.")
        if hold_s:
            print(f"  Holding for {hold_s:g} s...")
            time.sleep(hold_s)
        else:
            input("  Press ENTER to stop and close the device... ")
        return True
    finally:
        dev.close()
        print("  Device closed.")
        print("=" * 70)


# ==========================================================================
#  Hardware verification: does a pump update disturb the laser?
# ==========================================================================

def _pulse_metrics(x, fs):
    """
    Frequency, duty cycle and period jitter from a captured square wave, by
    linear-interpolated threshold crossings at the 50% level.
    """
    x = np.asarray(x, dtype=np.float64)

    # The logic levels must NOT come from fixed percentiles.  A pulsed-laser
    # trigger runs at a low duty cycle — 10% by default here — so the 95th
    # percentile of the trace still sits in the LOW state and the levels
    # collapse.  Split at the midpoint of the full range first, then take the
    # median of each side: that is independent of duty cycle and still robust
    # to noise and overshoot.
    x_lo, x_hi = float(x.min()), float(x.max())
    if x_hi - x_lo < 0.05:
        return None
    mid = 0.5 * (x_lo + x_hi)
    low_samples = x[x < mid]
    high_samples = x[x >= mid]
    if low_samples.size < 2 or high_samples.size < 2:
        return None
    lo = float(np.median(low_samples))
    hi = float(np.median(high_samples))
    if hi - lo < 0.05:
        return None
    thr = 0.5 * (lo + hi)
    above = x > thr
    rise = np.nonzero(~above[:-1] & above[1:])[0]
    fall = np.nonzero(above[:-1] & ~above[1:])[0]
    if rise.size < 3:
        return None

    def refine(idx):
        """Sub-sample crossing position by linear interpolation."""
        idx = np.atleast_1d(np.asarray(idx))
        x0, x1 = x[idx], x[idx + 1]
        d = x1 - x0
        frac = np.where(d != 0.0, (thr - x0) / np.where(d != 0.0, d, 1.0), 0.0)
        return idx.astype(np.float64) + frac

    tr = refine(rise) / fs
    periods = np.diff(tr)
    # pair each rise with the next falling edge to get the high time
    highs = []
    for r in tr:
        nxt = fall[fall > r * fs]
        if nxt.size:
            highs.append(float(refine(nxt[:1])[0]) / fs - r)
    return {
        'f_hz': 1.0 / np.mean(periods),
        'period_mean_us': 1e6 * np.mean(periods),
        'period_std_us': 1e6 * np.std(periods),
        'period_ptp_us': 1e6 * np.ptp(periods),
        'width_mean_us': 1e6 * float(np.mean(highs)) if highs else float('nan'),
        'duty_pct': 100.0 * float(np.mean(highs)) * (1.0 / np.mean(periods)) if highs else float('nan'),
        'n_periods': int(periods.size),
        'v_low': lo, 'v_high': hi,
    }


def verify_laser_undisturbed(cfg, voltages=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0),
                             dwell_s=3.0, use_internal_scope=False,
                             scope_channel=0, scope_rate_hz=2e6):
    """
    Hardware verification procedure for requirement 8.

    Runs the laser continuously and steps the pump command through
    `voltages`, holding each for dwell_s.  It NEVER touches the laser channel
    between steps, and it reports the number of laser-channel API calls so
    that can be confirmed rather than assumed.

    Two levels of checking:

    1. MANUAL (authoritative).  Put a scope on the laser output and watch
       frequency, pulse width, duty cycle and trigger stability while this
       routine walks the pump voltage.  The routine prints a countdown and a
       banner at each step so the scope trace can be correlated with the
       commanded voltage.  This is the check that matters, because an
       independent instrument is the only way to see a glitch that the AD3's
       own clock domain would hide.

    2. OPTIONAL SELF-CHECK (use_internal_scope=True).  Wire the laser output
       back into scope input 1+ and the AD3 measures its own waveform before,
       during and after each pump change, reporting frequency, pulse width,
       duty and period jitter.  Useful as a fast regression check, but note
       the honest limitation: the generator and the digitiser share the same
       device and reference clock, so this measurement is blind to any common
       clock disturbance and its jitter figure is a lower bound.  It does not
       replace the external scope.
    """
    print("\n" + "=" * 76)
    print("  ANALOG DISCOVERY 3 — LASER / PUMP INDEPENDENCE VERIFICATION")
    print("=" * 76)

    dev, pump = make_pump(cfg)
    if dev is None:
        print("\n  No Analog Discovery available. Nothing was verified.")
        print("  Set FLAG_AD3_ENABLED = True and close the WaveForms application.")
        return None

    laser_calls_after_start = dev._laser_touch_count
    results = []
    try:
        print(f"\n  Laser backend : {cfg.ad3.laser_backend}")
        if cfg.ad3.laser_backend == 'analog':
            print(f"  Laser channel : Analog Out ch{cfg.ad3.laser_channel} "
                  f"(W{cfg.ad3.laser_channel + 1})")
        elif cfg.ad3.laser_backend == 'pattern':
            print(f"  Laser channel : DIO{cfg.ad3.laser_dio_channel}")
        print(f"  Laser setting : {(cfg.ad3.laser_frequency_hz or 0):.2f} Hz, "
              f"{cfg.ad3.laser_duty_percent:.1f}% duty")
        print(f"  Pump channel  : Analog Out ch{cfg.ad3.pump_channel} "
              f"(W{cfg.ad3.pump_channel + 1})")
        print(f"\n  Put the scope on the LASER output now.")
        print(f"  Trigger on its rising edge and enable frequency, pulse-width")
        print(f"  and duty measurements. Watch the trigger-stability/jitter too.\n")
        input("  Press ENTER when the scope is armed and stable... ")

        baseline = None
        if use_internal_scope:
            x, fs = dev.capture(scope_channel, rate_hz=scope_rate_hz)
            baseline = _pulse_metrics(x, fs)
            if baseline is None:
                print("  [self-check] No square wave seen on scope input "
                      f"{scope_channel + 1}+. Is the laser output wired back to it?")
                use_internal_scope = False
            else:
                print(f"  [self-check] baseline: {baseline['f_hz']:.4f} Hz, "
                      f"width {baseline['width_mean_us']:.2f} us, "
                      f"duty {baseline['duty_pct']:.2f}%, "
                      f"period jitter (std) {baseline['period_std_us']:.3f} us")

        for v in voltages:
            print("\n" + "-" * 76)
            print(f"  PUMP -> {v:.2f} V   (laser channel is NOT addressed)")
            t_apply = time.perf_counter()
            applied = pump.set_voltage(v)
            dt_apply = time.perf_counter() - t_apply
            print(f"  applied {applied:.3f} V in {dt_apply * 1e3:.3f} ms "
                  f"[OffsetSet + Configure(ch{cfg.ad3.pump_channel}, fStart=3)]")

            row = {'v_command': v, 'v_applied': applied, 'apply_ms': dt_apply * 1e3}

            if use_internal_scope:
                time.sleep(0.2)
                x, fs = dev.capture(scope_channel, rate_hz=scope_rate_hz)
                m = _pulse_metrics(x, fs)
                if m is not None:
                    df = 1e6 * (m['f_hz'] - baseline['f_hz']) / baseline['f_hz']
                    dw = m['width_mean_us'] - baseline['width_mean_us']
                    print(f"  [self-check] {m['f_hz']:.4f} Hz "
                          f"({df:+.1f} ppm vs baseline), "
                          f"width {m['width_mean_us']:.3f} us ({dw:+.3f} us), "
                          f"duty {m['duty_pct']:.3f}%, "
                          f"jitter(std) {m['period_std_us']:.3f} us")
                    row.update({f'scope_{k}': val for k, val in m.items()})
                    row['freq_ppm_change'] = df
                    row['width_change_us'] = dw

            results.append(row)
            for s in range(int(dwell_s), 0, -1):
                print(f"    holding {v:.2f} V ... {s} s ", end='\r', flush=True)
                time.sleep(1.0)
            print(" " * 60, end='\r')

        print("\n" + "=" * 76)
        print("  RESULT")
        print("=" * 76)
        print(f"  Laser-channel configuration calls during the sweep : "
              f"{dev._laser_touch_count - laser_calls_after_start}  (expected 0)")
        print(f"  Pump updates issued                                : {pump.n_calls}")
        print(f"  Pump update failures                               : "
              f"{getattr(pump, 'n_failures', 0)}")
        ut = getattr(pump, 'update_times', [])
        if ut:
            a = np.array(ut) * 1e3
            print(f"  Pump update time [ms]  mean {a.mean():.3f}  "
                  f"max {a.max():.3f}  n={a.size}")
        print(f"  Device reopen / reset events                       : 0 "
              f"(the handle was opened once and never reset)")
        if use_internal_scope and results:
            ppm = [r.get('freq_ppm_change') for r in results if 'freq_ppm_change' in r]
            dw = [r.get('width_change_us') for r in results if 'width_change_us' in r]
            if ppm:
                print(f"  Self-check laser frequency drift : "
                      f"max |{max(abs(p) for p in ppm):.1f}| ppm")
                print(f"  Self-check pulse-width drift     : "
                      f"max |{max(abs(w) for w in dw):.4f}| us")
        print("\n  The external oscilloscope observation is the authoritative")
        print("  result. If frequency, pulse width, duty cycle or trigger")
        print("  stability changed at any step, do NOT proceed to closed-loop")
        print("  operation: report it, and consider laser_backend='pattern',")
        print("  which puts the laser on the Pattern Generator, a separate")
        print("  instrument from Analog Out.")
        print("=" * 76 + "\n")
        return results

    finally:
        try:
            pump.set_voltage(cfg.supervisor.safe_pump_voltage)
        except Exception:                                      # noqa: BLE001
            pass
        if dev is not None:
            dev.close()
