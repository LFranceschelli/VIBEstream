"""
VIBE — scriptable interface to VibeStream (event camera + rt-EBIV + Analog Discovery).

One object gives a Python script everything the VibeStream GUI does, without
the GUI:

    from vibe import VIBE

    with VIBE(biases={'bias_diff_on': 40, 'bias_diff_off': 150}) as vibe:
        vibe.record(T=2, f=500, path='run01.raw')           # blocking, 2 s

        for fld in vibe.velocity(f=500, T=5, window=48, step=24):
            print(fld.t_us, fld.u.mean(), fld.v.mean())       # px per frame

METHODS
    camera        list_cameras()  sensor_size()  read_biases()
    record        record(T, f, path, block=True)   stop()   read_meta(path)
    live          frames(f, ...)            generator of Frame
                  velocity(f, ...)          generator of VelocityField
    control loop  start(f, ...)  latest()  wait()  stop()  running
                  roi_mean(field, roi)  pid(kp, ki, kd, ...)  show(field)
    offline       frames_from_raw(path)  piv(frames)  save_frames(path, dir)
                  piv_offline(in_dir, out_dir)  detect_phase(path, f)  playback(path)
    Analog Disc.  laser_on(f)  laser_off()  set_voltage(v)  voltage
                  scope(ch)  laser_check()  verify_hardware()  ad3_open()  ad3_close()
    lifecycle     close()  /  with VIBE(...) as vibe:

It is built ON TOP of the VibeStream 5.2 library in this folder (lib/), not
beside it.  The numerics are the library's:

    correlation        ebiv_piv.CPUCorrelator / ebiv_gpu.GPUCorrelator
    validation         ebiv_utils.universal_outlier_detection
    laser phase        ebiv_utils.detect_laser_phase
    event accumulation ebiv_utils.fast_accumulate (numba)
    Analog Discovery   ebiv_hardware.AnalogDiscovery3 / AnalogOutPump
    PID                ebiv_control.PIDController
    offline pipeline   ebiv_utils.generate_centered_images / process_offline_piv

What is NEW here, because the library only had it inlined inside the GUI
stream loop, is the event -> pseudo-frame logic (_FrameBuilder) as a reusable
object, so that the same frames can feed a generator, a background thread or
an offline pass.

CONVENTIONS
    * Images are returned AFTER flip_x / flip_y.  Every ROI given to this
      class ([x0, x1, y0, y1], pixels, end-exclusive) refers to that flipped
      image.  (ebiv_utils.generate_centered_images crops BEFORE flipping;
      VIBE does not, so that one ROI means the same pixels everywhere.)
    * Displacements u, v are in PIXELS PER FRAME.  v is positive DOWNWARD
      (image rows), as in the rest of VibeStream and in the saved .mat files.
    * One laser pulse per frame is assumed: the frame separation is 1/f.
      Pulse-pair illumination is NOT handled here.
    * The camera is opened for each operation and released at its end
      (the pattern ebiv_utils uses).  Biases are applied at every open.

Everything that touches hardware releases it on every exit path: normal end,
exception, Ctrl-C, breaking out of a generator, close().  close() also parks
the Analog Discovery outputs at 0 V (laser trigger low, pump 0 V).
"""

import os
import sys
import gc
import json
import time
import queue
import logging
import threading
import collections
from ctypes import c_int
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import numpy as np

# The VibeStream library imports its modules by plain name; make sure this
# folder is importable even when vibe.py is imported from elsewhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import ebiv_utils as _U                                   # noqa: E402
import ebiv_hardware as _HW                               # noqa: E402
from ebiv_config import AD3Config, PIDConfig, ControlSystemConfig  # noqa: E402
from ebiv_config import __version__ as _VIBESTREAM_RELEASE          # noqa: E402
from ebiv_piv import CPUCorrelator                        # noqa: E402
from ebiv_control import PIDController                    # noqa: E402

__version__ = "1.0.0"

log = logging.getLogger("vibe")

CameraSetupError = _U.CameraSetupError
AD3Error = _HW.AD3Error


# ==========================================================================
#  Results
# ==========================================================================

@dataclass
class Frame:
    """One pseudo-frame.  image is uint8, full sensor, after flips."""
    t_us: int           # camera time of the frame: pulse centre (triggered) or batch end
    image: np.ndarray
    lag_s: float = 0.0  # how far the acquisition is behind the sensor (live only)
    index: int = 0


@dataclass
class VelocityField:
    """
    One rt-EBIV vector field.

    x, y      vector positions, pixels, in the (flipped) full image.
    u, v      displacement, px per frame.  Outliers replaced by the local
              median when validation is on (what the GUI displays).
    u_raw,
    v_raw     displacement before validation.
    valid     outlier mask (True = accepted), or None without validation.
    cc        normalised correlation peak per vector, or None.
    t_us      camera time of the NEWEST frame in the pair.
    lag_s     acquisition lag at that frame (live only; 0 offline).
    """
    x: np.ndarray
    y: np.ndarray
    u: np.ndarray
    v: np.ndarray
    u_raw: np.ndarray
    v_raw: np.ndarray
    valid: Optional[np.ndarray]
    cc: Optional[np.ndarray]
    t_us: int
    f_hz: float
    seq: int = 0
    lag_s: float = 0.0
    t_wall: float = 0.0     # time.perf_counter() when the newest frame was complete
    image: Optional[np.ndarray] = None   # newest frame (full, flipped), for show()

    @property
    def dt_s(self) -> float:
        """Frame separation (one pulse per frame)."""
        return 1.0 / self.f_hz

    @property
    def magnitude(self) -> np.ndarray:
        return np.hypot(self.u, self.v)

    def to_ms(self, px_per_mm: float) -> Tuple[np.ndarray, np.ndarray]:
        """(u, v) in m/s for an optical resolution in px/mm.  v stays positive down."""
        k = self.f_hz / float(px_per_mm) / 1000.0
        return self.u * k, self.v * k

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in ('x', 'y', 'u', 'v', 'u_raw', 'v_raw',
                                            't_us', 'f_hz', 'seq', 'lag_s')}
        if self.valid is not None:
            d['valid'] = self.valid
        if self.cc is not None:
            d['cc'] = self.cc
        return d


@dataclass
class RecordResult:
    path: str
    meta_path: str
    duration_s: float        # measured on the CAMERA clock
    n_events: int
    wall_s: float            # how long it took on the PC clock
    complete: bool           # False if stopped early / timed out / interrupted
    meta: dict = field(default_factory=dict)


# ==========================================================================
#  Event -> pseudo-frame
# ==========================================================================

class _FrameBuilder:
    """
    Turns batches of events into accumulation images.

    trigger='none'      one frame per batch (the batch length is 1/f).
    trigger='auto'      laser phase measured from the event rate over the
                        first `calib_us`, then windows of duty_cycle/f
                        centred on every pulse.  No frame is produced while
                        calibrating (the GUI shows untriggered frames there;
                        a measurement should not mix the two).
    trigger='external'  window centres from trigger timestamps passed in
                        with add_triggers(); between triggers the schedule
                        advances by one period.
    phase_us=<number>   fixed, known phase (used offline): behaves like
                        'auto' with the calibration already done.

    Same logic as the inline loop in ebiv_utils.stream_camera, with one
    generalisation: every complete window in a batch is emitted, not only the
    first, so a batch longer than one period does not lose frames.

    push() returns a list of (t_center_us, counts) where counts is a float32
    COPY of the accumulation (events per pixel).
    """

    def __init__(self, width, height, f_hz, trigger='none', duty_cycle=0.8,
                 phase_us=None, calib_us=500_000):
        if trigger not in ('none', 'auto', 'external'):
            raise ValueError("trigger must be 'none', 'auto' or 'external'")
        if not (0.0 < duty_cycle <= 1.0):
            raise ValueError("duty_cycle must be in (0, 1]")
        self.w, self.h = int(width), int(height)
        self.period = int(round(1e6 / float(f_hz)))
        self.half = int(duty_cycle * self.period / 2)
        self.trigger = trigger
        self.buf = np.zeros((self.h, self.w), dtype=np.float32)
        self.calib_us = int(calib_us)
        self._calib = []
        self._calib_t0 = None
        self.phase_us = None if phase_us is None else float(phase_us)
        self.center = None
        self.triggers = collections.deque(maxlen=4096)
        self.n_resync = 0
        if self.phase_us is not None:
            self.trigger = 'auto'

    # ------------------------------------------------------------------
    @property
    def ready(self) -> bool:
        if self.trigger == 'none':
            return True
        if self.trigger == 'external':
            return self.center is not None
        return self.phase_us is not None

    def add_triggers(self, ts):
        for t in np.asarray(ts).ravel():
            self.triggers.append(int(t))

    def _emit(self, stamp):
        out = self.buf.copy()
        self.buf.fill(0.0)
        return int(stamp), out

    def _next_external(self, after):
        for t_ in self.triggers:
            if t_ > after:
                return t_
        return None

    # ------------------------------------------------------------------
    def push(self, t, x, y, t_end=None):
        frames = []
        n = len(t)

        if self.trigger == 'none':
            if n:
                _U.fast_accumulate(self.buf, x, y)
            stamp = t_end if t_end is not None else (int(t[-1]) if n else 0)
            frames.append(self._emit(stamp))
            return frames

        if n == 0:
            return frames
        t0, t1 = int(t[0]), int(t[-1])

        # --- auto: calibrate the phase once --------------------------------
        if self.trigger == 'auto' and self.phase_us is None:
            if self._calib_t0 is None:
                self._calib_t0 = t0
            self._calib.append(np.asarray(t).copy())
            if t1 - self._calib_t0 < self.calib_us:
                return frames
            all_t = np.concatenate(self._calib)
            self._calib.clear()
            peak, _, _ = _U.detect_laser_phase(all_t, self.period)
            self.phase_us = float(peak)
            # first centre strictly after this batch (as ebiv_utils does)
            self.center = t1 - ((t1 - int(self.phase_us)) % self.period) + self.period
            log.info("Auto-trigger calibrated: laser phase %.0f us within a %d us "
                     "period. First window centre at t = %d us.",
                     self.phase_us, self.period, self.center)
            return frames

        if self.trigger == 'auto' and self.center is None:
            # fixed phase given (offline): first centre at or after the data
            k = int(np.ceil((t0 - self.half - self.phase_us) / self.period))
            self.center = int(round(self.phase_us)) + max(k, 0) * self.period

        if self.trigger == 'external' and self.center is None:
            if not self.triggers:
                return frames
            self.center = self.triggers[-1]

        # --- accumulation windows centred on the laser pulses ---------------
        # The stream paused (no light, nothing arrived): jump by whole periods.
        # The phase is unchanged; only the counter was stale.
        if t0 - (self.center + self.half) > self.period:
            n_skip = (t0 - self.center) // self.period
            self.center += n_skip * self.period
            self.buf.fill(0.0)
            self.n_resync += 1
            if self.n_resync <= 3 or self.n_resync % 50 == 0:
                log.info("Trigger schedule re-anchored %d period(s) forward "
                         "(event stream paused). [%d so far]", n_skip, self.n_resync)

        while True:
            lo = self.center - self.half
            hi = self.center + self.half
            i0 = int(np.searchsorted(t, lo, side='left'))
            i1 = int(np.searchsorted(t, hi, side='left'))
            if i1 > i0:
                _U.fast_accumulate(self.buf, x[i0:i1], y[i0:i1])
            if t1 < hi:
                break                       # window continues in the next batch
            frames.append(self._emit(self.center))
            if self.trigger == 'external':
                nxt = self._next_external(self.center)
                self.center = nxt if nxt is not None else self.center + self.period
            else:
                self.center += self.period
        return frames


# ==========================================================================
#  Background result holder
# ==========================================================================

class _Latest:
    """Newest VelocityField, published as one object, with a wait()."""

    def __init__(self):
        self._cv = threading.Condition()
        self._fld = None

    def publish(self, fld):
        with self._cv:
            self._fld = fld
            self._cv.notify_all()

    def read(self):
        with self._cv:
            return self._fld

    def wait_newer(self, seq, timeout):
        end = time.perf_counter() + (timeout if timeout is not None else 1e9)
        with self._cv:
            while self._fld is None or self._fld.seq <= seq:
                left = end - time.perf_counter()
                if left <= 0:
                    return None
                self._cv.wait(left)
            return self._fld


# ==========================================================================
#  VIBE
# ==========================================================================

class VIBE:
    """
    Parameters
    ----------
    biases : dict, optional
        Camera biases, e.g. {'bias_diff_on': 40, 'bias_diff_off': 150,
        'bias_hpf': 70, 'bias_fo': 0, 'bias_refr': 90, 'bias_diff': 0}.
        Applied every time the camera is opened.
    serial : str, optional
        Camera serial number.  None = first camera found.
    roi : [x0, x1, y0, y1], optional
        Default processing ROI (pixels, end-exclusive, in the flipped image).
    flip_x, flip_y : bool
        Mirror the image left-right / up-down.
    max_events_per_pixel : int
        Contrast clamp of the live pseudo-frames (as MAX_EVENTS_PER_PIXEL).
    output_dir : str
        Where relative paths (recordings, frames) go.
    ad3 : dict or AD3Config, optional
        Analog Discovery settings (see ebiv_config.AD3Config): device_index,
        laser_backend, laser_channel, laser_amplitude_v, laser_offset_v,
        laser_duty_percent, pump_channel, pump_v_min, pump_v_max, ...
        The device is opened on first use, not here.
    ad3_mock : bool
        True: never touch a real Analog Discovery.  set_voltage() drives a
        MockPump and laser calls only log.  For dry runs of a control loop.
    smooth_sigma : float, optional
        Gaussian smoothing (std in px) applied to every pseudo-frame, live and
        offline, before correlation.  None = off.
    """

    TRIGGERS = ('none', 'auto', 'external')

    def __init__(self, biases=None, serial=None, roi=None, flip_x=False, flip_y=False,
                 max_events_per_pixel=1, output_dir='.', ad3=None, ad3_mock=False,
                 smooth_sigma=None):
        self.biases = dict(biases) if biases else {}
        self.serial = serial
        self.roi = list(roi) if roi is not None else None
        self.flip_x = bool(flip_x)
        self.flip_y = bool(flip_y)
        self.max_events_per_pixel = int(max_events_per_pixel)
        # Gaussian smoothing (std, px) of every pseudo-frame, live and offline
        # (the paper uses 0.75 px on binary frames).  None = off.
        self.smooth_sigma = float(smooth_sigma) if smooth_sigma else None
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        if isinstance(ad3, AD3Config):
            self._ad3_cfg = ad3
        else:
            self._ad3_cfg = AD3Config(**(ad3 or {}))
        self._ad3_cfg.enabled = True
        self.ad3_mock = bool(ad3_mock)
        self._ad3 = None            # AnalogDiscovery3
        self._pump = None           # AnalogOutPump / MockPump
        self._laser_running = False
        self._laser_f = None

        self._busy = threading.Lock()   # one camera operation at a time
        self._stop_evt = threading.Event()
        self._rec_thread = None
        self._rec_result = None
        self._bg_threads = []
        self._latest = _Latest()
        self._bg_error = None

    # ------------------------------------------------------------------
    #  housekeeping
    # ------------------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        """Stop everything, release the camera, park the Analog Discovery at 0 V."""
        try:
            self.stop()
        finally:
            self.ad3_close()

    def __repr__(self):
        return (f"VIBE(serial={self.serial!r}, roi={self.roi}, flip=({self.flip_x},"
                f"{self.flip_y}), ad3={'open' if self._ad3 else 'closed'}"
                f"{' MOCK' if self.ad3_mock else ''}, busy={self.busy})")

    @property
    def busy(self) -> bool:
        return self._busy.locked()

    @staticmethod
    def version():
        return {'vibe': __version__, 'vibestream': _VIBESTREAM_RELEASE}

    def _path(self, p):
        return p if os.path.isabs(p) else os.path.join(self.output_dir, p)

    # ==================================================================
    #  CAMERA
    # ==================================================================
    @staticmethod
    def list_cameras() -> List[str]:
        """Serial numbers of the connected cameras."""
        if not _U._HAS_METAVISION:
            raise CameraSetupError(
                "Metavision (metavision_hal / metavision_core) is not importable "
                "in this Python. Run tools/check_env.py with this interpreter.")
        return list(_U.mv.DeviceDiscovery.list())

    def _open_device(self):
        if not _U._HAS_METAVISION:
            raise CameraSetupError(
                "Metavision (metavision_hal / metavision_core) is not importable "
                "in this Python, so no event camera can be opened. Run "
                "tools/check_env.py with this interpreter (ReadMe: 'IF THE "
                "CAMERA WORKED BEFORE AND NOW GIVES A WINDOWS DLL DIALOG').")
        devs = _U.mv.DeviceDiscovery.list()
        if not devs:
            raise CameraSetupError("No camera found. Check the USB3 connection "
                                   "and MV_HAL_PLUGIN_PATH.")
        target = self.serial if self.serial else devs[0]
        if self.serial and self.serial not in devs:
            raise CameraSetupError(f"Camera {self.serial!r} not found. Connected: {devs}")
        device = _U.mv.DeviceDiscovery.open(target)
        if device is None:
            raise CameraSetupError(f"Could not open camera {target!r} (in use by "
                                   "another program?).")
        self._apply_biases(device)
        return device, target

    def _apply_biases(self, device):
        if not self.biases:
            return
        ll = device.get_i_ll_biases()
        if ll is None:
            log.warning("This camera exposes no bias interface; biases not set.")
            return
        for name, val in self.biases.items():
            try:
                ll.set(name, int(val))
            except Exception as exc:                            # noqa: BLE001
                log.warning("Could not set bias %s=%s: %s", name, val, exc)

    def read_biases(self) -> dict:
        """Open the camera, apply self.biases, and return what the camera reports."""
        with self._busy_guard("read_biases"):
            device, _ = self._open_device()
            try:
                ll = device.get_i_ll_biases()
                if ll is None:
                    return {}
                try:
                    return dict(ll.get_all_biases())
                except Exception:                                # noqa: BLE001
                    return {k: ll.get(k) for k in self.biases}
            finally:
                del device
                gc.collect()

    def sensor_size(self) -> Tuple[int, int]:
        """(width, height) of the connected sensor."""
        with self._busy_guard("sensor_size"):
            device, _ = self._open_device()
            it = None
            try:
                it = _U.EventsIterator.from_device(device=device, delta_t=10_000)
                h, w = it.get_size()
                return int(w), int(h)
            finally:
                del it, device
                gc.collect()

    class _BusyGuard:
        def __init__(self, owner, what):
            self.owner, self.what = owner, what

        def __enter__(self):
            if not self.owner._busy.acquire(blocking=False):
                raise RuntimeError(
                    f"{self.what}: the camera is already in use by another VIBE "
                    "operation (a background stream or recording). Call stop() first.")
            return self

        def __exit__(self, *exc):
            self.owner._busy.release()
            return False

    def _busy_guard(self, what):
        return VIBE._BusyGuard(self, what)

    # ------------------------------------------------------------------
    #  RECORD
    # ------------------------------------------------------------------
    def record(self, T=2.0, f=None, path=None, block=True, log_triggers=False,
               timeout_s=None) -> Optional[RecordResult]:
        """
        Record raw events to a .raw file.

        T      duration in seconds, measured on the CAMERA clock (event
               timestamps), not the PC clock.  None = until stop() (needs
               block=False).
        f      laser / acquisition frequency in Hz.  Stored in the metadata
               for the later pseudo-image generation (one per pulse).  Event cameras have no
               frame rate: f does not down-sample anything.  If VIBE is
               already driving the laser (laser_on()), it is retuned to f;
               a laser that is off is never switched on implicitly.
        path   .raw path; default rec_YYYYmmdd_HHMMSS.raw in output_dir.
               A JSON sidecar <path>.json holds the metadata.
        block  False: record in the background and return None; stop() ends
               it and returns the RecordResult.
        log_triggers  enable the camera's trigger-in so external trigger
               events are written into the .raw file.
        timeout_s  PC-clock safety limit (default T + 10 s): with no light,
               no events arrive and the camera clock seen through them stops.
        """
        if T is None and block:
            raise ValueError("T=None records until stop(); use block=False.")
        if T is not None and T <= 0:
            raise ValueError("T must be > 0")
        path = self._path(path or time.strftime("rec_%Y%m%d_%H%M%S.raw"))
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        if f is not None:
            self._sync_laser(f)

        if block:
            with self._busy_guard("record"):
                self._stop_evt.clear()
                return self._record_impl(T, f, path, log_triggers, timeout_s)

        if not self._busy.acquire(blocking=False):
            raise RuntimeError("record: the camera is busy. Call stop() first.")
        self._stop_evt.clear()
        self._rec_result = None

        def _run():
            try:
                self._rec_result = self._record_impl(T, f, path, log_triggers, timeout_s)
            except Exception as exc:                            # noqa: BLE001
                log.exception("Background recording failed.")
                self._bg_error = exc
            finally:
                self._busy.release()

        self._rec_thread = threading.Thread(target=_run, name="vibe-record", daemon=True)
        self._rec_thread.start()
        log.info("Recording to %s in the background%s. Call stop() to end it.",
                 path, f" for {T:g} s" if T else "")
        return None

    def _record_impl(self, T, f, path, log_triggers, timeout_s):
        device, cam_id = self._open_device()
        stream = it = None
        logging_on = False
        n_ev = 0
        t_first = t_last = None
        complete = False
        wall0 = time.perf_counter()
        if timeout_s is None:
            timeout_s = (T + 10.0) if T is not None else None
        try:
            if log_triggers:
                self._enable_trigger_in(device)
            stream = device.get_i_events_stream()
            if stream is None:
                raise CameraSetupError("The camera has no events-stream interface.")
            stream.log_raw_data(path)
            logging_on = True
            log.info("Recording %s -> %s", f"{T:g} s" if T else "until stop()", path)
            # The iterator has to be consumed: that is what pulls the events
            # out of the camera buffer and into the file.
            it = _U.EventsIterator.from_device(device=device, delta_t=10_000)
            for evs in it:
                if len(evs):
                    n_ev += len(evs)
                    if t_first is None:
                        t_first = int(evs['t'][0])
                    t_last = int(evs['t'][-1])
                if self._stop_evt.is_set():
                    log.info("Recording stopped by stop().")
                    complete = T is None        # open-ended recording ends here by design
                    break
                if T is not None and t_first is not None and (t_last - t_first) >= T * 1e6:
                    complete = True
                    break
                if timeout_s is not None and time.perf_counter() - wall0 > timeout_s:
                    log.warning("Recording hit the PC-clock timeout (%.1f s) before %s s "
                                "of events arrived. Is the laser on / is there light?",
                                timeout_s, T)
                    break
        except KeyboardInterrupt:
            log.warning("Recording interrupted (Ctrl-C). The partial file is kept.")
            raise
        finally:
            if logging_on:
                try:
                    stream.stop_log_raw_data()
                except Exception:                                # noqa: BLE001
                    log.exception("stop_log_raw_data failed; the file may be truncated.")
            if stream is not None:
                try:
                    stream.stop()
                except Exception:                                # noqa: BLE001
                    pass
            del it, stream, device
            gc.collect()
            dur = ((t_last - t_first) * 1e-6) if t_first is not None else 0.0
            wall = time.perf_counter() - wall0
            meta = self._metadata(kind='record', f=f, T_requested=T, camera=cam_id,
                                  duration_s=dur, n_events=n_ev, complete=complete,
                                  log_triggers=log_triggers)
            meta_path = path + '.json'
            self._write_json(meta_path, meta)
            level = logging.INFO if complete else logging.WARNING
            log.log(level, "Recorded %.3f s of camera time, %d events (%.1f Mev/s) in "
                    "%.2f s wall -> %s%s", dur, n_ev, n_ev / max(dur, 1e-9) / 1e6,
                    wall, path, "" if complete else "  [INCOMPLETE]")
        return RecordResult(path, meta_path, dur, n_ev, wall, complete, meta)

    def _enable_trigger_in(self, device):
        trig = device.get_i_trigger_in()
        if trig is None:
            log.warning("This camera has no trigger-in interface.")
            return None
        mv = _U.mv
        for ch_name in ('MAIN', 'AUX'):
            try:
                trig.enable(getattr(mv.I_TriggerIn.Channel, ch_name))
            except Exception:                                    # noqa: BLE001
                pass
        return trig

    # ------------------------------------------------------------------
    #  LIVE FRAMES / VELOCITY  (generators)
    # ------------------------------------------------------------------
    def frames(self, f, T=None, n=None, trigger='auto', duty_cycle=0.8,
               record=None, trigger_polarity=1):
        """
        Live pseudo-frames (generator of Frame).

        f          laser / acquisition frequency, Hz (window = 1/f).
        T, n       stop after T seconds of camera time or n frames (either;
                   None = until you break out of the loop or call stop()).
        trigger    'auto' (default: phase from the event rate), 'external'
                   (camera trigger-in), or 'none' (fixed 1/f batches).
        duty_cycle fraction of the period accumulated around each pulse.
        record     optional .raw path: log the raw events while streaming.

        Breaking out of the for-loop releases the camera.
        """
        with self._busy_guard("frames"):
            self._stop_evt.clear()
            yield from self._frames_impl(f, T, n, trigger, duty_cycle, record,
                                         trigger_polarity)

    def _frames_impl(self, f, T, n, trigger, duty_cycle, record, trigger_polarity):
        if trigger not in self.TRIGGERS:
            raise ValueError(f"trigger must be one of {self.TRIGGERS}")
        if f is None or f <= 0:
            raise ValueError("f (Hz) is required")
        self._sync_laser(f)
        _warmup_jit()           # compile numba BEFORE the camera clock starts
        period_us = int(round(1e6 / f))
        device, cam_id = self._open_device()
        stream = it = None
        logging_on = False
        ext_cb_ts = collections.deque(maxlen=4096)
        try:
            use_it_triggers = False
            if trigger == 'external':
                trig = self._enable_trigger_in(device)
                if trig is None:
                    log.warning("trigger='external' but no trigger input: using 'auto'.")
                    trigger = 'auto'
            if record:
                stream = device.get_i_events_stream()
                rpath = self._path(record)
                stream.log_raw_data(rpath)
                logging_on = True
                self._write_json(rpath + '.json', self._metadata(
                    kind='stream_record', f=f, camera=cam_id, trigger=trigger,
                    duty_cycle=duty_cycle))
                log.info("Logging raw events to %s while streaming.", rpath)

            it = _U.EventsIterator.from_device(device=device, delta_t=period_us)
            h, w = it.get_size()
            fb = _FrameBuilder(w, h, f, trigger, duty_cycle)

            if trigger == 'external':
                if hasattr(it, 'get_ext_trigger_events'):
                    use_it_triggers = True
                else:
                    try:
                        device.get_i_trigger_in().add_callback(ext_cb_ts.append)
                    except Exception:                            # noqa: BLE001
                        log.warning("Cannot read external triggers with this SDK: "
                                    "using 'auto'.")
                        fb = _FrameBuilder(w, h, f, 'auto', duty_cycle)

            acq_t0_wall = acq_t0_evt = None
            lag = lag_ema = 0.0
            lag_warned = False
            t_start = None
            count = 0
            for evs in it:
                if self._stop_evt.is_set():
                    break
                nev = len(evs)
                if nev:
                    t_evt = float(evs['t'][-1]) * 1e-6
                    now = time.perf_counter()
                    if acq_t0_wall is None:
                        acq_t0_wall, acq_t0_evt = now, t_evt
                    else:
                        # drift of the PC clock against the camera clock since the
                        # first batch: growth means this loop is not keeping up
                        lag = max(0.0, (now - acq_t0_wall) - (t_evt - acq_t0_evt))
                        lag_ema += 0.05 * (lag - lag_ema)
                        if lag_ema > 0.5 and not lag_warned:
                            lag_warned = True
                            log.warning(
                                "ACQUISITION LAG %.2f s and growing: your loop is slower "
                                "than the sensor, events are queueing in the SDK. Do less "
                                "per frame (smaller ROI, lower f), or use start()/latest() "
                                "which drops fields instead of falling behind.", lag_ema)
                if use_it_triggers:
                    try:
                        tr = it.get_ext_trigger_events()
                        if len(tr):
                            fb.add_triggers(tr['t'][tr['p'] == trigger_polarity])
                            it.clear_ext_trigger_events()
                    except Exception:                            # noqa: BLE001
                        pass
                elif ext_cb_ts:
                    fb.add_triggers(list(ext_cb_ts))
                    ext_cb_ts.clear()

                t_end = None
                try:
                    t_end = int(it.get_current_time())
                except Exception:                                # noqa: BLE001
                    pass
                for t_c, counts in fb.push(evs['t'], evs['x'], evs['y'], t_end=t_end):
                    img = self._finalize(counts, mode='clip')
                    if t_start is None:
                        t_start = t_c
                    yield Frame(t_c, img, lag, count)
                    count += 1
                    if n is not None and count >= n:
                        return
                    if T is not None and (t_c - t_start) >= T * 1e6:
                        return
                    if self._stop_evt.is_set():
                        return
        finally:
            if logging_on:
                try:
                    stream.stop_log_raw_data()
                except Exception:                                # noqa: BLE001
                    log.exception("stop_log_raw_data failed.")
            del it, stream, device
            gc.collect()

    def velocity(self, f, T=None, n=None, window=48, step=24, roi=None,
                 trigger='auto', duty_cycle=0.8, validate=True, val_threshold=2.0,
                 val_epsilon=0.1, triple=False, subpixel=True, gpu=False,
                 record=None):
        """
        Live rt-EBIV (generator of VelocityField).

        Each field is computed synchronously in YOUR loop: nothing is dropped,
        so if the loop body is slower than 1/f the acquisition falls behind
        (field.lag_s grows, and a warning is logged at 0.5 s).  For a control
        loop that must stay current, use start() / latest() instead.

        window, step   interrogation window and node distance, px.
        roi            [x0, x1, y0, y1]; default self.roi, else full sensor.
        validate       universal outlier detection (Westerweel & Scarano).
        triple         three-frame correlation (as PIV_TRIPLE_CORR).
        gpu            use the torch/CUDA correlator if available.
        """
        with self._busy_guard("velocity"):
            self._stop_evt.clear()
            proc = None
            buf = collections.deque(maxlen=3 if triple else 2)
            seq = 0
            t_first = None
            inner = self._frames_impl(f, None, None, trigger, duty_cycle, record, 1)
            try:
                for fr in inner:
                    if proc is None:
                        proc = _PIVProc(fr.image.shape,
                                        roi if roi is not None else self.roi,
                                        window, step, triple, subpixel, gpu,
                                        validate, val_threshold, val_epsilon, f)
                    buf.append(fr)
                    if len(buf) < buf.maxlen:
                        continue
                    if t_first is None:
                        t_first = buf[0].t_us
                    seq += 1
                    yield proc.run([b.image for b in buf], fr.t_us, seq, fr.lag_s,
                                   time.perf_counter() - fr.lag_s)
                    if n is not None and seq >= n:
                        return
                    if T is not None and (fr.t_us - t_first) >= T * 1e6:
                        return
            finally:
                inner.close()           # release the camera NOW, not at garbage collection

    # ------------------------------------------------------------------
    #  BACKGROUND STREAM  (for control loops)
    # ------------------------------------------------------------------
    def start(self, f, window=48, step=24, roi=None, trigger='auto', duty_cycle=0.8,
              validate=True, val_threshold=2.0, val_epsilon=0.1, triple=False,
              subpixel=True, gpu=False, record=None, max_rate_hz=None,
              on_field=None):
        """
        Start rt-EBIV in background threads and return immediately.

        Read the newest field with latest() or wait(); stop with stop().
        Acquisition and correlation run in separate threads with a one-slot
        queue, exactly like the GUI: when the correlator is busy, the new
        pair is DROPPED rather than queued, so latest() is always current.

        max_rate_hz   cap on how often a pair is correlated (frees the GIL
                      for acquisition; see PIVConfig.rt_max_rate_hz).
        on_field      optional callback(field), called in the PIV thread.
                      Keep it short.
        """
        if not self._busy.acquire(blocking=False):
            raise RuntimeError("start: the camera is busy. Call stop() first.")
        self._stop_evt.clear()
        self._latest = _Latest()
        self._bg_error = None
        q = queue.Queue(maxsize=1)
        min_dt = (1.0 / max_rate_hz) if max_rate_hz else 0.0
        ctx = {'proc': None}
        started = threading.Event()

        def _acq():
            inner = self._frames_impl(f, None, None, trigger, duty_cycle, record, 1)
            try:
                buf = collections.deque(maxlen=3 if triple else 2)
                next_due = -1e9
                for fr in inner:
                    started.set()
                    buf.append(fr)
                    if len(buf) < buf.maxlen:
                        continue
                    t_frame = time.perf_counter() - fr.lag_s
                    if min_dt > 0.0:
                        # deadline scheduling, as ebiv_utils.stream_camera
                        if t_frame < next_due:
                            continue
                        next_due += min_dt
                        if next_due <= t_frame:
                            next_due = t_frame + min_dt
                    if ctx['proc'] is None:
                        ctx['proc'] = _PIVProc(fr.image.shape,
                                               roi if roi is not None else self.roi,
                                               window, step, triple, subpixel, gpu,
                                               validate, val_threshold, val_epsilon, f)
                    try:
                        q.put_nowait(([b.image for b in buf], fr.t_us, fr.lag_s, t_frame))
                    except queue.Full:
                        pass
            except Exception as exc:                            # noqa: BLE001
                log.exception("Background acquisition failed.")
                self._bg_error = exc
            finally:
                inner.close()
                started.set()
                self._stop_evt.set()
                try:
                    q.put_nowait(None)
                except queue.Full:
                    try:
                        q.get_nowait()
                        q.put_nowait(None)
                    except Exception:                            # noqa: BLE001
                        pass

        def _piv():
            seq = 0
            while True:
                task = q.get()
                if task is None:
                    break
                imgs, t_us, lag, t_frame = task
                try:
                    seq += 1
                    fld = ctx['proc'].run(imgs, t_us, seq, lag, t_frame)
                    self._latest.publish(fld)
                    if on_field is not None:
                        on_field(fld)
                except Exception:                                # noqa: BLE001
                    log.exception("rt-EBIV failed on field %d; continuing.", seq)

        def _supervise(a, p):
            a.join()
            p.join(timeout=5.0)
            self._busy.release()

        ta = threading.Thread(target=_acq, name="vibe-acq", daemon=True)
        tp = threading.Thread(target=_piv, name="vibe-piv", daemon=True)
        ts = threading.Thread(target=_supervise, args=(ta, tp), name="vibe-sup", daemon=True)
        self._bg_threads = [ta, tp, ts]
        ta.start()
        tp.start()
        ts.start()
        started.wait(timeout=10.0)
        if self._bg_error is not None:
            err = self._bg_error
            self.stop()
            raise err
        log.info("Background rt-EBIV started at f = %g Hz (trigger=%s).", f, trigger)

    def latest(self) -> Optional[VelocityField]:
        """Newest field from start(), or None if none yet.  Never blocks."""
        self._raise_bg_error()
        return self._latest.read()

    def wait(self, timeout=1.0, newer_than=None) -> Optional[VelocityField]:
        """Block until a field newer than `newer_than` (a seq, or a field) exists."""
        self._raise_bg_error()
        if isinstance(newer_than, VelocityField):
            newer_than = newer_than.seq
        if newer_than is None:
            cur = self._latest.read()
            newer_than = cur.seq if cur is not None else 0
        return self._latest.wait_newer(newer_than, timeout)

    @property
    def running(self) -> bool:
        return any(t.is_alive() for t in self._bg_threads[:1])

    def _raise_bg_error(self):
        if self._bg_error is not None:
            err, self._bg_error = self._bg_error, None
            raise RuntimeError(f"background acquisition stopped: {err}") from err

    def stop(self) -> Optional[RecordResult]:
        """Stop a background stream and/or a background recording.  Safe to call any time."""
        self._stop_evt.set()
        for t in self._bg_threads:
            if t is not threading.current_thread():
                t.join(timeout=10.0)
        self._bg_threads = []
        res = None
        if self._rec_thread is not None:
            self._rec_thread.join(timeout=15.0)
            self._rec_thread = None
            res = self._rec_result
        return res

    # ------------------------------------------------------------------
    #  OFFLINE
    # ------------------------------------------------------------------
    def read_meta(self, raw_path) -> dict:
        """The JSON sidecar written by record(), or {} if there is none."""
        p = self._path(raw_path) + '.json'
        if os.path.exists(p):
            with open(p) as fh:
                return json.load(fh)
        return {}

    def detect_phase(self, raw_path, f, search_s=1.0) -> float:
        """Laser phase (us within a period) from the first `search_s` of a .raw file."""
        raw_path = self._path(raw_path)
        period = int(round(1e6 / f))
        it = _U.EventsIterator(input_path=raw_path, delta_t=50_000)
        ts = []
        try:
            for evs in it:
                if len(evs):
                    ts.append(np.asarray(evs['t']).copy())
                if it.get_current_time() > search_s * 1e6:
                    break
        finally:
            del it
        if not ts:
            raise ValueError(f"No events in the first {search_s} s of {raw_path}")
        peak, _, _ = _U.detect_laser_phase(np.concatenate(ts), period)
        return float(peak)

    def frames_from_raw(self, raw_path, f=None, n=None, duty_cycle=0.8, roi=None,
                        normalize='max', gaussian=None, phase_us=None, search_s=1.0):
        """
        Pseudo-frames from a .raw file, one per laser pulse (generator of Frame).

        Same algorithm as ebiv_utils.generate_centered_images (phase from a
        phase-folded histogram of the first `search_s`, windows of
        duty_cycle/f centred on each pulse), but returns arrays instead of
        writing .tif and draws no plot.

        f          defaults to the value in the recording's JSON sidecar.
        normalize  'max' (each frame / its max, as the offline pipeline) or
                   'clip' (clamp at max_events_per_pixel, as the live stream).
        roi        crop (after flips) BEFORE normalising, like the offline
                   pipeline.  None = full frame.
        gaussian   None or (ksize, sigma), e.g. ((5, 5), 1.0).
        """
        raw_path = self._path(raw_path)
        if f is None:
            f = self.read_meta(raw_path).get('f_hz')
            if f is None:
                raise ValueError("f is not given and the recording has no metadata.")
        if phase_us is None:
            phase_us = self.detect_phase(raw_path, f, search_s)
            log.info("Laser phase in %s: %.0f us.", os.path.basename(raw_path), phase_us)
        it = _U.EventsIterator(input_path=raw_path, delta_t=10_000)
        h, w = it.get_size()
        fb = _FrameBuilder(w, h, f, 'auto', duty_cycle, phase_us=phase_us)
        count = 0
        try:
            for evs in it:
                if not len(evs):
                    continue
                for t_c, counts in fb.push(evs['t'], evs['x'], evs['y']):
                    img = self._finalize(counts, mode=normalize, roi=roi, gaussian=gaussian)
                    yield Frame(t_c, img, 0.0, count)
                    count += 1
                    if n is not None and count >= n:
                        return
        finally:
            del it

    def save_frames(self, raw_path, out_dir, f=None, n=100, duty_cycle=0.8, roi=None,
                    gaussian=None, prefix="frame_"):
        """Write .tif pseudo-images, one per laser pulse (as FLAG_IMAGE_GEN).  Returns the paths."""
        import cv2
        out_dir = self._path(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        paths = []
        for fr in self.frames_from_raw(raw_path, f, n, duty_cycle, roi, 'max', gaussian):
            p = os.path.join(out_dir, f"{prefix}{fr.index:05d}.tif")
            cv2.imwrite(p, fr.image)
            paths.append(p)
        log.info("Saved %d frames to %s", len(paths), out_dir)
        return paths

    def piv(self, frames, window=48, step=24, roi=None, f=None, validate=True,
            val_threshold=2.0, val_epsilon=0.1, triple=False, subpixel=True, gpu=False):
        """
        Pairwise (or triple) FFT correlation over a sequence of frames, with
        the same engine as the live stream.  Returns a list of VelocityField.

        frames  iterable of Frame or of 2-D arrays.
        f       frequency for the field's time base; taken from Frame.t_us
                spacing if not given.
        For multi-frame pyramidal PIV on .tif folders use piv_offline().
        """
        imgs, stamps = [], []
        for k, fr in enumerate(frames):
            if isinstance(fr, Frame):
                imgs.append(fr.image)
                stamps.append(fr.t_us)
            else:
                imgs.append(np.asarray(fr))
                stamps.append(k)
        need = 3 if triple else 2
        if len(imgs) < need:
            raise ValueError(f"need at least {need} frames")
        if f is None:
            d = np.diff(stamps)
            f = 1e6 / float(np.median(d)) if len(d) and np.median(d) > 1 else 1.0
        proc = _PIVProc(imgs[0].shape, roi if roi is not None else None, window, step,
                        triple, subpixel, gpu, validate, val_threshold, val_epsilon, f)
        out = []
        for i in range(need - 1, len(imgs)):
            out.append(proc.run(imgs[i - need + 1:i + 1], int(stamps[i]), len(out) + 1))
        return out

    def piv_offline(self, input_dir, output_dir, window=64, step=32, levels=3,
                    validate=True, val_threshold=2.0, val_epsilon=0.1, gpu=False):
        """Multi-frame pyramidal PIV on a .tif folder (as FLAG_PIV_PROCESS).  Writes .mat."""
        out = self._path(output_dir)
        os.makedirs(out, exist_ok=True)
        return _U.process_offline_piv(self._path(input_dir), out, window_size=window,
                                      node_distance=step, apply_validation=validate,
                                      val_threshold=val_threshold, val_epsilon=val_epsilon,
                                      pyramid_levels=levels, use_gpu=gpu)

    def playback(self, raw_path, f=200.0):
        """Replay a .raw file in an OpenCV window ([q] to stop)."""
        return _U.check_raw_video(self._path(raw_path), accum_time_us=int(1e6 / f),
                                  max_events_per_pixel=self.max_events_per_pixel,
                                  flip_x=self.flip_x, flip_y=self.flip_y)

    # ------------------------------------------------------------------
    #  image finalisation
    # ------------------------------------------------------------------
    def _finalize(self, counts, mode='clip', roi=None, gaussian=None):
        import cv2
        img = counts
        if self.flip_x:
            img = img[:, ::-1]
        if self.flip_y:
            img = img[::-1, :]
        if roi is not None:
            x0, x1, y0, y1 = _clip_roi(roi, img.shape[1], img.shape[0])
            img = img[y0:y1, x0:x1]
        img = np.ascontiguousarray(img, dtype=np.float32)
        if mode == 'clip':
            m = float(self.max_events_per_pixel)
            np.clip(img, 0, m, out=img)
            img *= 255.0 / m
        elif mode == 'max':
            mx = float(img.max())
            if mx > 0:
                img *= 255.0 / mx
        else:
            raise ValueError("normalize must be 'clip' or 'max'")
        out = img.astype(np.uint8)
        if gaussian is None and self.smooth_sigma:
            gaussian = ((0, 0), self.smooth_sigma)
        if gaussian:
            k, s = gaussian
            out = cv2.GaussianBlur(out, tuple(k), float(s))
        return out

    def processing_settings(self):
        """The settings that shape a pseudo-frame (stored with trained models)."""
        return dict(flip_x=self.flip_x, flip_y=self.flip_y,
                    max_events_per_pixel=self.max_events_per_pixel,
                    smooth_sigma=self.smooth_sigma)

    # ==================================================================
    #  ANALOG DISCOVERY
    # ==================================================================
    @property
    def ad3_config(self) -> AD3Config:
        return self._ad3_cfg

    def ad3_open(self, **overrides):
        """
        Open the Analog Discovery (once).  Keyword overrides update the
        AD3Config fields, e.g. ad3_open(device_index=0, laser_duty_percent=5).

        The WaveForms desktop application must be CLOSED: it holds an
        exclusive lock on the device.
        """
        if overrides and self._ad3 is not None:
            raise RuntimeError("ad3_open: the device is already open; call ad3_close() "
                               "before changing its configuration.")
        for k, v in overrides.items():
            if not hasattr(self._ad3_cfg, k):
                raise AttributeError(f"AD3Config has no field {k!r}")
            setattr(self._ad3_cfg, k, v)
        self._ad3_cfg.__post_init__()
        if self.ad3_mock or self._ad3 is not None:
            return self._ad3
        self._ad3 = _HW.AnalogDiscovery3(self._ad3_cfg)
        return self._ad3

    def ad3_close(self):
        """Laser trigger and pump output to 0 V, then close the device."""
        if self._pump is not None:
            try:
                self._pump.close()
            except Exception:                                    # noqa: BLE001
                pass
        self._pump = None
        if self._ad3 is not None:
            try:
                self._ad3.close()           # parks every output at 0 V first
            finally:
                self._ad3 = None
        self._laser_running = False
        self._laser_f = None

    def laser_on(self, f=None, amplitude_v=None, offset_v=None, duty_percent=None):
        """
        Start (or retune) the laser trigger waveform.

        f            pulse frequency, Hz.  Required the first time.
        amplitude_v, offset_v   square wave swings offset +- amplitude.
        duty_percent pulse width as % of the period.

        Retuning stops ONLY the laser channel and starts it again; the pump
        channel is not addressed.
        """
        c = self._ad3_cfg
        if f is not None:
            c.laser_frequency_hz = float(f)
        if amplitude_v is not None:
            c.laser_amplitude_v = float(amplitude_v)
        if offset_v is not None:
            c.laser_offset_v = float(offset_v)
        if duty_percent is not None:
            c.laser_duty_percent = float(duty_percent)
        c.__post_init__()
        if c.laser_frequency_hz is None:
            raise ValueError("laser_on: give f (Hz) the first time.")
        if self.ad3_mock:
            log.info("[mock] laser ON at %g Hz", c.laser_frequency_hz)
            self._laser_running, self._laser_f = True, c.laser_frequency_hz
            return
        dev = self.ad3_open()
        if self._laser_running:
            self._laser_stop_hw(dev)
        dev._laser_started = False          # allow the library to start it again
        dev.start_laser()
        self._laser_running = True
        self._laser_f = c.laser_frequency_hz

    def laser_off(self):
        """Stop the laser waveform (trigger line driven to 0 V)."""
        if self.ad3_mock:
            if self._laser_running:
                log.info("[mock] laser OFF")
            self._laser_running = False
            return
        if self._ad3 is not None and self._laser_running:
            self._laser_stop_hw(self._ad3)
        self._laser_running = False

    @property
    def laser_running(self) -> bool:
        return self._laser_running

    @property
    def laser_frequency(self) -> Optional[float]:
        return self._laser_f if self._laser_running else None

    def _laser_stop_hw(self, dev):
        c = self._ad3_cfg
        if c.laser_backend == 'analog':
            dev.park_outputs_safe(channels=[int(c.laser_channel)])
        elif c.laser_backend == 'pattern':
            try:
                dev._lib.FDwfDigitalOutConfigure(dev.hdwf, c_int(0))
                log.info("Laser pattern (DIO%d) stopped.", c.laser_dio_channel)
            except Exception as exc:                             # noqa: BLE001
                log.error("Could not stop the laser pattern: %s", exc)

    def _sync_laser(self, f):
        """
        Before an acquisition at f: if VIBE is ALREADY driving the laser at a
        different frequency, retune it to f.  A laser that is not running is
        never started implicitly: switching on a laser is the user's call
        (laser_on()).
        """
        if not self._laser_running or self._ad3_cfg.laser_backend == 'external':
            return
        if self._laser_f is None or abs(self._laser_f - f) > 1e-9:
            log.info("Retuning the laser from %s to %g Hz to match the acquisition.",
                     self._laser_f, f)
            self.laser_on(f)

    def set_voltage(self, v) -> float:
        """
        Set the analog output (pump channel, AD3Config.pump_channel) to v volts.

        Clamped to [pump_v_min, pump_v_max]; returns the voltage actually
        applied.  Non-finite values are refused (AD3Error / ValueError).
        """
        if self._pump is None:
            c = self._ad3_cfg
            v0 = float(np.clip(v, c.pump_v_min, c.pump_v_max)) if np.isfinite(v) else 0.0
            if self.ad3_mock:
                self._pump = _HW.MockPump(c.pump_v_min, c.pump_v_max, v0)
            else:
                self._pump = _HW.AnalogOutPump(self.ad3_open(), c, initial_voltage=v0)
        return self._pump.set_voltage(v)

    @property
    def voltage(self) -> Optional[float]:
        """Last voltage applied by set_voltage(), or None."""
        return None if self._pump is None else self._pump.voltage

    def scope(self, channel=0, n=8192, rate_hz=1e6, v_range=10.0):
        """Single-shot capture on an AD3 analog input.  Returns (t_s, volts)."""
        if self.ad3_mock:
            raise AD3Error("scope() needs a real Analog Discovery (ad3_mock=True).")
        x, fs = self.ad3_open().capture(channel=channel, n_samples=n, rate_hz=rate_hz,
                                        v_range=v_range)
        return np.arange(len(x)) / fs, x

    def laser_check(self, scope_channel=0, n=16384, rate_hz=None):
        """
        Measure the laser trigger with the AD3's own scope (wire W1 -> 1+).
        Returns {'f_hz', 'duty_pct', 'period_std_us', 'v_low', 'v_high', ...}
        or None if no square wave is seen.  The AD3 measuring itself cannot
        see a glitch in its own clock domain; an external scope is the authority.
        """
        f = self._laser_f or self._ad3_cfg.laser_frequency_hz or 100.0
        if rate_hz is None:
            rate_hz = min(2e6, n * f / 6.0)
        t, x = self.scope(scope_channel, n, rate_hz)
        return _HW._pulse_metrics(x, rate_hz)

    def verify_hardware(self, voltages=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0), dwell_s=3.0,
                        **kw):
        """
        The FLAG_VERIFY_HARDWARE procedure (steps the pump, laser must stay
        undisturbed).  Closes VIBE's own AD3 handle first, because the
        procedure opens the device itself.
        """
        self.ad3_close()
        cfg = ControlSystemConfig()
        cfg.ad3 = self._ad3_cfg
        cfg.ad3.enabled = True
        return _HW.verify_laser_undisturbed(cfg, voltages=voltages, dwell_s=dwell_s, **kw)

    # ==================================================================
    #  CONTROL-LOOP HELPERS
    # ==================================================================
    @staticmethod
    def pid(kp=0.0, ki=0.0, kd=0.0, v_min=0.0, v_max=5.0, slew_v_per_s=2.0,
            derivative_tau_s=0.05, anti_windup='clamp', max_dt=1.0) -> PIDController:
        """
        The VibeStream PID (anti-windup, slew limit, filtered derivative on
        measurement).  u = pid.update(target, measurement, dt).u_command.
        Gains in V per velocity unit: tune from an open-loop calibration.
        """
        cfg = PIDConfig(kp=kp, ki=ki, kd=kd, v_min=v_min, v_max=v_max,
                        slew_rate_v_per_s=slew_v_per_s,
                        derivative_filter_tau_s=derivative_tau_s,
                        anti_windup=anti_windup)
        return PIDController(cfg, max_dt=max_dt)

    @staticmethod
    def roi_mean(fld: VelocityField, roi=None, component='u', min_valid=0.5,
                 cc_min=None) -> Tuple[float, bool, int]:
        """
        Mean of one component over the vectors whose centre lies in roi.

        Uses the RAW displacements and EXCLUDES rejected vectors (it does not
        average the median replacements; audit bug B4).
        component  'u', 'v', '-u', '-v' or 'mag'.
        Returns (value in px/frame, ok, n_valid); ok is False when fewer than
        min_valid of the vectors in the ROI are valid.
        """
        if roi is None:
            inside = np.ones_like(fld.x, dtype=bool)
        else:
            x0, x1, y0, y1 = roi
            inside = (fld.x >= x0) & (fld.x < x1) & (fld.y >= y0) & (fld.y < y1)
        n_in = int(inside.sum())
        if n_in == 0:
            raise ValueError(f"no vector centre inside roi {roi}")
        ok_mask = inside.copy()
        if fld.valid is not None:
            ok_mask &= fld.valid
        if cc_min is not None and fld.cc is not None:
            ok_mask &= fld.cc >= cc_min
        comp = {'u': fld.u_raw, '-u': -fld.u_raw, 'v': fld.v_raw, '-v': -fld.v_raw,
                'mag': np.hypot(fld.u_raw, fld.v_raw)}[component]
        n_ok = int(ok_mask.sum())
        if n_ok == 0:
            return float('nan'), False, 0
        return float(comp[ok_mask].mean()), n_ok >= min_valid * n_in, n_ok

    def show(self, frame, fld: Optional[VelocityField] = None, scale=4.0, skip=1,
             title="VIBE", wait_ms=1):
        """
        Draw a frame with optional vectors in an OpenCV window:
            vibe.show(fld)            a VelocityField (its own newest frame)
            vibe.show(frame)          a Frame or a 2-D array
            vibe.show(frame, fld)     both
        Returns the key pressed (255 if none).  Costs time: call it every Nth
        field in loops that must keep up with the sensor.
        """
        import cv2
        if isinstance(frame, VelocityField):
            fld, frame = frame, frame.image
        img = frame.image if isinstance(frame, Frame) else frame
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if fld is not None and fld.x.size:
            for j in range(0, fld.x.shape[0], skip):
                for i in range(0, fld.x.shape[1], skip):
                    x, y = float(fld.x[j, i]), float(fld.y[j, i])
                    ok = fld.valid is None or fld.valid[j, i]
                    cv2.arrowedLine(bgr, (int(x), int(y)),
                                    (int(x + scale * fld.u[j, i]), int(y + scale * fld.v[j, i])),
                                    (0, 255, 0) if ok else (0, 0, 255), 1, tipLength=0.3)
        cv2.imshow(title, bgr)
        return cv2.waitKey(wait_ms) & 0xFF

    # ------------------------------------------------------------------
    #  metadata
    # ------------------------------------------------------------------
    def _metadata(self, kind, f=None, **extra):
        return dict(kind=kind, created=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    f_hz=f, biases=self.biases, roi=self.roi, flip_x=self.flip_x,
                    flip_y=self.flip_y, max_events_per_pixel=self.max_events_per_pixel,
                    laser={'driven_by_vibe': self._laser_running and not self.ad3_mock,
                           'frequency_hz': self._laser_f,
                           'duty_percent': self._ad3_cfg.laser_duty_percent},
                    software=self.version(), **extra)

    @staticmethod
    def _write_json(path, data):
        try:
            with open(path, 'w') as fh:
                json.dump(data, fh, indent=2, default=_json_default)
        except Exception as exc:                                 # noqa: BLE001
            log.warning("Could not write metadata %s: %s", path, exc)


# ==========================================================================
#  PIV processor (shared by velocity(), start() and piv())
# ==========================================================================

class _PIVProc:
    def __init__(self, img_shape, roi, window, step, triple, subpixel, gpu,
                 validate, val_threshold, val_epsilon, f_hz):
        H, W = img_shape
        if roi is None:
            roi = [0, W, 0, H]
        self.x0, self.x1, self.y0, self.y1 = _clip_roi(roi, W, H)
        shape = (self.y1 - self.y0, self.x1 - self.x0)
        self.validate = bool(validate)
        self.thr, self.eps = float(val_threshold), float(val_epsilon)
        self.f_hz = float(f_hz)
        self.corr = None
        if gpu:
            if getattr(_U, '_HAS_GPU', False):
                try:
                    self.corr = _U.GPUCorrelator(window_size=window, node_distance=step,
                                                 frame_shape=shape, triple_corr=triple,
                                                 subpixel=subpixel, compute_quality=True)
                    log.info("rt-EBIV on the GPU.")
                except Exception as exc:                         # noqa: BLE001
                    log.warning("GPU correlator failed (%s); using the CPU.", exc)
            else:
                log.warning("gpu=True but torch+CUDA is not available here; using the CPU.")
        if self.corr is None:
            self.corr = CPUCorrelator(window, step, shape, triple_corr=triple,
                                      subpixel=subpixel, compute_quality=True)
        gy, gx = self.corr.grid_y, self.corr.grid_x
        xc = self.x0 + np.arange(gx) * step + window // 2
        yc = self.y0 + np.arange(gy) * step + window // 2
        self.X, self.Y = np.meshgrid(xc.astype(np.float32), yc.astype(np.float32))

    def run(self, images, t_us, seq, lag_s=0.0, t_wall=0.0) -> VelocityField:
        crop = [im[self.y0:self.y1, self.x0:self.x1] for im in images]
        U, V, CC = self.corr.correlate(crop)
        if self.validate:
            Uc, Vc, valid = _U.universal_outlier_detection(U, V, self.thr, self.eps)
            Uc, Vc = Uc.astype(np.float32), Vc.astype(np.float32)
        else:
            Uc, Vc, valid = U, V, None
        return VelocityField(self.X, self.Y, Uc, Vc, U, V, valid, CC, int(t_us),
                             self.f_hz, seq, float(lag_s), float(t_wall), images[-1])


_JIT_READY = False


def _warmup_jit():
    """Trigger numba compilation of fast_accumulate for event dtypes (uint16)."""
    global _JIT_READY
    if _JIT_READY:
        return
    buf = np.zeros((2, 2), dtype=np.float32)
    for dt in (np.uint16, np.int16, np.int32, np.int64):
        try:
            _U.fast_accumulate(buf, np.zeros(1, dt), np.zeros(1, dt))
        except Exception:                                        # noqa: BLE001
            pass
    _JIT_READY = True


def _clip_roi(roi, W, H):
    x0, x1, y0, y1 = [int(v) for v in roi]
    x0, x1 = max(0, x0), min(W, x1)
    y0, y1 = max(0, y0), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"empty ROI {roi} for a {W}x{H} image")
    return x0, x1, y0, y1


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
