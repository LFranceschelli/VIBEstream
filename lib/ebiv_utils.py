"""
EBIV core library — Release 5.2.

Release 4.0's offline pipeline (image generation, pyramidal PIV, validation,
plotting) is preserved.  The live streaming path has been rewritten to add the
closed-loop controller and to fix the defects listed in AUDIT_R4_to_R5.md.

The camera SDK is imported lazily so that the control law, the PIV engine and
the whole test-suite can be imported on a machine with no Metavision
installation.
"""

import sys
import os
import glob
import time
import collections
import cv2
import numpy as np
import gc
import logging
import threading
import queue
import matplotlib.pyplot as plt
from scipy.fft import rfft2, irfft2
from scipy.io import savemat
from scipy.ndimage import median_filter
from ebiv_profiler import PipelineProfiler
from ebiv_piv import extract_windows as _extract_windows_fast

# --- Camera SDK (optional at import time) ---------------------------------
try:
    import metavision_hal as mv
    from metavision_core.event_io import EventsIterator
    _HAS_METAVISION = True
except Exception:                                              # noqa: BLE001
    mv = None
    EventsIterator = None
    _HAS_METAVISION = False

# --- Numba (optional): fast_accumulate falls back to a numpy scatter-add ---
try:
    from numba import njit
    _HAS_NUMBA = True
except Exception:                                              # noqa: BLE001
    _HAS_NUMBA = False

    def njit(*a, **k):                                          # noqa: D401
        """No-op decorator used when numba is unavailable."""
        def wrap(f):
            return f
        return wrap(a[0]) if a and callable(a[0]) else wrap

# Module-level "null" profiler — used when no profiler is passed
_NULL_PROF = PipelineProfiler(enabled=False)

# GPU backend (optional).  Release 5.0 is CPU-only by design; the import is
# kept so that Release 4.0's offline GPU path remains callable unchanged.
try:
    from ebiv_gpu import GPUCorrelator, is_gpu_available
    _HAS_GPU = is_gpu_available()
except ImportError:
    _HAS_GPU = False


class CameraSetupError(Exception):
    pass

if _HAS_NUMBA:
    @njit(cache=True)
    def fast_accumulate(frame, x, y):
        """JIT-compiled loop running at C-speed"""
        for i in range(len(x)):
            frame[y[i], x[i]] += 1.0
else:
    def fast_accumulate(frame, x, y):
        """numpy fallback when numba is unavailable (slower, same result)."""
        np.add.at(frame, (y, x), 1.0)


# =========================================================================
#  BATCH FFT CROSS-CORRELATION (vectorized over the entire grid)
# =========================================================================
def _extract_windows(frame, window_size, node_distance, grid_y, grid_x):
    """
    All interrogation windows of a frame as (N, ws, ws) float32.

    Release 5.0 change: Release 4.0 built this with a Python double loop over
    every grid node.  This delegates to ebiv_piv.extract_windows, which uses a
    strided view and one C-level copy.  The returned elements are IDENTICAL;
    test_release5.py asserts bit-exactness against the Release 4.0 loop.
    Measured 3.0-4.1x faster on this stage (AUDIT_R4_to_R5.md §3).
    """
    return _extract_windows_fast(frame, window_size, node_distance, grid_y, grid_x)


def batch_cross_correlate(frames, window_size, node_distance, prof=_NULL_PROF):
    """
    Vectorized FFT cross-correlation over the full grid.
    frames: list of 2 or 3 uint8/float32 2D arrays.
    Returns U, V displacement fields.
    """
    h, w = frames[0].shape
    grid_y = (h - window_size) // node_distance + 1
    grid_x = (w - window_size) // node_distance + 1

    if len(frames) == 3:
        with prof.measure("window_extraction"):
            W0 = _extract_windows(frames[0], window_size, node_distance, grid_y, grid_x)
            W1 = _extract_windows(frames[1], window_size, node_distance, grid_y, grid_x)
            W2 = _extract_windows(frames[2], window_size, node_distance, grid_y, grid_x)

        with prof.measure("mean_subtract"):
            W0 -= W0.mean(axis=(1, 2), keepdims=True)
            W1 -= W1.mean(axis=(1, 2), keepdims=True)
            W2 -= W2.mean(axis=(1, 2), keepdims=True)

        with prof.measure("fft_forward"):
            F0 = rfft2(W0)
            F1 = rfft2(W1)
            F2 = rfft2(W2)

        with prof.measure("cross_power"):
            cross_power = (np.conj(F0) * F1) + (np.conj(F1) * F2)
    else:
        with prof.measure("window_extraction"):
            W1 = _extract_windows(frames[0], window_size, node_distance, grid_y, grid_x)
            W2 = _extract_windows(frames[1], window_size, node_distance, grid_y, grid_x)

        with prof.measure("mean_subtract"):
            W1 -= W1.mean(axis=(1, 2), keepdims=True)
            W2 -= W2.mean(axis=(1, 2), keepdims=True)

        with prof.measure("fft_forward"):
            F1 = rfft2(W1)
            F2 = rfft2(W2)

        with prof.measure("cross_power"):
            cross_power = np.conj(F1) * F2

    with prof.measure("fft_inverse"):
        R = np.real(irfft2(cross_power))

    with prof.measure("fftshift"):
        R = np.fft.fftshift(R, axes=(1, 2))

    with prof.measure("peak_finding"):
        R_flat = R.reshape(R.shape[0], -1)
        peak_indices = np.argmax(R_flat, axis=1)
        cy_all, cx_all = np.unravel_index(peak_indices, (window_size, window_size))

        U = (cx_all - (window_size // 2)).reshape(grid_y, grid_x).astype(np.float32)
        # Image convention: V is the row displacement, positive DOWNWARD
        # (row index increases downward). This matches the .mat data and MATLAB.
        V = (cy_all - (window_size // 2)).reshape(grid_y, grid_x).astype(np.float32)

    return U, V


# =========================================================================
#  TRIGGER DETECTION — auto-detect laser pulse timing from event rate
# =========================================================================
def detect_laser_phase(event_timestamps, period_us, bin_width_us=20):
    """
    Phase-fold event timestamps at the given period and find the burst peak.
    Returns (peak_phase_us, phase_histogram, bin_centers).
    """
    num_bins = period_us // bin_width_us
    phases = event_timestamps % period_us
    counts, bin_edges = np.histogram(phases, bins=num_bins, range=(0, period_us))
    bin_centers = bin_edges[:-1] + bin_width_us / 2
    peak_idx = np.argmax(counts)
    peak_phase_us = bin_centers[peak_idx]
    return peak_phase_us, counts, bin_centers

def open_device_with_biases(biases=None):
    """Helper to open the camera and apply biases."""
    if not _HAS_METAVISION or mv is None:
        # Without this the next line is an AttributeError on None, which the
        # caller's `except CameraSetupError` does not catch and which says
        # nothing useful about the actual problem.
        raise CameraSetupError(
            "the Metavision SDK is not available in this Python environment "
            "(import of metavision_hal / metavision_core failed), so no event "
            "camera can be opened. Everything that does not need the camera — "
            "the GUI, the control law, the test-suite, the benchmark — still "
            "works.")
    devs = mv.DeviceDiscovery.list()
    if not devs:
        raise CameraSetupError("No devices found. Check connection and MV_HAL_PLUGIN_PATH.")

    device = mv.DeviceDiscovery.open(devs[0])
    if device is None:
        raise CameraSetupError("Failed to open the device.")

    if biases:
        i_ll_biases = device.get_i_ll_biases()
        if i_ll_biases is not None:
            for b_name, b_val in biases.items():
                try:
                    i_ll_biases.set(b_name, int(b_val))
                    logging.info(f"Bias set: {b_name} = {b_val}")
                except Exception as e:
                    logging.warning(f"Could not set bias {b_name}: {e}")
        else:
            logging.warning("Bias interface not available on this device.")

    return device


def stream_camera(output_dir, accum_time_us=10000, biases=None, max_events_per_pixel=1,
                  roi=None, piv_window_size=64, piv_node_distance=32, piv_triple_corr=False,
                  trigger_mode='none', trigger_period_us=None, trigger_duty_cycle=0.8,
                  prof=_NULL_PROF, use_gpu=False,
                  display_fps=25, arrow_skip=1, arrow_scale=4, save_rt_piv=False,
                  flip_x=False, flip_y=False,
                  cfg=None, run_mode='ebiv', run_name=None,
                  blackout_background=True, show_strip_chart=True,
                  plot_history_s=60.0, plot_zoom_window_s=12.0,
                  frame_smooth_sigma=None, hr=None):
    """
    Live event-camera stream with real-time EBIV and, new in Release 5.0, an
    optional closed-loop water-jet controller.

    frame_smooth_sigma : Gaussian smoothing (px) of each pseudo-frame after the
        flips, before the correlation.  None/0 = off (Release 5.2 behaviour).
    hr : optional vibe_hr_live.HRLive.  The PIV worker feeds it every LR field;
        [h] switches the display between LR and HR vectors.

    Release 4.0 parameters keep their meaning.  `roi` is the DisplayROI; if a
    ControlSystemConfig is supplied it takes precedence and also carries the
    ControlROI.

    run_mode
        'ebiv'        Release 4.0 behaviour: measure and display only.  No
                      pump backend is created and no voltage is ever issued.
        'manual'      rtEBIV plus open-loop pump control from the keyboard.
        'calibration' rtEBIV plus the programmed open-loop voltage schedule
                      in cfg.calibration (steps or sweep).  Logs the data
                      needed to build the V_command -> U_control curve.
        'closed_loop' rtEBIV plus the PID.  Starts DISARMED in MANUAL; press
                      [c] to arm.  The loop is never armed automatically.

    Keyboard
        [q] quit                [d] double exposure      [s] record .raw
        [p] RT-EBIV on/off      [v] outlier validation   [w] save .tif
        [f] save RT-PIV .mat    [b] background blackout  [g] strip chart
        [c] arm / disarm the closed loop
        [+]/[-] manual pump voltage step        [0] pump to safe voltage
        [x] EMERGENCY: force the SAFE state
    """
    from ebiv_config import ControlSystemConfig
    from ebiv_piv import CPUCorrelator
    from ebiv_control import (ControlROIEstimator, ControlSupervisor, ControlState,
                              MeasurementHolder)
    from ebiv_hardware import make_pump, open_device_and_start_laser
    from ebiv_logger import ExperimentLogger
    from ebiv_runtime import (FieldHolder, LatencyStats, AsyncFileSaver,
                              CalibrationDriver, rt_piv_worker, control_thread_main)
    import ebiv_viz as viz

    if run_mode not in ('ebiv', 'manual', 'calibration', 'closed_loop'):
        raise ValueError(f"unknown run_mode {run_mode!r}")

    # ------------------------------------------------------------------
    #  Configuration
    # ------------------------------------------------------------------
    if cfg is None:
        cfg = ControlSystemConfig()
        cfg.roi.display_roi = roi
        cfg.piv.window_size = piv_window_size
        cfg.piv.node_distance = piv_node_distance
        cfg.piv.triple_corr = piv_triple_corr
        cfg.supervisor.visualization_rate_hz = display_fps
    cfg.cross_check()

    want_gpu = bool(use_gpu) or bool(getattr(cfg.piv, 'use_gpu', False))

    control_enabled = run_mode != 'ebiv'
    # Throttle for the "that key does nothing here" notice, so a held key
    # cannot flood the log.
    _last_inert_key_msg = [0.0]
    run_name = run_name or f"{run_mode}_{time.strftime('%Y%m%d_%H%M%S')}"

    # ------------------------------------------------------------------
    #  Resolve the velocity units ONCE, before anything measures anything.
    # ------------------------------------------------------------------
    _report_velocity_units(cfg, accum_time_us, control_enabled)

    logging.info("Opening camera for live stream...")
    try:
        device = open_device_with_biases(biases)
    except CameraSetupError as e:
        logging.error(e)
        return

    i_events_stream = device.get_i_events_stream()

    # ------------------------------------------------------------------
    #  Trigger mode setup (unchanged from Release 4.0 except where noted)
    # ------------------------------------------------------------------
    use_trigger = trigger_mode in ('external', 'auto')
    if use_trigger and trigger_period_us is None:
        raise ValueError("trigger_period_us is required when trigger_mode != 'none'")

    trigger_accum_half_width_us = int(trigger_duty_cycle * trigger_period_us / 2) \
        if trigger_period_us else 0

    auto_phase_offset_us = None
    auto_calibration_done = False
    auto_calib_timestamps = []
    AUTO_CALIB_DURATION_US = 500_000
    auto_calib_start_t = None
    trigger_next_center_us = None

    # BUG FIX (B6): Release 4.0 appended every external trigger timestamp to an
    # unbounded list and then scanned the whole list with a list comprehension
    # once per frame, so both memory and per-frame cost grew without limit over
    # a long run.  A bounded deque plus a single forward scan fixes both.
    ext_trigger_times = collections.deque(maxlen=4096)
    if trigger_mode == 'external':
        i_trigger = device.get_i_trigger_in()
        if i_trigger is not None:
            def _trigger_cb(ts):
                ext_trigger_times.append(ts)
            i_trigger.add_callback(_trigger_cb)
            logging.info("External trigger callback registered.")
        else:
            logging.warning("No trigger input available on device — falling back to 'auto'.")
            trigger_mode = 'auto'

    mv_iterator = EventsIterator.from_device(device=device, delta_t=accum_time_us)
    height, width = mv_iterator.get_size()

    # ------------------------------------------------------------------
    #  ROIs and PIV engine
    # ------------------------------------------------------------------
    display_roi, control_roi = cfg.roi.validate(width, height)
    dx0, dx1, dy0, dy1 = display_roi
    dh, dw = dy1 - dy0, dx1 - dx0
    logging.info("DisplayROI %s (%dx%d px) | ControlROI %s (%dx%d px)",
                 display_roi, dw, dh, control_roi,
                 control_roi[1] - control_roi[0], control_roi[3] - control_roi[2])
    if cfg.roi.control_roi is None and control_enabled:
        logging.warning("No ControlROI was configured: the controller will use the "
                        "ENTIRE DisplayROI, including any quiescent surroundings. "
                        "Set cfg.roi.control_roi to the part of the jet you actually "
                        "want to control.")
    _report_flip(flip_x, flip_y, width, height, display_roi, control_roi,
                 cfg.measurement.component, control_enabled)

    # ------------------------------------------------------------------
    #  Correlation backend.
    #
    #  GPUCorrelator.correlate() has the same signature and the same
    #  semantics as CPUCorrelator.correlate() — sub-pixel and CC included —
    #  so rt_piv_worker, the ControlROI estimator and the PID are all
    #  indifferent to which one is in use.  The CPU stays the default: the
    #  GPU has to be asked for.
    # ------------------------------------------------------------------
    correlator = None
    if want_gpu:
        if not _HAS_GPU:
            logging.warning(
                "use_gpu was requested but torch with CUDA is not available "
                "in THIS interpreter (%s). Falling back to the CPU. Run "
                "tools/check_gpu.py with this same python to see why.",
                sys.executable)
        else:
            try:
                correlator = GPUCorrelator(
                    window_size=cfg.piv.window_size,
                    node_distance=cfg.piv.node_distance,
                    frame_shape=(dh, dw),
                    triple_corr=cfg.piv.triple_corr,
                    subpixel=cfg.piv.subpixel,
                    compute_quality=cfg.piv.compute_quality)
                logging.info("Real-time correlation backend: GPU.")
                if cfg.piv.fast_mean_removal:
                    logging.warning(
                        "fast_mean_removal has no effect on the GPU backend; "
                        "the mean is always removed in the spatial domain.")
                if control_enabled:
                    logging.warning(
                        "CLOSED-LOOP RUN ON THE GPU BACKEND. It computes the "
                        "same sub-pixel displacement and CC as the CPU (agrees "
                        "to ~3e-6), so this is not a correctness downgrade. "
                        "What is NOT established is latency: a GPU buys "
                        "throughput, and the loop cares about per-field delay "
                        "and its jitter. Watch the latency summary and compare "
                        "against a CPU run before trusting gains tuned here.")
            except Exception as exc:                           # noqa: BLE001
                logging.warning("GPU backend init failed (%s: %s). Using the CPU.",
                                type(exc).__name__, exc)
                correlator = None

    if correlator is None:
        correlator = CPUCorrelator(
            window_size=cfg.piv.window_size, node_distance=cfg.piv.node_distance,
            frame_shape=(dh, dw), triple_corr=cfg.piv.triple_corr,
            fft_workers=cfg.piv.fft_workers, subpixel=cfg.piv.subpixel,
            compute_quality=cfg.piv.compute_quality,
            fast_mean_removal=cfg.piv.fast_mean_removal)

    estimator = None
    if control_enabled:
        estimator = ControlROIEstimator(
            cfg.measurement, display_roi, control_roi,
            cfg.piv.window_size, cfg.piv.node_distance,
            correlator.grid_y, correlator.grid_x)

    # --- Pre-allocated buffers ---
    frame_buf = np.zeros((height, width), dtype=np.float32)
    prev_frame = np.zeros((height, width), dtype=np.uint8)
    prev_prev_frame = np.zeros((height, width), dtype=np.uint8)
    trigger_frame_buf = np.zeros((height, width), dtype=np.float32)

    # State toggles
    double_exposure = False
    recording_raw = False
    # control modes need PIV running; so does HR estimation, which the user
    # asked for by enabling it
    rt_piv_active = control_enabled or (hr is not None)
    rt_val_active = cfg.piv.validation
    raw_record_count = 0
    saving_tifs = False
    tif_save_count = 0
    raw_img_dir = os.path.join(output_dir, "RawImg")
    show_chart = bool(show_strip_chart) and control_enabled
    # The cross-stream profile is OFF until [u] asks for it: it is a setup
    # tool, and it costs a reduction over the whole grid on every displayed
    # frame.  It needs the estimator, so it only exists in a control mode.
    profile = None
    show_profile = False
    blackout = bool(blackout_background)

    piv_queue = queue.Queue(maxsize=1)
    field_holder = FieldHolder()
    meas_holder = MeasurementHolder()
    stats = LatencyStats(report_period_s=cfg.logging.latency_report_period_s)
    saver = AsyncFileSaver()

    # ------------------------------------------------------------------
    #  Pump, controller, logger, chart
    # ------------------------------------------------------------------
    pump = None
    supervisor = None
    logger_exp = None
    chart = None
    calibration = None
    control_stop = threading.Event()
    control_thread = None

    # ------------------------------------------------------------------
    #  The LASER first, in EVERY mode.
    #
    #  Illumination belongs to the acquisition, not to the controller: plain
    #  'ebiv' measurement needs the laser just as much as a closed-loop run
    #  does.  Release 5.0 had this whole block behind `control_enabled`, so
    #  running in 'ebiv' mode with the Analog Discovery enabled silently
    #  produced no laser at all.
    # ------------------------------------------------------------------
    ad3_device = open_device_and_start_laser(cfg)
    if cfg.ad3.enabled and ad3_device is None:
        logging.warning("Continuing WITHOUT the Analog Discovery. EBIV will "
                        "see whatever illumination you have by other means.")

    if control_enabled:
        ad3_device, pump = make_pump(cfg, device=ad3_device)
        logger_exp = ExperimentLogger(
            output_dir, run_name, cfg,
            extra_header={'run_mode': run_mode,
                          'display_roi': display_roi,
                          'control_roi': control_roi,
                          'sensor': [width, height],
                          'grid': [correlator.grid_y, correlator.grid_x],
                          'accum_time_us': accum_time_us,
                          'f_acq_hz': 1e6 / accum_time_us,
                          'control_nodes': estimator.n_total})
        supervisor = ControlSupervisor(cfg, pump, logger=logger_exp)
        if run_mode in ('manual', 'calibration'):
            # STREAM_MODE decides WHO commands the voltage; CAL_MODE only
            # describes the programme, and it applies to 'calibration' alone.
            # Without force_manual a 'manual' run would execute the CAL_MODE
            # schedule and overwrite every key press.
            calibration = CalibrationDriver(cfg.calibration,
                                            cfg.pid.v_min, cfg.pid.v_max,
                                            force_manual=(run_mode == 'manual'))
            calibration.set_manual(cfg.supervisor.startup_pump_voltage)
            if run_mode == 'manual':
                logging.info("Pump under MANUAL control: %s",
                             calibration.describe())
            else:
                logging.info("Open-loop programme: %s", calibration.describe())
        profile = viz.ProfileView(
            estimator, display_roi, cfg.piv.window_size, cfg.piv.node_distance,
            cfg.measurement.component,
            velocity_scale=cfg.measurement.velocity_scale,
            units=cfg.measurement.velocity_units)
        if show_chart:
            chart = viz.StripChart(
                history_s=plot_history_s,
                zoom_window_s=plot_zoom_window_s,
                units=cfg.measurement.velocity_units,
                v_limits=(cfg.pid.v_min, cfg.pid.v_max))

    # ------------------------------------------------------------------
    #  Threads
    # ------------------------------------------------------------------
    piv_thread = threading.Thread(
        target=rt_piv_worker,
        args=(piv_queue, field_holder, meas_holder, correlator, estimator,
              cfg, prof, logger_exp, stats, hr),
        daemon=True)
    piv_thread.start()

    if control_enabled:
        control_thread = threading.Thread(
            target=control_thread_main,
            args=(supervisor, meas_holder, cfg, logger_exp, chart,
                  control_stop, stats, calibration),
            daemon=True)
        control_thread.start()

    from ebiv_config import __version__ as _release
    logging.info("--- LIVE STREAM STARTED (Release %s, mode=%s) ---",
                 _release, run_mode)
    logging.info("Trigger mode: %s", trigger_mode)
    logging.info("Controls: [q]uit [d]bl-exp [s]record [p]RT-EBIV [v]alidation "
                 "[w]tif [f]piv-mat [b]ackground [g]raph [u]profile"
                 + (" [h]HR/LR vectors" if hr is not None else ""))
    if control_enabled:
        logging.info("Control:  [c] arm/disarm closed loop | [0] safe voltage "
                     "| [x] EMERGENCY SAFE")
        if cfg.reference.kind == 'live':
            logging.info(
                "          [+]/[-] move the TARGET VELOCITY by %g %s while the "
                "loop is closed, and the pump voltage while it is open. "
                "[t] snaps the target onto the current measurement.",
                cfg.reference.live_step, cfg.measurement.velocity_units)
        else:
            logging.info(
                "          [+]/[-] move the pump voltage while the loop is "
                "open. Reference kind is %r, so once armed the target follows "
                "its own trajectory and the keys do nothing.",
                cfg.reference.kind)

    # Display decimation
    f_acq = 1e6 / accum_time_us
    vis_rate = cfg.supervisor.visualization_rate_hz
    skip_frames = max(1, int(f_acq / max(vis_rate, 1e-6)))
    frame_count = 0
    frames_ready = 0
    logging.info("Display throttle: %.1f FPS (skip_frames=%d); EBIV runs as fast "
                 "as the correlator allows.", vis_rate, skip_frames)

    saving_rt_piv = save_rt_piv
    rt_piv_dir = os.path.join(output_dir, "RT_PIV")
    rt_piv_save_count = 0
    last_saved_seq = -1
    if saving_rt_piv:
        os.makedirs(rt_piv_dir, exist_ok=True)

    overlay = viz.VectorOverlay(
        correlator.grid_y, correlator.grid_x, cfg.piv.window_size,
        cfg.piv.node_distance, origin=(dx0, dy0),
        arrow_skip=arrow_skip, arrow_scale=arrow_scale)

    def _finalize_frame(raw_float_frame):
        """Clip, normalise to uint8, apply X/Y flip.  Returns a NEW array.

        The divide and the multiply are done IN PLACE.  The expression this
        replaces, ((raw / max) * 255).astype(uint8), allocated two full-size
        float32 temporaries per call -- 3.7 MB each on a 1280x720 sensor, at
        f_acq times a second.  Same operations in the same order, so the
        result is bit-identical (asserted in the test-suite for several
        max_events_per_pixel values); only the temporaries are gone.
        Measured saving ~0.2 ms per frame, which is real but modest -- do not
        expect it to rescue an acquisition loop on its own.

        The caller's buffer IS mutated, which is safe because every caller
        refills it before the next accumulation.  The RETURN value is a fresh
        array, which matters: prev_frame and prev_prev_frame must stay valid
        while the buffer is reused, and the PIV worker holds views into them.
        """
        np.clip(raw_float_frame, 0, max_events_per_pixel, out=raw_float_frame)
        np.divide(raw_float_frame, max_events_per_pixel, out=raw_float_frame)
        np.multiply(raw_float_frame, 255.0, out=raw_float_frame)
        out = raw_float_frame.astype(np.uint8)
        if flip_x:
            out = cv2.flip(out, 1)
        if flip_y:
            out = cv2.flip(out, 0)
        if frame_smooth_sigma:
            out = cv2.GaussianBlur(out, (0, 0), float(frame_smooth_sigma))
        return out

    if _HAS_NUMBA:
        logging.info("Event accumulator: numba JIT.")
    else:
        logging.warning(
            "Event accumulator: NUMPY FALLBACK (numba is not installed in "
            "this interpreter). It is several times slower. If the "
            "acquisition lag reported below grows, install numba: "
            "pip install numba")

    # ------------------------------------------------------------------
    #  ACQUISITION LAG
    #
    #  The camera stamps every event with its own clock.  Comparing how far
    #  that clock has advanced against how far the wall clock has advanced
    #  says whether this loop is keeping up with the sensor.  If the body of
    #  the loop costs more than delta_t, events pile up inside the Metavision
    #  SDK and every frame we build is older than the last -- without bound.
    #
    #  Nothing else in the pipeline could see this: t_frame used to be
    #  stamped when the frame was ASSEMBLED, so a frame built from events
    #  that had waited ten seconds in a buffer still looked "20 ms old" to
    #  the profiler AND to the supervisor's staleness gate.
    #
    #  It measures DRIFT from the first batch, so a constant pipeline delay
    #  present from the very first frame is not included.  Growth is the
    #  signal, and growth is what breaks a control loop.
    # ------------------------------------------------------------------
    # PIV rate cap.  See PIVConfig.rt_max_rate_hz: an unthrottled PIV worker
    # competes with this loop for the GIL, and this loop is the one that must
    # keep up with the sensor.
    _piv_rate = getattr(cfg.piv, 'rt_max_rate_hz', None)
    _piv_min_dt = (1.0 / float(_piv_rate)) if _piv_rate else 0.0
    _piv_next_due = -1e9
    if _piv_min_dt > 0.0:
        logging.info("RT-PIV rate capped at %.1f Hz (control rate is %.1f Hz); "
                     "frames in between are still accumulated, just not "
                     "correlated.", float(_piv_rate),
                     cfg.supervisor.control_rate_hz)

    trigger_resyncs = 0
    _t_stream_start = time.perf_counter()
    _n_piv_queued = 0

    acq_t0_wall = None
    acq_t0_evt = None
    acq_lag_s = 0.0
    acq_lag_ema = 0.0
    acq_lag_warned = False

    display_frame_bgr = None

    try:
        for evs in mv_iterator:
            frame_count += 1
            curr_frame = None
            curr_center_us = None      # pulse centre of curr_frame (triggered modes)

            if len(evs) > 0:
                _t_evt = float(evs['t'][-1]) * 1e-6      # camera clock, seconds
                _t_now = time.perf_counter()
                if acq_t0_wall is None:
                    acq_t0_wall, acq_t0_evt = _t_now, _t_evt
                else:
                    acq_lag_s = max(0.0, (_t_now - acq_t0_wall)
                                    - (_t_evt - acq_t0_evt))
                    acq_lag_ema += 0.05 * (acq_lag_s - acq_lag_ema)
                    if stats is not None:
                        stats.record('acquisition_lag', acq_lag_s)
                    if acq_lag_ema > 0.5 and not acq_lag_warned:
                        acq_lag_warned = True
                        logging.warning(
                            "ACQUISITION LAG %.2f s and growing: this loop is "
                            "not keeping up with the sensor, so events are "
                            "queueing inside the Metavision SDK and what you "
                            "see is that far behind reality. A controller "
                            "cannot work on this. Reduce the cost per frame: "
                            "shrink DISPLAY_ROI, lower F_ACQ, lower the "
                            "display rate, install numba, or fix the USB3 "
                            "connection.", acq_lag_ema)

            # ==========================================================
            # 1. FAST LOOP: event accumulation
            # ==========================================================
            if not use_trigger:
                with prof.measure("event_accumulation"):
                    frame_buf.fill(0)
                    if len(evs) > 0:
                        fast_accumulate(frame_buf, evs['x'], evs['y'])
                with prof.measure("frame_finalize"):
                    curr_frame = _finalize_frame(frame_buf)
            else:
                if len(evs) == 0:
                    continue
                ev_t, ev_x, ev_y = evs['t'], evs['x'], evs['y']
                chunk_t_start = int(ev_t[0])
                chunk_t_end = int(ev_t[-1])

                if trigger_mode == 'auto' and not auto_calibration_done:
                    if auto_calib_start_t is None:
                        auto_calib_start_t = chunk_t_start
                    auto_calib_timestamps.append(ev_t.copy())
                    if chunk_t_end - auto_calib_start_t >= AUTO_CALIB_DURATION_US:
                        all_ts = np.concatenate(auto_calib_timestamps)
                        peak_phase, _, _ = detect_laser_phase(all_ts, trigger_period_us)
                        auto_phase_offset_us = peak_phase
                        auto_calibration_done = True
                        trigger_next_center_us = (
                            chunk_t_end
                            - ((chunk_t_end - int(auto_phase_offset_us)) % trigger_period_us)
                            + trigger_period_us)
                        logging.info("Auto-trigger calibrated: laser phase = %.0f us. "
                                     "Next centre at t = %d us.",
                                     auto_phase_offset_us, trigger_next_center_us)
                        auto_calib_timestamps.clear()
                    with prof.measure("event_accumulation"):
                        frame_buf.fill(0)
                        fast_accumulate(frame_buf, ev_x, ev_y)
                    with prof.measure("frame_finalize"):
                        curr_frame = _finalize_frame(frame_buf)

                else:
                    if trigger_mode == 'external' and ext_trigger_times:
                        if trigger_next_center_us is None:
                            trigger_next_center_us = ext_trigger_times[-1]

                    if trigger_next_center_us is None:
                        with prof.measure("event_accumulation"):
                            frame_buf.fill(0)
                            fast_accumulate(frame_buf, ev_x, ev_y)
                        with prof.measure("frame_finalize"):
                            curr_frame = _finalize_frame(frame_buf)
                    else:
                        # ----------------------------------------------
                        #  RE-ANCHOR THE SCHEDULE IF IT HAS FALLEN BEHIND
                        #
                        #  The laser runs at a fixed rate, so the phase is
                        #  measured once and never changes.  But the schedule
                        #  is a COUNTER that advances one period per produced
                        #  frame, and an empty batch is skipped entirely
                        #  (`continue` above).  Block the light for a second --
                        #  a hand in front of the sheet -- and no events arrive,
                        #  so the counter stops while the camera clock does not.
                        #  When events return, the counter is hundreds of
                        #  periods in the past and used to walk forward one
                        #  period per BATCH, emitting garbage the whole way.
                        #
                        #  Jumping by a whole number of periods preserves the
                        #  phase exactly, which is the point: the calibration
                        #  is still valid, only the anchor was stale.
                        # ----------------------------------------------
                        if chunk_t_end - trigger_next_center_us > trigger_period_us:
                            n_skip = ((chunk_t_end - trigger_next_center_us)
                                      // trigger_period_us)
                            trigger_next_center_us += n_skip * trigger_period_us
                            trigger_resyncs += 1
                            if trigger_resyncs <= 3 or trigger_resyncs % 50 == 0:
                                logging.info(
                                    "Auto-trigger re-anchored %d period(s) "
                                    "forward (the event stream paused). The "
                                    "laser phase is unchanged; only the "
                                    "counter was stale. [%d so far]",
                                    n_skip, trigger_resyncs)

                        t_lo = trigger_next_center_us - trigger_accum_half_width_us
                        t_hi = trigger_next_center_us + trigger_accum_half_width_us
                        # Event timestamps are monotonic, so the accumulation
                        # window is a CONTIGUOUS RANGE.  searchsorted finds it
                        # with two binary searches and gives views; the boolean
                        # mask it replaces built three full-length temporaries
                        # and then two more copies through fancy indexing, on
                        # every batch.  Identical event subset: side='left'
                        # reproduces (t >= t_lo) & (t < t_hi) exactly.
                        with prof.measure("trigger_masking"):
                            lo_i = int(np.searchsorted(ev_t, t_lo, side='left'))
                            hi_i = int(np.searchsorted(ev_t, t_hi, side='left'))
                        if hi_i > lo_i:
                            with prof.measure("event_accumulation"):
                                fast_accumulate(trigger_frame_buf,
                                                ev_x[lo_i:hi_i], ev_y[lo_i:hi_i])

                        if chunk_t_end >= t_hi:
                            with prof.measure("frame_finalize"):
                                curr_frame = _finalize_frame(trigger_frame_buf)
                            curr_center_us = trigger_next_center_us
                            trigger_frame_buf.fill(0)
                            if trigger_mode == 'external' and ext_trigger_times:
                                nxt = None
                                for t_ in ext_trigger_times:
                                    if t_ > trigger_next_center_us:
                                        nxt = t_
                                        break
                                trigger_next_center_us = (
                                    nxt if nxt is not None
                                    else trigger_next_center_us + trigger_period_us)
                            else:
                                trigger_next_center_us += trigger_period_us
                        else:
                            continue

            if curr_frame is None:
                continue

            # Backdated by the acquisition lag so that t_frame means WHEN THE
            # LIGHT ARRIVED.  Everything downstream keys off this: the latency
            # statistics, and the supervisor's staleness gate.
            t_frame = time.perf_counter() - acq_lag_s
            frames_ready += 1

            # ==========================================================
            # 2. Queue a PIV pair
            # ==========================================================
            # BUG FIX (B1): Release 4.0 correlated the very first real frame
            # against a zero-filled buffer.  A correlation plane of all zeros
            # makes argmax return index 0, which maps to a displacement of
            # (-ws/2, -ws/2) -- a large, entirely spurious velocity that used
            # to be published and drawn.  Nothing is queued until enough real
            # frames have been accumulated.
            need = 3 if cfg.piv.triple_corr else 2
            need = max(need, cfg.piv.min_frames_before_output)
            # DEADLINE scheduling, not "time since the last one".  Frames
            # arrive on a 1/f_acq grid, so a >= test against the previous
            # queue time always waits for the NEXT grid point and the cap
            # silently undershoots: at 125 Hz a 20 Hz cap delivered 17.9 Hz,
            # and a 10 Hz cap fell below the control rate it was meant to
            # feed.  Advancing a deadline by exactly 1/rate keeps the AVERAGE
            # correct; individual intervals still jitter by one frame period.
            _due = (_piv_min_dt <= 0.0) or (t_frame >= _piv_next_due)
            if rt_piv_active and frames_ready >= need and _due:
                if _piv_min_dt > 0.0:
                    _piv_next_due += _piv_min_dt
                    if _piv_next_due <= t_frame:      # fell behind: resync
                        _piv_next_due = t_frame + _piv_min_dt
                with prof.measure("roi_crop"):
                    if cfg.piv.triple_corr:
                        f_piv = [prev_prev_frame[dy0:dy1, dx0:dx1],
                                 prev_frame[dy0:dy1, dx0:dx1],
                                 curr_frame[dy0:dy1, dx0:dx1]]
                    else:
                        f_piv = [prev_frame[dy0:dy1, dx0:dx1],
                                 curr_frame[dy0:dy1, dx0:dx1]]
                with prof.measure("piv_queue_put"):
                    try:
                        # frame index on the laser-period grid when the pulse
                        # time is known, so a paused stream shows up as a gap
                        f_idx = (int(round(curr_center_us / trigger_period_us))
                                 if curr_center_us is not None else frames_ready)
                        piv_queue.put_nowait((f_piv, t_frame, rt_val_active, f_idx))
                        _n_piv_queued += 1
                    except queue.Full:
                        pass    # solver busy: keep acquiring, drop this pair

            prev_prev_frame, prev_frame = prev_frame, curr_frame

            # ==========================================================
            # 3. Optional saving — off the hot path
            # ==========================================================
            if saving_tifs:
                saver.submit('tif',
                             os.path.join(raw_img_dir, f"live_frame_{tif_save_count:05d}.tif"),
                             curr_frame)
                tif_save_count += 1

            if saving_rt_piv:
                fld = field_holder.read()
                # BUG FIX (B5): only save when a NEW field exists, and never
                # call os.makedirs or savemat from the acquisition loop.
                if fld is not None and fld[4] != last_saved_seq:
                    U_, V_, M_, CC_, seq_, _ = fld
                    payload = {"U": U_, "V": V_, "M": M_,
                               "grid_x": correlator.grid_x,
                               "grid_y": correlator.grid_y,
                               "window_size": cfg.piv.window_size,
                               "node_distance": cfg.piv.node_distance,
                               "display_roi": np.asarray(display_roi),
                               "control_roi": np.asarray(control_roi),
                               "frame_index": frame_count, "field_seq": seq_}
                    if CC_ is not None:
                        payload["CC"] = CC_
                    saver.submit('mat',
                                 os.path.join(rt_piv_dir, f"rt_piv_{rt_piv_save_count:05d}.mat"),
                                 payload)
                    rt_piv_save_count += 1
                    last_saved_seq = seq_

            if prof.tick():
                prof.report(title="RT Pipeline Profile")
            stats.maybe_report(_runtime_extra_lines(supervisor, pump, correlator))

            # ==========================================================
            # 4. SLOW LOOP: UI
            # ==========================================================
            if frame_count % skip_frames:
                continue

            with prof.measure("display_render"):
                base = cv2.add(curr_frame, prev_prev_frame) if double_exposure else curr_frame
                display_frame_bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

            if rt_piv_active:
                fld = field_holder.read()
                hr_uv = (None, None)
                if hr is not None and hr.show and hr.overlay is not None:
                    hr_uv = hr.field()
                if hr_uv[0] is not None:
                    with prof.measure("arrow_drawing"):
                        if blackout:
                            display_frame_bgr[:] = 0
                        hr.overlay.draw(display_frame_bgr, hr_uv[0], hr_uv[1])
                elif fld is not None:
                    U_, V_, M_, _, _, _ = fld
                    with prof.measure("arrow_drawing"):
                        if blackout:
                            display_frame_bgr[:] = 0
                        overlay.draw(display_frame_bgr, U_, V_)
                    if profile is not None and show_profile:
                        profile.update(U_, V_)
                cv2.putText(display_frame_bgr,
                            f"RT-EBIV ACTIVE{' + VALIDATION' if rt_val_active else ''}",
                            (20, height - 52), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 2, cv2.LINE_AA)

            # DisplayROI and ControlROI outlines
            if display_roi != [0, width, 0, height]:
                cv2.rectangle(display_frame_bgr, (dx0, dy0), (dx1 - 1, dy1 - 1),
                              (110, 110, 110), 1)
                # bottom-left of the rectangle: the status panel occupies the
                # top-left of the window
                cv2.putText(display_frame_bgr, "DisplayROI",
                            (dx0 + 2, min(dy1 + 14, height - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (110, 110, 110), 1,
                            cv2.LINE_AA)
            if control_enabled:
                viz.draw_control_roi(display_frame_bgr, control_roi, offset=(0, 0))
                viz.draw_status(display_frame_bgr, supervisor,
                                units=cfg.measurement.velocity_units,
                                extra_lines=_hud_extra(run_mode, calibration, pump,
                                                       acq_lag_ema))

            # Resolution-enhancement banner, top right, always shown so it is
            # obvious whether the HR estimate is running and which vectors
            # are on screen.
            if hr is not None:
                b_txt, b_col = hr.banner()
                cv2.putText(display_frame_bgr, hr.hud(), (20, height - 156),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 1, cv2.LINE_AA)
            else:
                b_txt, b_col = "HR estimation OFF - LR vectors only", (160, 160, 160)
            (tw, th), _ = cv2.getTextSize(b_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            bx = max(0, width - tw - 24)
            cv2.rectangle(display_frame_bgr, (bx - 8, 8), (width - 8, 16 + th + 12),
                          (0, 0, 0), -1)
            cv2.rectangle(display_frame_bgr, (bx - 8, 8), (width - 8, 16 + th + 12), b_col, 2)
            cv2.putText(display_frame_bgr, b_txt, (bx, 16 + th), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, b_col, 2, cv2.LINE_AA)
            if double_exposure:
                cv2.putText(display_frame_bgr, "DOUBLE EXPOSURE", (20, height - 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if recording_raw:
                cv2.putText(display_frame_bgr, "RECORDING .RAW", (20, height - 104),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if saving_tifs:
                cv2.putText(display_frame_bgr, f"SAVING .TIF [{tif_save_count}]",
                            (20, height - 78), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 100, 0), 2)
            if saving_rt_piv:
                cv2.putText(display_frame_bgr, f"SAVING RT-PIV [{rt_piv_save_count}]",
                            (20, height - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 255), 2)
            if use_trigger:
                lbl = f"TRIGGER: {trigger_mode.upper()} | DC={trigger_duty_cycle:.0%}"
                if trigger_mode == 'auto' and not auto_calibration_done:
                    lbl += " (calibrating...)"
                cv2.putText(display_frame_bgr, lbl, (width - 420, height - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

            with prof.measure("imshow_and_waitkey"):
                cv2.imshow("EBIV Live Stream", display_frame_bgr)
                if chart is not None and show_chart:
                    chart.maybe_show(cfg.supervisor.plot_rate_hz)
                if profile is not None and show_profile:
                    profile.maybe_show(cfg.supervisor.plot_rate_hz)
                key = cv2.waitKey(1) & 0xFF

            # ==========================================================
            # 5. Keyboard
            # ==========================================================
            if key == 0xFF:
                continue
            if key == ord('q'):
                logging.info("Quitting live stream...")
                break
            elif key == ord('d'):
                double_exposure = not double_exposure
            elif key == ord('h'):
                if hr is None:
                    logging.warning("[h] shows the HR estimate: enable 'HR estimation' "
                                    "(tab HR estimation) and give a trained model.")
                else:
                    hr.show = not hr.show
                    logging.info("Display: %s vectors", "HR" if hr.show else "LR")
            elif key == ord('p'):
                rt_piv_active = not rt_piv_active
                logging.info("Real-Time PIV: %s", rt_piv_active)
                if control_enabled and not rt_piv_active:
                    logging.warning("RT-EBIV switched OFF while a controller is "
                                    "running: the measurement will go stale and "
                                    "the supervisor will HOLD, then go SAFE.")
            elif key == ord('v'):
                rt_val_active = not rt_val_active
                logging.info("RT-PIV validation: %s", rt_val_active)
            elif key == ord('b'):
                blackout = not blackout
                logging.info("Background blackout: %s", blackout)
            elif key == ord('u'):
                if profile is None:
                    logging.warning(
                        "[u] needs the ControlROI estimator, which only exists "
                        "in a control mode. Use STREAM_MODE 'manual', "
                        "'calibration' or 'closed_loop'.")
                else:
                    show_profile = not show_profile
                    if show_profile:
                        profile.reset()
                        logging.info(
                            "Cross-stream profile ON. It spans the whole "
                            "DisplayROI so you can see the shear layers on "
                            "either side of the ControlROI; press [u] again to "
                            "close it, [U] to restart the averaging.")
                    else:
                        cv2.destroyWindow(profile.window_name)
            elif key == ord('U') and profile is not None:
                profile.reset()
                logging.info("Cross-stream profile averaging restarted.")
            elif key == ord('g') and chart is not None:
                show_chart = not show_chart
                if not show_chart:
                    cv2.destroyWindow(chart.window_name)
            elif key == ord('s'):
                recording_raw = not recording_raw
                if recording_raw:
                    raw_path = os.path.join(output_dir, f"live_record_{raw_record_count:03d}.raw")
                    i_events_stream.log_raw_data(raw_path)
                    logging.info("STARTED recording raw file to: %s", raw_path)
                    raw_record_count += 1
                else:
                    i_events_stream.stop_log_raw_data()
                    logging.info("STOPPED recording raw file.")
            elif key == ord('w'):
                saving_tifs = not saving_tifs
                if saving_tifs:
                    os.makedirs(raw_img_dir, exist_ok=True)
                    logging.info("STARTED saving .tif -> %s", raw_img_dir)
                else:
                    logging.info("STOPPED saving .tif. Total: %d", tif_save_count)
            elif key == ord('f'):
                saving_rt_piv = not saving_rt_piv
                if saving_rt_piv:
                    os.makedirs(rt_piv_dir, exist_ok=True)
                    logging.info("STARTED saving RT-PIV fields -> %s", rt_piv_dir)
                else:
                    logging.info("STOPPED saving RT-PIV fields. Total: %d",
                                 rt_piv_save_count)
            elif control_enabled:
                if key == ord('c'):
                    if supervisor.state == ControlState.CLOSED_LOOP:
                        supervisor.disarm("operator pressed [c]")
                    elif supervisor.state == ControlState.SAFE:
                        logging.warning("Controller is in SAFE. Press [0] to return "
                                        "to MANUAL first, then [c] to arm.")
                    else:
                        if not np.isfinite(supervisor.u_filtered):
                            logging.warning("Refusing to arm: no valid velocity "
                                            "measurement yet.")
                        else:
                            supervisor.arm()
                elif key in (ord('+'), ord('=')):
                    _operator_bump(supervisor, cfg, calibration, +1)
                elif key in (ord('-'), ord('_')):
                    _operator_bump(supervisor, cfg, calibration, -1)
                elif key == ord('t'):
                    u = supervisor.target_to_measurement()
                    if u is None:
                        logging.warning(
                            "[t] sets the live target to the measured velocity. "
                            "It needs REFERENCE_KIND = 'live' and a valid "
                            "measurement; neither is true right now.")
                    else:
                        logging.info("Target snapped to the measurement: "
                                     "%.4g %s", u,
                                     cfg.measurement.velocity_units)
                elif key == ord('0'):
                    supervisor.state = ControlState.MANUAL
                    v = supervisor.set_manual_voltage(cfg.supervisor.safe_pump_voltage)
                    if calibration is not None:
                        calibration.set_manual(v)
                    logging.info("Pump set to the configured safe voltage %.3f V "
                                 "(MANUAL).", v)
                elif key == ord('x'):
                    supervisor.go_safe("operator pressed [x]")

            elif key in (ord('c'), ord('+'), ord('='), ord('-'), ord('_'),
                         ord('0'), ord('x'), ord('t')):
                # A controller key pressed while no controller is running.
                # Silence here is genuinely confusing: nothing moves and there
                # is no way to tell "inert by design" from "broken", which is
                # exactly how the AD3 problem presented in the first place.
                now_k = time.perf_counter()
                if now_k - _last_inert_key_msg[0] > 2.0:
                    _last_inert_key_msg[0] = now_k
                    logging.warning(
                        "[%s] does nothing: STREAM_MODE is 'ebiv', which "
                        "measures and displays but runs no controller, so no "
                        "pump object exists. The laser is unaffected. Use "
                        "'manual' to drive the pump by hand, 'calibration' "
                        "for the programmed voltage schedule, or "
                        "'closed_loop' for the PID.", chr(key))

    except KeyboardInterrupt:
        logging.warning("Interrupted by user (Ctrl-C).")
    except Exception:
        logging.exception("Unhandled exception in the acquisition loop.")
    finally:
        # ------------------------------------------------------------------
        #  Controlled shutdown.  The pump reaches its configured safe state
        #  on EVERY exit path: normal quit, exception, or Ctrl-C.
        # ------------------------------------------------------------------
        logging.info("Shutting down...")
        if control_thread is not None:
            control_stop.set()
            control_thread.join(timeout=3.0)
        if supervisor is not None:
            supervisor.shutdown("stream ended")
        try:
            piv_queue.put_nowait(None)
        except queue.Full:
            try:
                piv_queue.get_nowait()
                piv_queue.put_nowait(None)
            except Exception:
                pass
        piv_thread.join(timeout=3.0)

        if recording_raw:
            try:
                i_events_stream.stop_log_raw_data()
            except Exception:
                pass

        saver.close()
        cv2.destroyAllWindows()
        if hr is not None:
            try:
                hr.close()
            except Exception:                                  # noqa: BLE001
                pass

        if pump is not None:
            try:
                pump.close()
            except Exception:
                pass
        if ad3_device is not None:
            ad3_device.close()

        if logger_exp is not None:
            if chart is not None:
                try:
                    chart.save(os.path.join(logger_exp.dir, f"{run_name}_history.png"))
                except Exception:
                    pass
            # The live loops time themselves into `stats`; hand those numbers
            # to the logger so they reach <run>_timing.csv and the .mat.
            logger_exp.adopt_stats(stats)
            # Record one final row so the log ends with the state and the
            # voltage the hardware was actually left at, not with the last
            # control step before shutdown.
            if supervisor is not None:
                try:
                    logger_exp.log_step(supervisor, None, 0.0, None)
                except Exception:
                    pass
            logger_exp.report_latency("Closed-loop timing (whole run)")
            logger_exp.close(sup=supervisor)

        _final_summary(stats, supervisor, pump, correlator, rt_piv_save_count,
                       f_acq_hz=1e6 / accum_time_us,
                       run_s=time.perf_counter() - _t_stream_start,
                       n_piv=_n_piv_queued,
                       control_rate_hz=cfg.supervisor.control_rate_hz,
                       n_batches=frame_count)

        prof.report(title="RT Pipeline Profile (Final)")
        prof.save_csv(os.path.join(output_dir, "rt_profile.csv"))


# ==========================================================================
#  Image flip
# ==========================================================================

def unflip_roi(roi, width, height, flip_x, flip_y):
    """
    Map a rectangle given in FLIPPED-image coordinates back to raw-sensor
    coordinates.  Both are half-open [x0, x1) x [y0, y1).

    cv2.flip maps pixel column x -> width-1-x, so the interval [x0, x1)
    becomes [width-x1, width-x0).  Same for rows.
    """
    x0, x1, y0, y1 = roi
    if flip_x:
        x0, x1 = width - x1, width - x0
    if flip_y:
        y0, y1 = height - y1, height - y0
    return [int(x0), int(x1), int(y0), int(y1)]


def _report_flip(flip_x, flip_y, width, height, display_roi, control_roi,
                 component, control_enabled):
    """
    Make the flip's consequences explicit at start-up.

    Two of them bite in practice:

      * the flip is applied to the FULL sensor frame inside _finalize_frame,
        BEFORE the ROI crop, so DISPLAY_ROI and CONTROL_ROI are in
        FLIPPED-image coordinates, not raw-sensor coordinates;
      * flipping left-right negates U (and top-bottom negates V), so the
        streamwise component that the controller should use changes with it.
    """
    if not (flip_x or flip_y):
        return
    lines = ["", "-" * 70, "  Image flip ACTIVE", "-" * 70,
             f"  flip_x (left-right) : {flip_x}",
             f"  flip_y (top-bottom) : {flip_y}",
             "",
             "  The flip is applied to the full sensor frame BEFORE the ROI crop,",
             "  so DISPLAY_ROI and CONTROL_ROI are in FLIPPED-image coordinates.",
             "  The regions you configured correspond to these RAW-SENSOR regions:",
             f"    DisplayROI  {display_roi}  ->  raw "
             f"{unflip_roi(display_roi, width, height, flip_x, flip_y)}",
             f"    ControlROI  {control_roi}  ->  raw "
             f"{unflip_roi(control_roi, width, height, flip_x, flip_y)}"]
    if control_enabled:
        lines += ["",
                  f"  Sign: flip_x negates U, flip_y negates V. "
                  f"CONTROL_COMPONENT is '{component}'.",
                  "  If the feedback comes out with the wrong sign, prefer changing",
                  "  CONTROL_COMPONENT ('u' <-> '-u') over adding a flip: it costs",
                  "  nothing and leaves the ROI coordinates and the saved data alone."]
    lines.append("-" * 70)
    logging.warning("\n".join(lines))


# ==========================================================================
#  Velocity units
# ==========================================================================

def _report_velocity_units(cfg, accum_time_us, control_enabled):
    """
    Resolve px/frame -> display units and report the conversion.

    Everything downstream — set-point, PID gains, filter, limits, logs, HUD —
    is expressed in the resolved unit, so this is printed prominently rather
    than buried: a unit mistake here silently rescales the whole controller.
    """
    m = cfg.measurement
    frame_dt = accum_time_us * 1e-6
    info = m.resolve_units(frame_dt, cfg.ad3.laser_frequency_hz)

    lines = ["", "-" * 70, "  Velocity units", "-" * 70]
    lines.append(f"  frame separation dt : {info['dt_s'] * 1e6:.1f} us "
                 f"({info['f_from_dt_hz']:.2f} Hz)  [{info['dt_source']}]")

    if m.calibration_px_per_mm is None:
        lines.append(f"  calibration         : none -> working in "
                     f"{m.velocity_units}")
        if m.velocity_units == 'px/s':
            lines.append(f"  1 px/frame          = {1.0 / info['dt_s']:.4g} px/s   "
                         f"(displacement / dt)")
            lines.append("  This is a VELOCITY: changing f_acq does not change what "
                         "the")
            lines.append("  set-point, the measurement or the PID gains mean.")
        else:
            lines.append("  WARNING: px/frame scales with 1/f_acq, so a set-point "
                         "and a gain")
            lines.append("  set tuned at one acquisition rate are WRONG at another. "
                         "Set")
            lines.append("  UNCALIBRATED_UNITS = 'px/s' unless you are reproducing "
                         "an old run.")
        lines.append("  Set CALIBRATION_PX_PER_MM in EBIV_Main.py to work in m/s.")
    else:
        lines.append(f"  resolution          : {m.calibration_px_per_mm:.4g} px/mm")
        lines.append(f"  1 px/frame          = {m.velocity_scale:.6g} m/s")
        lines.append(f"  1 m/s               = {info['px_per_unit']:.4g} px/frame")
        lines.append(f"  1 m/s               = "
                     f"{info['px_per_unit'] / info['dt_s']:.4g} px/s")
        if info.get('overrode_velocity_scale'):
            lines.append("  NOTE: an explicit VELOCITY_SCALE was overridden by the "
                         "px/mm calibration.")
        lines.append("  PID gains are in V per (m/s). A controller previously tuned "
                     "in px/frame")
        lines.append(f"  needs its gains multiplied by {info['px_per_unit']:.4g} "
                     "to mean the same thing.")

    if cfg.ad3.enabled:
        lines.append("")
        if getattr(cfg.ad3, '_laser_freq_derived', False):
            lines.append(f"  laser rate          : {cfg.ad3.laser_frequency_hz:.2f} Hz "
                         f"= f_acq (locked; one pulse per pseudo-frame)")
        else:
            lines.append(f"  laser rate          : {cfg.ad3.laser_frequency_hz:.2f} Hz "
                         f"SET EXPLICITLY (f_acq is {info['f_from_dt_hz']:.2f} Hz)")
            lines.append("  It will NOT follow f_acq. Set LASER_FREQUENCY_HZ = None, or "
                         "clear the")
            lines.append("  Frequency box on the Hardware tab, to lock it to the "
                         "acquisition rate.")

    if info.get('laser_mismatch'):
        lines.append("")
        lines.append(f"  WARNING: laser_frequency_hz = {cfg.ad3.laser_frequency_hz:.2f} Hz "
                     f"but the frame separation implies {info['f_from_dt_hz']:.2f} Hz.")
        lines.append("  With one laser pulse per pseudo-frame these must match. "
                     "Either the")
        lines.append("  laser frequency or f_acq is wrong, or you are using pulse "
                     "pairs — in")
        lines.append("  which case set PULSE_SEPARATION_S explicitly to silence this.")

    # Sanity-check the set-point against the correlation's dynamic range.
    if control_enabled and cfg.reference.kind in ('constant', 'step', 'sine'):
        try:
            peak = abs(cfg.reference.value)
            if cfg.reference.kind == 'step':
                peak = max(peak, abs(cfg.reference.value + cfg.reference.step_amplitude))
            elif cfg.reference.kind == 'sine':
                peak += abs(cfg.reference.amplitude)
            px = m.to_px(peak)
            ws = cfg.piv.window_size
            lines.append("")
            lines.append(f"  peak set-point      : {peak:.4g} {m.velocity_units} "
                         f"= {px:.2f} px/frame")
            lines.append(f"  interrogation window: {ws} px "
                         f"(correlation wraps beyond +-{ws // 2} px; "
                         f"the usual working limit is ~{ws // 4} px)")
            # displacement [px] = U * dt * px_per_m, so a LARGER dt (lower
            # f_acq) gives a LARGER displacement, and vice versa.
            if px > ws / 2:
                lines.append("  ERROR-LEVEL WARNING: the set-point exceeds the "
                             "correlation range. It")
                lines.append("  CANNOT be measured; the peak will wrap and the loop "
                             "will chase noise.")
                lines.append(f"  Fix by shortening dt (RAISE f_acq above "
                             f"{info['f_from_dt_hz'] * px / (ws / 4):.0f} Hz for the "
                             f"quarter-window rule), or by")
                lines.append("  enlarging window_size, or by reducing the "
                             "magnification (fewer px/mm).")
            elif px > ws / 4:
                lines.append("  WARNING: the set-point is beyond the usual "
                             "quarter-window rule of")
                lines.append("  thumb. Correlation quality degrades with displacement; "
                             "check CC.")
                lines.append(f"  Raising f_acq to ~{info['f_from_dt_hz'] * px / (ws / 4):.0f} Hz "
                             f"would bring it back to {ws // 4} px.")
            elif px < 0.5:
                lines.append("  WARNING: the set-point is under 0.5 px/frame. "
                             "Sub-pixel noise will")
                lines.append("  dominate the feedback signal. Lengthen dt (LOWER "
                             "f_acq) or increase")
                lines.append("  the magnification so the per-pulse displacement is "
                             "larger.")
        except Exception:                                      # noqa: BLE001
            pass

    lines.append("-" * 70)
    logging.info("\n".join(lines))
    return info


# ==========================================================================
#  Small helpers for the status displays
# ==========================================================================

def _operator_bump(supervisor, cfg, calibration, sign):
    """[+] / [-].

    What these keys move depends on what is actually commandable at that
    moment, and the log line always says which one moved:

      loop CLOSED (or HOLD) and reference kind 'live'
          -> the TARGET VELOCITY.  The PID owns the voltage; asking the
             operator to also drive the voltage would be fighting it.
      loop CLOSED and any other reference kind
          -> nothing, with an explanation.  The target is following its own
             trajectory and silently overriding it would corrupt the run.
      loop OPEN (MANUAL / SAFE)
          -> the pump VOLTAGE, exactly as before.  There is no target to
             move, so the voltage is the only thing to command.
    """
    # Imported here, not at module scope: ebiv_utils keeps its control-side
    # imports local so that the acquisition half stays importable on a machine
    # with no controller dependencies.
    from ebiv_control import ControlState

    armed = supervisor.state in (ControlState.CLOSED_LOOP, ControlState.HOLD)
    if armed:
        u = supervisor.bump_target(sign)
        if u is not None:
            logging.info("Target velocity: %.4g %s", u,
                         cfg.measurement.velocity_units)
            return
        logging.warning(
            "The loop is closed and the reference kind is %r, so the target "
            "follows its own trajectory: [+]/[-] do nothing. Use "
            "REFERENCE_KIND = 'live' to steer the target by hand, or press "
            "[c] to disarm and command the voltage directly.",
            cfg.reference.kind)
        return

    v = supervisor.set_manual_voltage(
        supervisor.manual_voltage + sign * cfg.calibration.manual_step_v)
    if calibration is not None:
        calibration.set_manual(v)
    logging.info("Manual pump set-point: %.3f V", v)


def _hud_extra(run_mode, calibration, pump, acq_lag_s=None):
    lines = [f"mode {run_mode}"]
    if acq_lag_s is not None and acq_lag_s > 0.05:
        lines.append(f"ACQ LAG {acq_lag_s:.2f} s  (display is behind reality)")
    if pump is not None:
        lines.append(f"pump {getattr(pump, 'name', '?')}")
    if calibration is not None and calibration.mode != "manual":
        lines.append("open-loop programme running"
                     if not calibration.finished else "programme finished")
    return lines


def _runtime_extra_lines(supervisor, pump, correlator):
    if supervisor is None:
        return ()
    c = supervisor.counters
    line = (f"controller: state={supervisor.state} steps={c['steps']} "
            f"valid={c['valid']} hold={c['hold']} stale={c['stale']} "
            f"invalid={c['invalid']} safe={c['safe']}")
    out = [line]
    live = supervisor.live_target
    if live is not None:
        out.append(f"live target: {live.value:.4g} "
                   f"([+]/[-] step {live.step:g}, [t] = snap to measurement)")
    if pump is not None and getattr(pump, 'n_clamped', 0):
        out.append(f"pump: {pump.n_clamped} of {pump.n_calls} commands clamped "
                   f"to the hardware limits")
    return out


def _acq_verdict(rows, f_acq_hz, run_s, n_piv, control_rate_hz, n_batches=None):
    """Turn the acquisition-lag samples into a plain statement.

    The raw numbers do not say whether the loop is keeping up; the reader has
    to divide a sample count by a run duration to find out.  Say it directly,
    because 'is the acquisition falling behind' is the single question that
    decides whether closed-loop control is possible at all.
    """
    lag = next((r for r in rows if r[0] == 'acquisition_lag'), None)
    if lag is None or run_s <= 0:
        return []
    # BUG FIX: LatencyStats keeps a deque(maxlen=4000), so lag[1] SATURATES on
    # any run longer than 4000/f_acq seconds and the achieved rate came out far
    # too low -- a 33 s run at ~150 Hz reported 120 Hz, and the advice derived
    # from it was correspondingly wrong.  Use the true batch counter.
    n = n_batches if n_batches else lag[1]
    achieved = n / run_s
    out = ["  " + "-" * 70,
           f"  acquisition: {achieved:.1f} batches/s achieved vs "
           f"{f_acq_hz:.1f} Hz requested"]
    if achieved < 0.95 * f_acq_hz:
        deficit = 1.0 / achieved - 1.0 / f_acq_hz
        out.append(f"  THE LOOP IS NOT KEEPING UP: {deficit * 1e3:.1f} ms per "
                   f"batch over budget.")
        out.append(f"  Every batch it falls further behind, so the lag GROWS "
                   f"without bound")
        out.append(f"  (mean {lag[2]:.0f} ms, max {lag[5]:.0f} ms in this run). "
                   f"A controller cannot")
        out.append(f"  work on this, and the staleness gate will refuse the "
                   f"measurement.")
        suggest = f_acq_hz * achieved / f_acq_hz
        out.append(f"  Fix, in order of effect:")
        out.append(f"    - lower F_ACQ to about {achieved * 0.8:.0f} Hz "
                   f"(gives ~25% headroom)")
        if n_piv and control_rate_hz and n_piv / max(run_s, 1e-9) > 2 * control_rate_hz:
            out.append(f"    - cap the PIV rate: it ran at "
                       f"{n_piv / run_s:.1f} Hz to feed a "
                       f"{control_rate_hz:.1f} Hz controller, and it competes")
            out.append(f"      with this loop for the GIL. Try "
                       f"PIV_RT_MAX_RATE_HZ = {2.5 * control_rate_hz:.0f}")
        out.append(f"    - shrink DISPLAY_ROI")
        out.append(f"    - fix the USB3 connection")
    else:
        out.append(f"  The loop is keeping up (lag mean {lag[2]:.0f} ms, "
                   f"max {lag[5]:.0f} ms).")
    return out


def _final_summary(stats, supervisor, pump, correlator, n_saved,
                   f_acq_hz=None, run_s=None, n_piv=None, control_rate_hz=None,
                   n_batches=None):
    rows = stats.snapshot()
    from ebiv_config import __version__ as _rel
    lines = ["", "=" * 74, f"  RELEASE {_rel} RUN SUMMARY", "=" * 74]
    if rows:
        lines.append(f"  {'stage':<30s} {'n':>7s} {'mean':>8s} {'p50':>8s} "
                     f"{'p95':>8s} {'max':>8s}")
        lines.append("  " + "-" * 70)
        for k, n, m, p50, p95, mx in rows:
            lines.append(f"  {k:<30s} {n:>7d} {m:>7.2f}ms {p50:>7.2f}ms "
                         f"{p95:>7.2f}ms {mx:>7.2f}ms")
    d = next((r for r in rows if r[0] == 'control_interval'), None)
    if d:
        lines.append("  " + "-" * 70)
        lines.append(f"  effective control rate: {1000.0 / d[2]:.2f} Hz "
                     f"(nominal interval p95 {d[4]:.2f} ms, max {d[5]:.2f} ms)")
    L = next((r for r in rows if r[0] == 'latency_frame_to_actuation'), None)
    if L:
        lines.append(f"  latency of measurements the controller USED: "
                     f"mean {L[2]:.1f} ms, p95 {L[4]:.1f} ms, max {L[5]:.1f} ms "
                     f"(n={L[1]})")
    A = next((r for r in rows if r[0] == 'measurement_age_at_step'), None)
    if A:
        lines.append(f"  age of whatever it LOOKED AT, every step: "
                     f"mean {A[2]:.1f} ms, p95 {A[4]:.1f} ms, max {A[5]:.1f} ms "
                     f"(n={A[1]})")
        if L and A[2] > 3.0 * L[2]:
            lines.append("  The gap between those two is time the controller "
                         "spent with no fresh")
            lines.append("  measurement. Raise the RT-PIV rate cap, or check "
                         "why PIV output stalls.")
    if f_acq_hz:
        lines.extend(_acq_verdict(rows, f_acq_hz, run_s or 0.0,
                                  n_piv, control_rate_hz, n_batches))
    if supervisor is not None:
        lines.append(f"  controller counters: {supervisor.counters}")
        lines.append(f"  final state: {supervisor.state}, "
                     f"final command: {supervisor.v_command:.3f} V")
    if pump is not None:
        lines.append(f"  pump backend '{getattr(pump, 'name', '?')}': "
                     f"{getattr(pump, 'n_calls', 0)} commands, "
                     f"{getattr(pump, 'n_clamped', 0)} clamped, "
                     f"{getattr(pump, 'n_failures', 0)} failed")
    if n_saved:
        lines.append(f"  RT-PIV snapshots saved: {n_saved}")
    lines.append("=" * 74)
    logging.info("\n".join(lines))


def record_raw_file(duration_sec, filename="output.raw", biases=None):
    """Records raw events directly from the hardware for a set duration."""
    device = open_device_with_biases(biases)

    stream = device.get_i_events_stream()
    if stream is None:
        raise CameraSetupError("Could not get events stream interface.")

    logging.info(f"Recording to {filename} for {duration_sec} seconds...")
    stream.log_raw_data(filename)

    # We MUST iterate to pull events from the camera buffer into the file
    it = EventsIterator.from_device(device=device)

    start_time = time.time()
    for evs in it:
        if time.time() - start_time >= duration_sec:
            logging.info("Recording duration reached.")
            break

    # Stop logging
    stream.stop_log_raw_data()
    stream.stop()

    # Explicitly release resources so the camera isn't locked for the next step
    del it
    del device
    gc.collect()

    logging.info(f"Successfully saved raw data to {filename}")


def check_raw_video(filename, accum_time_us=5000, max_events_per_pixel=4,
                    flip_x=False, flip_y=False):
    """Plays back a recorded .raw file using the same enhanced contrast as the stream."""
    if not os.path.exists(filename):
        logging.error(f"Raw file not found: {filename}")
        return

    logging.info(f"Playing back {filename}... Press 'q' to stop early.")

    it = EventsIterator(input_path=filename, delta_t=accum_time_us)
    height, width = it.get_size()

    # Use the same CLAHE enhancement so playback looks identical to live stream
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    frame = np.zeros((height, width), dtype=np.float32)  # reusable buffer
    for evs in it:
        frame.fill(0)
        if len(evs) > 0:
            fast_accumulate(frame, evs['x'], evs['y'])
            np.clip(frame, 0, max_events_per_pixel, out=frame)
            frame[:] = (frame / max_events_per_pixel) * 255.0

        display_frame = frame.astype(np.uint8)
        # Apply flips before CLAHE/resize so the displayed image matches
        # what the rest of the pipeline (RT, image gen, offline PIV) sees.
        if flip_x:
            display_frame = cv2.flip(display_frame, 1)
        if flip_y:
            display_frame = cv2.flip(display_frame, 0)
        enhanced_frame = clahe.apply(display_frame)

        view_img = cv2.resize(enhanced_frame, (0, 0), fx=1.5, fy=1.5)
        cv2.imshow("Raw File Playback", view_img)

        # Calculate delay to roughly match the accumulation time (real-time playback)
        delay_ms = max(1, int(accum_time_us / 1000))
        if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
            logging.info("Playback stopped by user.")
            break

    cv2.destroyAllWindows()
    logging.info("Playback finished.")


def generate_centered_images(raw_file_path, output_dir, f_acq, Nimg, prefix="frame_",
                             apply_gaussian=False, gaussian_kernel=(5, 5), gaussian_sigma=0.0,
                             burst_search_max_sec=1.0, roi=None, duty_cycle=0.8,
                             prof=_NULL_PROF, flip_x=False, flip_y=False, flip_state=None):
    """
    Generates phase-locked event frames from a .raw recording.

    duty_cycle (0..1): fraction of the inter-pulse period to accumulate,
        centered on the detected laser burst peak.
        E.g. duty_cycle=0.8 at 200 Hz → accumulate 4000 µs out of 5000 µs,
        symmetric around the pulse.  duty_cycle=1.0 recovers the full-period
        accumulation (original behaviour).
    """
    if not os.path.exists(raw_file_path):
        logging.error(f"Raw file not found: {raw_file_path}")
        return

    period_us = int((1 / f_acq) * 1e6)  # e.g., 5000 us for 200 Hz
    bin_width_us = 20  # Sharp zoom: 20us bins to match your C++ reference
    num_bins = period_us // bin_width_us

    # Accumulation window derived from duty cycle
    accum_window_us = int(duty_cycle * period_us)
    accum_half_us = accum_window_us // 2

    # =========================================================
    # PASS 1: Phase-Folded Histogram (Find the burst peak)
    # =========================================================
    logging.info(f"Scanning first {burst_search_max_sec}s to compute phase offset...")

    hist_it = EventsIterator(input_path=raw_file_path, delta_t=50_000)
    phase_histogram = np.zeros(num_bins, dtype=np.float64)

    with prof.measure("phase_detection"):
        for evs in hist_it:
            if hist_it.get_current_time() > burst_search_max_sec * 1e6:
                break
            if len(evs) > 0:
                phases = evs['t'] % period_us
                counts, _ = np.histogram(phases, bins=num_bins, range=(0, period_us))
                phase_histogram += counts

    if np.max(phase_histogram) == 0:
        logging.error("No events found to build histogram.")
        return

    # Normalize to 1.0
    rel_histogram = phase_histogram / np.max(phase_histogram)
    bin_edges = np.linspace(0, period_us, num_bins + 1)
    bin_centers = bin_edges[:-1] + (bin_width_us / 2)

    # Find the Peak
    peak_idx = np.argmax(rel_histogram)
    peak_time_us = bin_centers[peak_idx]

    # Window start: half the accumulation window BEFORE the peak (not half period)
    phase_offset_us = int((peak_time_us - accum_half_us) % period_us)

    logging.info(f"Peak detected at: {peak_time_us:.0f} µs within period.")
    logging.info(f"Duty cycle: {duty_cycle:.0%} → accumulation window: {accum_window_us} µs "
                 f"(±{accum_half_us} µs around pulse).")
    logging.info(f"Window Start (t0) set to: {phase_offset_us} µs.")

    # Show the Phase Plot with accumulation window highlighted
    with prof.measure("histogram_plot"):
        plt.figure(figsize=(12, 4))
        plt.bar(bin_edges[:-1], rel_histogram, width=bin_width_us, align='edge', color='black')
        plt.axvline(x=peak_time_us, color='red', linestyle='-', label='Burst Peak')
        # Show the accumulation window
        win_lo = (peak_time_us - accum_half_us) % period_us
        win_hi = (peak_time_us + accum_half_us) % period_us
        plt.axvline(x=win_lo, color='green', linestyle='--', linewidth=2, label=f'Accum window ({duty_cycle:.0%})')
        plt.axvline(x=win_hi, color='green', linestyle='--', linewidth=2)
        # Shade the accumulation region
        if win_lo < win_hi:
            plt.axvspan(win_lo, win_hi, alpha=0.15, color='green')
        else:
            # Window wraps around the period boundary
            plt.axvspan(win_lo, period_us, alpha=0.15, color='green')
            plt.axvspan(0, win_hi, alpha=0.15, color='green')
        plt.title(f"Phase-Folded Zoom (Period: {period_us} µs, DC: {duty_cycle:.0%})")
        plt.xlabel("Event time modulo period [µs]")
        plt.ylabel("Event count (rel.)")
        plt.xlim(0, period_us)
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "event_histogram_centering.png"))

        plt.show(block=False)
        plt.pause(0.1)

    # =========================================================
    # PASS 2: Duty-Cycle-Aware Frame Accumulation
    # =========================================================
    # Each frame is centered on a pulse.  The grid of pulse centers advances
    # by period_us.  Only events within ±accum_half_us of each center are
    # accumulated.
    #
    # Absolute pulse center for frame i:
    #   center_i = first_center + i * period_us
    # Accumulation window:
    #   [center_i - accum_half_us,  center_i + accum_half_us)
    #
    logging.info(f"Generating {Nimg} images. Accum window: {accum_window_us} µs "
                 f"(out of {period_us} µs period).")
    if roi:
        logging.info(f"Applying ROI crop [start_x, end_x, start_y, end_y]: {roi}")

    # Flip is baked into the .tif here. Mark the shared guard so a downstream
    # offline PIV in the SAME run does not flip again (would cancel out).
    if (flip_x or flip_y):
        logging.info(f"Image flip baked into .tif: flip_x={flip_x}, flip_y={flip_y}.")
        if flip_state is not None:
            flip_state['done'] = True

    img_it = EventsIterator(input_path=raw_file_path, delta_t=10_000)
    height, width = img_it.get_size()

    frames_saved = 0
    # First pulse center: peak_time_us is the phase *within* a period.
    # We need the first absolute center that falls within the data.
    first_center = int(peak_time_us)  # first center relative to t=0
    t_center = first_center
    t_lo = t_center - accum_half_us
    t_hi = t_center + accum_half_us
    frame = np.zeros((height, width), dtype=np.float32)

    for evs in img_it:
        if len(evs) == 0:
            continue

        ev_t = evs['t']

        # Fast-forward if data jumps ahead (e.g. gaps in the recording)
        if ev_t[0] > t_hi:
            n_skip = (int(ev_t[0]) - t_center) // period_us
            if n_skip > 0:
                t_center += n_skip * period_us
                t_lo = t_center - accum_half_us
                t_hi = t_center + accum_half_us
                frame.fill(0)

        # Process events that overlap the current (or future) windows
        remaining_t = ev_t
        remaining_x = evs['x']
        remaining_y = evs['y']

        while len(remaining_t) > 0 and remaining_t[-1] >= t_hi:
            # Accumulate events in [t_lo, t_hi)
            with prof.measure("offline_event_accumulation"):
                mask_lo = remaining_t >= t_lo
                mask_hi = remaining_t < t_hi
                mask = mask_lo & mask_hi
                if np.any(mask):
                    fast_accumulate(frame, remaining_x[mask], remaining_y[mask])

            # Split: advance past t_hi
            split_idx = np.searchsorted(remaining_t, t_hi)

            # --- Save the frame ---
            with prof.measure("offline_frame_finalize"):
                if roi is not None:
                    sx, ex, sy, ey = roi
                    sx, ex = max(0, sx), min(width, ex)
                    sy, ey = max(0, sy), min(height, ey)
                    frame_crop = frame[sy:ey, sx:ex]
                else:
                    frame_crop = frame

                if frame_crop.max() > 0:
                    frame_norm = (frame_crop / frame_crop.max()) * 255.0
                else:
                    frame_norm = frame_crop

                frame_uint8 = frame_norm.astype(np.uint8)

                if apply_gaussian:
                    frame_uint8 = cv2.GaussianBlur(frame_uint8, gaussian_kernel, gaussian_sigma)

                # Apply flips so saved .tif matches the streaming/offline convention.
                if flip_x:
                    frame_uint8 = cv2.flip(frame_uint8, 1)
                if flip_y:
                    frame_uint8 = cv2.flip(frame_uint8, 0)

            out_path = os.path.join(output_dir, f"{prefix}{frames_saved:05d}.tif")
            cv2.imwrite(out_path, frame_uint8)

            frames_saved += 1
            logging.info(f"Generated snapshot {frames_saved}/{Nimg} | "
                         f"Pulse center: {t_center} µs | Window: [{t_lo}, {t_hi}) µs")

            if frames_saved >= Nimg:
                break

            # Advance to next pulse
            frame.fill(0)
            t_center += period_us
            t_lo = t_center - accum_half_us
            t_hi = t_center + accum_half_us
            remaining_t = remaining_t[split_idx:]
            remaining_x = remaining_x[split_idx:]
            remaining_y = remaining_y[split_idx:]

        if frames_saved >= Nimg:
            break

        # Accumulate leftover events that fall within the current window
        if len(remaining_t) > 0:
            mask = (remaining_t >= t_lo) & (remaining_t < t_hi)
            if np.any(mask):
                fast_accumulate(frame, remaining_x[mask], remaining_y[mask])

    logging.info(f"Successfully saved {frames_saved} images to {output_dir}/.")

    # Profiler report for offline image generation
    prof.report(title="Offline Image Generation Profile")
    prof_csv = os.path.join(output_dir, "imggen_profile.csv")
    prof.save_csv(prof_csv)


def universal_outlier_detection(U, V, threshold=2.0, epsilon=0.1):
    """
    Normalized Median Test for PIV outlier detection.
    Based on Westerweel & Scarano (2005) - Universal outlier detection for PIV data.

    NOTE: Uses scipy.ndimage.median_filter (C implementation) instead of
    generic_filter(np.median, ...) for ~3× speedup. The 3×3 median_filter
    includes the center pixel, which is a minor deviation from the strict
    8-neighbor formulation but negligible in practice for outlier detection.
    """
    # 1. Local median (3×3 window, C-compiled)
    Um = median_filter(U.astype(np.float64), size=3, mode='reflect')
    Vm = median_filter(V.astype(np.float64), size=3, mode='reflect')

    # 2. Absolute residuals of the vectors w.r.t the local median
    rU = np.abs(U - Um)
    rV = np.abs(V - Vm)

    # 3. Median of the residuals in the neighborhood
    rmU = median_filter(rU, size=3, mode='reflect')
    rmV = median_filter(rV, size=3, mode='reflect')

    # 4. Normalized residuals
    normU = rU / (rmU + epsilon)
    normV = rV / (rmV + epsilon)

    # 5. Combined magnitude of normalized residuals
    norm_total = np.sqrt(normU**2 + normV**2)

    # 6. Create mask of valid vectors
    valid_mask = norm_total < threshold

    # 7. Replace outliers with local median
    U_clean = np.where(valid_mask, U, Um)
    V_clean = np.where(valid_mask, V, Vm)

    return U_clean, V_clean, valid_mask


def subpixel_peak(R, cy, cx):
    """
    Standard 3-point Gaussian sub-pixel interpolation.
    Falls back to parabolic if values are negative or zero.
    """
    try:
        if 0 < cx < R.shape[1] - 1:
            c_left, c, c_right = R[cy, cx - 1], R[cy, cx], R[cy, cx + 1]
            if c_left > 0 and c > 0 and c_right > 0:
                dx = (np.log(c_left) - np.log(c_right)) / (2 * (np.log(c_left) - 2 * np.log(c) + np.log(c_right)))
            else:  # Parabolic fallback
                dx = (c_left - c_right) / (2 * (c_left - 2 * c + c_right))
        else:
            dx = 0.0

        if 0 < cy < R.shape[0] - 1:
            c_up, c, c_down = R[cy - 1, cx], R[cy, cx], R[cy + 1, cx]
            if c_up > 0 and c > 0 and c_down > 0:
                dy = (np.log(c_up) - np.log(c_down)) / (2 * (np.log(c_up) - 2 * np.log(c) + np.log(c_down)))
            else:
                dy = (c_up - c_down) / (2 * (c_up - 2 * c + c_down))
        else:
            dy = 0.0

    except Exception:
        dx, dy = 0.0, 0.0

    # Clamp to [-1, 1] to prevent wild jumps
    return np.clip(dx, -1.0, 1.0), np.clip(dy, -1.0, 1.0)


def batch_subpixel_peak(R_batch, cy_all, cx_all):
    """
    Vectorized 3-point Gaussian sub-pixel interpolation over a batch.
    R_batch: (N, ws, ws), cy_all/cx_all: (N,) integer peak indices.
    Returns dx, dy arrays of shape (N,).
    """
    N = R_batch.shape[0]
    ws = R_batch.shape[1]
    dx = np.zeros(N, dtype=np.float64)
    dy = np.zeros(N, dtype=np.float64)

    idx = np.arange(N)

    # --- X sub-pixel ---
    valid_x = (cx_all > 0) & (cx_all < ws - 1)
    if np.any(valid_x):
        vi = idx[valid_x]
        c_left = R_batch[vi, cy_all[vi], cx_all[vi] - 1]
        c_cent = R_batch[vi, cy_all[vi], cx_all[vi]]
        c_right = R_batch[vi, cy_all[vi], cx_all[vi] + 1]
        # Gaussian where all positive
        gauss_ok = (c_left > 0) & (c_cent > 0) & (c_right > 0)
        gi = vi[gauss_ok]
        if len(gi) > 0:
            cl = np.log(c_left[gauss_ok])
            cc = np.log(c_cent[gauss_ok])
            cr = np.log(c_right[gauss_ok])
            denom = 2.0 * (cl - 2 * cc + cr)
            safe = np.abs(denom) > 1e-12
            dx[gi[safe]] = np.clip((cl[safe] - cr[safe]) / denom[safe], -1, 1)
        # Parabolic fallback
        pi = vi[~gauss_ok]
        if len(pi) > 0:
            cl2 = c_left[~gauss_ok]
            cc2 = c_cent[~gauss_ok]
            cr2 = c_right[~gauss_ok]
            denom2 = 2.0 * (cl2 - 2 * cc2 + cr2)
            safe2 = np.abs(denom2) > 1e-12
            dx[pi[safe2]] = np.clip((cl2[safe2] - cr2[safe2]) / denom2[safe2], -1, 1)

    # --- Y sub-pixel ---
    valid_y = (cy_all > 0) & (cy_all < ws - 1)
    if np.any(valid_y):
        vi = idx[valid_y]
        c_up = R_batch[vi, cy_all[vi] - 1, cx_all[vi]]
        c_cent = R_batch[vi, cy_all[vi], cx_all[vi]]
        c_down = R_batch[vi, cy_all[vi] + 1, cx_all[vi]]
        gauss_ok = (c_up > 0) & (c_cent > 0) & (c_down > 0)
        gi = vi[gauss_ok]
        if len(gi) > 0:
            cu = np.log(c_up[gauss_ok])
            cc = np.log(c_cent[gauss_ok])
            cd = np.log(c_down[gauss_ok])
            denom = 2.0 * (cu - 2 * cc + cd)
            safe = np.abs(denom) > 1e-12
            dy[gi[safe]] = np.clip((cu[safe] - cd[safe]) / denom[safe], -1, 1)
        pi = vi[~gauss_ok]
        if len(pi) > 0:
            cu2 = c_up[~gauss_ok]
            cc2 = c_cent[~gauss_ok]
            cd2 = c_down[~gauss_ok]
            denom2 = 2.0 * (cu2 - 2 * cc2 + cd2)
            safe2 = np.abs(denom2) > 1e-12
            dy[pi[safe2]] = np.clip((cu2[safe2] - cd2[safe2]) / denom2[safe2], -1, 1)

    return dx, dy


def process_offline_piv(input_dir, output_dir, window_size=64, node_distance=32,
                        apply_validation=True, val_threshold=2.0, val_epsilon=0.1,
                        pyramid_levels=3, prof=_NULL_PROF, use_gpu=False,
                        flip_x=False, flip_y=False, flip_state=None):
    """
    Reads sequential .tif files, computes Pyramidal FFT PIV vector fields
    (dt, 2dt, 3dt...), applies homothetic scaling to the correlation planes,
    and uses sub-pixel interpolation to find the peak.

    When use_gpu=True and CUDA is available, the entire pyramidal correlation
    (FFT, cross-power, warp, peak, sub-pixel) runs on the GPU.
    """
    tif_files = sorted(glob.glob(os.path.join(input_dir, "*.tif")))

    # Need N+1 frames for N pyramid levels (e.g., levels=3 needs 4 frames: t, t+1, t+2, t+3)
    frames_needed = pyramid_levels + 1
    if len(tif_files) < frames_needed:
        logging.error(f"Need at least {frames_needed} TIFF files for {pyramid_levels} pyramid levels.")
        return

    logging.info(f"Processing PIV... Pyramidal Correlation Levels: {pyramid_levels}")

    # ------------------------------------------------------------------
    # Flip handling (shared WasFlipped guard).
    # If image generation already baked the flip into the .tif during THIS
    # run, flip_state['done'] is True and we must NOT flip again (it would
    # cancel out). If no upstream flip happened this run, we apply it here.
    # NOTE: across separate runs the guard resets, so re-processing .tif that
    # were already flipped in a previous run would double-flip — accepted
    # limitation of the in-memory guard.
    # ------------------------------------------------------------------
    already_flipped = bool(flip_state is not None and flip_state.get('done'))
    do_flip_x = flip_x and not already_flipped
    do_flip_y = flip_y and not already_flipped
    if (flip_x or flip_y) and already_flipped:
        logging.info("Offline PIV: flip already baked upstream (image-gen) this run "
                     "— skipping to avoid double-flip.")
    elif (do_flip_x or do_flip_y):
        logging.info(f"Offline PIV: applying flip on .tif read "
                     f"(flip_x={do_flip_x}, flip_y={do_flip_y}).")

    # GPU correlator (if requested and available)
    gpu_corr = None
    if use_gpu and _HAS_GPU:
        try:
            gpu_corr = GPUCorrelator(window_size=window_size, node_distance=node_distance)
            logging.info("Offline PIV: GPU engine activated.")
            logging.warning(
                "GPU pyramidal PIV: CC (peak correlation) is NOT computed by "
                "this kernel and will be saved as NaN. Anything downstream "
                "that filters on CC must use the CPU path.")
            logging.warning(
                "GPU pyramidal PIV: the homothetic scaling was wrong in "
                "Release 4.0 (audit B7/B8) and is fixed in 5.2, but the fix "
                "was verified on CPU tensors, not on CUDA. Run "
                "tools/check_gpu.py on this machine once before trusting "
                "these vectors.")
        except Exception as e:
            logging.warning(f"GPU init failed, falling back to CPU: {e}")
            gpu_corr = None
    elif use_gpu and not _HAS_GPU:
        logging.warning("use_gpu=True but no CUDA GPU detected. Falling back to CPU.")

    U_list = []
    V_list = []
    CC_list = []  # Track peak CC for each snapshot
    outliers_count = 0
    total_vectors = 0

    first_frame_path = tif_files[0]
    sample_frame = cv2.imread(first_frame_path, cv2.IMREAD_GRAYSCALE)
    h, w = sample_frame.shape

    grid_y = (h - window_size) // node_distance + 1
    grid_x = (w - window_size) // node_distance + 1

    x_centers = np.arange(grid_x) * node_distance + (window_size // 2)
    y_centers = np.arange(grid_y) * node_distance + (window_size // 2)
    X, Y = np.meshgrid(x_centers, y_centers)

    # Center coordinate of the correlation plane (CPU path only)
    center_c = window_size // 2

    # Pre-compute homothetic warp matrices for CPU path
    if gpu_corr is None:
        warp_matrices = {}
        for k in range(2, pyramid_levels + 1):
            S = 1.0 / k
            warp_matrices[k] = np.array([[S, 0, center_c * (1 - S)],
                                          [0, S, center_c * (1 - S)]], dtype=np.float32)

    N_windows = grid_y * grid_x

    # Loop with a sliding window for the full sequence
    for i in range(len(tif_files) - pyramid_levels):
        frames = [cv2.imread(tif_files[i + k], cv2.IMREAD_GRAYSCALE).astype(np.float32)
                  for k in range(pyramid_levels + 1)]

        # Apply flip on read only if not already baked upstream this run.
        if do_flip_x:
            frames = [cv2.flip(f, 1) for f in frames]
        if do_flip_y:
            frames = [cv2.flip(f, 0) for f in frames]

        # =============================================================
        # GPU PATH: entire pyramidal correlation on GPU
        # =============================================================
        if gpu_corr is not None:
            with prof.measure("gpu_pyramidal_correlation"):
                U, V = gpu_corr.pyramidal_correlate(frames, pyramid_levels, prof=prof)
            # BUG FIX (B7): GPUCorrelator.pyramidal_correlate returns only
            # (U, V), but the savemat call below writes "CC".  In Release 4.0
            # this raised NameError on the very first snapshot, so the offline
            # GPU path could not complete a single frame.  CC is filled with
            # NaN here to make the absence explicit rather than crashing or
            # silently reusing the previous frame's values.
            CC = np.full((grid_y, grid_x), np.nan, dtype=np.float32)

        # =============================================================
        # CPU PATH: original vectorized implementation
        # =============================================================
        else:
            with prof.measure("offline_window_extraction"):
                W0 = _extract_windows(frames[0], window_size, node_distance, grid_y, grid_x)
                W0 -= W0.mean(axis=(1, 2), keepdims=True)

            # Compute energy (L2 norm) of template window
            energy_W0 = np.sum(W0 ** 2, axis=(1, 2))

            with prof.measure("offline_fft_forward"):
                F0 = rfft2(W0)

            R_sum = np.zeros((N_windows, window_size, window_size), dtype=np.float32)
            energy_sum = np.zeros(N_windows, dtype=np.float32)  # Track accumulated energy

            for k in range(1, pyramid_levels + 1):
                with prof.measure("offline_window_extraction"):
                    Wk = _extract_windows(frames[k], window_size, node_distance, grid_y, grid_x)
                    Wk -= Wk.mean(axis=(1, 2), keepdims=True)

                # Compute energy of search window for normalization
                energy_Wk = np.sum(Wk ** 2, axis=(1, 2))
                energy_sum += energy_Wk

                with prof.measure("offline_fft_forward"):
                    Fk = rfft2(Wk)

                with prof.measure("offline_cross_power"):
                    cross_power = np.conj(F0) * Fk

                with prof.measure("offline_fft_inverse"):
                    R_k = np.real(irfft2(cross_power))

                with prof.measure("offline_fftshift"):
                    R_k = np.fft.fftshift(R_k, axes=(1, 2))

                if k == 1:
                    R_sum += R_k
                else:
                    with prof.measure("offline_homothetic_warp"):
                        M_warp = warp_matrices[k]
                        for n in range(N_windows):
                            R_sum[n] += cv2.warpAffine(R_k[n], M_warp,
                                                       (window_size, window_size),
                                                       flags=cv2.INTER_LINEAR)

            with prof.measure("offline_peak_finding"):
                R_flat = R_sum.reshape(N_windows, -1)
                peak_indices = np.argmax(R_flat, axis=1)
                peak_cc_raw = np.max(R_flat, axis=1)  # Raw correlation values
                cy_all, cx_all = np.unravel_index(peak_indices, (window_size, window_size))

                # Normalize by energy (L2 norm) to get correlation coefficient between 0-1
                # CC = correlation / sqrt(energy_template * energy_search)
                normalization = np.sqrt(energy_W0 * energy_sum + 1e-10)  # Small epsilon to avoid division by zero
                peak_cc = peak_cc_raw / normalization

            with prof.measure("offline_subpixel"):
                dx_all, dy_all = batch_subpixel_peak(R_sum, cy_all, cx_all)

            U = ((cx_all.astype(np.float64) + dx_all) - center_c).reshape(grid_y, grid_x)
            # Image convention: V is the row displacement, positive DOWNWARD
            # (row index increases downward). This is what gets saved to .mat and
            # is the sign MATLAB expects.
            V = ((cy_all.astype(np.float64) + dy_all) - center_c).reshape(grid_y, grid_x)
            CC = peak_cc.reshape(grid_y, grid_x)  # Reshape peak CC as a grid

        # Validation Step (always CPU — tiny field)
        if apply_validation:
            with prof.measure("offline_validation"):
                U, V, valid_mask = universal_outlier_detection(U, V, val_threshold, val_epsilon)
                outliers_count += np.sum(~valid_mask)
                total_vectors += valid_mask.size

        # Save individual snapshot
        with prof.measure("offline_mat_save"):
            snapshot_mat_path = os.path.join(output_dir, f"piv_snapshot_{i:05d}.mat")
            savemat(snapshot_mat_path, {
                "X": X, "Y": Y, "U": U, "V": V, "CC": CC,
                "window_size": window_size, "node_distance": node_distance,
                "pyramid_levels": pyramid_levels
            })

        U_list.append(U)
        V_list.append(V)
        CC_list.append(CC)

        if (i + 1) % 5 == 0 or i == len(tif_files) - pyramid_levels - 1:
            logging.info(f"Processed frame {i + 1}/{len(tif_files) - pyramid_levels}")

    # =========================================================
    # CALCULATE AVERAGES AND SAVE / PLOT
    # =========================================================
    with prof.measure("offline_averaging"):
        U_avg = np.mean(U_list, axis=0)
        V_avg = np.mean(V_list, axis=0)
        CC_avg = np.mean(CC_list, axis=0)  # Average peak CC

        mat_path_avg = os.path.join(output_dir, "offline_piv_average.mat")
        savemat(mat_path_avg, {"X": X, "Y": Y, "U_avg": U_avg, "V_avg": V_avg, "CC_avg": CC_avg})

    with prof.measure("offline_piv_plot"):
        # --- Image-convention display ----------------------------------------
        # U, V are stored in image convention: x increases to the right, y (rows)
        # increases DOWNWARD, V positive = downward motion. This is exactly what
        # is saved to .mat (so MATLAB sees the same sign).
        #
        # To draw arrows that point in the TRUE physical direction we only flip
        # the DISPLAY axis (invert_yaxis): the plot's y now increases downward,
        # matching the camera. A vector (U, V) with V>0 then renders pointing
        # down — i.e. the real motion direction. The saved data is NOT touched;
        # this is a viewer setting only.
        quiver_skip = 1
        Xq = X[::quiver_skip, ::quiver_skip]
        Yq = Y[::quiver_skip, ::quiver_skip]
        Uq = U_avg[::quiver_skip, ::quiver_skip]
        Vq = V_avg[::quiver_skip, ::quiver_skip]

        # Auto length scale: make a typical vector span ~one node spacing so the
        # arrows are visible regardless of the displacement magnitude.
        mag = np.hypot(Uq, Vq)
        typ = np.median(mag[mag > 0]) if np.any(mag > 0) else 1.0
        q_scale = max(typ, 1e-6) / float(node_distance)  # data units per arrow length

        def _overlay_quiver(ax):
            ax.quiver(Xq, Yq, Uq, Vq, color='black',
                      angles='xy', scale_units='xy', scale=q_scale,
                      width=0.002, pivot='mid')

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        c1 = ax1.pcolormesh(X, Y, U_avg, cmap='jet', shading='auto')
        _overlay_quiver(ax1)
        ax1.set_title(f"Averaged U (Pyramid Levels: {pyramid_levels})")
        ax1.set_xlabel("x [px]"); ax1.set_ylabel("y [px]")
        ax1.axis('equal'); ax1.invert_yaxis()
        fig.colorbar(c1, ax=ax1)

        c2 = ax2.pcolormesh(X, Y, V_avg, cmap='jet', shading='auto')
        _overlay_quiver(ax2)
        ax2.set_title(f"Averaged V (Pyramid Levels: {pyramid_levels})")
        ax2.set_xlabel("x [px]"); ax2.set_ylabel("y [px]")
        ax2.axis('equal'); ax2.invert_yaxis()
        fig.colorbar(c2, ax=ax2)

        c3 = ax3.pcolormesh(X, Y, CC_avg, cmap='viridis', shading='auto')
        _overlay_quiver(ax3)
        ax3.set_title("Average Peak Correlation Coefficient (CC)")
        ax3.set_xlabel("x [px]"); ax3.set_ylabel("y [px]")
        ax3.axis('equal'); ax3.invert_yaxis()
        fig.colorbar(c3, ax=ax3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "piv_pcolor_UV_CC_average.png"), dpi=300)
        # Draw without blocking so all cleanup/reporting below still runs; the
        # blocking show() at the very end keeps the window open until you close it.
        plt.draw()
        plt.pause(0.1)

    # Release GPU resources
    if gpu_corr is not None:
        gpu_corr.release()

    # Final profiler report and CSV export
    prof.report(title="Offline PIV Profile")
    prof_csv = os.path.join(output_dir, "offline_piv_profile.csv")
    prof.save_csv(prof_csv)

    # Keep the figure open until the user closes it manually. This is the last
    # statement, so GPU release, profiler report, and CSV export have all run.
    plt.show()