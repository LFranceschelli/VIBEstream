"""
================================================================================
  EBIV — Event-Based Imaging Velocimetry, Release 5.2
  THIS IS THE ONLY FILE YOU NEED TO OPEN.
================================================================================

  How to use it:
      1. Set ONE flag in section 1 below to True.
      2. Set the parameters in sections 2-8 that the flag needs.
      3. Run:   python EBIV_Main.py

  If the laser does not come on, set FLAG_TEST_LASER = True first.  That does
  nothing but open the Analog Discovery and start the laser, so you can tell a
  hardware problem from a software one in ten seconds.

  Everything else in this folder is a library that this file drives.  You do
  not need to open any of them:

      ebiv_config.py    the parameter definitions
      ebiv_utils.py     acquisition, streaming, offline image generation, PIV
      ebiv_piv.py       the correlation engine
      ebiv_control.py   filter, reference, PID, supervisor
      ebiv_hardware.py  the Analog Discovery
      ebiv_runtime.py   the threads
      ebiv_logger.py    the experiment log
      ebiv_viz.py       the live display and the strip chart
      ebiv_session.py   glue: runs whatever the flags asked for
      ebiv_gui.py       the optional parameter window
      ebiv_profiler.py  the Release 4.0 profiler
      ebiv_gpu.py       the GPU backend (PIV_USE_GPU in section 4)
      lib/              all of the above.  You never open this folder.
      tools/            tests, benchmark, simulation.  Never needed to run an
                        experiment.

  Or just double-click VibeStream.bat, which opens the parameter window.

  READ THIS BEFORE THE FIRST HARDWARE RUN
      * Close the WaveForms desktop application.  It takes an EXCLUSIVE lock
        on the Analog Discovery, so Python cannot open the device while it is
        running.  Because Python then owns the device, Python also generates
        the laser waveform: set the LASER_* values in section 6 to whatever
        you were setting by hand in the WaveForms GUI.
      * DEVICE_INDEX: use -1 (first available).  Indices are 0-based, so with
        one device plugged in only -1 and 0 are valid.
      * Run FLAG_TEST_LASER once to confirm the laser starts.
      * Run STREAM_MODE = 'calibration' before choosing any PID gain.
================================================================================
"""

import os
import sys
import logging

# The library lives in lib/ so that this folder stays readable.  Nothing else
# changes: every module still imports its neighbours by plain name.
_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB = os.path.join(_HERE, "lib")
if os.path.isdir(_LIB) and _LIB not in sys.path:
    sys.path.insert(0, _LIB)


def _suppress_windows_dll_dialogs():
    """Turn a blocking DLL-loader message box into an ordinary error.

    The Metavision HAL plugins are C++ DLLs loaded at run time.  When one of
    them cannot be loaded, Windows puts up a MODAL dialog ("the procedure
    entry point could not be found in ...") and the process stops dead until
    somebody clicks OK.  During an unattended run that is a hang, not a
    failure, and the log says nothing.

    SEM_FAILCRITICALERRORS | SEM_NOOPENFILEERRORBOX suppresses the box.  The
    load still fails - it just fails as a normal exception that reaches the
    log, which is what we want.  No-op off Windows.
    """
    if os.name != "nt":
        return
    try:
        import ctypes
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x8000)
    except Exception:                                              # noqa: BLE001
        pass                                    # diagnostics must never break a run


_suppress_windows_dll_dialogs()

from ebiv_config import (Session, RunConfig,
                         ControlSystemConfig, ROIConfig, MeasurementConfig,
                         FilterConfig, ReferenceConfig, PIDConfig,
                         SupervisorConfig, AD3Config, CalibrationConfig,
                         LoggingConfig, PIVConfig)


# =============================================================================
#  1. WHAT DO YOU WANT TO DO?          Set the flags.
# =============================================================================

# --- the normal things -------------------------------------------------------
FLAG_STREAM       = True    # live camera.  Uses STREAM_MODE below.
FLAG_RECORD       = False   # record events from the camera to a .raw file
FLAG_PLAYBACK     = False   # replay that .raw file so you can look at it
FLAG_IMAGE_GEN    = False   # .raw  ->  phase-locked .tif frames
FLAG_PIV_PROCESS  = False   # offline pyramidal PIV on those .tif frames

# FLAG_STREAM is exclusive: it runs the live camera and nothing else.
# The other four are the offline chain; whichever are True run one after
# another, in the order listed above, in a single go.

# --- what the live stream should do -----------------------------------------
STREAM_MODE = 'calibration'
#   'ebiv'         measure and display only.  The laser STILL RUNS if the
#                  Analog Discovery is enabled in section 6.  No pump control
#                  at all: [c] [+] [-] [0] [x] [t] do NOTHING in this mode,
#                  because no pump object and no control thread are created.
#   'manual'       + drive the pump by hand with [+] and [-].  Use this to
#                  check the sign of CONTROL_COMPONENT before closing a loop.
#   'calibration'  + run the programmed voltage schedule from section 8.
#                    DO THIS BEFORE CHOOSING PID GAINS.
#   'closed_loop'  + the PID.  Starts disarmed; press [c] to close the loop.

# --- hardware checks and offline tools --------------------------------------
FLAG_TEST_LASER      = False  # just start the laser and hold it.  START HERE
                              # if the laser does not come on.
FLAG_VERIFY_HARDWARE = False  # scope procedure: step the pump, watch the laser
FLAG_SIMULATE        = False  # closed loop against a simulated pump, no HW
FLAG_SELFTEST        = False  # run the software test-suite

# --- the parameter window ----------------------------------------------------
FLAG_GUI = False              # True: ignore the flags above and open the
                              # window instead, starting from these values


# =============================================================================
#  2. ACQUISITION
# =============================================================================
F_ACQ                = 100      # Hz.  Accumulation window is 1/F_ACQ.
                                # Also the frame separation used for m/s.
MAX_EVENTS_PER_PIXEL = 1        # contrast clamp
DURATION_SEC         = 2        # length of a .raw recording

TRIGGER_MODE       = 'auto'     # 'none' | 'external' | 'auto'
TRIGGER_DUTY_CYCLE = 0.5

FLIP_X = False                  # mirror left-right.  Negates U, and puts the
FLIP_Y = False                  # ROIs below into flipped coordinates.
                                # For a pure SIGN change use CONTROL_COMPONENT
                                # = '-u' instead; it costs nothing.

CAMERA_BIASES = {
    'bias_diff_on': 60, 'bias_diff_off': 140, 'bias_hpf': 70,
    'bias_fo': 0, 'bias_refr': 90, 'bias_diff': 0,
}

# --- where everything is written --------------------------------------------
OUTPUT_BASE_FOLDER = os.path.join(os.path.expanduser("~"), "VIBEstream_data")
ACQ_NAME           = "Test"
RAW_FILENAME       = "Test.raw"
#   <base>/Raw/<raw filename>        recordings
#   <base>/RawImg/<acq name>/        generated .tif frames
#   <base>/Out/<acq name>/           PIV results, profiles, control logs


# =============================================================================
#  3. REGIONS OF INTEREST                       (sensor pixels, or None)
# =============================================================================
DISPLAY_ROI = None              # [x0, x1, y0, y1] where vectors are computed.
                                # None = full sensor.  CROPPING IS THE BIGGEST
                                # SPEED LEVER: cost scales with the number of
                                # interrogation windows.
CONTROL_ROI = [200, 800, 300, 500]              # [x0, x1, y0, y1] the part of the jet the
                                # controller uses.  Must be inside DISPLAY_ROI.
                                # None = the whole DisplayROI.


# =============================================================================
#  4. PIV
# =============================================================================
PIV_WINDOW_SIZE    = 64         # interrogation window [px]
PIV_NODE_DISTANCE  = 48         # grid spacing [px]
PIV_TRIPLE_CORR    = False
PIV_FFT_WORKERS    = 1          # 1 = Release 4.0.  Try 2-4 on a large grid.
PIV_SUBPIXEL       = True       # False reproduces R4's integer-pixel output
PIV_VALIDATION     = True       # Westerweel-Scarano outlier detection
PIV_VAL_THRESHOLD  = 2.0
PIV_VAL_EPSILON    = 0.1
PIV_PYRAMID_LEVELS = 3          # offline PIV only

PIV_USE_GPU        = False      # torch + CUDA for the correlation.  Applies
                                # to the LIVE STREAM (closed loop included)
                                # and to the offline pyramidal PIV.
                                # Falls back to the CPU with a log line if
                                # there is no GPU, so leaving it True is safe.
                                # Live: U, V and CC match the CPU to ~3e-6,
                                # so the PID sees the same measurement.  The
                                # LATENCY is not established - compare the
                                # latency summary against a CPU run before
                                # trusting gains tuned on the GPU.
                                # Offline: CC is written as NaN.
                                # Run tools/check_gpu.py once first.

# offline frame generation
N_IMAGES             = 200
IMG_PREFIX           = "frame_"
APPLY_GAUSSIAN       = True
GAUSSIAN_KERNEL      = 5        # odd
GAUSSIAN_SIGMA       = 1.0
BURST_SEARCH_MAX_SEC = 1.0


# =============================================================================
#  5. THE FEEDBACK QUANTITY AND ITS UNITS       (control modes only)
# =============================================================================
CONTROL_COMPONENT = '-u'         # 'u' = jet flows LEFT->RIGHT in the image.
                                # '-u' = right->left.  V is positive DOWNWARD,
                                # so an upward jet is '-v'.
CONTROL_STATISTIC = 'mean'      # 'mean' | 'median'

MIN_VALID_FRACTION      = 0.5   # of the ControlROI nodes
MIN_VALID_VECTORS       = 4
USE_CORRELATION_GATE    = True
MIN_CORRELATION         = 0.05
MAX_ABS_DISPLACEMENT_PX = None  # None = disabled

# Set the resolution and EVERYTHING (set-point, gains, logs, plots) is in m/s.
# Leave it None to work in px/frame and assume nothing about the optics.
UNCALIBRATED_UNITS = 'px/s'     # 'px/s' | 'px/frame'
                                # What the feedback quantity means when there
                                # is NO px/mm calibration below.
                                #   'px/s'     displacement / dt -> a VELOCITY.
                                #              Change F_ACQ and the set-point,
                                #              the measurement and the PID
                                #              gains keep the same meaning.
                                #   'px/frame' the raw displacement, as
                                #              Release 4.0-5.1 reported it. It
                                #              scales with 1/F_ACQ, so gains
                                #              tuned at one rate are wrong at
                                #              another. Use only to reproduce
                                #              an old run.
                                # With CALIBRATION_PX_PER_MM set, everything is
                                # m/s either way and this is ignored.

CALIBRATION_PX_PER_MM = 5    # e.g. 20.0
PULSE_SEPARATION_S    = None    # None = 1/F_ACQ, correct when the laser fires
                                # once per pseudo-frame


# =============================================================================
#  6. ANALOG DISCOVERY — LASER AND PUMP
# =============================================================================
AD3_ENABLED  = True      # True: Python opens the device, generates the laser
                          # waveform, and (in control modes) drives the pump.
                          # THE WAVEFORMS APPLICATION MUST BE CLOSED.
DEVICE_INDEX = -1         # -1 = first available.  0-based otherwise.

# --- laser: transcribe what you set by hand in the WaveForms GUI -------------
LASER_BACKEND      = 'analog'   # 'analog' | 'pattern' | 'external'
LASER_CHANNEL      = 0          # Analog Out ch0 = W1
LASER_DIO_CHANNEL  = 0          # only when LASER_BACKEND = 'pattern'
LASER_FREQUENCY_HZ = None       # None = FOLLOW F_ACQ.  Leave it alone and the
                                # laser fires once per pseudo-frame, so
                                # 1/f_laser IS the frame separation dt that
                                # the velocity is computed from.  Change
                                # F_ACQ and the laser follows automatically.
                                #
                                # Put a NUMBER here only if the laser really
                                # is not one pulse per frame:
                                #   - an external laser you do not drive from
                                #     the AD3 (LASER_BACKEND='external');
                                #   - pulse-pair illumination, in which case
                                #     also set PULSE_SEPARATION_S, because dt
                                #     is then NOT 1/F_ACQ.
LASER_AMPLITUDE_V  = 2.5        # HALF peak-to-peak: the wave swings
LASER_OFFSET_V     = 2.5        # OFFSET +- AMPLITUDE, so 2.5/2.5 gives 0..5 V
LASER_DUTY_PERCENT = 10.0

# --- pump: EcoDrift 4.3 auxiliary 0-10 V input, driven over 0-5 V ------------
PUMP_CHANNEL = 1                # Analog Out ch1 = W2.  Must differ from the
                                # laser channel.
PUMP_V_MIN   = 0.0
PUMP_V_MAX   = 5.0


# =============================================================================
#  7. CONTROLLER                                (control modes only)
# =============================================================================
# --- temporal filter (adds phase lag; printed in every run header) ----------
FILTER_KIND   = 'ema'           # 'none' | 'ma' | 'ema'
FILTER_WINDOW = 5               # samples, for 'ma'
FILTER_TAU_S  = 0.10            # seconds, for 'ema'

# --- set-point ---------------------------------------------------------------
REFERENCE_KIND       = 'constant'   # constant|live|step|sine|file|random
#   'live' = you set the target yourself during the run with [+] and [-].
#   Those keys move the TARGET VELOCITY and the PID finds the voltage.  That
#   is NOT the same as STREAM_MODE = 'manual', where [+]/[-] move the pump
#   voltage directly and no controller runs at all.
REFERENCE_VALUE      = 375.0        # in the display units (px/s by default,
                                    # m/s with a px/mm calibration); for
                                    # 'live', the target the run starts at.
                                    # NOTE: this was 5.0 when the units were
                                    # px/frame. 5 px/frame at F_ACQ = 75 Hz is
                                    # 375 px/s -- the same physical speed.
REF_LIVE_STEP        = 25.0         # how much one key press moves the target,
                                    # in the display units (was 0.5 px/frame)
REF_LIVE_MIN         = 0.0          # lowest target you can dial in (None = no limit)
REF_LIVE_MAX         = None         # highest target you can dial in (None = no limit)
                                    # SET THIS once you know what the pump can
                                    # actually deliver.
REF_STEP_TIME_S      = 20.0
REF_STEP_AMPLITUDE   = 2.0
REF_SINE_AMPLITUDE   = 1.5
REF_SINE_FREQ_HZ     = 0.05
REF_FILEPATH         = None         # two columns: time_s, target
REF_RANDOM_SEED      = 12345
REF_RANDOM_STD       = 1.0
REF_RANDOM_CUTOFF_HZ = 0.05

# --- PID.  Gains are VOLTS per (display unit).  CALIBRATE FIRST. ------------
PID_KP = 0.0
PID_KI = 0.0
PID_KD = 0.0
PID_V_MIN = 0.0
PID_V_MAX = 5.0
PID_SLEW_V_PER_S     = 2.0      # None disables
PID_D_ON_MEASUREMENT = True
PID_D_FILTER_TAU_S   = 0.05
PID_ANTI_WINDUP      = 'clamp'  # 'clamp'|'back_calc'|'none'
PID_INTEGRAL_LIMIT   = None

# --- rates, dropouts, safe state --------------------------------------------
CONTROL_RATE_HZ       = 10.0    # independent of the EBIV rate
VISUALIZATION_RATE_HZ = 20.0    # image window
PLOT_RATE_HZ          = 5.0     # strip chart
HOLD_TIMEOUT_S        = 0.5     # short dropout: hold the output silently
SAFE_TIMEOUT_S        = 3.0     # longer: go to the safe voltage and disarm
MAX_MEASUREMENT_AGE_S = 0.5
SAFE_PUMP_VOLTAGE     = 0.0     # DO NOT assume 0 V is right for your facility
STARTUP_PUMP_VOLTAGE  = 0.0


# =============================================================================
#  8. OPEN-LOOP CALIBRATION, LOGGING, DISPLAY
# =============================================================================
CAL_MODE             = 'steps'  # 'manual' | 'steps' | 'sweep'
CAL_VOLTAGES         = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
CAL_DWELL_S          = 30.0
CAL_SETTLE_S         = 15.0     # discarded head of each dwell
CAL_SWEEP_START      = 0.0
CAL_SWEEP_STOP       = 5.0
CAL_SWEEP_DURATION_S = 180.0
CAL_SWEEP_RETURN     = True
CAL_MANUAL_STEP_V    = 0.1

LOG_ENABLED             = True
LOG_FORMATS             = ('csv', 'mat')
LOG_FLUSH_EVERY         = 50
LATENCY_REPORT_PERIOD_S = 10.0

ARROW_SKIP          = 1         # draw one arrow every N grid nodes
ARROW_SCALE         = 4         # arrow length scaling
BLACKOUT_BACKGROUND = True      # blank the pseudo-frame behind the vectors
SHOW_STRIP_CHART    = True
PLOT_HISTORY_S      = 60.0      # how much of the past the strip chart draws
PLOT_ZOOM_WINDOW_S  = 12.0      # the y-axis is scaled on the LAST N SECONDS
                                # only.  A run that starts at 0 and settles at
                                # 0.5 m/s would otherwise keep 0 in the pane
                                # for the whole history and squash the few-
                                # percent variations you are working with.
                                # Smaller = zooms in harder and sooner.
                                # 0 = scale on everything drawn (pre-5.2).
SAVE_RT_PIV         = False     # save every RT velocity field to .mat

FLAG_PROFILE         = False    # Release 4.0 per-stage profiler
PROFILE_REPORT_EVERY = 200


# =============================================================================
#  Nothing below here needs editing.
# =============================================================================

def build_session():
    """Pack the values above into the objects the library uses."""
    control = ControlSystemConfig(
        roi=ROIConfig(display_roi=DISPLAY_ROI, control_roi=CONTROL_ROI),
        measurement=MeasurementConfig(
            component=CONTROL_COMPONENT, statistic=CONTROL_STATISTIC,
            min_valid_fraction=MIN_VALID_FRACTION,
            min_valid_vectors=MIN_VALID_VECTORS,
            use_correlation_gate=USE_CORRELATION_GATE,
            min_correlation=MIN_CORRELATION,
            max_abs_displacement_px=MAX_ABS_DISPLACEMENT_PX,
            uncalibrated_units=UNCALIBRATED_UNITS,
            calibration_px_per_mm=CALIBRATION_PX_PER_MM,
            pulse_separation_s=PULSE_SEPARATION_S),
        filt=FilterConfig(kind=FILTER_KIND, window=FILTER_WINDOW,
                          tau_s=FILTER_TAU_S),
        reference=ReferenceConfig(
            kind=REFERENCE_KIND, value=REFERENCE_VALUE,
            live_step=REF_LIVE_STEP, live_min=REF_LIVE_MIN, live_max=REF_LIVE_MAX,
            t_step_s=REF_STEP_TIME_S, step_amplitude=REF_STEP_AMPLITUDE,
            amplitude=REF_SINE_AMPLITUDE, freq_hz=REF_SINE_FREQ_HZ,
            filepath=REF_FILEPATH, random_seed=REF_RANDOM_SEED,
            random_std=REF_RANDOM_STD, random_cutoff_hz=REF_RANDOM_CUTOFF_HZ),
        pid=PIDConfig(
            kp=PID_KP, ki=PID_KI, kd=PID_KD, v_min=PID_V_MIN, v_max=PID_V_MAX,
            slew_rate_v_per_s=PID_SLEW_V_PER_S,
            derivative_on_measurement=PID_D_ON_MEASUREMENT,
            derivative_filter_tau_s=PID_D_FILTER_TAU_S,
            anti_windup=PID_ANTI_WINDUP, integral_limit=PID_INTEGRAL_LIMIT),
        supervisor=SupervisorConfig(
            control_rate_hz=CONTROL_RATE_HZ,
            visualization_rate_hz=VISUALIZATION_RATE_HZ,
            plot_rate_hz=PLOT_RATE_HZ, hold_timeout_s=HOLD_TIMEOUT_S,
            safe_timeout_s=SAFE_TIMEOUT_S,
            max_measurement_age_s=MAX_MEASUREMENT_AGE_S,
            safe_pump_voltage=SAFE_PUMP_VOLTAGE,
            startup_pump_voltage=STARTUP_PUMP_VOLTAGE),
        ad3=AD3Config(
            enabled=AD3_ENABLED, device_index=DEVICE_INDEX,
            laser_backend=LASER_BACKEND, laser_channel=LASER_CHANNEL,
            laser_dio_channel=LASER_DIO_CHANNEL,
            laser_frequency_hz=LASER_FREQUENCY_HZ,
            laser_amplitude_v=LASER_AMPLITUDE_V,
            laser_offset_v=LASER_OFFSET_V,
            laser_duty_percent=LASER_DUTY_PERCENT,
            pump_channel=PUMP_CHANNEL, pump_v_min=PUMP_V_MIN,
            pump_v_max=PUMP_V_MAX),
        calibration=CalibrationConfig(
            mode=CAL_MODE, voltages=CAL_VOLTAGES, dwell_s=CAL_DWELL_S,
            settle_s=CAL_SETTLE_S, v_start=CAL_SWEEP_START,
            v_stop=CAL_SWEEP_STOP, sweep_duration_s=CAL_SWEEP_DURATION_S,
            sweep_return=CAL_SWEEP_RETURN, manual_step_v=CAL_MANUAL_STEP_V),
        logging=LoggingConfig(
            enabled=LOG_ENABLED, formats=LOG_FORMATS,
            flush_every=LOG_FLUSH_EVERY,
            latency_report_period_s=LATENCY_REPORT_PERIOD_S),
        piv=PIVConfig(
            window_size=PIV_WINDOW_SIZE, node_distance=PIV_NODE_DISTANCE,
            triple_corr=PIV_TRIPLE_CORR, fft_workers=PIV_FFT_WORKERS,
            subpixel=PIV_SUBPIXEL, compute_quality=True,
            validation=PIV_VALIDATION, val_threshold=PIV_VAL_THRESHOLD,
            val_epsilon=PIV_VAL_EPSILON, use_gpu=PIV_USE_GPU),
    )
    run = RunConfig(
        mode='stream' if FLAG_STREAM else 'offline',
        run_mode=STREAM_MODE,
        do_record=FLAG_RECORD, do_playback=FLAG_PLAYBACK,
        do_image_gen=FLAG_IMAGE_GEN, do_piv_process=FLAG_PIV_PROCESS,
        f_acq=F_ACQ, max_events_per_pixel=MAX_EVENTS_PER_PIXEL,
        duration_sec=DURATION_SEC,
        trigger_mode=TRIGGER_MODE, trigger_duty_cycle=TRIGGER_DUTY_CYCLE,
        flip_x=FLIP_X, flip_y=FLIP_Y, camera_biases=CAMERA_BIASES,
        output_base_folder=OUTPUT_BASE_FOLDER, acq_name=ACQ_NAME,
        raw_filename=RAW_FILENAME,
        n_images=N_IMAGES, img_prefix=IMG_PREFIX,
        apply_gaussian=APPLY_GAUSSIAN, gaussian_kernel=GAUSSIAN_KERNEL,
        gaussian_sigma=GAUSSIAN_SIGMA,
        burst_search_max_sec=BURST_SEARCH_MAX_SEC,
        pyramid_levels=PIV_PYRAMID_LEVELS,
        arrow_skip=ARROW_SKIP, arrow_scale=ARROW_SCALE,
        blackout_background=BLACKOUT_BACKGROUND,
        show_strip_chart=SHOW_STRIP_CHART,
        plot_history_s=PLOT_HISTORY_S,
        plot_zoom_window_s=PLOT_ZOOM_WINDOW_S,
        profile=FLAG_PROFILE, profile_report_every=PROFILE_REPORT_EVERY,
        save_rt_piv=SAVE_RT_PIV)
    session = Session(run=run, control=control)
    # LASER_FREQUENCY_HZ = None means "follow F_ACQ".  Resolve it HERE, not
    # only in Session.validate(), because FLAG_TEST_LASER and
    # FLAG_VERIFY_HARDWARE go straight to the hardware without validating a
    # run, and the device needs a concrete number.
    session.resolve_laser_frequency()
    return session


def build_config():
    """Kept so the tools in tools/ and older scripts still work."""
    return build_session().control


def _add_tools_to_path():
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools")
    if d not in sys.path:
        sys.path.insert(0, d)
    return d


def main(force_gui=False):
    """
    force_gui : True when started by VibeStream.bat (or with --gui), which
    always opens the parameter window whatever FLAG_GUI says in this file.
    """
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    session = build_session()
    want_gui = FLAG_GUI or force_gui

    # ---- software-only tools -------------------------------------------
    if FLAG_SELFTEST and not force_gui:
        _add_tools_to_path()
        import test_release5
        raise SystemExit(test_release5.main())

    if FLAG_SIMULATE and not force_gui:
        _add_tools_to_path()
        import simulate_control
        raise SystemExit(simulate_control.main(session.control))

    # ---- hardware checks ------------------------------------------------
    if FLAG_TEST_LASER and not force_gui:
        from ebiv_hardware import quick_laser_test
        if not session.control.ad3.enabled:
            logging.error("FLAG_TEST_LASER needs AD3_ENABLED = True "
                          "(section 6).")
            return
        quick_laser_test(session.control)
        return

    if FLAG_VERIFY_HARDWARE and not force_gui:
        from ebiv_hardware import verify_laser_undisturbed
        verify_laser_undisturbed(session.control,
                                 voltages=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0),
                                 dwell_s=4.0, use_internal_scope=False)
        return

    # ---- the parameter window -------------------------------------------
    if want_gui:
        from ebiv_gui import launch
        launch(session=session)
        return

    # ---- the actual run --------------------------------------------------
    if not (FLAG_STREAM or FLAG_RECORD or FLAG_PLAYBACK or FLAG_IMAGE_GEN
            or FLAG_PIV_PROCESS):
        logging.error("Nothing to do: every flag in section 1 is False. Set "
                      "FLAG_STREAM = True for the live camera, or one of the "
                      "offline flags.")
        return

    if FLAG_STREAM and (FLAG_RECORD or FLAG_PLAYBACK or FLAG_IMAGE_GEN
                        or FLAG_PIV_PROCESS):
        logging.warning("FLAG_STREAM is True, so the live camera runs and the "
                        "offline flags are ignored this time. Set FLAG_STREAM "
                        "= False to run the offline chain.")

    from ebiv_session import run_session
    run_session(session)


if __name__ == "__main__":
    #   python EBIV_Main.py           obey the flags above
    #   python EBIV_Main.py --gui     always open the window (VibeStream.bat)
    main(force_gui=("--gui" in sys.argv))
