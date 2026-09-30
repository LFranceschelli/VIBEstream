"""
EBIV Release 5.2 — parameter GUI.

    python ebiv_gui.py            open the window
    python ebiv_gui.py preset.json

The window is deliberately plain: tkinter, which ships with every CPython and
with Anaconda, so there is no new dependency to install on the lab machine.

Structure
    1. Choose the MODE:
         Live streaming        -> the camera path, as usual, with the
                                  streaming sub-mode (EBIV only / manual /
                                  calibration / closed loop) next to it.
         Offline chain         -> record .raw, play it back, generate
                                  phase-locked frames, run the pyramidal PIV.
                                  Any subset, run in that order, in one go.
    2. Resolution enhancement (panel 2, always visible): tick 'Estimate the
       HR field live' and pick a model.  The coloured status line says, as
       you edit ANY setting, whether the live stream will show HR vectors:
         grey  OFF            green ON (model matches the live settings)
         amber ON, check the listed non-critical differences
         red   Run will refuse (no model, or trained with another
               frequency / ROI / window / step / flip / trigger mode)
         blue  the offline chain will train a model
       Menu 'Resolution enhancement': Train a model... (dialog: settings,
       progress, Stop, per-method test errors, 'Use this model live'),
       Load a model..., Model info... (trained vs current settings),
       How it works...
    3. Edit the parameters on the tabs.  Only the tabs relevant to the chosen
       mode are shown.
    4. Validate, then Run.

Every widget is bound to a field of the Session dataclasses by an explicit
path such as 'control.piv.window_size'.  There is no second copy of the
parameter list: PARAM_SPEC below is checked against the dataclasses at
start-up, so a field that gets renamed, or a spec entry that points nowhere,
is reported immediately rather than silently doing nothing.  The test-suite
runs the same check, and also reports config fields that no tab exposes.

While a run is in progress the window is hidden and the pipeline owns the
process: the live stream opens its own OpenCV windows, and the offline steps
print to the console.  The window comes back when the run ends, with the log
captured on the Log tab.  Running the pipeline in a worker thread so the GUI
could stay live is a future item; it needs care because OpenCV HighGUI
and matplotlib both want the main thread.

IMPORTS: only the standard library and ebiv_config are imported here, so the
window appears immediately.  numpy, OpenCV and the camera SDK are pulled in
by ebiv_session only when Run is pressed.
"""

import os
import sys
import json
import logging
import textwrap
import traceback
from dataclasses import is_dataclass

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

from ebiv_config import Session

APP_TITLE = "VibeStream — EBIV run configuration"


def _app_folder():
    """
    Where user-facing files belong: next to EBIV_Main.py.

    This module normally lives in lib/, so the folder the user actually looks
    at is one level up.  Without this the auto-saved preset would be written
    inside lib/, where nobody would think to look for it — or to delete it.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(here).lower() == "lib":
        return os.path.dirname(here)
    return here


LAST_PRESET = os.path.join(_app_folder(), "VibeStream_last_settings.json")

# Which tabs each mode shows, in order.
# The resolution enhancement has no tab: its live settings are in the header
# panel "2. Resolution enhancement" (always visible, with a status line), and
# its training in the dialog opened from the "Resolution enhancement" menu.
TABS_STREAM = ["Run", "Acquisition", "PIV & ROIs", "Feedback", "Controller",
               "Hardware", "Calibration", "Display", "Log"]
TABS_OFFLINE = ["Run", "Acquisition", "PIV & ROIs", "Offline", "Log"]


# ==========================================================================
#  Parameter specification
#
#  (tab, group, path, label, kind, extra, help)
#    kind: float int bool text choice folder file roi optfloat floats biases
#    extra: choices for 'choice'
# ==========================================================================

def P(tab, group, path, label, kind, extra=None, help=""):
    return dict(tab=tab, group=group, path=path, label=label, kind=kind,
                extra=extra, help=help)


PARAM_SPEC = [
    # ---------------- Run ----------------
    P("Run", "Output", "run.output_base_folder", "Output base folder", "folder",
      help="Raw/, RawImg/<acq name>/ and Out/<acq name>/ are created here."),
    P("Run", "Output", "run.acq_name", "Acquisition name", "text",
      help="Sub-folder name for this acquisition's frames and results."),
    P("Run", "Output", "run.raw_filename", "Raw file name", "text",
      help="File inside Raw/. Written by 'record', read by playback and "
           "frame generation."),
    P("Run", "Diagnostics", "run.profile", "Per-stage profiler", "bool",
      help="Release 4.0 profiler. Keeps every sample: use it for short "
           "diagnostic runs, not long experiments."),
    P("Run", "Diagnostics", "run.profile_report_every", "Report every N frames", "int"),
    P("Run", "Diagnostics", "run.save_rt_piv", "Save RT-PIV fields (.mat)", "bool",
      help="Streaming only. Also toggled at run time with [f]."),

    # ---------------- Acquisition ----------------
    P("Acquisition", "Timing", "run.f_acq", "Acquisition frequency [Hz]", "float",
      help="Accumulation window is 1/f_acq. This is also the frame separation "
           "used to convert px/frame to m/s."),
    P("Acquisition", "Timing", "run.max_events_per_pixel", "Max events per pixel", "int",
      help="Contrast clamp before normalising to uint8."),
    P("Acquisition", "Timing", "run.duration_sec", "Recording duration [s]", "float",
      help="Offline 'record' step only."),
    P("Acquisition", "Trigger", "run.trigger_mode", "Trigger mode", "choice",
      ["none", "external", "auto"],
      help="none = fixed-dt accumulation. external = camera trigger-in. "
           "auto = detect the laser phase from the event rate."),
    P("Acquisition", "Trigger", "run.trigger_duty_cycle", "Trigger duty cycle", "float",
      help="Fraction of the pulse period accumulated, centred on the pulse."),
    P("Acquisition", "Orientation", "run.flip_x", "Flip left-right", "bool",
      help="Mirrors the FULL frame before the ROI crop, so the ROIs are then "
           "in flipped coordinates. For a pure sign change prefer the "
           "feedback component '-u'."),
    P("Acquisition", "Orientation", "run.flip_y", "Flip top-bottom", "bool",
      help="Same caveat as flip left-right; negates V."),
    P("Acquisition", "Frames", "run.frame_smooth_sigma", "Frame smoothing sigma [px]", "float",
      help="Gaussian smoothing of every live pseudo-frame before the correlation "
           "(0 = off). The paper uses 0.75 px on binary frames. HR training applies "
           "the same value, so change it only together with the HR model."),
    P("Acquisition", "Camera biases", "run.camera_biases", "Biases", "biases",
      help="Prophesee low-level biases, applied when the device is opened."),

    # ---------------- PIV & ROIs ----------------
    P("PIV & ROIs", "Correlation", "control.piv.window_size",
      "Interrogation window [px]", "int",
      help="Correlation wraps beyond +-window/2 px; the usual working limit "
           "is a quarter of it."),
    P("PIV & ROIs", "Correlation", "control.piv.node_distance",
      "Node distance [px]", "int", help="Grid spacing between vectors."),
    P("PIV & ROIs", "Correlation", "control.piv.triple_corr",
      "Triple correlation (3 frames)", "bool"),
    P("PIV & ROIs", "Correlation", "control.piv.fft_workers", "FFT worker threads", "int",
      help="1 reproduces Release 4.0. More helps on large grids and hurts on "
           "small ones. Measure with bench_release5.py."),
    P("PIV & ROIs", "Correlation", "control.piv.subpixel", "Sub-pixel interpolation", "bool",
      help="ON is recommended: Release 4.0's real-time path was integer-pixel. "
           "OFF reproduces it bit-for-bit."),
    P("PIV & ROIs", "Correlation", "control.piv.compute_quality",
      "Correlation quality (CC)", "bool",
      help="Required by the ControlROI correlation gate."),
    P("PIV & ROIs", "Correlation", "control.piv.fast_mean_removal",
      "Fast mean removal (not bit-exact)", "bool",
      help="Zeroes the FFT DC bin instead of subtracting the spatial mean. "
           "Faster, ~1e-7 relative difference. See AUDIT section 3."),
    P("PIV & ROIs", "Correlation", "control.piv.rt_max_rate_hz",
      "Cap the RT-PIV rate [Hz]", "optfloat",
      help="Blank = unlimited (Release 4.0 behaviour). The PIV worker and the "
           "acquisition loop share the GIL, so a PIV thread running flat out "
           "starves acquisition and events queue inside the camera SDK - which "
           "shows up as the display lagging reality by seconds. The controller "
           "only needs control_rate_hz; 2-4x that is plenty."),

    P("PIV & ROIs", "Correlation", "control.piv.use_gpu",
      "GPU correlation (torch + CUDA)", "bool",
      help="Applies to the live stream AND the offline pyramidal PIV. The "
           "live GPU path returns sub-pixel U, V and CC exactly like the CPU "
           "(agrees to ~3e-6), so the closed loop can use it - but its "
           "LATENCY is unmeasured, so compare against a CPU run before "
           "trusting gains tuned on it. Falls back to the CPU with a log line "
           "if torch/CUDA is missing. Offline: CC is written as NaN. Run "
           "tools/check_gpu.py once on this machine first."),
    P("PIV & ROIs", "Validation", "control.piv.validation",
      "Westerweel-Scarano validation", "bool"),
    P("PIV & ROIs", "Validation", "control.piv.val_threshold", "Threshold", "float"),
    P("PIV & ROIs", "Validation", "control.piv.val_epsilon", "Epsilon", "float"),
    P("PIV & ROIs", "Validation", "control.piv.min_frames_before_output",
      "Frames to prime before output", "int",
      help="No field is published until this many real frames exist. Release "
           "4.0 correlated the first frame against a zero buffer and reported "
           "a spurious -window/2 displacement."),
    P("PIV & ROIs", "Regions", "control.roi.display_roi",
      "DisplayROI  [x0, x1, y0, y1]", "roi",
      help="Where vectors are computed and drawn. Empty = full sensor. "
           "Cropping is the biggest performance lever."),
    P("PIV & ROIs", "Regions", "control.roi.control_roi",
      "ControlROI  [x0, x1, y0, y1]", "roi",
      help="Must be inside the DisplayROI. Only these vectors feed the "
           "controller. Drawn on the live image."),

    # ---------------- Offline ----------------
    P("Offline", "Frame generation", "run.n_images", "Number of frames", "int"),
    P("Offline", "Frame generation", "run.img_prefix", "File prefix", "text"),
    P("Offline", "Frame generation", "run.burst_search_max_sec",
      "Burst search window [s]", "float",
      help="How much of the recording is scanned to find the laser phase."),
    P("Offline", "Frame generation", "run.apply_gaussian", "Gaussian blur", "bool"),
    P("Offline", "Frame generation", "run.gaussian_kernel", "Kernel size (odd)", "int"),
    P("Offline", "Frame generation", "run.gaussian_sigma", "Sigma", "float"),
    P("Offline", "Offline PIV", "run.pyramid_levels", "Pyramid levels", "int",
      help="Correlations at dt, 2dt, ... are homothetically scaled and summed. "
           "Needs levels+1 frames."),

    # ---------------- Feedback ----------------
    P("Feedback", "Control quantity", "control.measurement.component",
      "Streamwise component", "choice", ["u", "-u", "v", "-v", "magnitude"],
      help="'u' = jet left to right. V is positive DOWNWARD, so an upward jet "
           "is '-v'. 'magnitude' cannot distinguish forward from reverse flow."),
    P("Feedback", "Control quantity", "control.measurement.statistic",
      "Spatial statistic", "choice", ["mean", "median"]),
    P("Feedback", "Validity", "control.measurement.min_valid_fraction",
      "Min valid fraction", "float",
      help="Fraction of ControlROI nodes that must be valid for the "
           "measurement to be usable."),
    P("Feedback", "Validity", "control.measurement.min_valid_vectors",
      "Min valid vectors", "int"),
    P("Feedback", "Validity", "control.measurement.use_correlation_gate",
      "Correlation quality gate", "bool",
      help="Rejects the flat-correlation-plane failure mode that Release 4.0 "
           "reported as a large spurious velocity."),
    P("Feedback", "Validity", "control.measurement.min_correlation",
      "Min correlation", "float"),
    P("Feedback", "Units", "control.measurement.uncalibrated_units",
      "Units when no px/mm calibration", "choice", ["px/s", "px/frame"],
      help="Only used when no px/mm calibration is set. 'px/s' divides the "
           "displacement by dt, so the number is a VELOCITY and changing "
           "f_acq does not change what the set-point and the PID gains mean. "
           "'px/frame' is the raw displacement, as Release 4.0-5.1 reported "
           "it, and it scales with 1/f_acq. With a px/mm calibration set, "
           "everything is m/s either way."),

    P("Feedback", "Validity", "control.measurement.max_abs_displacement_px",
      "Max |displacement| [px]", "optfloat",
      help="Empty = disabled. Sensible value: a little below window/2."),
    P("Feedback", "Units", "control.measurement.calibration_px_per_mm",
      "Resolution [px/mm]", "optfloat",
      help="Set it and the whole loop works in m/s. Empty = px/frame."),
    P("Feedback", "Units", "control.measurement.pulse_separation_s",
      "Pulse separation dt [s]", "optfloat",
      help="Empty = derived as 1/f_acq, which equals 1/f_laser when the laser "
           "fires once per pseudo-frame. Set it only for pulse pairs."),

    # ---------------- Controller ----------------
    P("Controller", "Temporal filter", "control.filt.kind", "Filter", "choice",
      ["none", "ma", "ema"],
      help="Filtering adds phase lag straight into the loop. The choice is "
           "printed in every run header."),
    P("Controller", "Temporal filter", "control.filt.window",
      "Moving-average length [samples]", "int",
      help="Group delay is about (N-1)/2 samples."),
    P("Controller", "Temporal filter", "control.filt.tau_s", "EMA tau [s]", "float",
      help="Group delay is about tau seconds."),

    P("Controller", "Reference", "control.reference.kind", "Trajectory", "choice",
      ["constant", "live", "step", "sine", "file", "random"],
      help="'live' = you set the target by hand during the run with [+]/[-]. "
           "The keys move the TARGET VELOCITY and the PID finds the voltage; "
           "that is different from run mode 'manual', where the keys move the "
           "voltage and there is no controller."),
    P("Controller", "Reference", "control.reference.value", "Base value", "float",
      help="In the display units: m/s if a px/mm resolution is set, else "
           "px/frame.  For 'live' this is the target the run starts at."),
    P("Controller", "Reference", "control.reference.live_step",
      "Live: [+]/[-] step", "float",
      help="How much one key press moves the target, in display units."),
    P("Controller", "Reference", "control.reference.live_min",
      "Live: minimum target", "optfloat",
      help="Lowest target the operator can dial in. Empty = unbounded."),
    P("Controller", "Reference", "control.reference.live_max",
      "Live: maximum target", "optfloat",
      help="Highest target the operator can dial in. Empty = unbounded. "
           "Worth setting: it stops a held-down key walking the target far "
           "past anything the pump can deliver."),
    P("Controller", "Reference", "control.reference.t_step_s", "Step time [s]", "float"),
    P("Controller", "Reference", "control.reference.step_amplitude", "Step size", "float"),
    P("Controller", "Reference", "control.reference.amplitude", "Sine amplitude", "float"),
    P("Controller", "Reference", "control.reference.freq_hz", "Sine frequency [Hz]", "float"),
    P("Controller", "Reference", "control.reference.phase_rad", "Sine phase [rad]", "float"),
    P("Controller", "Reference", "control.reference.filepath",
      "Time series file", "file",
      help="Two columns: time_s, target. .csv, .txt or .npy."),
    P("Controller", "Reference", "control.reference.file_loop", "Loop the file", "bool"),
    P("Controller", "Reference", "control.reference.random_std", "Random std", "float"),
    P("Controller", "Reference", "control.reference.random_cutoff_hz",
      "Random bandwidth [Hz]", "float"),
    P("Controller", "Reference", "control.reference.random_seed", "Random seed", "int"),

    P("Controller", "PID", "control.pid.kp", "Kp  [V per unit]", "float",
      help="Gains depend on the velocity unit. Run a calibration first; the "
           "shipped default is zero on purpose."),
    P("Controller", "PID", "control.pid.ki", "Ki  [V per unit per s]", "float"),
    P("Controller", "PID", "control.pid.kd", "Kd  [V s per unit]", "float"),
    P("Controller", "PID", "control.pid.v_min", "Output min [V]", "float"),
    P("Controller", "PID", "control.pid.v_max", "Output max [V]", "float"),
    P("Controller", "PID", "control.pid.slew_rate_v_per_s",
      "Slew limit [V/s]", "optfloat", help="Empty = no slew limiting."),
    P("Controller", "PID", "control.pid.derivative_on_measurement",
      "Derivative on measurement", "bool",
      help="Avoids the set-point kick on a step."),
    P("Controller", "PID", "control.pid.derivative_filter_tau_s",
      "Derivative filter tau [s]", "float", help="0 disables it."),
    P("Controller", "PID", "control.pid.anti_windup", "Anti-windup", "choice",
      ["clamp", "back_calc", "none"]),
    P("Controller", "PID", "control.pid.back_calc_gain", "Back-calculation gain", "float",
      help="Only used when anti-windup is 'back_calc'."),
    P("Controller", "PID", "control.pid.integral_limit",
      "Integral limit [V]", "optfloat", help="Empty = unbounded."),

    P("Controller", "Rates and safety", "control.supervisor.control_rate_hz",
      "Control rate [Hz]", "float",
      help="Independent of the EBIV rate. Set it from the measured pump step "
           "response, not from the camera rate."),
    P("Controller", "Rates and safety", "control.supervisor.hold_timeout_s",
      "Hold timeout [s]", "float",
      help="Short dropouts: the last voltage is held and the PID does not "
           "integrate."),
    P("Controller", "Rates and safety", "control.supervisor.safe_timeout_s",
      "Safe timeout [s]", "float",
      help="Longer dropouts: drive to the safe voltage and disarm."),
    P("Controller", "Rates and safety", "control.supervisor.max_measurement_age_s",
      "Max measurement age [s]", "float"),
    P("Controller", "Rates and safety", "control.supervisor.safe_pump_voltage",
      "Safe voltage [V]", "float",
      help="Do NOT assume 0 V is right for your facility. Decide it."),
    P("Controller", "Rates and safety", "control.supervisor.startup_pump_voltage",
      "Startup voltage [V]", "float"),

    # ---------------- Hardware ----------------
    P("Hardware", "Device", "control.ad3.enabled", "Use the Analog Discovery", "bool",
      help="OFF = mock pump, no voltage reaches any hardware. The WaveForms "
           "desktop application must be CLOSED when this is ON."),
    P("Hardware", "Device", "control.ad3.device_index", "Device index", "int",
      help="-1 = first available."),
    P("Hardware", "Laser", "control.ad3.laser_backend", "Laser backend", "choice",
      ["analog", "pattern", "external"],
      help="analog = square wave on an Analog Out channel. pattern = a DIO "
           "pin. external = something else drives it."),
    P("Hardware", "Laser", "control.ad3.laser_channel", "Analog Out channel", "int",
      help="0 = W1, 1 = W2. Must differ from the pump channel."),
    P("Hardware", "Laser", "control.ad3.laser_dio_channel", "DIO channel", "int"),
    P("Hardware", "Laser", "control.ad3.laser_frequency_hz",
      "Frequency [Hz]  (blank = follow acquisition)", "optfloat",
      help="Leave BLANK and the laser follows the acquisition frequency, one "
           "pulse per pseudo-frame, so 1/f_laser is exactly the frame "
           "separation dt. That is what you want in almost every run. Enter a "
           "number only for an external laser, or for pulse-pair "
           "illumination - and in that case also set the pulse separation on "
           "the Feedback tab, because dt is then not 1/f_acq."),

P("Hardware", "Laser", "control.ad3.laser_amplitude_v", "Amplitude [V]", "float",
      help="HALF peak-to-peak: the wave swings offset +- amplitude."),
    P("Hardware", "Laser", "control.ad3.laser_offset_v", "Offset [V]", "float",
      help="Amplitude 2.5 with offset 2.5 gives 0..5 V, right at the +-5 V rail."),
    P("Hardware", "Laser", "control.ad3.laser_duty_percent", "Duty cycle [%]", "float"),
    P("Hardware", "Laser", "control.ad3.laser_idle_low",
      "Laser OFF when the channel is stopped", "bool",
      help="ON stops the AD3 driving the pin when the channel is not running. "
           "OFF parks it at the configured offset, which with offset 2.5 V is "
           "a continuously-triggered laser. Leave this ON. Note the pin is "
           "left high-impedance, so a pulled-up laser input can still read "
           "HIGH: shutdown also drives a real 0 V first, and a pull-down "
           "resistor at the laser input is the only hard guarantee."),

P("Hardware", "Pump", "control.ad3.pump_channel", "Analog Out channel", "int",
      help="0 = W1, 1 = W2. Must differ from the laser channel."),
    P("Hardware", "Pump", "control.ad3.pump_v_min", "Hardware min [V]", "float",
      help="Enforced in the backend; a command outside these limits is "
           "clamped and logged, never sent."),
    P("Hardware", "Pump", "control.ad3.pump_v_max", "Hardware max [V]", "float"),

    # ---------------- Calibration ----------------
    P("Calibration", "Open-loop programme", "control.calibration.mode", "Mode", "choice",
      ["manual", "steps", "sweep"],
      help="manual = [+]/[-] keys. steps = hold each voltage. sweep = ramp."),
    P("Calibration", "Open-loop programme", "control.calibration.voltages",
      "Step voltages [V]", "floats", help="Comma separated."),
    P("Calibration", "Open-loop programme", "control.calibration.dwell_s",
      "Dwell per step [s]", "float"),
    P("Calibration", "Open-loop programme", "control.calibration.settle_s",
      "Settle (discarded) [s]", "float"),
    P("Calibration", "Open-loop programme", "control.calibration.v_start",
      "Sweep start [V]", "float"),
    P("Calibration", "Open-loop programme", "control.calibration.v_stop",
      "Sweep stop [V]", "float"),
    P("Calibration", "Open-loop programme", "control.calibration.sweep_duration_s",
      "Sweep duration [s]", "float"),
    P("Calibration", "Open-loop programme", "control.calibration.sweep_return",
      "Sweep back down", "bool", help="Exposes hysteresis."),
    P("Calibration", "Open-loop programme", "control.calibration.manual_step_v",
      "Manual key step [V]", "float"),
    P("Calibration", "Logging", "control.logging.enabled", "Experiment logging", "bool"),
    P("Calibration", "Logging", "control.logging.flush_every", "CSV flush every N rows", "int"),
    P("Calibration", "Logging", "control.logging.latency_report_period_s",
      "Timing report period [s]", "float"),

    # ---------------- HR estimation (live) ----------------
    P("HR estimation", "Live", "run.hr.enabled", "Estimate the HR field live", "bool",
      help="Runs the trained estimator on every rt-EBIV field. [h] in the stream "
           "window switches the vectors between LR and HR. The model must have been "
           "trained with the CURRENT Acquisition and PIV & ROIs settings; if not, the "
           "run refuses to start and says which setting differs."),
    P("HR estimation", "Live", "run.hr.model_path", "Model file (.npz)", "file",
      help="Written by Resolution enhancement > Train a model (or the offline "
           "step 'train HR model'), in Out/<acq name>/; training sets this path."),
    P("HR estimation", "Live", "run.hr.method", "Estimator", "choice",
      ["kf", "lse", "lse_vr"],
      help="kf = direct Kalman filter (Method I). lse = LSE + KF (Method II). "
           "lse_vr = variance-rescaled LSE + KF (Method III). All three are in "
           "every trained model."),
    P("HR estimation", "Live", "run.hr.steady_state", "Steady-state gain", "bool",
      help="Use the converged Kalman gain: identical estimate after the first "
           "~100-200 fields, ~100x cheaper per step. OFF = the MATLAB scripts' "
           "time-varying filter."),
    P("HR estimation", "Live", "run.hr.show_hr", "Show HR vectors at start", "bool",
      help="Toggled at run time with [h]."),

    # ---------------- Display ----------------
    P("Display", "Rates", "control.supervisor.visualization_rate_hz",
      "Image window [Hz]", "float",
      help="Cosmetic only. It can never determine control timing."),
    P("Display", "Rates", "control.supervisor.plot_rate_hz", "Strip chart [Hz]", "float"),
    P("Display", "Vectors", "run.arrow_skip", "Draw every Nth node", "int"),
    P("Display", "Vectors", "run.arrow_scale", "Arrow scale [px/px]", "float"),
    P("Display", "Vectors", "run.blackout_background", "Black background", "bool",
      help="Release 4.0 blanked the pseudo-frame behind the vectors. "
           "Toggled at run time with [b]."),
    P("Display", "Vectors", "run.show_strip_chart", "Show the strip chart", "bool",
      help="Toggled at run time with [g]."),
    P("Display", "Strip chart", "run.plot_history_s", "History shown [s]", "float",
      help="How much of the past the chart draws."),
    P("Display", "Strip chart", "run.plot_zoom_window_s", "Y-scale window [s]", "float",
      help="The y-axis is scaled on the last N seconds only, so a start-up "
           "transient from 0 stops holding the scale open once it is this "
           "old. Small (5-15 s) zooms hard into the current operating point; "
           "0 scales on the whole history, as before 5.2."),
]



# Training settings: shown in the dialog "Resolution enhancement > Train a
# model...", not on a tab.  Same (tab, group, path, ...) format; 'tab' is
# unused, 'group' is the frame in the dialog.
HR_TRAIN_SPEC = [
    P("HR training", "Source", "run.hr.source", "Parameters from", "radio",
      [("raw", "a recording (.raw): LR + HR fields, POD and operators all computed here"),
       ("fields", "LR / HR fields from elsewhere (LR.mat + HR.mat, or a training set "
                  ".npz): POD and operators here"),
       ("operators", "operators from elsewhere (.mat of tools/matlab/vibe_export_model.m): "
                     "only converted")],
      help="External data are mapped onto the live convention through the LR grid: "
           "their LR nodes must be the live LR nodes (same ROI, window, step). See "
           "'How it works'."),
    P("HR training", "External data", "run.hr.ext_lr_file", "LR fields (.mat / .npz)", "file",
      help="LR.mat of the MATLAB pipeline (X, Y, U, V; v7 or v7.3), or a training set "
           "*_trainingset.npz saved by an earlier training (then HR is not needed)."),
    P("HR training", "External data", "run.hr.ext_hr_file", "HR fields (.mat)", "file",
      help="HR.mat, snapshot-paired with LR.mat (same number, same instants)."),
    P("HR training", "External data", "run.hr.ext_model_file", "Operators (.mat)", "file",
      help="Written by tools/matlab/vibe_export_model.m at the end of "
           "Proc_Main_FullKF_DEF.m (KF) or Proc_Main_EPOD_DEF.m (LSE or LSE+VR). "
           "Only the estimator(s) in the file are available."),
    P("HR training", "External data", "run.hr.ext_y_up", "Y axis points up", "bool",
      help="ON: the files' Y grows upwards (DaVis, most PIV codes) and v is positive "
           "up; the image rows grow downwards, so v changes sign. OFF: image convention."),
    P("HR training", "External data", "run.hr.ext_velocity_units", "Velocity units", "choice",
      ["m/s", "px/frame", "unit/s"],
      help="m/s: X, Y in mm. unit/s: the length unit of X, Y per second. The px per "
           "unit comes from matching the LR grid to the live grid."),
    P("HR training", "External data", "run.hr.ext_f_hz", "Rate of the data [Hz] (0 = f_acq)",
      "float", help="Must equal the live acquisition frequency; the model is stamped "
                    "with it, so a different f_acq is refused later."),
    P("HR training", "External data", "run.hr.ext_sensor_w", "Sensor width [px]", "int",
      help="Only used when the ROI is empty (full sensor), to place the live grid."),
    P("HR training", "External data", "run.hr.ext_sensor_h", "Sensor height [px]", "int"),
    P("HR training", "Recording", "run.hr.train_raw", "Raw file (blank = Run tab)", "file",
      help="Recording used to build the LR/HR training set. The LR fields are "
           "computed with the live PIV & ROIs settings; the phase-locked frames use "
           "the Acquisition settings (trigger mode must be auto/external)."),
    P("HR training", "Recording", "run.hr.train_start_frame", "First frame", "int"),
    P("HR training", "Recording", "run.hr.train_n_frames", "Frames (0 = all)", "int",
      help="The split below must fit: n_train + n_val + n_test + 2*gap fields."),
    P("HR training", "HR processing", "run.hr.hr_window", "Window [px]", "int",
      help="Paper: 32 px (jet), 64 px (channel)."),
    P("HR training", "HR processing", "run.hr.hr_step", "Step [px]", "int",
      help="75% overlap = window / 4."),
    P("HR training", "HR processing", "run.hr.hr_levels", "Frame separations", "int",
      help="Pairs per field. centred: separations 1, 3, 5.. dt (2*levels frames); "
           "forward: 1, 2, 3.. dt (levels+1 frames)."),
    P("HR training", "HR processing", "run.hr.hr_stencil", "Stencil", "choice",
      ["centred", "forward"],
      help="centred: every pair is centred on the LR pair's time (time-aligned "
           "with the LR field). forward: the offline PIV's stencil, anchored at "
           "the first frame."),
    P("HR training", "HR processing", "run.hr.hr_combine", "Combine separations", "choice",
      ["peaks", "planes"],
      help="peaks: sub-pixel peak per pair, weighted by separation^2 (more "
           "accurate in the tests). planes: sum of rescaled correlation planes, "
           "as the offline PIV."),
    P("HR training", "HR processing", "run.hr.hr_predictor", "Window offset (predictor)", "bool",
      help="First pass with 2x windows, then each pair's windows are offset by "
           "the predicted displacement. Needed whenever separation x displacement "
           "approaches window/2 (the jet at ~13 px/frame with 32 px windows)."),
    P("HR training", "Split (fields)", "run.hr.n_train", "Training", "int",
      help="Consecutive fields: F is a one-step model."),
    P("HR training", "Split (fields)", "run.hr.gap", "Gap between blocks", "int"),
    P("HR training", "Split (fields)", "run.hr.n_val", "Validation", "int",
      help="Used for the measurement-noise covariances R."),
    P("HR training", "Split (fields)", "run.hr.n_test", "Test", "int",
      help="Only for the error report (delta vs HR-LOR and cubic). 0 = none."),
    P("HR training", "POD", "run.hr.rank", "HR rank (0 = elbow)", "int"),
    P("HR training", "POD", "run.hr.truncate_lr", "Truncate LR (elbow)", "bool",
      help="FlagLRRankLimit of the MATLAB script. OFF = full LR rank (paper Sec. 2.1)."),
    P("HR training", "POD", "run.hr.elbow_threshold", "Elbow threshold", "float",
      help="pod_elbow_rank: first k with lambda_(k+1)/lambda_k >= threshold."),
    P("HR training", "POD", "run.hr.elbow_smooth", "Elbow smoothing", "choice",
      ["none", "movmean", "fir"],
      help="Smoothing of F(k) before the threshold test. With 'none' a pair of "
           "nearly degenerate modes (convected structures) can stop the "
           "truncation early."),
    P("HR training", "POD", "run.hr.elbow_span", "movmean span", "int"),
    P("HR training", "POD", "run.hr.lambda_c", "Ridge lambda (C, M)", "float"),
    P("HR training", "POD", "run.hr.vr_max_gain", "VR gain cap (Method III)", "float",
      help="Cap on the variance-rescaling gain Gamma = std(true)/std(LSE) per mode "
           "(Proc.MaxGain, 10 in the scripts). Modes the LSE barely predicts get "
           "the cap: their estimate is mostly noise amplified by it. The training "
           "report says how many modes hit the cap."),
    P("HR training", "Kalman tuning (test and live)", "run.hr.tune_q", "Q multiplier", "float",
      help="Proc.tuneQ: Q x this. < 1 trusts the model F more. Applies to the "
           "test-block report AND to the live estimator, without retraining."),
    P("HR training", "Kalman tuning (test and live)", "run.hr.tune_r", "R multiplier", "float",
      help="Proc.tuneR: R x this. > 1 trusts the LR measurement less (smoother, "
           "more lag). Applies to the test-block report AND live."),
    P("HR training", "Output", "run.hr.model_name", "Model file name", "text",
      help="Saved in Out/<acq name>/."),
    P("HR training", "Output", "run.hr.u_ref", "U_ref for delta [px/frame] (0 = auto)", "float"),
    P("HR training", "Output", "run.hr.save_training_set", "Save the training set (.npz)", "bool",
      help="LR and HR fields, to retrain without reprocessing the recording."),
    P("HR training", "Output", "run.hr.export_matlab", "Also export LR.mat / HR.mat", "bool",
      help="MATLAB-pipeline layout, px/frame, v positive down."),
]

# The live HR fields are drawn in the header panel, not on a tab.
HEADER_TAB = "HR estimation"


# ==========================================================================
#  Path access
# ==========================================================================

def get_path(session, path):
    obj = session
    parts = path.split('.')
    for p in parts[:-1]:
        obj = getattr(obj, p)
    return getattr(obj, parts[-1])


def set_path(session, path, value):
    obj = session
    parts = path.split('.')
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], value)


# Fields deliberately not given a widget, with the reason.  Anything NOT in
# this map and not in PARAM_SPEC is an oversight, and the test-suite says so —
# that is how a newly added config field gets noticed instead of quietly
# becoming unreachable from the GUI.
INTENTIONALLY_HIDDEN = {
    'run.mode': "chosen by the mode radio buttons in the header",
    'run.run_mode': "chosen by the streaming sub-mode combo in the header",
    'run.do_record': "chosen by the offline step checkboxes in the header",
    'run.do_playback': "chosen by the offline step checkboxes in the header",
    'run.do_image_gen': "chosen by the offline step checkboxes in the header",
    'run.do_piv_process': "chosen by the offline step checkboxes in the header",
    'run.do_hr_train': "chosen by the offline step checkboxes in the header",
    'control.measurement.velocity_scale':
        "superseded by the px/mm resolution; exposing both invites confusion",
    'control.measurement.velocity_units':
        "set automatically from the px/mm resolution",
    'control.reference.random_clip':
        "an optional (lo, hi) pair; edit it in a preset file if you need it",
    'control.logging.formats':
        "csv and mat are both wanted in practice; edit a preset to change it",
}


def check_spec(session=None):
    """
    Validate PARAM_SPEC against the dataclasses.

    Returns (bad_paths, unaccounted_fields).  bad_paths are spec entries that
    do not resolve — the GUI refuses to start if there are any.
    unaccounted_fields are config fields that neither have a widget nor appear
    in INTENTIONALLY_HIDDEN.  The test-suite asserts both are empty.
    """
    session = session or Session()
    bad = []
    for s in PARAM_SPEC + HR_TRAIN_SPEC:
        try:
            get_path(session, s['path'])
        except AttributeError:
            bad.append(s['path'])

    exposed = ({s['path'] for s in PARAM_SPEC + HR_TRAIN_SPEC}
               | set(INTENTIONALLY_HIDDEN))
    unexposed = []

    def walk(obj, prefix):
        for fname in getattr(obj, '__dataclass_fields__', {}):
            full = f"{prefix}{fname}"
            val = getattr(obj, fname)
            if is_dataclass(val):
                walk(val, full + ".")
            elif full not in exposed:
                unexposed.append(full)

    walk(session.run, "run.")
    walk(session.control, "control.")
    return bad, unexposed


# ==========================================================================
#  Small widgets
# ==========================================================================

class ToolTip:
    """Minimal hover tooltip."""

    def __init__(self, widget, text, wrap=460):
        self.widget, self.text, self.wrap = widget, text, wrap
        self.tip = None
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")

    def _show(self, _=None):
        if self.tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tk.Label(tw, text=self.text, justify="left", background="#ffffe0",
                 relief="solid", borderwidth=1, wraplength=self.wrap,
                 font=("TkDefaultFont", 8)).pack(ipadx=4, ipady=2)

    def _hide(self, _=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


class ScrollFrame(ttk.Frame):
    """A frame that scrolls vertically, for tabs taller than the window."""

    def __init__(self, parent):
        super().__init__(parent)
        canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        vsb = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.inner = ttk.Frame(canvas)
        self.inner.bind("<Configure>",
                        lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        win = canvas.create_window((0, 0), window=self.inner, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(win, width=e.width))
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self.canvas = canvas

        # Wheel scrolling is bound only while the pointer is over THIS canvas.
        #
        # The obvious implementation — bind_all in every ScrollFrame — is
        # wrong: there is one ScrollFrame per tab, so a single wheel notch
        # fires every tab's handler and scrolls all ten canvases at once.  The
        # visible tab looks right, but every hidden tab is silently scrolled
        # too and is found at a strange offset when you switch to it.
        def _wheel(event):
            if event.num == 5:
                delta = 1
            elif event.num == 4:
                delta = -1
            else:
                delta = -1 * (event.delta // 120 if abs(event.delta) >= 120
                              else (1 if event.delta > 0 else -1))
            canvas.yview_scroll(delta, "units")
            return "break"

        def _enter(_):
            for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                canvas.bind_all(seq, _wheel)

        def _leave(_):
            for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                canvas.unbind_all(seq)

        for w in (canvas, self.inner):
            w.bind("<Enter>", _enter, add="+")
            w.bind("<Leave>", _leave, add="+")


# ==========================================================================
#  Field bindings
# ==========================================================================

class Field:
    """One parameter: its spec, its tk variable(s) and the conversions."""

    def __init__(self, spec, parent, row):
        self.spec = spec
        self.kind = spec['kind']
        self.vars = []
        label = ttk.Label(parent, text=spec['label'])
        label.grid(row=row, column=0, sticky="nw" if spec['kind'] == 'radio' else "w",
                   padx=(6, 10), pady=3)
        self.label = label
        if spec['help']:
            ToolTip(label, spec['help'])

        box = ttk.Frame(parent)
        box.grid(row=row, column=1, sticky="ew", pady=3)
        parent.columnconfigure(1, weight=1)
        self.box = box

        k = self.kind
        if k == 'radio':
            v = tk.StringVar()
            for val, text in spec['extra']:
                ttk.Radiobutton(box, text=text, value=val, variable=v).pack(anchor="w")
            self.vars = [v]
        elif k == 'bool':
            v = tk.BooleanVar()
            ttk.Checkbutton(box, variable=v).pack(side="left")
            self.vars = [v]
        elif k == 'choice':
            v = tk.StringVar()
            ttk.Combobox(box, textvariable=v, values=list(spec['extra']),
                         state="readonly", width=16).pack(side="left")
            self.vars = [v]
        elif k == 'roi':
            for i, nm in enumerate(("x0", "x1", "y0", "y1")):
                ttk.Label(box, text=nm).pack(side="left", padx=(0 if i == 0 else 6, 2))
                v = tk.StringVar()
                ttk.Entry(box, textvariable=v, width=7).pack(side="left")
                self.vars.append(v)
            ttk.Label(box, text="  (leave empty = full / none)",
                      foreground="#666").pack(side="left", padx=6)
        elif k == 'biases':
            self.bias_keys = ['bias_diff_on', 'bias_diff_off', 'bias_hpf',
                              'bias_fo', 'bias_refr', 'bias_diff']
            for i, bk in enumerate(self.bias_keys):
                r, c = divmod(i, 3)
                ttk.Label(box, text=bk).grid(row=r, column=2 * c, sticky="e",
                                             padx=(0 if c == 0 else 10, 3), pady=1)
                v = tk.StringVar()
                ttk.Entry(box, textvariable=v, width=6).grid(row=r, column=2 * c + 1,
                                                             sticky="w", pady=1)
                self.vars.append(v)
        elif k in ('folder', 'file'):
            v = tk.StringVar()
            ttk.Entry(box, textvariable=v).pack(side="left", fill="x", expand=True)
            ttk.Button(box, text="...", width=3,
                       command=self._browse).pack(side="left", padx=4)
            self.vars = [v]
        else:
            v = tk.StringVar()
            width = 40 if k == 'text' and 'name' in spec['path'] else (
                34 if k in ('text', 'floats') else 12)
            ttk.Entry(box, textvariable=v, width=width).pack(side="left")
            self.vars = [v]

        if spec['help'] and k != 'bool':
            ToolTip(box, spec['help'])

    # ------------------------------------------------------------------

    def set_enabled(self, on):
        """Grey the field out (ttk state: keeps a combobox read-only when re-enabled)."""
        todo = [self.label, self.box]
        while todo:
            w = todo.pop()
            todo.extend(w.winfo_children())
            try:
                w.state(['!disabled'] if on else ['disabled'])
            except (AttributeError, tk.TclError):
                pass

    def _browse(self):
        if self.kind == 'folder':
            p = filedialog.askdirectory(title=self.spec['label'])
        else:
            p = filedialog.askopenfilename(title=self.spec['label'])
        if p:
            self.vars[0].set(p)

    # ------------------------------------------------------------------

    def load(self, value):
        k = self.kind
        if k == 'bool':
            self.vars[0].set(bool(value))
        elif k == 'roi':
            vals = list(value) if value else ["", "", "", ""]
            for v, x in zip(self.vars, vals):
                v.set("" if x is None or x == "" else str(int(x)))
        elif k == 'biases':
            for v, bk in zip(self.vars, self.bias_keys):
                v.set(str((value or {}).get(bk, 0)))
        elif k == 'floats':
            self.vars[0].set(", ".join(f"{float(x):g}" for x in (value or [])))
        elif k in ('optfloat',):
            self.vars[0].set("" if value is None else f"{value:g}")
        elif k == 'float':
            self.vars[0].set("" if value is None else f"{float(value):g}")
        elif k == 'int':
            self.vars[0].set("" if value is None else str(int(value)))
        else:
            self.vars[0].set("" if value is None else str(value))

    def collect(self):
        """Read the widget(s) back to a Python value.  Raises ValueError."""
        k = self.kind
        lbl = self.spec['label']
        try:
            if k == 'bool':
                return self.vars[0].get()
            if k in ('choice', 'radio'):
                return self.vars[0].get()
            if k == 'roi':
                raw = [v.get().strip() for v in self.vars]
                if all(r == "" for r in raw):
                    return None
                if any(r == "" for r in raw):
                    raise ValueError("give all four values, or leave all four empty")
                return [int(float(r)) for r in raw]
            if k == 'biases':
                return {bk: int(float(v.get())) for bk, v in
                        zip(self.bias_keys, self.vars)}
            if k == 'floats':
                s = self.vars[0].get().strip()
                if not s:
                    return []
                return [float(x) for x in s.replace(";", ",").split(",") if x.strip()]
            s = self.vars[0].get().strip()
            if k == 'optfloat':
                return None if s == "" else float(s)
            if k == 'float':
                return float(s)
            if k == 'int':
                return int(float(s))
            if k == 'file':
                return s or None
            return s
        except ValueError as exc:
            raise ValueError(f"{lbl}: {exc}") from None


# ==========================================================================
#  Log capture
# ==========================================================================

class TextHandler(logging.Handler):
    """Buffers log records; the GUI flushes them into the Log tab."""

    def __init__(self, maxlines=4000):
        super().__init__()
        self.lines = []
        self.maxlines = maxlines

    def emit(self, record):
        try:
            self.lines.append(self.format(record))
            if len(self.lines) > self.maxlines:
                del self.lines[:len(self.lines) - self.maxlines]
        except Exception:                                      # noqa: BLE001
            pass


# ==========================================================================
#  Resolution enhancement: status of the live estimation
# ==========================================================================

HR_LABELS = {'kf': "KF (Method I)", 'lse': "LSE + KF (Method II)",
             'lse_vr': "LSE+VR + KF (Method III)"}

# level -> (background, foreground) of the status line
HR_COLOURS = {'off': ("#e4e4e4", "#303030"), 'info': ("#dbe8f7", "#1d4f91"),
              'ok': ("#d6f0d6", "#135c13"), 'warn': ("#fff0c2", "#6b4a00"),
              'error': ("#f8d4d2", "#8f1a1a")}


def model_brief(hdr):
    """One line about a model, from vibe_hr.read_model_header()."""
    rep = hdr.get('test_report') or {}
    txt = (f"r = {hdr.get('r_hr')} HR modes ({100 * hdr.get('hr_energy', float('nan')):.0f}% "
           f"energy), r_LR = {hdr.get('r_lr_modes')}, trained {hdr.get('file_mtime', '?')} "
           f"on {hdr.get('n_train', '?')} fields")
    if hdr.get('source') == 'imported operators':
        txt += f", IMPORTED from {os.path.basename(str(hdr.get('source_file', '?')))}"
    elif hdr.get('source') == 'external fields':
        txt += " (external LR/HR fields)"
    d = [(k, rep.get(f'delta_{k}')) for k in ('kf', 'lse', 'lse_vr')]
    d = [f"{HR_LABELS[k].split(' (')[0]} {v:.3f}" for k, v in d if v is not None]
    if d:
        cu = rep.get('delta_cubic')
        txt += "; test error delta: " + ", ".join(d) + (
            f" (cubic interpolation {cu:.3f})" if cu is not None else "")
    return txt


def hr_status(mode, enabled, model_path, method, live=None, train_step=False,
              model_name=None, tune=(1.0, 1.0)):
    """
    What the live resolution enhancement will do with these settings.
    Returns (level, text): level in 'off', 'info', 'ok', 'warn', 'error'.
    'error' means Run would refuse (or run without it); it mirrors the checks
    of Session.validate() and vibe_hr_live.HRLive.  Needs numpy (vibe_hr) only
    when a model file is given.
    """
    hdr, err = None, None
    path = str(model_path or '').strip()
    if path:
        if not os.path.exists(path):
            err = f"the model file does not exist: {path}"
        else:
            try:
                import vibe_hr
                hdr = vibe_hr.read_model_header(path)
            except Exception as exc:                              # noqa: BLE001
                err = f"the model file cannot be read ({type(exc).__name__}: {exc})"

    mism, crit = [], []
    if hdr is not None and live is not None:
        import vibe_hr
        mism = vibe_hr.check_processing(hdr, live)
        crit = [m for m in mism if vibe_hr.is_critical(m)]

    if mode == 'offline':
        if train_step:
            return 'info', ("The offline chain will TRAIN a new HR model (Out/<acq name>/"
                            f"{model_name or 'hr_model.npz'}). The estimation itself runs "
                            "in Live streaming.")
        return 'off', ("Offline chain: the HR estimation is not used here (it runs in Live "
                       "streaming). Resolution enhancement > Train a model... builds one.")

    if not enabled:
        txt = "OFF: the live stream shows the LR rt-EBIV vectors only."
        if hdr is not None and not crit:
            txt += (" A compatible model is selected; tick 'Estimate the HR field live' "
                    "to use it.")
        elif hdr is not None:
            txt += " (The selected model does not match the current settings.)"
        return 'off', txt

    if not path:
        return 'error', ("ON, but no model is selected: Run will refuse. Train a model or "
                         "load one (Resolution enhancement menu).")
    if err:
        return 'error', f"ON, but {err}. Run will refuse."
    if method not in hdr.get('methods', ['kf']):
        return 'error', (f"ON, but this model has no '{method}' estimator (it has "
                         f"{', '.join(hdr.get('methods', []))}). Retrain it.")
    if crit:
        return 'error', ("ON, but the model is INCOMPATIBLE with the current settings; Run "
                         "will refuse:  " + ";  ".join(crit) +
                         ".  Retrain, or restore the settings it was trained with "
                         "(Model info...).")
    head = f"ON: {HR_LABELS.get(method, method)}. {model_brief(hdr)}."
    if tuple(tune) != (1.0, 1.0):
        head += f" Tuning Q x {tune[0]:g}, R x {tune[1]:g}."
    extra = []
    if hdr.get('units') not in (None, 'px/frame'):
        extra.append(f"model units {hdr.get('units')!r}, the stream gives px/frame")
    if mism or extra:
        return 'warn', head + "  CHECK: " + ";  ".join(mism + extra)
    return 'ok', head + "  In the stream window [h] switches HR / LR vectors."


# ==========================================================================
#  Main window
# ==========================================================================

class EbivGUI:

    def __init__(self, root, session=None, preset_path=None):
        self.root = root
        self.session = session or Session()
        self.fields = {}
        self.tab_frames = {}
        self.preset_path = preset_path

        bad, _ = check_spec(self.session)
        if bad:
            raise RuntimeError(
                "The parameter specification does not match the configuration "
                "dataclasses. These paths do not resolve:\n  " + "\n  ".join(bad))

        root.title(APP_TITLE)
        root.geometry(f"1000x{min(880, max(600, root.winfo_screenheight() - 90))}")
        root.minsize(820, 560)

        self._hr_fp = None
        self._hr_after = None
        self.hr_level, self.hr_text = 'off', ""
        self.train_dialog = None

        self._build_header()
        self._build_hr_panel()
        self._build_notebook()
        self._build_footer()
        self._build_menubar()

        # Capture the log for the Log tab.  Any handler left behind by an
        # earlier window in this process is removed first: without this,
        # opening the GUI a second time (or reopening it after a run) leaves
        # both handlers attached and every line is captured twice.
        root_logger = logging.getLogger()
        for h in [h for h in root_logger.handlers if isinstance(h, TextHandler)]:
            root_logger.removeHandler(h)
        self.log_handler = TextHandler()
        self.log_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        root_logger.addHandler(self.log_handler)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.refresh()
        self._on_mode_change()
        # the status line re-checks the model against the widgets as they are
        # edited (the first check imports numpy: after the window is up)
        self._hr_after = root.after(300, self._poll_hr_status)

    def _on_close(self):
        """Detach the log handler before the window goes away."""
        if self.train_dialog is not None and self.train_dialog.busy:
            if not messagebox.askyesno(
                    "HR training running",
                    "A training is running. Stop it and quit?", parent=self.root):
                return
            self.train_dialog.stop()
        if self._hr_after is not None:
            try:
                self.root.after_cancel(self._hr_after)
            except tk.TclError:
                pass
            self._hr_after = None
        try:
            logging.getLogger().removeHandler(self.log_handler)
        except Exception:                                      # noqa: BLE001
            pass
        self.root.destroy()

    # ------------------------------------------------------------------

    def _build_header(self):
        f = ttk.LabelFrame(self.root, text="1.  What do you want to run?")
        f.pack(fill="x", padx=10, pady=(10, 6))

        self.mode_var = tk.StringVar(value=self.session.run.mode)
        row = ttk.Frame(f)
        row.pack(fill="x", padx=8, pady=6)

        ttk.Radiobutton(row, text="Live streaming", value="stream",
                        variable=self.mode_var,
                        command=self._on_mode_change).pack(side="left")
        self.run_mode_var = tk.StringVar(value=self.session.run.run_mode)
        self.run_mode_box = ttk.Combobox(
            row, textvariable=self.run_mode_var, state="readonly", width=14,
            values=["ebiv", "manual", "calibration", "closed_loop"])
        self.run_mode_box.pack(side="left", padx=(8, 0))
        ToolTip(self.run_mode_box,
                "ebiv = measure and display only, no pump backend is created.\n"
                "manual = open-loop pump control from the keyboard.\n"
                "calibration = the programmed open-loop voltage schedule.\n"
                "closed_loop = the PID; starts disarmed, press [c] to arm.")

        ttk.Separator(row, orient="vertical").pack(side="left", fill="y", padx=16)

        ttk.Radiobutton(row, text="Offline chain", value="offline",
                        variable=self.mode_var,
                        command=self._on_mode_change).pack(side="left")

        self.step_vars = {}
        steps = ttk.Frame(f)
        steps.pack(fill="x", padx=8, pady=(0, 8))
        self.steps_frame = steps
        ttk.Label(steps, text="steps:").pack(side="left", padx=(2, 6))
        for key, label, tip in (
                ('do_record', "record .raw",
                 "Record events from the camera to Raw/<raw file name>."),
                ('do_playback', "play back",
                 "Replay the .raw so you can check the recording."),
                ('do_image_gen', "generate frames",
                 "Phase-locked .tif frames into RawImg/<acq name>/."),
                ('do_piv_process', "offline PIV",
                 "Pyramidal PIV on those frames, results into Out/<acq name>/."),
                ('do_hr_train', "train HR model",
                 "Build the LR/HR training set from the .raw and train the three "
                 "HR estimators, with the settings of Resolution enhancement > "
                 "Train a model... (the dialog can also run the training on its "
                 "own, with progress and results). The new model is selected in "
                 "panel 2.")):
            v = tk.BooleanVar(value=getattr(self.session.run, key))
            cb = ttk.Checkbutton(steps, text=label, variable=v)
            cb.pack(side="left", padx=6)
            ToolTip(cb, tip + "  Selected steps run in this order, in one go.")
            self.step_vars[key] = v

    def _build_hr_panel(self):
        """Panel 2: the live HR settings and a status line saying what will happen."""
        import tkinter.font as tkfont
        f = ttk.LabelFrame(self.root, text="2.  Resolution enhancement  "
                                           "(LR rt-EBIV field -> HR field, live)")
        f.pack(fill="x", padx=10, pady=(0, 6))
        self.hr_frame = f
        bold = tkfont.nametofont("TkDefaultFont").copy()
        bold.configure(weight="bold")
        self._bold = bold
        self.hr_status_lbl = tk.Label(f, text="checking...", anchor="w", justify="left",
                                      font=bold, padx=8, pady=5, wraplength=900)
        self.hr_status_lbl.pack(fill="x", padx=6, pady=(4, 4))
        self.hr_status_lbl.bind(
            "<Configure>", lambda e: self.hr_status_lbl.configure(
                wraplength=max(200, e.width - 20)))
        self._paint_hr_status('off', "checking...")

        specs = {s['path']: s for s in PARAM_SPEC if s['tab'] == HEADER_TAB}
        row = ttk.Frame(f)
        row.pack(fill="x", padx=2)
        for path in ('run.hr.enabled', 'run.hr.method', 'run.hr.steady_state',
                     'run.hr.show_hr'):
            cell = ttk.Frame(row)
            cell.pack(side="left", padx=(0, 14))
            self.fields[path] = Field(specs[path], cell, 0)
        row2 = ttk.Frame(f)
        row2.pack(fill="x", padx=2, pady=(0, 6))
        cell = ttk.Frame(row2)
        cell.pack(side="left", fill="x", expand=True)
        self.fields['run.hr.model_path'] = Field(specs['run.hr.model_path'], cell, 0)
        ttk.Button(row2, text="Model info...",
                   command=self.show_model_info).pack(side="right", padx=(4, 6))
        ttk.Button(row2, text="Train a model...",
                   command=self.open_train_dialog).pack(side="right", padx=4)
        missing = set(specs) - set(self.fields)
        if missing:                                  # a new HR field needs a place here
            raise RuntimeError(f"HR panel: no widget for {sorted(missing)}")

    def _build_menubar(self):
        mb = tk.Menu(self.root)
        fm = tk.Menu(mb, tearoff=0)
        fm.add_command(label="Load preset...", command=self.load_preset)
        fm.add_command(label="Save preset...", command=self.save_preset)
        fm.add_command(label="Reset to defaults", command=self.reset_defaults)
        fm.add_separator()
        fm.add_command(label="Quit", command=self._on_close)
        mb.add_cascade(label="File", menu=fm)
        hm = tk.Menu(mb, tearoff=0)
        hm.add_command(label="Train a model...", command=self.open_train_dialog)
        hm.add_command(label="Load a model...", command=self.load_hr_model)
        hm.add_command(label="Model info...", command=self.show_model_info)
        hm.add_separator()
        hm.add_checkbutton(label="Estimate the HR field live (streaming)",
                           variable=self.fields['run.hr.enabled'].vars[0])
        hm.add_separator()
        hm.add_command(label="How it works...", command=self.show_hr_help)
        mb.add_cascade(label="Resolution enhancement", menu=hm)
        self.root.config(menu=mb)
        self.menubar, self.hr_menu = mb, hm

    # ------------------------------------------------------------------
    #  Resolution enhancement
    # ------------------------------------------------------------------

    def _paint_hr_status(self, level, text):
        bg, fg = HR_COLOURS[level]
        mark = {'off': "OFF", 'info': "TRAIN", 'ok': "ON", 'warn': "ON*",
                'error': "!!"}[level]
        self.hr_status_lbl.configure(text=f"[{mark}]  {text}", bg=bg, fg=fg)

    def peek_session(self):
        """A copy of the session with the widgets' current values (no side effects)."""
        import copy
        peek = copy.deepcopy(self.session)
        for path, fld in self.fields.items():
            try:
                set_path(peek, path, fld.collect())
            except (ValueError, tk.TclError):
                pass
        peek.run.mode = self.mode_var.get()
        peek.run.run_mode = self.run_mode_var.get()
        for k, v in self.step_vars.items():
            setattr(peek.run, k, v.get())
        return peek

    def update_hr_status(self, force=False):
        """Recompute the status line if any widget (or the model file) changed."""
        mp = self.fields['run.hr.model_path'].vars[0].get().strip()
        try:
            mtime = os.path.getmtime(mp) if mp else None
        except OSError:
            mtime = None
        fp = (tuple(v.get() for fld in self.fields.values() for v in fld.vars),
              self.mode_var.get(), tuple(v.get() for v in self.step_vars.values()), mtime)
        if fp == self._hr_fp and not force:
            return self.hr_level
        self._hr_fp = fp
        try:
            peek = self.peek_session()
            h = peek.run.hr
            live = None
            if h.model_path and peek.run.mode == 'stream':
                from vibe_hr_live import live_settings_from_session
                live = live_settings_from_session(peek)
            level, text = hr_status(peek.run.mode, h.enabled, h.model_path, h.method, live,
                                    train_step=peek.run.do_hr_train,
                                    model_name=h.model_name, tune=(h.tune_q, h.tune_r))
        except Exception as exc:                               # noqa: BLE001
            level, text = 'warn', f"status unavailable ({type(exc).__name__}: {exc})"
        self.hr_level, self.hr_text = level, text
        self._paint_hr_status(level, text)
        return level

    def _poll_hr_status(self):
        try:
            self.update_hr_status()
        finally:
            try:
                self._hr_after = self.root.after(700, self._poll_hr_status)
            except tk.TclError:
                self._hr_after = None

    def open_train_dialog(self):
        if self.train_dialog is not None and self.train_dialog.alive:
            self.train_dialog.top.deiconify()
            self.train_dialog.top.lift()
            return self.train_dialog
        self.train_dialog = HRTrainDialog(self)
        return self.train_dialog

    def load_hr_model(self, path=None, ask_enable=True):
        path = path or filedialog.askopenfilename(
            title="Load an HR model", filetypes=[("HR model", "*.npz"), ("All", "*.*")])
        if not path:
            return None
        try:
            import vibe_hr
            vibe_hr.read_model_header(path)
        except Exception as exc:                               # noqa: BLE001
            messagebox.showerror("Not an HR model", f"{path}\n\n{type(exc).__name__}: {exc}")
            return None
        self.fields['run.hr.model_path'].vars[0].set(path)
        en = self.fields['run.hr.enabled'].vars[0]
        if ask_enable and not en.get() and messagebox.askyesno(
                "HR model loaded", "Use this model for the live estimation now?"):
            en.set(True)
        self.update_hr_status(force=True)
        self.status.set(f"HR model: {os.path.basename(path)}")
        return path

    def show_model_info(self):
        path = self.fields['run.hr.model_path'].vars[0].get().strip()
        if not path or not os.path.exists(path):
            messagebox.showinfo("Model info", "No model file is selected (or it does not "
                                "exist). Train one or load one from the Resolution "
                                "enhancement menu.")
            return None
        return ModelInfoDialog(self, path)

    def show_hr_help(self):
        top = tk.Toplevel(self.root)
        top.title("Resolution enhancement: how it works")
        top.geometry("760x620")
        t = scrolledtext.ScrolledText(top, wrap="word", padx=10, pady=8)
        t.pack(fill="both", expand=True)
        t.insert("end", HR_HELP)
        t.configure(state="disabled")
        ttk.Button(top, text="Close", command=top.destroy).pack(pady=6)
        return top

    def _build_notebook(self):
        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill="both", expand=True, padx=10, pady=4)

        order = []
        for s in PARAM_SPEC:
            if s['tab'] not in order and s['tab'] != HEADER_TAB:
                order.append(s['tab'])
        order.append("Log")

        for tab in order:
            sf = ScrollFrame(self.nb)
            self.tab_frames[tab] = sf
            if tab == "Log":
                self._build_log_tab(sf.inner)
                continue
            groups = {}
            for s in [x for x in PARAM_SPEC if x['tab'] == tab]:
                g = groups.get(s['group'])
                if g is None:
                    g = ttk.LabelFrame(sf.inner, text=s['group'])
                    g.pack(fill="x", padx=10, pady=(8, 2))
                    groups[s['group']] = g
                    g._row = 0
                self.fields[s['path']] = Field(s, g, g._row)
                g._row += 1

    def _build_log_tab(self, parent):
        bar = ttk.Frame(parent)
        bar.pack(fill="x", padx=8, pady=6)
        ttk.Button(bar, text="Refresh", command=self._refresh_log).pack(side="left")
        ttk.Button(bar, text="Clear",
                   command=lambda: (self.log_handler.lines.clear(),
                                    self._refresh_log())).pack(side="left", padx=6)
        ttk.Button(bar, text="Save to file...",
                   command=self._save_log).pack(side="left")
        self.log_text = scrolledtext.ScrolledText(parent, height=26, wrap="none",
                                                  font=("TkFixedFont", 8))
        self.log_text.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    def _build_footer(self):
        f = ttk.Frame(self.root)
        f.pack(fill="x", padx=10, pady=(2, 10))

        ttk.Button(f, text="Load preset...", command=self.load_preset).pack(side="left")
        ttk.Button(f, text="Save preset...", command=self.save_preset).pack(side="left", padx=6)
        ttk.Button(f, text="Reset to defaults",
                   command=self.reset_defaults).pack(side="left")

        self.run_btn = ttk.Button(f, text="Run", command=self.run)
        self.run_btn.pack(side="right")
        ttk.Button(f, text="Validate", command=self.validate).pack(side="right", padx=6)
        ttk.Button(f, text="Quit", command=self._on_close).pack(side="right", padx=(0, 16))

        self.status = tk.StringVar(value="Ready.")
        ttk.Label(self.root, textvariable=self.status, relief="sunken",
                  anchor="w").pack(fill="x", side="bottom")

    # ------------------------------------------------------------------

    def _on_mode_change(self):
        mode = self.mode_var.get()
        wanted = TABS_STREAM if mode == 'stream' else TABS_OFFLINE

        # ttk.Notebook.hide() leaves the tab in tabs(), so the notebook's own
        # idea of what exists no longer matches what is on screen.  forget()
        # and re-add() instead: the frames survive (they are children of the
        # notebook, not owned by the tab), the order is exactly `wanted`, and
        # tabs() then tells the truth — which the GUI tests rely on.
        try:
            current = self.nb.tab(self.nb.select(), "text")
        except tk.TclError:
            current = None
        for frame_id in list(self.nb.tabs()):
            self.nb.forget(frame_id)
        for tab in wanted:
            self.nb.add(self.tab_frames[tab], text=tab)
        if current in wanted:
            self.nb.select(self.tab_frames[current])

        state = "readonly" if mode == 'stream' else "disabled"
        self.run_mode_box.configure(state=state)
        for child in self.steps_frame.winfo_children():
            try:
                child.configure(state="normal" if mode == 'offline' else "disabled")
            except tk.TclError:
                pass
        self.status.set("Live streaming." if mode == 'stream'
                        else "Offline chain: the selected steps run in order.")

    # ------------------------------------------------------------------

    def refresh(self):
        """Session -> widgets.

        Show DECLARED intent, never a resolved value.  validate() and run()
        call Session.validate(), which resolves the laser rate and the
        velocity scale IN PLACE; without restoring first, a blank "follow
        f_acq" box would come back filled with whatever f_acq happened to be,
        and the next collect() would read that as an explicit override.  That
        is how a blanked field reappeared as a number after pressing Run.
        """
        self.session.restore_declared()
        for path, fld in self.fields.items():
            fld.load(get_path(self.session, path))
        self.mode_var.set(self.session.run.mode)
        self.run_mode_var.set(self.session.run.run_mode)
        for k, v in self.step_vars.items():
            v.set(getattr(self.session.run, k))
        self._hr_fp = None                       # status line: recheck on the next poll
        if self.train_dialog is not None and self.train_dialog.alive \
                and not self.train_dialog.busy:
            self.train_dialog.load()

    def collect(self):
        """Widgets -> session.  Raises ValueError listing every bad field."""
        errors = []
        for path, fld in self.fields.items():
            try:
                set_path(self.session, path, fld.collect())
            except ValueError as exc:
                errors.append(str(exc))
        self.session.run.mode = self.mode_var.get()
        self.session.run.run_mode = self.run_mode_var.get()
        for k, v in self.step_vars.items():
            setattr(self.session.run, k, v.get())
        if errors:
            raise ValueError("Could not read these fields:\n  - " +
                             "\n  - ".join(errors))
        # The widgets are now the user's intent, so any snapshot taken by an
        # earlier resolve() is stale and must not survive this edit.
        self.session.forget_resolution()
        return self.session

    # ------------------------------------------------------------------

    def validate(self, quiet=False):
        try:
            self.collect()
            # Session.validate() covers the acquisition parameters, the
            # control-system cross-checks, ROI containment and the reference
            # file — everything that can be known before the camera is opened.
            self.session.validate()
        except ValueError as exc:
            self.status.set("Configuration problem.")
            if not quiet:
                messagebox.showerror("Configuration problem", str(exc))
            return False
        self.status.set("Configuration is valid.")
        if not quiet:
            messagebox.showinfo("Validation", "Configuration is valid.")
        return True

    def run(self):
        if self._training_busy():
            return
        if not self.validate(quiet=True):
            self.validate()
            return
        if self.session.run.mode == 'stream' and self.update_hr_status(force=True) == 'error':
            messagebox.showerror("Resolution enhancement", self.hr_text +
                                 "\n\nUntick 'Estimate the HR field live' to stream "
                                 "without it.")
            return
        try:
            self.session.save(LAST_PRESET)
        except Exception:                                      # noqa: BLE001
            pass

        mode = self.session.run.mode
        if mode == 'stream' and self.session.run.run_mode == 'closed_loop' \
                and self.session.control.ad3.enabled:
            if not messagebox.askokcancel(
                    "Closed loop with real hardware",
                    "This will open the Analog Discovery and drive the pump.\n\n"
                    "Confirm:\n"
                    "  - the WaveForms desktop application is CLOSED\n"
                    "  - you have run the laser/pump verification with a scope\n"
                    "  - the safe voltage is right for your facility\n\n"
                    "The loop starts DISARMED; press [c] in the image window "
                    "to close it."):
                return

        self.status.set("Running — this window is hidden until the run ends.")
        self.root.update_idletasks()
        self.root.withdraw()

        result, error = None, None
        try:
            from ebiv_session import run_session
            result = run_session(self.session)
        except Exception as exc:                               # noqa: BLE001
            error = exc
            logging.exception("Run failed.")
        finally:
            self.root.deiconify()
            self.root.lift()
            self._refresh_log()

        if error is not None:
            self.status.set("Run failed — see the Log tab.")
            messagebox.showerror(
                "Run failed",
                f"{type(error).__name__}: {error}\n\n"
                "The full traceback is on the Log tab and in the console.")
        else:
            # The run can write back into the session (HR training sets the
            # model path): show it, so the next collect() does not undo it.
            try:
                self.refresh()
            except Exception:                                  # noqa: BLE001
                logging.exception("Could not refresh the window after the run.")
            self.status.set("Run finished.")
            msg = "Run finished.\n\n"
            if result and result.get('mode') == 'offline':
                for step, dt in result.get('steps', []):
                    msg += f"  {step:<12s} {dt:6.1f} s\n"
            if result:
                msg += f"\nOutput: {result.get('output_dir', '')}"
            if result and result.get('hr_model'):
                msg += (f"\n\nHR model: {result['hr_model']}\n"
                        "It is now selected in panel 2 (Resolution enhancement).")
            messagebox.showinfo("Finished", msg)

    # ------------------------------------------------------------------

    def _refresh_log(self):
        self.log_text.delete("1.0", "end")
        self.log_text.insert("end", "\n".join(self.log_handler.lines[-4000:]))
        self.log_text.see("end")

    def _save_log(self):
        p = filedialog.asksaveasfilename(defaultextension=".txt",
                                         initialfile="ebiv_log.txt")
        if p:
            with open(p, "w", encoding="utf-8") as f:
                f.write("\n".join(self.log_handler.lines))
            self.status.set(f"Log saved to {p}")

    def _training_busy(self):
        if self.train_dialog is not None and self.train_dialog.busy:
            messagebox.showwarning("HR training running",
                                   "Wait for the training to finish, or stop it first.")
            return True
        return False

    def load_preset(self):
        if self._training_busy():
            return
        p = filedialog.askopenfilename(title="Load preset",
                                       filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if not p:
            return
        try:
            session, ignored = Session.load(p)
        except Exception as exc:                               # noqa: BLE001
            messagebox.showerror("Could not load the preset", str(exc))
            return
        self.session = session
        self.preset_path = p
        self.refresh()
        self._on_mode_change()
        self.status.set(f"Loaded {os.path.basename(p)}")
        if ignored:
            messagebox.showwarning(
                "Preset loaded with unknown keys",
                "These entries are not part of this version's configuration "
                "and were ignored:\n\n  " + "\n  ".join(ignored[:25]))

    def save_preset(self):
        try:
            self.collect()
        except ValueError as exc:
            messagebox.showerror("Configuration problem", str(exc))
            return
        p = filedialog.asksaveasfilename(
            title="Save preset", defaultextension=".json",
            initialfile=os.path.basename(self.preset_path or "ebiv_preset.json"),
            filetypes=[("JSON", "*.json")])
        if not p:
            return
        self.session.save(p)
        self.preset_path = p
        self.status.set(f"Saved {os.path.basename(p)}")

    def reset_defaults(self):
        if self._training_busy():
            return
        if messagebox.askyesno("Reset", "Discard the current settings and "
                                        "return to the built-in defaults?"):
            self.session = Session()
            self.refresh()
            self._on_mode_change()
            self.status.set("Reset to defaults.")


# ==========================================================================
#  Resolution enhancement: training dialog, model info, help
# ==========================================================================

HR_HELP = """\
WHAT IT DOES
The live rt-EBIV gives a coarse (LR) velocity field at the acquisition rate.
A model trained on a recording of the same flow turns every LR field into an
estimate of the high-resolution (HR) field, in a reduced POD basis, with a
Kalman filter that also uses the time dynamics learned from the training data
(Franceschelli et al., the three estimators of Sec. 2.2).

TRAINING  (Resolution enhancement > Train a model...)
 1. A .raw recording of the flow is cut into phase-locked frames.
 2. Every field is processed twice:
      LR = the live rt-EBIV processing, with the CURRENT Acquisition and
           PIV & ROIs settings (exactly what the stream will compute);
      HR = multi-frame correlation with small overlapping windows (the
           training reference).
 3. POD of the LR and HR sets, truncation by the elbow criterion, then
    the dynamics F, Q and the measurement models of the three estimators;
    the noise covariances come from the validation block, the error report
    from the test block.  No MATLAB step.
 The training set is saved, so the POD/operator part can be redone without
 reprocessing the recording (see HR_Example.py).

PARAMETERS COMPUTED ELSEWHERE  (same dialog, section 1)
 - LR / HR fields (LR.mat + HR.mat of the MATLAB pipeline): the POD and the
   operators are computed here, with the test-block report.  A saved
   training set (*_trainingset.npz) can be reused the same way.
 - Operators: add one line to Proc_Main_FullKF_DEF.m or Proc_Main_EPOD_DEF.m
   (see tools/matlab/vibe_export_model.m) and import the .mat it writes.
   Nothing is recomputed; only the estimator(s) in the file are available.
 In both cases the file's LR nodes must be the live LR nodes (same ROI,
 window, step): matching the two grids fixes the array layout, the length
 scale and the velocity conversion to px/frame.  You declare the direction
 of the Y axis and the velocity units.  That the LR fields really come from
 the live rt-EBIV processing (validation, smoothing, ...) cannot be checked:
 the model is stamped with the live settings as DECLARED.

LIVE  (tick 'Estimate the HR field live', then Run in Live streaming)
 Each LR field costs one filter step (well under 1 ms with the steady-state
 gain); the HR field is reconstructed only for the display.  The banner at the
 top right of the stream window says whether the HR estimation is ON; [h]
 switches between HR and LR vectors.  Fields the correlator drops are
 bridged by prediction-only steps.

THE THREE ESTIMATORS
 KF (Method I)         the LR POD coefficients are the measurement, through
                       a linear map C fitted on the training set.
 LSE + KF (Method II)  a linear stochastic estimate LR -> HR is used as a
                       direct measurement of the HR state.
 LSE+VR + KF (III)     the same, with each mode rescaled to its training
                       variance (gain capped, see the training report).

LIMITS  (read before trusting the output)
 - A model is valid for the flow condition and the LR settings it was trained
   with.  A different frequency, ROI, window, step, flip or trigger mode makes
   it invalid: the status line turns red and Run refuses.  Other differences
   (validation, smoothing, duty cycle...) are reported as warnings.
 - The HR reference is itself a PIV measurement.  The error delta is computed
   against its projection on the r retained modes (HR-LOR), not against the
   full HR field: scales outside those r modes are not recovered, by
   construction.
 - The test block comes from the same recording as the training.  Another
   day, seeding or operating point can be worse; retrain when in doubt.
 - With variance rescaling, modes whose gain hits the cap are mostly
   amplified noise.
"""


def model_info_text(path, hdr, live=None):
    """Plain-text description of a model (read_model_header output) vs the live settings."""
    import vibe_hr
    L = []
    L.append(f"Model file   {path}")
    L.append(f"Saved        {hdr.get('file_mtime', '?')}   (vibe_hr {hdr.get('version', '?')})")
    src = hdr.get('source') or ('recording (.raw)' if hdr.get('lr_processing') else 'unknown')
    L.append(f"Source       {src}" + (f": {hdr.get('source_file')}" if hdr.get('source_file') else ""))
    cv = hdr.get('conversion')
    if isinstance(cv, dict):
        L.append(f"Conversion   {cv.get('px_per_unit', float('nan')):.4g} px per file unit, "
                 f"velocity x {cv.get('vel_scale', float('nan')):.4g} ({cv.get('velocity_units')}"
                 f" -> px/frame), Y axis {'up' if cv.get('y_up') else 'down'}, file LR grid "
                 f"{tuple(cv.get('lr_shape_file', ()))}")
    L.append("Estimators   " + ", ".join(HR_LABELS.get(m, m) for m in hdr.get('methods', [])))
    L.append(f"POD          r = {hdr.get('r_hr')} HR modes "
             f"({100 * hdr.get('hr_energy', float('nan')):.1f}% of the HR fluctuation energy)")
    L.append(f"             rank: {hdr.get('rank_mode', '?')}, elbow threshold "
             f"{hdr.get('threshold', '?')}, options {hdr.get('elbow') or {}}")
    L.append(f"             r_LR = {hdr.get('r_lr_modes')} LR modes "
             f"({'elbow' if hdr.get('truncate_lr', True) else 'full rank'}"
             + (f", {100 * hdr['lr_energy']:.1f}% energy" if 'lr_energy' in hdr else "") + ")")
    L.append(f"Grids        LR {tuple(hdr.get('lr_shape', ()))}  ->  HR {tuple(hdr.get('hr_shape', ()))}")
    L.append(f"Training     {hdr.get('n_train', '?')} fields (F, Q, C, M), R from "
             f"{hdr.get('n_val', '?')} validation fields")
    if isinstance(hdr.get('split'), dict):
        L.append("             split " + ", ".join(f"{k} {v}" for k, v in hdr['split'].items()))
    if hdr.get('gamma_range'):
        g0, g1 = hdr['gamma_range']
        L.append(f"Method III   gain Gamma in [{g0:.2f}, {g1:.2f}]; {hdr.get('gamma_n_capped', '?')} "
                 f"of {hdr.get('r_hr')} modes at the cap {hdr.get('max_gain', '?')}")
    rep = hdr.get('test_report') or {}
    if any(k.startswith('delta_') for k in rep):
        L.append("")
        L.append("TEST BLOCK  delta = rms vector error / U_ref vs HR-LOR (Eq. 34), lower is better;")
        L.append("            TKE ratio = fluctuation energy estimate / HR-LOR (1 = right energy)")
        L.append(f"            U_ref = {rep.get('u_ref', float('nan')):.4g} px/frame" +
                 (f", tuning Q x {hdr['test_tuning']['q_scale']:g}, R x "
                  f"{hdr['test_tuning']['r_scale']:g}" if hdr.get('test_tuning') else ""))
        rows = [("cubic interpolation", rep.get('delta_cubic'), None)] + [
            (HR_LABELS[m], rep.get(f'delta_{m}'), rep.get(f'tke_ratio_{m}'))
            for m in ('kf', 'lse', 'lse_vr')]
        for name, d, k in rows:
            if d is None:
                continue
            L.append(f"   {name:<28s} delta {d:7.4f}" + (f"   TKE ratio {k:5.2f}" if k is not None else ""))
    ref = hdr.get('lr_processing')
    L.append("")
    if not ref:
        L.append("LR PROCESSING: no record (model trained outside vibe_train); the "
                 "consistency with the live settings cannot be checked.")
    else:
        L.append("LR PROCESSING   trained with            now (main window)       " +
                 ("   [DECLARED at import, not verified]" if hdr.get('lr_processing_declared')
                  else ""))
        for k, v in ref.items():
            if live is None or k not in live:
                now, verdict = "-", ""
            else:
                now = live[k]
                bad = vibe_hr.check_processing(hdr, {k: now})
                verdict = ("ok" if not bad else
                           "DIFFERENT: Run refuses" if vibe_hr.is_critical(bad[0]) else
                           "different (warning)")
            L.append(f"   {k:<22s} {str(v):<22s}  {str(now):<22s}  {verdict}")
    ts = hdr.get('training_settings')
    if ts:
        L.append("")
        L.append("HR PROCESSING (training)")
        for k in ('hr_window', 'hr_step', 'hr_levels', 'hr_stencil', 'hr_combine',
                  'hr_predictor', 'start_frame', 'n_frames'):
            if k in ts:
                L.append(f"   {k:<22s} {ts[k]}")
    return "\n".join(L)


class ModelInfoDialog:

    def __init__(self, app, path):
        import vibe_hr
        from vibe_hr_live import live_settings_from_session
        self.app = app
        hdr = vibe_hr.read_model_header(path)
        try:
            live = live_settings_from_session(app.peek_session())
        except Exception:                                      # noqa: BLE001
            live = None
        self.text = model_info_text(path, hdr, live)
        top = self.top = tk.Toplevel(app.root)
        top.title(f"HR model: {os.path.basename(path)}")
        top.geometry("900x640")
        top.transient(app.root)
        t = scrolledtext.ScrolledText(top, wrap="none", font="TkFixedFont", padx=8, pady=6)
        t.pack(fill="both", expand=True)
        t.insert("end", self.text)
        t.configure(state="disabled")
        bar = ttk.Frame(top)
        bar.pack(fill="x", pady=6)
        ttk.Button(bar, text="Copy", command=self._copy).pack(side="left", padx=8)
        ttk.Button(bar, text="Close", command=top.destroy).pack(side="right", padx=8)

    def _copy(self):
        self.top.clipboard_clear()
        self.top.clipboard_append(self.text)


class HRTrainDialog:
    """
    Resolution enhancement > Train a model...

    The LR side is NOT editable here: it is the live Acquisition / PIV & ROIs
    settings of the main window (shown read-only and re-checked every second),
    so the model always matches the stream.  The training runs in a worker
    thread (ebiv_session.train_hr_model on a COPY of the session); progress
    comes back through a queue polled by the Tk loop.  While it runs the
    dialog grabs the input, so the settings cannot change under it.
    """

    # which training fields matter for which source (others are greyed out)
    ONLY = {
        'raw': {'run.hr.train_raw', 'run.hr.train_start_frame', 'run.hr.train_n_frames',
                'run.hr.hr_window', 'run.hr.hr_step', 'run.hr.hr_levels', 'run.hr.hr_stencil',
                'run.hr.hr_combine', 'run.hr.hr_predictor'},
        'fields': {'run.hr.ext_lr_file', 'run.hr.ext_hr_file'},
        'operators': {'run.hr.ext_model_file'},
    }
    EXTERNAL = {'run.hr.ext_y_up', 'run.hr.ext_velocity_units', 'run.hr.ext_f_hz',
                'run.hr.ext_sensor_w', 'run.hr.ext_sensor_h'}
    FITTING = {'run.hr.n_train', 'run.hr.gap', 'run.hr.n_val', 'run.hr.n_test', 'run.hr.rank',
               'run.hr.truncate_lr', 'run.hr.elbow_threshold', 'run.hr.elbow_smooth',
               'run.hr.elbow_span', 'run.hr.lambda_c', 'run.hr.vr_max_gain', 'run.hr.u_ref',
               'run.hr.save_training_set', 'run.hr.export_matlab'}

    @classmethod
    def relevant(cls, path, source):
        for src, paths in cls.ONLY.items():
            if path in paths:
                return source == src
        if path in cls.EXTERNAL:
            return source != 'raw'
        if path in cls.FITTING:
            return source != 'operators'
        return True

    PHASES = {'build': "Step 1/2  building the training set (LR + multi-frame HR PIV "
                       "of every field)",
              'load': "Step 1/2  loading the external fields, converting them to the "
                      "live convention",
              'import': "Importing the operators, converting them to the live convention",
              'train': "Step 2/2  POD, operators, test-block evaluation (cannot be "
                       "stopped; usually seconds to a few minutes)",
              'done': "Saving"}

    def __init__(self, app):
        self.app = app
        self.fields = {}
        self.busy = False
        self.alive = True
        self.phase = None
        self.result = None
        self.error = None
        self.thread = None
        self.stop_event = None
        self.q = None
        self._after = None
        self._plan_after = None
        top = self.top = tk.Toplevel(app.root)
        top.title("Resolution enhancement: train a model")
        top.geometry(f"960x{min(1000, max(640, top.winfo_screenheight() - 90))}")
        top.minsize(780, 620)
        top.transient(app.root)
        top.protocol("WM_DELETE_WINDOW", self.close)

        srcf = ttk.LabelFrame(top, text="1.  Where do the resolution-enhancement "
                                        "parameters come from?")
        srcf.pack(fill="x", padx=10, pady=(10, 4))
        src_spec = next(x for x in HR_TRAIN_SPEC if x['path'] == 'run.hr.source')
        self.fields['run.hr.source'] = Field(src_spec, srcf, 0)
        self.fields['run.hr.source'].vars[0].trace_add(
            "write", lambda *_: self._on_source())

        lf = ttk.LabelFrame(top, text="2.  LR input = the LIVE settings  (edit them on the "
                                      "Acquisition and PIV & ROIs tabs of the main window)")
        lf.pack(fill="x", padx=10, pady=4)
        self.plan_var = tk.StringVar()
        self.plan_lbl = tk.Label(lf, textvariable=self.plan_var, justify="left", anchor="w",
                                 font="TkFixedFont")
        self.plan_lbl.pack(side="left", fill="x", expand=True, padx=8, pady=4)

        of = ttk.LabelFrame(top, text="3.  Settings  (hover a label for help; greyed = not "
                                      "used by this source)")
        of.pack(fill="both", expand=True, padx=10, pady=4)
        sf = ScrollFrame(of)
        sf.pack(fill="both", expand=True)
        order = []
        for sp in HR_TRAIN_SPEC:
            if sp['group'] not in order and sp['group'] != "Source":
                order.append(sp['group'])
        sf.inner.columnconfigure(0, weight=1)
        sf.inner.columnconfigure(1, weight=1)
        self.groups = {}
        for grp in order:
            g = ttk.LabelFrame(sf.inner, text=grp)
            self.groups[grp] = g
            for r, sp in enumerate([x for x in HR_TRAIN_SPEC if x['group'] == grp]):
                self.fields[sp['path']] = Field(sp, g, r)

        rf = ttk.LabelFrame(top, text="4.  Run")
        rf.pack(fill="x", padx=10, pady=4)
        row = ttk.Frame(rf)
        row.pack(fill="x", padx=6, pady=(6, 2))
        self.train_btn = ttk.Button(row, text="Train", command=self.start)
        self.train_btn.pack(side="left")
        self.stop_btn = ttk.Button(row, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=6)
        self.phase_var = tk.StringVar(value="Not started.")
        ttk.Label(row, textvariable=self.phase_var).pack(side="left", padx=10)
        self.pb = ttk.Progressbar(rf, mode="determinate", maximum=100)
        self.pb.pack(fill="x", padx=8, pady=2)
        self.detail_var = tk.StringVar(value="")
        ttk.Label(rf, textvariable=self.detail_var, foreground="#555").pack(
            fill="x", padx=8, pady=(0, 6))

        res = ttk.LabelFrame(top, text="5.  Results on the test block")
        res.pack(fill="x", padx=10, pady=4)
        self.tree = ttk.Treeview(res, columns=("delta", "tke"), show="tree headings",
                                 height=4)
        self.tree.heading("#0", text="Estimator")
        self.tree.heading("delta", text="delta vs HR-LOR (Eq. 34)")
        self.tree.heading("tke", text="TKE ratio est / ref")
        self.tree.column("#0", width=260)
        self.tree.column("delta", width=180, anchor="e")
        self.tree.column("tke", width=160, anchor="e")
        self.tree.pack(fill="x", padx=6, pady=(6, 2))
        self.res_var = tk.StringVar(value="No model trained in this window yet.")
        ttk.Label(res, textvariable=self.res_var, justify="left", wraplength=860).pack(
            fill="x", padx=8, pady=(2, 6))

        bar = ttk.Frame(top)
        bar.pack(fill="x", padx=10, pady=(4, 10))
        self.use_btn = ttk.Button(bar, text="Use this model live", command=self.use_live,
                                  state="disabled")
        self.use_btn.pack(side="left")
        ttk.Button(bar, text="Close", command=self.close).pack(side="right")
        ttk.Button(bar, text="How it works...", command=app.show_hr_help).pack(
            side="right", padx=6)

        self.load()
        self._on_source()
        self._plan_after = top.after(1000, self._poll_plan)

    # ------------------------------------------------------------------
    def load(self, session=None):
        session = session or self.app.session
        for path, fld in self.fields.items():
            fld.load(get_path(session, path))

    @property
    def source(self):
        return self.fields['run.hr.source'].vars[0].get() or 'raw'

    def _on_source(self):
        src = self.source
        for path, fld in self.fields.items():
            fld.set_enabled(self.relevant(path, src))
        # show only the groups this source uses, two per row
        shown = [g for grp, g in self.groups.items()
                 if any(self.relevant(sp['path'], src) for sp in HR_TRAIN_SPEC
                        if sp['group'] == grp)]
        for g in self.groups.values():
            g.grid_remove()
        for i, g in enumerate(shown):
            g.grid(row=i // 2, column=i % 2, sticky="nsew", padx=6, pady=4)
        self.train_btn.configure(text="Import" if src == 'operators' else "Train")
        if not self.busy:
            self.refresh_plan()

    def collect(self, session=None):
        session = session or self.app.session
        errors = []
        for path, fld in self.fields.items():
            try:
                set_path(session, path, fld.collect())
            except ValueError as exc:
                errors.append(str(exc))
        if errors:
            raise ValueError("Could not read these fields:\n  - " + "\n  - ".join(errors))
        return session

    def plan(self):
        """What Train would use now (widgets of both windows, nothing committed)."""
        from ebiv_session import hr_training_plan
        peek = self.app.peek_session()
        try:
            self.collect(peek)
        except ValueError as exc:
            p = hr_training_plan(peek)
            p['problems'].insert(0, str(exc))
            return p
        return hr_training_plan(peek)

    def refresh_plan(self):
        try:
            p = self.plan()
        except Exception as exc:                               # noqa: BLE001
            self.plan_var.set(f"cannot check the settings: {type(exc).__name__}: {exc}")
            self.plan_lbl.configure(fg=HR_COLOURS['error'][1])
            self.train_btn.configure(state="disabled")
            return None
        lr = p['lr']
        roi = "full sensor" if lr['roi'] is None else str(lr['roi'])
        exists = os.path.exists(p['model'])
        h = self.app.peek_session().run.hr
        txt = (f"f_acq {lr['f_hz']:g} Hz, trigger {lr['trigger']} (duty {lr['duty_cycle']:g})"
               f"  |  ROI {roi}\n"
               f"window {lr['window']} px, step {lr['step']} px, validation "
               f"{'on' if lr['validate'] else 'off'}, sub-pixel {'on' if lr['subpixel'] else 'off'}"
               f"  |  smoothing {lr['smooth_sigma']:g} px, flip x/y "
               f"{'yes' if lr['flip_x'] else 'no'}/{'yes' if lr['flip_y'] else 'no'}\n")
        if p['source'] != 'raw':
            txt += ("(external data: DECLARED to come from this processing; the LR grid "
                    "is checked, the rest cannot be)\n")
        for lbl, pth, ok in p['inputs']:
            txt += f"{lbl:<10s} {pth or '(none)'}  [{'found' if ok else 'NOT FOUND'}]\n"
        if p.get('conversion'):
            txt += "conversion " + textwrap.fill(p['conversion'], 110,
                                                 subsequent_indent=" " * 11) + "\n"
        txt += f"model  ->  {p['model']}" + ("  [exists: will be replaced]" if exists else "")
        if p['source'] != 'operators':
            txt += (f"\nsplit needs >= {p['fields_needed']} fields  (train {h.n_train} | gap "
                    f"{h.gap} | val {h.n_val} | gap {h.gap} | test {h.n_test})")
        if p['problems']:
            txt += "\n\nCANNOT TRAIN:\n  - " + "\n  - ".join(p['problems'])
        self.plan_var.set(txt)
        self.plan_lbl.configure(fg=HR_COLOURS['error'][1] if p['problems'] else "#202020")
        if not self.busy:
            self.train_btn.configure(state="disabled" if p['problems'] else "normal")
        return p

    def _poll_plan(self):
        if not self.alive:
            return
        if not self.busy:
            self.refresh_plan()
        try:
            self._plan_after = self.top.after(1000, self._poll_plan)
        except tk.TclError:
            pass

    # ------------------------------------------------------------------
    def start(self):
        import copy
        import queue
        import threading
        if self.busy:
            return False
        try:
            self.app.collect()
            self.collect()
        except ValueError as exc:
            messagebox.showerror("Configuration problem", str(exc), parent=self.top)
            return False
        import ebiv_session
        p = ebiv_session.hr_training_plan(self.app.session)
        if p['problems']:
            messagebox.showerror("HR training cannot start",
                                 "\n".join(p['problems']), parent=self.top)
            return False
        if os.path.exists(p['model']) and not getattr(self, 'assume_yes', False) and \
                not messagebox.askyesno("Replace the model?",
                                        f"{p['model']}\nexists. Replace it?", parent=self.top):
            return False
        # heavy imports in the main thread, before the worker starts
        import vibe_train                                      # noqa: F401
        import vibe_import                                     # noqa: F401
        if p['source'] == 'raw':
            import vibe                                        # noqa: F401
        sess = copy.deepcopy(self.app.session)
        self.q = queue.Queue()
        self.stop_event = threading.Event()
        self.result, self.error, self.phase = None, None, None
        self.busy = True
        self._t0 = __import__('time').time()
        self._need = p['fields_needed']
        for iid in self.tree.get_children():
            self.tree.delete(iid)
        self.res_var.set("Running...")
        self.train_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.use_btn.configure(state="disabled")
        self.pb.configure(mode="indeterminate", value=0)
        self.pb.start(20)
        try:
            self.top.grab_set()
        except tk.TclError:
            pass
        logging.info("HR %s started from the dialog: %s -> %s",
                     "import" if p['source'] == 'operators' else "training",
                     ", ".join(x[1] for x in p['inputs']), p['model'])
        self.thread = threading.Thread(target=self._worker, args=(sess,), daemon=True,
                                       name="hr-train")
        self.thread.start()
        self._after = self.top.after(150, self._poll)
        return True

    def _worker(self, sess):
        import ebiv_session
        q = self.q
        try:
            rep = ebiv_session.train_hr_model(
                sess, progress=lambda i, n, eta: q.put(('prog', i, n, eta)),
                stop_event=self.stop_event, on_phase=lambda ph: q.put(('phase', ph)))
            q.put(('ok', rep))
        except ebiv_session.TrainingCancelled as exc:
            q.put(('cancel', str(exc)))
        except Exception as exc:                               # noqa: BLE001
            logging.exception("HR training failed.")
            q.put(('error', f"{type(exc).__name__}: {exc}"))

    def _poll(self):
        import queue
        if not self.alive:
            return
        done = None
        try:
            while True:
                msg = self.q.get_nowait()
                if msg[0] == 'phase':
                    self.phase = msg[1]
                    self.phase_var.set(self.PHASES.get(msg[1], msg[1]))
                    if msg[1] in ('train', 'import'):
                        self.stop_btn.configure(state="disabled")
                        self.pb.stop()
                        self.pb.configure(mode="indeterminate")
                        self.pb.start(20)
                        self.detail_var.set("")
                elif msg[0] == 'prog':
                    _, i, n, eta = msg
                    if n and n > 0:
                        self.pb.stop()
                        self.pb.configure(mode="determinate", maximum=n, value=i)
                        self.detail_var.set(f"{i} / {n} fields" + (
                            f", about {eta / 60:.1f} min left" if eta == eta else ""))
                    else:
                        self.detail_var.set(f"{i} fields so far (whole recording; the "
                                            f"split needs {self._need})")
                else:
                    done = msg
        except queue.Empty:
            pass
        if done is None:
            self._after = self.top.after(150, self._poll)
            return
        self._finish(done)

    def _finish(self, done):
        import time
        self.busy = False
        try:
            self.top.grab_release()
        except tk.TclError:
            pass
        self.pb.stop()
        self.stop_btn.configure(state="disabled")
        dt = (time.time() - self._t0) / 60
        kind = done[0]
        if kind == 'ok':
            rep = self.result = done[1]
            path = rep['model_path']
            self.app.session.run.hr.model_path = path
            self.app.fields['run.hr.model_path'].vars[0].set(path)
            self.pb.configure(mode="determinate", maximum=1, value=1)
            self.phase_var.set(f"Done in {dt:.1f} min.  Model saved and selected in panel 2.")
            self.detail_var.set(path)
            self._show_results(rep)
            self.use_btn.configure(state="normal")
            self.app.update_hr_status(force=True)
            self.app.status.set(f"HR model {'imported' if rep.get('source') == 'operators' else 'trained'}: "
                                f"{os.path.basename(path)}")
        elif kind == 'cancel':
            self.pb.configure(mode="determinate", value=0)
            self.phase_var.set(f"Stopped after {dt:.1f} min: {done[1]}")
            self.res_var.set("No model was saved.")
        else:
            self.error = done[1]
            self.pb.configure(mode="determinate", value=0)
            self.phase_var.set("FAILED (details on the Log tab of the main window).")
            self.res_var.set(done[1])
            if not getattr(self, 'quiet', False):
                messagebox.showerror("HR training failed", done[1], parent=self.top)
        self.refresh_plan()

    def _show_results(self, rep):
        if rep.get('source') == 'operators':
            g = rep.get('gamma_range')
            self.res_var.set(
                f"Imported {', '.join(HR_LABELS.get(m, m) for m in rep.get('methods', []))}: "
                f"r = {rep.get('r')} HR modes ({100 * rep.get('hr_energy', float('nan')):.1f}% "
                f"energy), r_LR = {rep.get('r_lr')}"
                + (f", gain in [{g[0]:.2f}, {g[1]:.2f}]" if g else "") + ".\n"
                f"Conversion: {rep.get('conversion')}.\nNo test report (no data here): compare "
                "the live HR field with a reference before trusting it.")
            return
        rows = [("Cubic interpolation (baseline)", 'cubic')] + [
            (HR_LABELS[m], m) for m in ('kf', 'lse', 'lse_vr')]
        deltas = {k: rep.get(f'delta_{k}') for _, k in rows}
        est = [v for k, v in deltas.items() if k != 'cubic' and v is not None]
        best = min(est) if est else None
        for name, k in rows:
            d = deltas[k]
            if d is None:
                continue
            tke = rep.get(f'tke_ratio_{k}')
            self.tree.insert("", "end", text=name + ("   <- lowest" if d == best else ""),
                             values=(f"{d:.4f}", "-" if tke is None else f"{tke:.2f}"))
        txt = (f"r = {rep.get('r')} HR modes ({100 * rep.get('hr_energy', float('nan')):.1f}% "
               f"of the HR fluctuation energy), r_LR = {rep.get('r_lr')}, "
               f"{rep.get('n_fields')} fields built.")
        g = rep.get('gamma_range')
        if g:
            txt += (f"  Method III gain in [{g[0]:.2f}, {g[1]:.2f}], "
                    f"{rep.get('gamma_n_capped')} mode(s) at the cap.")
        if best is None:
            txt += "  No test block (n_test = 0): no error report."
        if rep.get('conversion'):
            txt += f"\nExternal fields converted: {rep['conversion']}."
        else:
            txt += (f"\ndelta: mean rms vector error / U_ref ({rep.get('u_ref', float('nan')):.3g} "
                    "px/frame) against HR-LOR, i.e. the HR reference projected on the r modes: "
                    "it says how well those modes are recovered, not the error against the full "
                    "HR field. Same recording as the training. TKE ratio < 1: less fluctuation "
                    "energy than the reference.")
        self.res_var.set(txt)

    # ------------------------------------------------------------------
    def stop(self):
        if self.busy and self.stop_event is not None:
            self.stop_event.set()
            self.stop_btn.configure(state="disabled")
            self.phase_var.set("Stopping after the current field...")

    def use_live(self):
        if not self.result:
            return
        a = self.app
        a.fields['run.hr.model_path'].vars[0].set(self.result['model_path'])
        a.fields['run.hr.enabled'].vars[0].set(True)
        meths = self.result.get('methods')
        if meths and a.fields['run.hr.method'].vars[0].get() not in meths:
            a.fields['run.hr.method'].vars[0].set(meths[0])
        if a.mode_var.get() != 'stream':
            a.mode_var.set('stream')
            a._on_mode_change()
        a.update_hr_status(force=True)
        a.status.set("Live streaming with HR estimation: check panel 2, then Run.")
        self.close()
        a.root.lift()

    def close(self):
        if self.busy:
            messagebox.showinfo("HR training running",
                                "Press Stop first (the training-set build stops after the "
                                "current field), or wait for the end.", parent=self.top)
            return False
        try:
            self.collect()
        except ValueError as exc:
            if not messagebox.askyesno("Discard?", f"{exc}\n\nClose and discard these "
                                       "values?", parent=self.top):
                return False
        self.alive = False
        for a in (self._after, self._plan_after):
            if a is not None:
                try:
                    self.top.after_cancel(a)
                except tk.TclError:
                    pass
        self.top.destroy()
        self.app._hr_fp = None
        return True


# ==========================================================================

def launch(session=None, preset_path=None):
    """Open the window.  Returns when the user quits."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    if session is None:
        if preset_path and os.path.exists(preset_path):
            session, _ = Session.load(preset_path)
        elif os.path.exists(LAST_PRESET):
            try:
                session, _ = Session.load(LAST_PRESET)
                logging.info("Restored the settings from the last session.")
            except Exception:                                  # noqa: BLE001
                session = Session()
        else:
            session = Session()

    root = tk.Tk()
    try:
        ttk.Style().theme_use('vista' if sys.platform.startswith('win') else 'clam')
    except tk.TclError:
        pass
    app = EbivGUI(root, session, preset_path)
    root.mainloop()
    return app.session


def main():
    """Console entry point (`vibestream` after pip install): optional preset path."""
    launch(preset_path=sys.argv[1] if len(sys.argv) > 1 else None)


if __name__ == "__main__":
    main()
