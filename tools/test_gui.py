"""
EBIV Release 5.1 — GUI tests.

Widget-level checks that need tkinter.  They are also run automatically by
test_release5.py section 11 when the interpreter has tkinter; this file exists
so they can be run on their own, and so they can be run under a virtual
display on a headless machine:

    python test_gui.py
    xvfb-run -a python test_gui.py           # headless

No camera, no Analog Discovery, and nothing is executed: only the parameter
window is built, filled, read back and torn down.
"""

# Run from anywhere: put the parent folder (the library) on the path.
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)                      # EBIV_Main.py
_sys.path.insert(0, _os.path.join(_ROOT, 'lib'))  # the library


import os
import sys
import json
import tempfile

_PASS, _FAIL = [], []


def check(name, cond, detail=""):
    (_PASS if cond else _FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
    return bool(cond)


def run():
    print("=" * 74)
    from ebiv_config import __version__ as _rel
    print(f"  EBIV Release {_rel} — GUI tests")
    print("=" * 74)

    try:
        import tkinter as tk
    except Exception as exc:                                   # noqa: BLE001
        print(f"  tkinter is not available in this interpreter ({exc}).")
        print("  Install it (python3-tk on Linux; it ships with Anaconda and")
        print("  with the python.org Windows installer) and re-run.")
        return 1

    try:
        root = tk.Tk()
    except Exception as exc:                                   # noqa: BLE001
        print(f"  No display available ({exc}). Run under xvfb-run, or on a "
              f"machine with a desktop.")
        return 1

    from ebiv_gui import (EbivGUI, check_spec, PARAM_SPEC, INTENTIONALLY_HIDDEN,
                          get_path, set_path, TABS_STREAM, TABS_OFFLINE, Field)
    from ebiv_config import Session

    # --- specification integrity ---------------------------------------
    bad, unaccounted = check_spec()
    check("every GUI parameter resolves to a real config field", not bad, str(bad))
    check("every config field is either on a tab or explicitly hidden",
          not unaccounted, str(unaccounted))
    check("no parameter is listed twice",
          len({s['path'] for s in PARAM_SPEC}) == len(PARAM_SPEC))
    check("every parameter has a label and a known kind",
          all(s['label'] and s['kind'] in
              ('float', 'int', 'bool', 'text', 'choice', 'folder', 'file',
               'roi', 'optfloat', 'floats', 'biases') for s in PARAM_SPEC))
    bad_choice = [s['path'] for s in PARAM_SPEC
                  if s['kind'] == 'choice' and not s['extra']]
    check("every choice parameter lists its options", not bad_choice, str(bad_choice))
    print(f"        ({len(PARAM_SPEC)} parameters exposed, "
          f"{len(INTENTIONALLY_HIDDEN)} deliberately hidden)")

    # --- choice options must match what the config accepts ---------------
    session = Session()
    bad_opts = []
    for s in PARAM_SPEC:
        if s['kind'] != 'choice':
            continue
        for opt in s['extra']:
            try:
                set_path(session, s['path'], opt)
                # re-run the owning dataclass's validation
                owner = session
                for part in s['path'].split('.')[:-1]:
                    owner = getattr(owner, part)
                if hasattr(owner, '__post_init__'):
                    owner.__post_init__()
            except Exception:                                  # noqa: BLE001
                bad_opts.append(f"{s['path']}={opt}")
    check("every offered choice is actually accepted by the config",
          not bad_opts, str(bad_opts))

    # --- build the window ------------------------------------------------
    app = EbivGUI(root, Session())
    root.update_idletasks()
    check("the window builds", app.nb is not None)
    check("a widget exists for every parameter",
          len(app.fields) == len(PARAM_SPEC),
          f"{len(app.fields)} widgets for {len(PARAM_SPEC)} parameters")

    # --- tab visibility follows the mode ----------------------------------
    def visible_tabs():
        return [app.nb.tab(t, "text") for t in app.nb.tabs()]

    app.mode_var.set("stream")
    app._on_mode_change()
    root.update_idletasks()
    check("streaming shows the streaming tabs",
          visible_tabs() == TABS_STREAM, str(visible_tabs()))
    check("the streaming sub-mode is selectable in streaming mode",
          str(app.run_mode_box.cget("state")) == "readonly")

    app.mode_var.set("offline")
    app._on_mode_change()
    root.update_idletasks()
    check("the offline chain shows only the offline tabs",
          visible_tabs() == TABS_OFFLINE, str(visible_tabs()))
    check("the streaming sub-mode is disabled in offline mode",
          str(app.run_mode_box.cget("state")) == "disabled")
    check("controller and hardware tabs are hidden in offline mode",
          "Controller" not in visible_tabs() and "Hardware" not in visible_tabs())

    app.mode_var.set("stream")
    app._on_mode_change()

    # --- round-trip through the widgets -----------------------------------
    app.session = Session()
    app.refresh()
    before = Session().to_dict()
    after = app.collect().to_dict()
    check("defaults survive a widget round-trip unchanged", before == after)
    if before != after:
        a = json.dumps(before, sort_keys=True, indent=1).splitlines()
        b = json.dumps(after, sort_keys=True, indent=1).splitlines()
        for x, y in zip(a, b):
            if x != y:
                print(f"          {x.strip()}   ->   {y.strip()}")

    # --- edited values survive the round-trip -----------------------------
    edits = {
        'run.f_acq': 250.0,
        'run.acq_name': "JetTest_01",
        'run.flip_x': True,
        'control.piv.window_size': 64,
        'control.measurement.component': '-u',
        'control.measurement.calibration_px_per_mm': 18.5,
        'control.pid.kp': 1.25,
        'control.pid.slew_rate_v_per_s': None,
        'control.roi.display_roi': [100, 900, 50, 500],
        'control.roi.control_roi': [300, 600, 200, 400],
        'control.calibration.voltages': [0.0, 0.5, 1.0, 2.5],
        'run.camera_biases': {'bias_diff_on': 55, 'bias_diff_off': 130,
                              'bias_hpf': 71, 'bias_fo': 2, 'bias_refr': 88,
                              'bias_diff': 1},
    }
    for path, value in edits.items():
        app.fields[path].load(value)
    got = app.collect()
    mismatches = [f"{p}: set {v!r} got {get_path(got, p)!r}"
                  for p, v in edits.items() if get_path(got, p) != v]
    check("edited values of every widget kind survive the round-trip",
          not mismatches, "; ".join(mismatches))

    # --- empty optional fields become None, not 0 -------------------------
    for path in ('control.pid.slew_rate_v_per_s',
                 'control.measurement.calibration_px_per_mm',
                 'control.measurement.max_abs_displacement_px'):
        app.fields[path].vars[0].set("")
    app.fields['control.roi.control_roi'].load(None)
    got = app.collect()
    check("an empty optional field reads back as None, not zero",
          all(get_path(got, p) is None for p in
              ('control.pid.slew_rate_v_per_s',
               'control.measurement.calibration_px_per_mm',
               'control.measurement.max_abs_displacement_px'))
          and got.control.roi.control_roi is None)

    # --- bad input is reported, not swallowed -----------------------------
    app.fields['run.f_acq'].vars[0].set("not a number")
    try:
        app.collect()
        raised = False
        msg = ""
    except ValueError as exc:
        raised, msg = True, str(exc)
    check("a non-numeric entry is reported with the field name",
          raised and "Acquisition frequency" in msg, msg.replace("\n", " ")[:80])
    app.fields['run.f_acq'].vars[0].set("500")

    # --- a partially filled ROI is rejected --------------------------------
    f = app.fields['control.roi.display_roi']
    f.vars[0].set("100")
    f.vars[1].set("")
    f.vars[2].set("")
    f.vars[3].set("")
    try:
        app.collect()
        roi_raised = False
    except ValueError:
        roi_raised = True
    check("a half-filled ROI is rejected rather than guessed at", roi_raised)
    f.load([100, 900, 50, 500])

    # --- validation catches a ControlROI outside the DisplayROI ------------
    app.fields['control.roi.display_roi'].load([100, 500, 100, 400])
    app.fields['control.roi.control_roi'].load([600, 700, 150, 300])
    check("validation rejects a ControlROI outside the DisplayROI",
          not app.validate(quiet=True))
    app.fields['control.roi.control_roi'].load([200, 400, 150, 300])
    check("validation accepts a ControlROI inside the DisplayROI",
          app.validate(quiet=True))

    # --- edits survive a mode switch ---------------------------------------
    app.fields['run.f_acq'].vars[0].set("321")
    app.fields['control.pid.kp'].vars[0].set("7.5")
    app.fields['control.roi.display_roi'].load([100, 900, 50, 500])
    app.fields['control.roi.control_roi'].load([300, 600, 200, 400])
    app.mode_var.set("offline")
    app._on_mode_change()
    app.mode_var.set("stream")
    app._on_mode_change()
    root.update_idletasks()
    s = app.collect()
    check("edits survive switching mode and back",
          s.run.f_acq == 321.0 and s.control.pid.kp == 7.5
          and s.control.roi.control_roi == [300, 600, 200, 400],
          f"f_acq={s.run.f_acq} kp={s.control.pid.kp}")

    # --- the mouse wheel must scroll ONE canvas, not all of them -------------
    def all_canvases(w, acc=None):
        acc = [] if acc is None else acc
        for ch in w.winfo_children():
            if isinstance(ch, tk.Canvas):
                acc.append(ch)
            all_canvases(ch, acc)
        return acc

    cans = all_canvases(root)
    check("no global wheel binding exists while the pointer is elsewhere",
          not root.bind_all("<MouseWheel>"),
          f"{len(cans)} scrollable tabs")
    app.nb.select(app.tab_frames["Controller"])
    root.update_idletasks()
    target = app.tab_frames["Controller"].canvas
    target.event_generate("<Enter>")
    root.update_idletasks()
    before = [c.yview() for c in cans]
    for _ in range(3):
        root.event_generate("<MouseWheel>", delta=-120)
    root.update_idletasks()
    after = [c.yview() for c in cans]
    moved = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
    check("one wheel notch scrolls exactly the tab under the pointer",
          len(moved) == 1 and cans[moved[0]] is target,
          f"{len(moved)} of {len(cans)} canvases moved")
    target.event_generate("<Leave>")
    root.update_idletasks()
    check("the wheel binding is released when the pointer leaves",
          not root.bind_all("<MouseWheel>"))

    # --- a second window must not duplicate the log handler -----------------
    import logging as _lg
    n_before = len([h for h in _lg.getLogger().handlers
                    if h.__class__.__name__ == 'TextHandler'])
    second = tk.Toplevel()
    app2 = EbivGUI(second, Session())
    n_after = len([h for h in _lg.getLogger().handlers
                   if h.__class__.__name__ == 'TextHandler'])
    check("opening a second window does not duplicate the log capture",
          n_before == 1 and n_after == 1, f"{n_before} -> {n_after}")
    app2._on_close()
    check("closing a window detaches its log handler",
          not [h for h in _lg.getLogger().handlers
               if h.__class__.__name__ == 'TextHandler'])

    # --- validation of self-inconsistent ROIs --------------------------------
    app.fields['control.roi.display_roi'].load([500, 100, 50, 400])
    check("an inverted DisplayROI is rejected", not app.validate(quiet=True))
    app.fields['control.roi.display_roi'].load([0, 40, 0, 40])
    app.fields['control.roi.control_roi'].load(None)
    app.fields['control.piv.window_size'].vars[0].set("48")
    check("a DisplayROI smaller than the interrogation window is rejected",
          not app.validate(quiet=True))
    app.fields['control.roi.display_roi'].load([100, 900, 50, 500])
    app.fields['control.roi.control_roi'].load([300, 600, 200, 400])
    check("a sane configuration validates again", app.validate(quiet=True))

    # --- reference='file' needs a file ---------------------------------------
    app.fields['control.reference.kind'].vars[0].set("file")
    check("reference kind 'file' with no path is rejected",
          not app.validate(quiet=True))
    app.fields['control.reference.kind'].vars[0].set("constant")

    # --- preset save/load through the GUI's own session ---------------------
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "gui_preset.json")
        app.collect().save(p)
        loaded, ignored = Session.load(p)
        check("the GUI's session saves and reloads as a preset",
              loaded.to_dict() == app.session.to_dict() and not ignored)

    # ==================================================================
    #  Release 5.2 additions
    #
    #  Every parameter added in 5.2 must be reachable from the window, bound
    #  to a real field, and survive a save/load.  The subtle one is the last
    #  group: resolve_units() and resolve_laser_frequency() OVERWRITE the
    #  fields they read, and the GUI keeps ONE Session alive across runs, so
    #  a preset written after a run must still record what the user asked
    #  for, not what was derived from that run's f_acq.
    # ==================================================================
    print("\n--- Release 5.2 parameters ---")
    NEW = {
        'control.reference.kind':                 'choice',
        'control.reference.live_step':            'float',
        'control.reference.live_min':             'optfloat',
        'control.reference.live_max':             'optfloat',
        'control.piv.use_gpu':                    'bool',
        'control.ad3.laser_frequency_hz':         'optfloat',
        'control.ad3.laser_idle_low':             'bool',
        'control.measurement.uncalibrated_units': 'choice',
    }
    spec = {p['path']: p for p in PARAM_SPEC}
    missing = [k for k in NEW if k not in spec]
    check("every Release 5.2 parameter is on a tab", not missing, str(missing))
    wrong = [k for k, v in NEW.items() if k in spec and spec[k]['kind'] != v]
    check("each uses the right widget kind", not wrong,
          str([(k, spec[k]['kind']) for k in wrong]))
    nohelp = [k for k in NEW if k in spec and not spec[k]['help']]
    check("each has hover help", not nohelp, str(nohelp))
    check("'live' is offered as a trajectory",
          'live' in (spec['control.reference.kind']['extra'] or []))
    check("both unit bases are offered",
          set(spec['control.measurement.uncalibrated_units']['extra'] or [])
          == {'px/s', 'px/frame'})

    # Widget round-trip, blank included: a blank optfloat MUST come back as
    # None, because None is what "follow f_acq" and "unbounded" are encoded as.
    rt_root = tk.Tk(); rt_root.withdraw()
    bad_rt = []
    cases = {'optfloat': [None, 12.5], 'float': [3.25], 'bool': [True, False]}
    for path, kind in NEW.items():
        e = spec[path]
        vals = cases.get(kind) or [(e['extra'] or ['x'])[0], (e['extra'] or ['x'])[-1]]
        f = Field(e, tk.Frame(rt_root), 0)
        for v in vals:
            f.load(v)
            got = f.collect()
            if not ((got is None and v is None) or got == v):
                bad_rt.append((path, v, got))
    rt_root.destroy()
    check("every Release 5.2 widget round-trips, blank included",
          not bad_rt, str(bad_rt))

    with tempfile.TemporaryDirectory() as d:
        s52 = Session()
        s52.run.f_acq = 90.0
        s52.control.reference.kind = 'live'
        s52.control.reference.live_step = 25.0
        s52.control.reference.live_max = 900.0
        s52.control.piv.use_gpu = True
        s52.control.ad3.laser_frequency_hz = None
        s52.resolve_laser_frequency()
        s52.control.measurement.resolve_units(1 / 90.0)
        path = os.path.join(d, 'p.json')
        s52.save(path)
        raw = json.load(open(path, encoding='utf-8'))
        check("a preset saved AFTER a run still stores 'follow f_acq'",
              raw['control']['ad3']['laser_frequency_hz'] is None,
              repr(raw['control']['ad3']['laser_frequency_hz']))
        check("a preset saved AFTER a run still stores the declared scale",
              raw['control']['measurement']['velocity_scale'] == 1.0,
              repr(raw['control']['measurement']['velocity_scale']))
        check("saving does not disturb the running session",
              s52.control.ad3.laser_frequency_hz == 90.0
              and s52.control.measurement.velocity_scale == 90.0)
        back, ign = Session.from_dict(raw)
        check("the preset reloads with nothing ignored", not ign, str(ign))
        check("the live reference and use_gpu survive the round-trip",
              back.control.reference.kind == 'live'
              and back.control.reference.live_step == 25.0
              and back.control.reference.live_max == 900.0
              and back.control.piv.use_gpu is True)
        back.run.f_acq = 30.0
        back.resolve_laser_frequency()
        back.control.measurement.resolve_units(1 / 30.0)
        check("a reloaded preset still follows a NEW f_acq",
              back.control.ad3.laser_frequency_hz == 30.0
              and back.control.measurement.velocity_scale == 30.0,
              f"laser={back.control.ad3.laser_frequency_hz}, "
              f"scale={back.control.measurement.velocity_scale}")

    # ------------------------------------------------------------------
    #  The reported sequence: blank the laser box, Validate, save, reopen.
    #
    #  validate() resolves the laser rate INTO the session, refresh() pushes
    #  the session back into the widgets, and collect() reads the widgets as
    #  the user's intent.  Without restore_declared() in refresh() and
    #  forget_resolution() in collect(), a blanked "follow f_acq" box came
    #  back filled with f_acq and then behaved as an explicit override.
    # ------------------------------------------------------------------
    print("\n--- blank the laser box, Validate, save, reopen ---")
    F_LASER = 'control.ad3.laser_frequency_hz'
    with tempfile.TemporaryDirectory() as d:
        preset = os.path.join(d, 'cycle.json')
        r3 = tk.Tk(); r3.withdraw()
        a3 = EbivGUI(r3)
        a3.fields['run.f_acq'].load(125.0)
        a3.fields[F_LASER].load(None)
        a3.collect()
        check("a blank laser box collects as None (follow f_acq)",
              a3.session.control.ad3.laser_frequency_hz is None)

        a3.validate(quiet=True)
        check("Validate resolves the laser rate in the session",
              a3.session.control.ad3.laser_frequency_hz == 125.0,
              repr(a3.session.control.ad3.laser_frequency_hz))

        a3.refresh()
        check("refresh() leaves the box BLANK, not filled with f_acq",
              a3.fields[F_LASER].collect() is None,
              f"box shows {a3.fields[F_LASER].collect()!r}")

        a3.collect()
        a3.session.save(preset)
        raw = json.load(open(preset, encoding='utf-8'))
        check("the saved preset stores null, not 125",
              raw['control']['ad3']['laser_frequency_hz'] is None,
              repr(raw['control']['ad3']['laser_frequency_hz']))
        check("the saved preset kept f_acq", raw['run']['f_acq'] == 125.0,
              repr(raw['run']['f_acq']))

        # Reopen: load that preset into a fresh window.
        r4 = tk.Tk(); r4.withdraw()
        a4 = EbivGUI(r4)
        sess4, ign4 = Session.from_dict(json.load(open(preset, encoding='utf-8')))
        a4.session = sess4
        a4.refresh()
        check("the reopened window shows the box blank",
              a4.fields[F_LASER].collect() is None,
              f"box shows {a4.fields[F_LASER].collect()!r}")
        check("the reopened window shows f_acq = 125",
              a4.fields['run.f_acq'].collect() == 125.0)

        # And it must ADAPT when f_acq changes.
        a4.fields['run.f_acq'].load(60.0)
        a4.collect()
        a4.validate(quiet=True)
        check("changing f_acq in the GUI re-derives the laser rate",
              a4.session.control.ad3.laser_frequency_hz == 60.0,
              repr(a4.session.control.ad3.laser_frequency_hz))
        a4.refresh()
        check("and the box is still blank after that",
              a4.fields[F_LASER].collect() is None)

        # An EXPLICIT number must survive the same cycle untouched.
        a4.fields[F_LASER].load(500.0)
        a4.collect(); a4.validate(quiet=True); a4.refresh()
        check("an explicit laser rate survives Validate and refresh",
              a4.fields[F_LASER].collect() == 500.0
              and a4.session.control.ad3.laser_frequency_hz == 500.0,
              f"box={a4.fields[F_LASER].collect()!r}")
        a4.fields['run.f_acq'].load(90.0)
        a4.collect(); a4.validate(quiet=True)
        check("an explicit laser rate does NOT follow f_acq",
              a4.session.control.ad3.laser_frequency_hz == 500.0,
              "pulse-pair and external-laser setups depend on this")
        r3.destroy(); r4.destroy()

    root.destroy()

    print(f"\n{'=' * 74}\n  {len(_PASS)} passed, {len(_FAIL)} failed")
    for n in _FAIL:
        print(f"    - {n}")
    print("=" * 74)
    return 1 if _FAIL else 0


if __name__ == "__main__":
    sys.exit(run())
