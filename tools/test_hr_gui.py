"""
HR estimation through the GUI path: ebiv_session.run_session() with a Session
exactly as the GUI builds it.

    1. offline chain, step 'train HR model' only: .raw -> model; the model
       path is written back into the session
    2. live stream with HR enabled: the real stream_camera() and PIV worker,
       fake camera, headless OpenCV; [h] toggles LR/HR vectors, [q] quits
    3. a model trained with a different PIV window is REFUSED
       Session.validate() refuses HR enabled without a model file
    4. the main window: panel-2 status line (off/ok/warn/error/info), the
       Resolution enhancement menu, Model info, the training dialog
       (worker thread, progress, Stop, results, "Use this model live")

    xvfb-run python tools/test_hr_gui.py        (needs no display otherwise)
"""

import os
import sys
import tempfile

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(_ROOT, 'lib'), os.path.join(_ROOT, 'tools')]

import numpy as np                  # noqa: E402
import test_vibe as T               # noqa: E402  (fake Metavision)
import test_hr_train as TT          # noqa: E402  (advected-particle recording)

_PASS, _FAIL = [], []


def check(name, cond, detail=""):
    (_PASS if cond else _FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
    return bool(cond)


def run():
    import cv2
    import ebiv_session
    import vibe_hr_live
    from ebiv_config import Session

    print("=" * 76)
    print("  HR estimation through run_session() (the GUI path)")
    print("=" * 76)
    NF = 1100
    ev, _ = TT.make_recording(NF)
    TT.install(ev, NF)
    T.FakeEventsIterator.SPEED = 1.0
    base = tempfile.mkdtemp(prefix="vibe_hrgui_")
    os.makedirs(os.path.join(base, "Raw"))
    open(os.path.join(base, "Raw", "synth.raw"), "wb").write(b"FAKE")

    s = Session()
    r, c = s.run, s.control
    r.output_base_folder, r.acq_name, r.raw_filename = base, "synth", "synth.raw"
    r.f_acq, r.trigger_mode, r.trigger_duty_cycle = 1e6 / T.PERIOD, 'auto', 0.8
    r.max_events_per_pixel, r.frame_smooth_sigma = 1, 0.75
    c.piv.window_size, c.piv.node_distance = 32, 32
    c.roi.display_roi = None
    h = r.hr
    h.hr_window, h.hr_step, h.hr_levels = 16, 4, 2
    h.n_train, h.gap, h.n_val, h.n_test = 450, 50, 150, 200

    # --- 1. offline training ------------------------------------------------
    print("\n[1] offline chain: train HR model")
    r.mode = 'offline'
    r.do_record = r.do_playback = r.do_image_gen = r.do_piv_process = False
    r.do_hr_train = True
    res = ebiv_session.run_session(s)
    check("training step ran and saved a model",
          res.get('hr_model') and os.path.exists(res['hr_model']), str(res.get('hr_model')))
    check("model path written back into the session", h.model_path == res.get('hr_model'))
    out = os.path.join(base, "Out", "synth")
    check("training set saved for retraining",
          os.path.exists(os.path.join(out, "hr_model_trainingset.npz")))

    # --- 2. live stream with HR ------------------------------------------------
    print("\n[2] live stream, HR enabled (fake camera, headless OpenCV)")
    created = []
    orig = vibe_hr_live.HRLive

    class SpyHR(orig):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            created.append(self)
            self.n_draw = 0
            d = self.overlay.draw

            def draw(*aa, **kk):
                self.n_draw += 1
                return d(*aa, **kk)
            self.overlay.draw = draw
    vibe_hr_live.HRLive = SpyHR

    shown, keys = [], []

    def fake_imshow(name, img):
        shown.append(name)

    def fake_waitkey(_):
        n = len(shown)
        k = {15: ord('h'), 30: ord('h'), 60: ord('q')}.get(n, 0xFF)
        if k != 0xFF:
            keys.append(chr(k))
        return k
    cv2.imshow, cv2.waitKey = fake_imshow, fake_waitkey
    cv2.destroyAllWindows = lambda *a: None
    cv2.destroyWindow = lambda *a: None

    r.mode, r.run_mode = 'stream', 'ebiv'
    h.enabled, h.method, h.steady_state, h.show_hr = True, 'lse_vr', True, False
    ebiv_session.run_session(s)
    hr = created[0] if created else None
    check("stream ran; HRLive created", hr is not None and len(shown) >= 60,
          f"{len(shown)} display frames, keys {keys}")
    check("estimator stepped on the live fields", hr is not None and hr.est.k > 50,
          f"{hr.est.k if hr else 0} KF steps, {hr.est.n_skipped if hr else 0} periods bridged")
    check("[h] switched the display to HR vectors (and back)",
          hr is not None and hr.n_draw > 5 and hr.show is False,
          f"HR drawn on {hr.n_draw if hr else 0} frames")
    x, _ = hr.latent()
    check("latent state finite, HR field on the HR grid",
          np.all(np.isfinite(x)) and hr.field()[0].shape == tuple(hr.model.hr_shape))

    # --- 3. mismatch refused ---------------------------------------------------
    print("\n[3] consistency")
    c.piv.window_size, c.piv.node_distance = 48, 48
    try:
        ebiv_session.run_session(s)
        check("model trained with another PIV window is refused", False)
    except ValueError as e:
        check("model trained with another PIV window is refused", "window" in str(e),
              str(e).splitlines()[1].strip())
    c.piv.window_size, c.piv.node_distance = 32, 32
    r.trigger_mode = 'none'
    try:
        ebiv_session.run_session(s)
        check("fixed-dt frames (trigger 'none') refused for a phase-locked model", False)
    except ValueError as e:
        check("fixed-dt frames (trigger 'none') refused for a phase-locked model",
              "phase_locked" in str(e))
    r.trigger_mode = 'auto'

    h.model_path = os.path.join(base, "missing.npz")
    try:
        s.validate()
        check("validate(): HR enabled without a model file", False)
    except ValueError as e:
        check("validate(): HR enabled without a model file", "model file" in str(e))

    vibe_hr_live.HRLive = orig
    h.model_path = res.get('hr_model')
    gui_section(s, base)
    print("\n" + "=" * 76)
    print(f"  {len(_PASS)} passed, {len(_FAIL)} failed")
    if _FAIL:
        print("  FAILED: " + "; ".join(_FAIL))
    print("=" * 76)
    return not _FAIL


def _pump(root, cond, timeout):
    import time
    t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        root.update()
        time.sleep(0.02)
    return cond()


def gui_section(s, base):
    """The main window: panel 2 status line, menu, model info, training dialog."""
    print("\n[4] GUI: status line, menu, model info, training dialog (tkinter)")
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception as exc:                                   # noqa: BLE001
        check("tkinter display available (run under xvfb-run)", False, str(exc))
        return
    import ebiv_gui as G
    shown = []
    G.messagebox.showerror = lambda *a, **k: shown.append(('error',) + a)
    G.messagebox.showinfo = lambda *a, **k: shown.append(('info',) + a)
    G.messagebox.showwarning = lambda *a, **k: shown.append(('warn',) + a)
    G.messagebox.askyesno = lambda *a, **k: True
    s.run.mode = 'stream'
    app = G.EbivGUI(root, s)
    root.update()
    F = app.fields

    check("no HR tab; HR fields live in panel 2",
          "HR estimation" not in G.TABS_STREAM and "HR training" not in G.TABS_OFFLINE
          and all(p in F for p in ('run.hr.enabled', 'run.hr.model_path', 'run.hr.method')))
    check("training fields are not main-window widgets (dialog only)",
          not any(sp['path'] in F for sp in G.HR_TRAIN_SPEC) and len(G.HR_TRAIN_SPEC) >= 20)
    labels = [app.hr_menu.entrycget(i, 'label') for i in range(app.hr_menu.index('end') + 1)
              if app.hr_menu.type(i) != 'separator']
    check("'Resolution enhancement' menu", labels[:3] == ["Train a model...", "Load a model...",
                                                          "Model info..."], str(labels))

    def status(**set_):
        for path, v in set_.items():
            path = path.replace('__', '.')
            fld = F[path]
            fld.load(v)
        return app.update_hr_status(force=True)

    F['run.hr.enabled'].load(False)
    lv = status()
    check("OFF -> grey status 'OFF'", lv == 'off' and app.hr_text.startswith("OFF"), app.hr_text[:70])
    check("colour follows the level", app.hr_status_lbl.cget('bg') == G.HR_COLOURS['off'][0])
    lv = status(run__hr__enabled=True, run__hr__method='lse_vr')
    check("ON with the matching model -> green", lv == 'ok', app.hr_text[:110])
    lv = status(control__piv__window_size=48)
    check("window changed -> red, names the setting", lv == 'error' and 'window' in app.hr_text,
          app.hr_text[:120])
    n_run = []
    import ebiv_session
    orig_rs = ebiv_session.run_session
    ebiv_session.run_session = lambda *a, **k: n_run.append(1)
    app.run()
    ebiv_session.run_session = orig_rs
    check("Run refuses while the status is red (before opening the camera)",
          not n_run and shown and shown[-1][0] == 'error')
    info = app.show_model_info()
    check("Model info: trained vs now, the difference flagged",
          info is not None and "DIFFERENT: Run refuses" in info.text and "window" in info.text)
    info.top.destroy()
    lv = status(control__piv__window_size=32, control__piv__validation=not s.control.piv.validation)
    check("non-critical difference (validation) -> amber warning", lv == 'warn', app.hr_text[-90:])
    status(control__piv__validation=s.control.piv.validation)
    lv = status(run__hr__model_path=os.path.join(base, "nope.npz"))
    check("missing model file -> red", lv == 'error' and 'does not exist' in app.hr_text)
    good = s.run.hr.model_path
    status(run__hr__model_path=good)
    app.mode_var.set('offline')
    app._on_mode_change()
    app.step_vars['do_hr_train'].set(True)
    lv = app.update_hr_status(force=True)
    check("offline + 'train HR model' step -> blue 'TRAIN' notice", lv == 'info')
    app.step_vars['do_hr_train'].set(False)
    app.mode_var.set('stream')
    app._on_mode_change()
    check("status back to green", app.update_hr_status(force=True) == 'ok')

    # ---- training dialog ------------------------------------------------------
    dlg = app.open_train_dialog()
    dlg.assume_yes = dlg.quiet = True
    dlg.fields['run.hr.train_raw'].load(os.path.join(base, "Raw", "missing.raw"))
    p = dlg.refresh_plan()
    check("dialog: missing recording -> CANNOT TRAIN, Train disabled",
          p['problems'] and str(dlg.train_btn.cget('state')) == 'disabled')
    dlg.fields['run.hr.train_raw'].load(None)
    dlg.fields['run.hr.model_name'].load("hr_model_gui.npz")
    p = dlg.refresh_plan()
    check("dialog: LR summary taken from the live settings",
          not p['problems'] and "window 32 px" in dlg.plan_var.get(), dlg.plan_var.get()[:80])
    ok = dlg.start()
    check("training runs in a worker thread, window stays responsive",
          ok and dlg.busy and dlg.thread.is_alive())
    phases = []
    _pump(root, lambda: (phases.append(dlg.phase) or not dlg.busy), 900)
    new = os.path.join(base, "Out", "synth", "hr_model_gui.npz")
    check("training finished; model saved", dlg.result is not None and os.path.exists(new),
          str(dlg.error))
    check("progress went through both phases", 'build' in phases and 'train' in phases)
    rows = [dlg.tree.item(i, 'text') for i in dlg.tree.get_children()]
    check("results table: cubic + 3 estimators", len(rows) == 4, str(rows))
    check("new model selected in panel 2", F['run.hr.model_path'].collect() == new)
    # stop
    dlg.fields['run.hr.model_name'].load("hr_model_stop.npz")
    dlg.start()
    dlg.stop()
    _pump(root, lambda: not dlg.busy, 300)
    check("Stop: build interrupted, nothing saved",
          dlg.result is None and dlg.error is None and "Stopped" in dlg.phase_var.get()
          and not os.path.exists(os.path.join(base, "Out", "synth",
                                                                   "hr_model_stop.npz")),
          dlg.phase_var.get())
    # source selection greys out what does not apply
    dlg.fields['run.hr.source'].load('operators')
    root.update()
    greyed = dlg.fields['run.hr.train_raw'].label.instate(['disabled'])
    active = not dlg.fields['run.hr.ext_model_file'].label.instate(['disabled'])
    check("source 'operators': button 'Import', recording fields greyed, file field active",
          str(dlg.train_btn.cget('text')) == 'Import' and greyed and active)
    dlg.fields['run.hr.ext_model_file'].load(os.path.join(base, "nope.mat"))
    p = dlg.refresh_plan()
    check("missing operator file -> CANNOT TRAIN",
          any('operators file not found' in x for x in p['problems']), str(p['problems'])[:80])
    dlg.fields['run.hr.source'].load('raw')
    root.update()
    check("back to 'raw': recording fields active again",
          not dlg.fields['run.hr.train_raw'].label.instate(['disabled'])
          and str(dlg.train_btn.cget('text')) == 'Train')
    dlg.fields['run.hr.model_name'].load("hr_model_gui.npz")
    dlg.result = {'model_path': new}
    F['run.hr.enabled'].load(False)
    dlg.use_live()
    check("'Use this model live' enables it and closes the dialog",
          F['run.hr.enabled'].collect() and not dlg.alive and app.update_hr_status(True) == 'ok')
    app._on_close()


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    sys.exit(0 if run() else 1)
