"""
EBIV Release 5.2 — session dispatcher.

One entry point, run_session(), that takes a Session (acquisition parameters
plus control-system parameters) and runs either

    mode='stream'    the live camera path: EBIV, and optionally the
                     open-loop or closed-loop pump controller;

    mode='offline'   the file-based chain, in order and in one go:
                     record .raw  ->  play it back  ->  generate
                     phase-locked frames  ->  pyramidal PIV.

Both the GUI and EBIV_Main call this, so there is exactly one place where a
run is assembled and exactly one set of semantics to keep straight.

The heavy imports (numpy, OpenCV, the camera SDK) happen inside the
functions, not at module load, so that ebiv_gui.py can import this module and
show its window instantly.
"""

import os
import time
import logging


# ==========================================================================
#  Paths
# ==========================================================================

def session_directories(run_cfg, create=True):
    """
    Resolve the Release 4.0 directory convention from a RunConfig.

        <base>/Raw                     .raw recordings
        <base>/RawImg/<acq_name>       generated .tif frames
        <base>/Out/<acq_name>          PIV output, profiles, control logs
    """
    base = run_cfg.output_base_folder
    dirs = {
        'raw': os.path.join(base, "Raw"),
        'img': os.path.join(base, "RawImg", run_cfg.acq_name),
        'out': os.path.join(base, "Out", run_cfg.acq_name),
    }
    if create:
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)
    dirs['raw_file'] = os.path.join(dirs['raw'], run_cfg.raw_filename)
    return dirs


# ==========================================================================
#  Dispatcher
# ==========================================================================

def run_session(session, run_name=None, on_progress=None):
    """
    Execute one session.

    on_progress : optional callable(step_name, index, total) used by the GUI
        to report which stage of the offline chain is running.

    Returns a dict summarising what ran.
    """
    session.validate()
    run = session.run
    cfg = session.control

    dirs = session_directories(run)
    from ebiv_config import __version__
    logging.info("Release %s | mode=%s | accumulation %d us (%.1f Hz) | out=%s",
                 __version__, run.mode, run.accum_time_us, run.f_acq, dirs['out'])

    from ebiv_profiler import PipelineProfiler
    prof = PipelineProfiler(enabled=run.profile,
                            report_every_n=run.profile_report_every)

    if run.mode == 'stream':
        return _run_stream(session, dirs, prof, run_name, on_progress)
    return _run_offline(session, dirs, prof, on_progress)


# ==========================================================================

def _run_stream(session, dirs, prof, run_name, on_progress):
    from ebiv_utils import stream_camera
    run, cfg = session.run, session.control

    if on_progress:
        on_progress(f"live stream ({run.run_mode})", 1, 1)

    hr = None
    if run.hr.enabled:
        from vibe_hr_live import HRLive, live_settings_from_session
        hr = HRLive(run.hr.model_path, method=run.hr.method,
                    steady_state=run.hr.steady_state, f_acq=run.f_acq,
                    live_settings=live_settings_from_session(session),
                    show=run.hr.show_hr, arrow_skip=max(1, run.arrow_skip),
                    arrow_scale=run.arrow_scale, q_scale=run.hr.tune_q,
                    r_scale=run.hr.tune_r)

    stream_camera(
        output_dir=dirs['out'],
        accum_time_us=run.accum_time_us,
        biases=run.camera_biases,
        max_events_per_pixel=run.max_events_per_pixel,
        trigger_mode=run.trigger_mode,
        trigger_period_us=run.trigger_period_us,
        trigger_duty_cycle=run.trigger_duty_cycle,
        prof=prof,
        display_fps=cfg.supervisor.visualization_rate_hz,
        arrow_skip=run.arrow_skip,
        arrow_scale=run.arrow_scale,
        save_rt_piv=run.save_rt_piv,
        flip_x=run.flip_x, flip_y=run.flip_y,
        cfg=cfg, run_mode=run.run_mode, run_name=run_name,
        blackout_background=run.blackout_background,
        show_strip_chart=run.show_strip_chart,
        plot_history_s=run.plot_history_s,
        plot_zoom_window_s=run.plot_zoom_window_s,
        frame_smooth_sigma=run.frame_smooth_sigma or None,
        hr=hr,
    )
    return {'mode': 'stream', 'run_mode': run.run_mode, 'output_dir': dirs['out']}


def _run_offline(session, dirs, prof, on_progress):
    from ebiv_utils import (record_raw_file, check_raw_video,
                            generate_centered_images, process_offline_piv)
    run, cfg = session.run, session.control

    # The flip is baked into the .tif by image generation; this shared guard
    # stops the offline PIV re-flipping the same frames in the same run.
    flip_state = {'done': False}

    steps = []
    if run.do_record:
        steps.append('record')
    if run.do_playback:
        steps.append('playback')
    if run.do_image_gen:
        steps.append('image_gen')
    if run.do_piv_process:
        steps.append('piv')
    if run.do_hr_train:
        steps.append('hr_train')

    done = []
    for i, step in enumerate(steps, start=1):
        if on_progress:
            on_progress(step, i, len(steps))
        t0 = time.perf_counter()
        logging.info("--- offline step %d/%d: %s ---", i, len(steps), step)

        if step == 'record':
            record_raw_file(duration_sec=run.duration_sec,
                            filename=dirs['raw_file'],
                            biases=run.camera_biases)

        elif step == 'playback':
            check_raw_video(filename=dirs['raw_file'],
                            accum_time_us=run.accum_time_us,
                            max_events_per_pixel=run.max_events_per_pixel,
                            flip_x=run.flip_x, flip_y=run.flip_y)

        elif step == 'image_gen':
            prof.reset()
            generate_centered_images(
                raw_file_path=dirs['raw_file'],
                output_dir=dirs['img'],
                f_acq=run.f_acq,
                Nimg=run.n_images,
                prefix=run.img_prefix,
                apply_gaussian=run.apply_gaussian,
                gaussian_kernel=(run.gaussian_kernel, run.gaussian_kernel),
                gaussian_sigma=run.gaussian_sigma,
                burst_search_max_sec=run.burst_search_max_sec,
                roi=cfg.roi.display_roi,
                duty_cycle=run.trigger_duty_cycle,
                prof=prof,
                flip_x=run.flip_x, flip_y=run.flip_y,
                flip_state=flip_state)

        elif step == 'piv':
            prof.reset()
            process_offline_piv(
                input_dir=dirs['img'],
                output_dir=dirs['out'],
                window_size=cfg.piv.window_size,
                node_distance=cfg.piv.node_distance,
                apply_validation=cfg.piv.validation,
                val_threshold=cfg.piv.val_threshold,
                val_epsilon=cfg.piv.val_epsilon,
                pyramid_levels=run.pyramid_levels,
                prof=prof,
                use_gpu=cfg.piv.use_gpu,
                flip_x=run.flip_x, flip_y=run.flip_y,
                flip_state=flip_state)

        elif step == 'hr_train':
            _train_hr(session, dirs)

        dt = time.perf_counter() - t0
        logging.info("--- %s finished in %.1f s ---", step, dt)
        done.append((step, dt))

    return {'mode': 'offline', 'steps': done,
            'output_dir': dirs['out'], 'image_dir': dirs['img'],
            'raw_file': dirs['raw_file'],
            'hr_model': run.hr.model_path if run.do_hr_train else None}


# ==========================================================================
#  HR training (offline step 'hr_train')
# ==========================================================================

def training_settings_from_session(session):
    """vibe_train.TrainingSettings with the LR part taken from the LIVE settings."""
    from vibe_train import TrainingSettings
    run, cfg, h = session.run, session.control, session.run.hr
    roi = cfg.roi.display_roi
    return TrainingSettings(
        f_hz=float(run.f_acq), roi=list(roi) if roi is not None else None,
        duty_cycle=float(run.trigger_duty_cycle),
        lr_window=int(cfg.piv.window_size), lr_step=int(cfg.piv.node_distance),
        lr_validate=bool(cfg.piv.validation), lr_subpixel=bool(cfg.piv.subpixel),
        val_threshold=float(cfg.piv.val_threshold), val_epsilon=float(cfg.piv.val_epsilon),
        hr_window=int(h.hr_window), hr_step=int(h.hr_step), hr_levels=int(h.hr_levels),
        hr_stencil=h.hr_stencil, hr_combine=h.hr_combine, hr_predictor=bool(h.hr_predictor),
        start_frame=int(h.train_start_frame),
        n_frames=int(h.train_n_frames) if h.train_n_frames > 0 else None)


class TrainingCancelled(RuntimeError):
    pass


def hr_training_plan(session):
    """
    What a training / import run would use, without running it (GUI dialog).
    Returns dict(problems=[...], source, inputs=[(label, path, exists)],
    model=<output path>, out_dir, lr=<live LR settings>, fields_needed,
    conversion=<one-line preview for external data, or None>).
    problems empty = ready.
    """
    run, cfg, h = session.run, session.control, session.run.hr
    dirs = session_directories(run, create=False)
    problems = list(h.validate(need_model=False))
    src = h.source
    raw = (h.train_raw or '').strip() or dirs['raw_file']
    inputs = []
    if src == 'raw':
        inputs.append(("recording", raw))
        if run.trigger_mode == 'none':
            problems.append("trigger mode 'none' (fixed-dt frames): the training needs "
                            "phase-locked frames, set the trigger to auto or external "
                            "(Acquisition tab)")
    elif src == 'fields':
        lrf = (h.ext_lr_file or '').strip()
        inputs.append(("LR fields", lrf))
        if not lrf.lower().endswith('.npz'):
            inputs.append(("HR fields", (h.ext_hr_file or '').strip()))
    else:
        inputs.append(("operators", (h.ext_model_file or '').strip()))
    inputs = [(lbl, pth, bool(pth) and os.path.exists(pth)) for lbl, pth in inputs]
    for lbl, pth, ok in inputs:
        if not pth:
            problems.append(f"no {lbl} file selected")
        elif not ok:
            problems.append(f"{lbl} file not found: {pth}")
    if not str(h.model_name or '').strip():
        problems.append("model file name is empty")
    if run.f_acq <= 0:
        problems.append("acquisition frequency must be > 0")
    roi = cfg.roi.display_roi
    lr = dict(f_hz=float(run.f_acq), trigger=run.trigger_mode,
              duty_cycle=float(run.trigger_duty_cycle),
              roi=list(roi) if roi is not None else None,
              window=int(cfg.piv.window_size), step=int(cfg.piv.node_distance),
              validate=bool(cfg.piv.validation), subpixel=bool(cfg.piv.subpixel),
              smooth_sigma=float(run.frame_smooth_sigma or 0.0),
              max_events_per_pixel=int(run.max_events_per_pixel),
              flip_x=bool(run.flip_x), flip_y=bool(run.flip_y))
    conversion = None
    if src != 'raw' and all(ok for _, _, ok in inputs) and not problems:
        try:
            import vibe_import as VI
            from vibe_hr_live import live_settings_from_session
            conversion = VI.preview(src, [p for _, p, _ in inputs],
                                    live_settings_from_session(session),
                                    _external_settings(h), (h.ext_sensor_w, h.ext_sensor_h))
        except Exception as exc:                                 # noqa: BLE001
            problems.append(f"external data: {exc}")
    need = h.n_train + h.n_val + h.n_test + 2 * h.gap
    return dict(problems=problems, source=src, inputs=inputs, raw=raw, out_dir=dirs['out'],
                model=os.path.join(dirs['out'], h.model_name), lr=lr, fields_needed=need,
                conversion=conversion)


def _external_settings(h):
    import vibe_import as VI
    return VI.ExternalSettings(y_up=bool(h.ext_y_up), velocity_units=h.ext_velocity_units,
                               f_hz=float(h.ext_f_hz))


def train_hr_model(session, progress=None, stop_event=None, on_phase=None):
    """
    The training / import step on its own (GUI: Resolution enhancement > Train
    a model).  Checks only what it needs, not the whole session, then runs
    _train_hr.  Raises ValueError listing the problems.
    """
    plan = hr_training_plan(session)
    if plan['problems']:
        raise ValueError("HR training cannot start:\n  - " + "\n  - ".join(plan['problems']))
    dirs = session_directories(session.run, create=True)
    return _train_hr(session, dirs, progress=progress, stop_event=stop_event,
                     on_phase=on_phase)


def _train_hr(session, dirs, progress=None, stop_event=None, on_phase=None):
    """
    Build or import the resolution-enhancement model, by run.hr.source:
        'raw'        .raw -> LR (live rt-EBIV settings) + HR (multi-frame) -> estimators
        'fields'     external LR / HR fields (or a training set .npz) -> estimators
        'operators'  external operators (vibe_export_model.m) -> converted model
    Saves <Out>/<model_name> (+ training set .npz, + LR.mat/HR.mat if asked)
    and points run.hr.model_path at the new model.  Returns a report dict.
    """
    import vibe_train as VT
    run, h = session.run, session.run.hr
    phase = on_phase or (lambda _p: None)
    stopped = lambda: stop_event is not None and stop_event.is_set()       # noqa: E731

    if h.source in ('fields', 'operators'):
        import vibe_import as VI
        from vibe_hr_live import live_settings_from_session
        live = live_settings_from_session(session)
        ext = _external_settings(h)
        sensor = (h.ext_sensor_w, h.ext_sensor_h)
        if h.source == 'operators':
            phase('import')
            model, conv = VI.import_operators(h.ext_model_file, live, ext, sensor)
            path = model.save(os.path.join(dirs['out'], h.model_name))
            h.model_path = path
            logging.info("HR model imported: %s  (%s)", path, model.summary())
            phase('done')
            return dict(source='operators', model_path=path, summary=model.summary(),
                        r=model.r, r_lr=model.r_lr, hr_energy=model.meta['hr_energy'],
                        methods=list(model.methods), conversion=conv.summary(),
                        gamma_range=model.meta.get('gamma_range'))
        phase('load')
        ds, conv = VI.fields_training_set(h.ext_lr_file, h.ext_hr_file, live, ext, sensor)
        if stopped():
            raise TrainingCancelled("stopped after loading; nothing was saved")
        conv_txt = conv.summary() if conv is not None else None
    else:
        from vibe import VIBE
        raw = (h.train_raw or '').strip() or dirs['raw_file']
        if not os.path.exists(raw):
            raise FileNotFoundError(f"HR training: raw file not found: {raw}")
        vibe = VIBE(biases=run.camera_biases, flip_x=run.flip_x, flip_y=run.flip_y,
                    max_events_per_pixel=run.max_events_per_pixel, output_dir=dirs['out'],
                    smooth_sigma=run.frame_smooth_sigma or None)
        st = training_settings_from_session(session)

        def _prog(i, n, eta):
            logging.info("HR training set: %d / %s fields%s", i, n if n > 0 else "?",
                         f", ~{eta / 60:.1f} min left" if eta == eta and n > 0 else "")
            if progress:
                progress(i, n, eta)

        logging.info("HR training from %s: LR = live settings (window %d, step %d, ROI %s); "
                     "HR = window %d, step %d, %d separations, %s stencil, %s, predictor %s",
                     raw, st.lr_window, st.lr_step, st.roi, st.hr_window, st.hr_step,
                     st.hr_levels, st.hr_stencil, st.hr_combine, st.hr_predictor)
        phase('build')
        try:
            ds = VT.build_training_set(vibe, raw, st, progress=_prog, stop_event=stop_event)
        except ValueError:
            if stopped():                    # stopped before the first field
                raise TrainingCancelled("stopped before the first field; nothing was saved")
            raise
        if stopped():
            raise TrainingCancelled(f"stopped after {len(ds['lr_u'])} fields; nothing was saved")
        conv_txt = None

    phase('train')
    base = os.path.splitext(h.model_name)[0]
    from_npz = h.source == 'fields' and str(h.ext_lr_file).lower().endswith('.npz')
    if h.save_training_set and not from_npz:
        VT.save_training_set(os.path.join(dirs['out'], base + "_trainingset.npz"), ds)
    if h.export_matlab:
        VT.export_matlab(os.path.join(dirs['out'], base + "_matlab"), ds)
    split = VT.SplitSettings(n_train=h.n_train, n_val=h.n_val, n_test=h.n_test, gap=h.gap)
    elbow = dict(smooth=h.elbow_smooth, span=h.elbow_span)
    model, report = VT.train_from_set(
        ds, split, u_ref=(h.u_ref or None), rank=(h.rank if h.rank > 0 else 'elbow'),
        truncate_lr=h.truncate_lr, threshold=h.elbow_threshold, elbow=elbow,
        lambda_c=h.lambda_c, max_gain=h.vr_max_gain, q_scale=h.tune_q, r_scale=h.tune_r)
    if ds.get('source'):
        model.meta['source'] = ds['source'].get('kind') if isinstance(ds['source'], dict) \
            else str(ds['source'])
        model.meta['source_detail'] = ds['source']
        if isinstance(ds['source'], dict) and ds['source'].get('kind') == 'external fields':
            model.meta['lr_processing_declared'] = True
    path = model.save(os.path.join(dirs['out'], h.model_name))
    h.model_path = path
    logging.info("HR model saved: %s  (%s)", path, model.summary())
    if 'delta_kf' in report:
        logging.info("Test block, delta vs HR-LOR (Eq. 34): cubic %s | KF %.4f | LSE %.4f | "
                     "LSE+VR %.4f  (u_ref %.3g px/frame)",
                     f"{report['delta_cubic']:.4f}" if 'delta_cubic' in report else "n/a",
                     report['delta_kf'], report['delta_lse'], report['delta_lse_vr'],
                     report['u_ref'])
    report = dict(report, source=h.source, model_path=path, summary=model.summary(),
                  n_fields=len(ds['lr_u']), conversion=conv_txt,
                  gamma_range=model.meta.get('gamma_range'),
                  gamma_n_capped=model.meta.get('gamma_n_capped'))
    phase('done')
    return report
