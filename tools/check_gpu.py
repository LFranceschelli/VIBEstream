"""Does the GPU pyramidal PIV agree with the CPU path ON THIS MACHINE?

    python tools/check_gpu.py
    python tools/check_gpu.py --levels 3 --window 64 --node 48

Run this ONCE on the machine with the GPU before you trust any GPU vector.

Why it exists
-------------
Release 4.0's offline GPU path carried two defects (AUDIT_R4_to_R5.md, B7 and
B8).  B7 raised NameError at the first snapshot.  B8 built the homothetic
rescaling backwards: `F.affine_grid` produces a SAMPLING grid, so its matrix
maps output coordinates back to input and therefore acts as the inverse of the
intended transform.  Asking for 1/k magnified the correlation plane by k
instead of shrinking it.  A third, subtler error sat underneath: with
align_corners=False the natural centre is pixel (ws-1)/2, while the CPU path
scales about ws//2 and measures from ws//2, so on an even window the two
differed by half a pixel — a systematic sub-pixel bias of 0.5*(1 - 1/k).

All three are fixed in Release 5.2.  The fix was verified against PyTorch on
CPU tensors, because the machine that wrote it has no CUDA.  The arithmetic is
device-independent, but "should be" is not "is", so this script closes the gap
by running the ACTUAL shipping pipeline — process_offline_piv — twice over the
same synthetic frames, once on each backend, and comparing the fields.

What a good result looks like
-----------------------------
Both backends recover the displacement that was synthesised, and the CPU-GPU
difference is at float32 interpolation level (a few times 1e-3 px), not at the
level of whole pixels.  A difference of ~0.25-0.4 px that GROWS with the number
of pyramid levels is the centre bug.  A difference of many pixels, or peaks
pinned at the plane edge, is the direction bug.
"""

import argparse
import os
import shutil
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB = os.path.join(os.path.dirname(_HERE), "lib")
if os.path.isdir(_LIB) and _LIB not in sys.path:
    sys.path.insert(0, _LIB)

import numpy as np                                              # noqa: E402
import cv2                                                      # noqa: E402

BAR = "-" * 74


def synth_frames(directory, n_frames, dx, dy, h=256, w=352, n_part=4000, seed=7):
    """Particle images translated by a known (dx, dy) per frame.

    A uniform translation is the right test here: the multi-dt pyramid assumes
    the displacement grows linearly with the frame separation, which is exactly
    what the homothetic rescaling undoes.  If the rescaling is wrong, levels
    beyond the first pull the summed peak away from the true displacement, and
    the error grows with the number of levels.
    """
    rng = np.random.default_rng(seed)
    pad = 60
    ys = rng.uniform(0, h + 2 * pad, n_part)
    xs = rng.uniform(0, w + 2 * pad, n_part)
    for k in range(n_frames):
        f = np.zeros((h + 2 * pad, w + 2 * pad), np.float32)
        yi = np.clip((ys + k * dy).astype(int), 0, f.shape[0] - 1)
        xi = np.clip((xs + k * dx).astype(int), 0, f.shape[1] - 1)
        np.add.at(f, (yi, xi), 255.0)
        f = cv2.GaussianBlur(f, (5, 5), 1.0)
        crop = f[pad:pad + h, pad:pad + w]
        crop = np.clip(crop, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(directory, f"frame_{k:04d}.tif"), crop)


def run_piv(img_dir, out_dir, use_gpu, args):
    from ebiv_utils import process_offline_piv
    os.makedirs(out_dir, exist_ok=True)
    process_offline_piv(
        input_dir=img_dir, output_dir=out_dir,
        window_size=args.window, node_distance=args.node,
        apply_validation=False,
        pyramid_levels=args.levels,
        use_gpu=use_gpu,
        flip_x=False, flip_y=False)
    return out_dir


def load_fields(out_dir):
    """Read back whatever process_offline_piv wrote, as (U, V) stacks."""
    from scipy.io import loadmat
    mats = sorted(f for f in os.listdir(out_dir) if f.endswith('.mat'))
    if not mats:
        raise RuntimeError(f"no .mat written in {out_dir}")
    Us, Vs = [], []
    for m in mats:
        d = loadmat(os.path.join(out_dir, m))
        if 'U' in d and 'V' in d:
            U, V = np.asarray(d['U'], float), np.asarray(d['V'], float)
            if U.ndim == 3:                      # a stack in one file
                for i in range(U.shape[-1]):
                    Us.append(U[..., i]); Vs.append(V[..., i])
            else:
                Us.append(U); Vs.append(V)
    if not Us:
        raise RuntimeError(f"no U/V arrays inside the .mat files in {out_dir}")
    return np.array(Us), np.array(Vs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--levels', type=int, default=3)
    ap.add_argument('--window', type=int, default=64)
    ap.add_argument('--node', type=int, default=48)
    ap.add_argument('--dx', type=float, default=2.5)
    ap.add_argument('--dy', type=float, default=-1.25)
    ap.add_argument('--keep', action='store_true', help="keep the temp folder")
    args = ap.parse_args()

    import logging
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")

    print(BAR)
    print("  Interpreter")
    print(BAR)
    # torch is installed PER INTERPRETER. Saying "torch is not installed"
    # without saying which python asked is how a working setup gets mistaken
    # for a broken one: VibeStream runs a venv, a bare `python` at the prompt
    # is usually something else entirely.
    print(f"  python      {sys.version.split()[0]}")
    print(f"  executable  {sys.executable}")
    print()
    print(BAR)
    print("  GPU availability")
    print(BAR)
    try:
        import torch
        print(f"  torch                  {torch.__version__}")
        print(f"  torch.cuda.is_available  {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  device                 {torch.cuda.get_device_name(0)}")
            print(f"  CUDA runtime           {torch.version.cuda}")
        else:
            print("\n  No CUDA device. The comparison below still runs, but the")
            print("  'GPU' column is PyTorch on the CPU. That validates the")
            print("  arithmetic and nothing about your CUDA installation.")
    except ImportError:
        print("  torch is NOT installed IN THIS INTERPRETER.")
        print()
        print("  That is a statement about the python printed above, not about")
        print("  the machine. Check the interpreter VibeStream.bat reports at")
        print("  startup, and re-run this with that same python:")
        print()
        print('      "<that python.exe>" tools\\check_gpu.py')
        print()
        print("  If it is genuinely absent there, install the CUDA build that")
        print("  matches your driver, e.g.")
        print("      pip install torch --index-url https://download.pytorch.org/whl/cu121")
        print(BAR)
        return 1

    from ebiv_gpu import HAS_TORCH, is_gpu_available
    print(f"  ebiv_gpu.is_gpu_available()  {is_gpu_available()}")

    tmp = tempfile.mkdtemp(prefix="vibestream_gpu_check_")
    try:
        img = os.path.join(tmp, "img")
        os.makedirs(img)
        n_frames = args.levels + 4
        synth_frames(img, n_frames, args.dx, args.dy)

        print()
        print(BAR)
        print(f"  Synthetic field: dx = {args.dx:+.3f} px/frame, "
              f"dy = {args.dy:+.3f} px/frame")
        print(f"  window {args.window}, node {args.node}, "
              f"pyramid levels {args.levels}, {n_frames} frames")
        print(BAR)

        run_piv(img, os.path.join(tmp, "cpu"), False, args)
        Uc, Vc = load_fields(os.path.join(tmp, "cpu"))

        # Force the GPU branch even when is_gpu_available() is False, so the
        # arithmetic can still be compared on a machine with no CUDA.
        import ebiv_utils
        saved = ebiv_utils._HAS_GPU
        ebiv_utils._HAS_GPU = HAS_TORCH
        try:
            run_piv(img, os.path.join(tmp, "gpu"), True, args)
        finally:
            ebiv_utils._HAS_GPU = saved
        Ug, Vg = load_fields(os.path.join(tmp, "gpu"))

        n = min(len(Uc), len(Ug))
        Uc, Vc, Ug, Vg = Uc[:n], Vc[:n], Ug[:n], Vg[:n]

        def stats(name, a, b, truth):
            d = np.abs(a - b)
            med = float(np.nanmedian(d))
            p95 = float(np.nanpercentile(d, 95))
            mx = float(np.nanmax(d))
            print(f"  {name}")
            print(f"      CPU  median {np.nanmedian(a):+8.4f}   "
                  f"error vs truth {np.nanmedian(a) - truth:+8.4f} px")
            print(f"      GPU  median {np.nanmedian(b):+8.4f}   "
                  f"error vs truth {np.nanmedian(b) - truth:+8.4f} px")
            print(f"      |CPU - GPU|  median {med:.5f}   "
                  f"p95 {p95:.5f}   max {mx:.5f} px")
            return med, p95, mx

        print()
        print(BAR)
        print(f"  Comparison over {n} snapshots")
        print(BAR)
        du = stats("U (streamwise)", Uc, Ug, args.dx)
        print()
        dv = stats("V (row, positive downward)", Vc, Vg, args.dy)

        print()
        print(BAR)
        print("  Verdict")
        print(BAR)
        med = max(du[0], dv[0])
        p95 = max(du[1], dv[1])
        mx = max(du[2], dv[2])

        # The p95 matters as much as the median, and for a reason worth
        # spelling out: the level-1 correlation term usually dominates the sum,
        # so a broken rescaling can leave MOST windows on the right peak while
        # throwing a large minority several pixels off.  Release 4.0's own code
        # does exactly that here — median 0.06 px, p95 20 px.  Judging on the
        # median alone would have called that "suspect" instead of "broken".
        if med < 0.02 and p95 < 0.05:
            print(f"  PASS  the backends agree: median {med:.5f} px, "
                  f"p95 {p95:.5f} px.")
            print("        Differences at this level are float32 and bilinear")
            print("        interpolation, not an algorithmic disagreement.")
            rc = 0
        elif med > 0.5 or p95 > 0.5:
            print(f"  FAIL  median {med:.4f} px, p95 {p95:.4f} px, max {mx:.4f} px.")
            if med < 0.5 <= p95:
                print("        Note the SMALL median with a LARGE p95: most")
                print("        vectors agree and a minority is badly wrong.")
                print("        That is the signature of a broken homothetic")
                print("        rescaling, not of noise.")
            print("        Do not use the GPU path. Send me this output.")
            rc = 1
        else:
            print(f"  SUSPECT  median {med:.4f} px, p95 {p95:.4f} px.")
            print("        A uniform bias near 0.5*(1 - 1/k) px that grows with")
            print("        --levels is the centre-convention bug. Compare")
            print("        --levels 1 against --levels 3 to tell.")
            rc = 2
        print(BAR)
        print("  Reminder: CC is NaN on the GPU path by construction.")
        print(BAR)
        return rc
    finally:
        if args.keep:
            print(f"\n  temp folder kept: {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
