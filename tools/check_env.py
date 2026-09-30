"""Report which Python environment this is, and whether it can drive the rig.

Run it from the place where the camera DOES work, then again from the
interpreter VibeStream.bat prints at startup, and compare.

    python tools/check_env.py

Why this exists: the Metavision HAL plugins are C++ DLLs loaded at run time.
Which ones HAL looks for is decided by MV_HAL_PLUGIN_PATH, and whether they
load is decided by DLL search order on PATH.  Both are process environment,
inherited from whatever launched Python.  So the same code can open the
camera from one shell and fail with a Windows "entry point not found" dialog
from another, with nothing about the EBIV code involved.

IMPORTANT: "No devices found" does NOT prove the plugins loaded.  A plugin
that fails to load leaves HAL with nothing to enumerate, which looks exactly
like an unplugged camera.  That is what the plugin probe below is for: it
loads each plugin DLL explicitly and reports the actual Windows error.
"""

import os
import sys

BAR = "-" * 70

# Places HAL looks when MV_HAL_PLUGIN_PATH is not set: the SDK's own install.
DEFAULT_PLUGIN_DIRS = [
    r"C:\Program Files\Prophesee\lib\metavision\hal\plugins",
    r"C:\Program Files\Prophesee\lib\hal\plugins",
]


def _line(label, value):
    print("  {:<22} {}".format(label, value))


def _suppress_dll_dialogs():
    """Stop a failing LoadLibrary from putting up a modal box that blocks us."""
    if os.name != "nt":
        return
    try:
        import ctypes
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x8000)
    except Exception:                                              # noqa: BLE001
        pass


def _probe_plugins():
    """Load every HAL plugin DLL explicitly and report what Windows says.

    This is the decisive test.  It works with the camera unplugged, because
    it is testing the loader, not the hardware.
    """
    if os.name != "nt":
        print("  (plugin probe is Windows-only)")
        return

    import ctypes

    raw = os.environ.get("MV_HAL_PLUGIN_PATH", "")
    dirs = [d for d in raw.split(os.pathsep) if d.strip()]
    if dirs:
        print("  MV_HAL_PLUGIN_PATH is set, so HAL uses ONLY these directories.")
    else:
        dirs = [d for d in DEFAULT_PLUGIN_DIRS if os.path.isdir(d)]
        print("  MV_HAL_PLUGIN_PATH is NOT set, so HAL falls back to its install")
        print("  directory.  A stale or missing value here is a common cause of")
        print("  the entry-point dialog: a shell where it is set works, and a")
        print("  process launched from Explorer with a stale environment does not.")

    if not dirs:
        print("  No plugin directory to probe.")
        return

    any_ok = False
    for d in dirs:
        print()
        print("  {}".format(d))
        if not os.path.isdir(d):
            print("      DIRECTORY DOES NOT EXIST")
            continue
        dlls = sorted(f for f in os.listdir(d) if f.lower().endswith(".dll"))
        if not dlls:
            print("      (no .dll in this directory)")
            continue
        for name in dlls:
            full = os.path.join(d, name)
            try:
                ctypes.WinDLL(full)
            except OSError as exc:
                print("      {:<40} FAILED TO LOAD".format(name))
                print("          {}".format(exc))
            else:
                any_ok = True
                print("      {:<40} loads OK".format(name))

    print()
    if any_ok:
        print("  At least one plugin loads, so HAL has something to enumerate with.")
    else:
        print("  NO plugin loaded.  This environment cannot see the camera even")
        print("  with it plugged in, and 'No devices found' here means nothing.")


def main():
    _suppress_dll_dialogs()

    print(BAR)
    print("  Interpreter")
    print(BAR)
    _line("python", sys.version.split()[0])
    _line("executable", sys.executable)
    _line("prefix", sys.prefix)
    _line("conda env", os.environ.get("CONDA_DEFAULT_ENV", "(not a conda env)"))
    _line("virtual env", os.environ.get("VIRTUAL_ENV", "(none)"))
    _line("PYTHONPATH", os.environ.get("PYTHONPATH", "(not set)"))

    print()
    print(BAR)
    print("  Packages")
    print(BAR)
    for name in ("numpy", "scipy", "cv2", "metavision_hal", "metavision_core"):
        try:
            mod = __import__(name)
        except Exception as exc:                                   # noqa: BLE001
            _line(name, "NOT AVAILABLE  ({}: {})".format(type(exc).__name__, exc))
        else:
            ver = getattr(mod, "__version__", "")
            where = getattr(mod, "__file__", "") or "(built-in)"
            _line(name, "{}  {}".format(ver, where).strip())

    print()
    print(BAR)
    print("  Metavision runtime environment")
    print(BAR)
    _line("MV_HAL_PLUGIN_PATH", os.environ.get("MV_HAL_PLUGIN_PATH", "(NOT SET)"))
    _line("MV_LOG_LEVEL", os.environ.get("MV_LOG_LEVEL", "(not set)"))

    interesting = []
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        low = entry.lower()
        if any(k in low for k in ("prophesee", "metavision", "openeb", "anaconda",
                                  "miniconda", "library\\bin", "condabin")):
            interesting.append(entry)
    if interesting:
        print()
        print("  PATH entries that decide which DLLs the plugins bind to,")
        print("  in search order:")
        for i, entry in enumerate(interesting, 1):
            print("    {:>2}. {}".format(i, entry))
        print()
        print("  If more than one Metavision/OpenEB install appears here, the")
        print("  plugins may bind to libraries from the OTHER install.  That is")
        print("  what an 'entry point not found' error is.")

    print()
    print(BAR)
    print("  Plugin load probe  (the decisive test - works with no camera)")
    print(BAR)
    _probe_plugins()

    print()
    print(BAR)
    print("  Camera enumeration")
    print(BAR)
    try:
        import metavision_hal as mv
    except Exception as exc:                                       # noqa: BLE001
        print("  metavision_hal did not import: {}: {}".format(type(exc).__name__, exc))
    else:
        try:
            devs = mv.DeviceDiscovery.list()
        except Exception as exc:                                   # noqa: BLE001
            print("  DeviceDiscovery.list() raised: {}: {}".format(
                type(exc).__name__, exc))
        else:
            if devs:
                print("  Devices found: {}".format(list(devs)))
            else:
                print("  No devices found.")
                print("  Read this together with the plugin probe above: with no")
                print("  camera connected, a healthy environment and a broken one")
                print("  both print this line.")

    print()
    print(BAR)
    print("  For HAL's own account of which plugins it tried, re-run with:")
    print("      set MV_LOG_LEVEL=TRACE")
    print("      python tools/check_env.py")
    print(BAR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
