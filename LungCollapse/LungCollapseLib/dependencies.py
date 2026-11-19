"""Installation of the prediction dependencies on first use (SLICER_EXTENSION.md, section 4).

- torch: from the PyTorch extension (PyTorchUtils), which picks the wheel for the platform;
- eurelaps_lung and its prediction dependencies (pyacvd, netgen-mesher): pip, pinned tag;
- never a second VTK: checked after the install.
"""

import importlib
import importlib.util
import os

import qt

import slicer

PACKAGE = "eurelaps_lung"
PACKAGE_VERSION = "0.2.0"
DEFAULT_SPEC = "eurelaps-lung @ git+https://github.com/edoardoedr/lung_collapse.git@v%s" % PACKAGE_VERSION
SETTING_SPEC = "LungCollapse/PackageSpec"


def package_spec():
    """pip requirement of the package: env LUNGCOLLAPSE_PACKAGE_SPEC, else the setting, else the pinned tag."""
    return (os.environ.get("LUNGCOLLAPSE_PACKAGE_SPEC")
            or qt.QSettings().value(SETTING_SPEC, "") or DEFAULT_SPEC)


def status():
    """{'torch': bool, 'package': bool, 'version': str|None, 'vtk_ok': bool, 'pytorch_extension': bool}"""
    out = {"torch": importlib.util.find_spec("torch") is not None,
           "package": importlib.util.find_spec(PACKAGE) is not None, "version": None,
           "pytorch_extension": importlib.util.find_spec("PyTorchUtils") is not None, "vtk_ok": vtk_ok()}
    if out["package"]:
        try:
            out["version"] = importlib.import_module(PACKAGE).__version__
        except Exception:
            out["version"] = None
    return out


def vtk_ok():
    """True when `import vtk` is Slicer's VTK (not a pip-installed second copy)."""
    import vtk
    path = os.path.realpath(vtk.__file__).replace("\\", "/")
    return path.startswith(os.path.realpath(slicer.app.slicerHome).replace("\\", "/")) and "site-packages" not in path


def _version_tuple(v):
    try:
        return tuple(int(x) for x in str(v).split(".")[:3])
    except ValueError:
        return (0,)


def missing():
    """What has to be installed: torch, and the package when absent or older than PACKAGE_VERSION."""
    s = status()
    m = []
    if not s["torch"]:
        m.append("torch")
    if not s["package"] or _version_tuple(s["version"]) < _version_tuple(PACKAGE_VERSION):
        m.append(PACKAGE)
    return m


def ensure(ask=True):
    """Install what is missing. Returns True when everything is there; False when the user declined."""
    s = status()
    todo = missing()
    if todo:
        msg = ("Lung Collapse needs:\n\n"
               + ("- PyTorch (torch), from the PyTorch extension\n" if "torch" in todo else "")
               + ("- the eurelaps_lung package %s%s (%s) and its dependencies (pyacvd, netgen-mesher)\n"
                  % (PACKAGE_VERSION, " (installed: %s)" % s["version"] if s["version"] else "", package_spec())
                  if PACKAGE in todo else "")
               + "\nInstall now? This can take several minutes; Slicer may look frozen meanwhile.")
        if ask and not slicer.util.confirmOkCancelDisplay(msg, windowTitle="Lung Collapse: dependencies"):
            return False
        if "torch" in todo:
            if not s["pytorch_extension"]:
                raise RuntimeError("Install the 'PyTorch' extension from the Extensions Manager, restart Slicer, "
                                   "then click Apply again.")
            import PyTorchUtils
            with slicer.util.WaitCursor():
                torch = PyTorchUtils.PyTorchUtilsLogic().installTorch(askConfirmation=False)
            if torch is None:
                raise RuntimeError("torch could not be installed (PyTorch extension)")
        if PACKAGE in todo:
            with slicer.util.WaitCursor():
                # a list: each item is one argument (the spec "name @ git+url" has spaces; a string is split on them)
                slicer.util.pip_install((["--upgrade"] if s["package"] else []) + [package_spec()])
            importlib.invalidate_caches()
            import sys
            for name in [m for m in sys.modules if m == PACKAGE or m.startswith(PACKAGE + ".")]:
                del sys.modules[name]                     # the new version on the next import
    if not vtk_ok():
        raise RuntimeError("A second VTK is installed in Slicer's Python (pip): uninstall it "
                           "(slicer.util.pip_uninstall('vtk')) and restart Slicer.")
    s = status()
    if not s["package"]:
        raise RuntimeError("%s is not importable after the install" % PACKAGE)
    if _version_tuple(s["version"]) < _version_tuple(PACKAGE_VERSION):
        raise RuntimeError("eurelaps_lung %s is installed; this extension needs %s: restart Slicer and click Apply "
                           "again" % (s["version"], PACKAGE_VERSION))
    return True
