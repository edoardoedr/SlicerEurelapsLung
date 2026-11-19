"""Process entry point of the real prediction:

    PythonSlicer run_predict.py --config <work>/config.json --collapse 40 --frames 30 --run pred_40 [--steps ...]

= `python -m eurelaps_lung.predict ...`, but with the model geometry (geometry -> hilum -> fem_setup) and,
with inputs.ct, the CT measure (ct) built in a child process of its own when needed, and the forward run
(torch) here.

Why: the pip wheel of pykdtree (used by pyacvd in the geometry step) bundles its own libomp, and so does
torch. In one process the two OpenMP runtimes abort (OMP Error #15) or deadlock (SLICER_EXTENSION.md 10.5):
seen on Slicer 5.13 arm64 (torch 2.15) and 5.12 x86_64 (torch 2.2.2). In two processes each has one
runtime only. The output is the same as one `predict.run`.
"""

import argparse
import subprocess
import sys
from pathlib import Path

GEOMETRY_STEPS = ("geometry", "hilum", "fem_setup")


def geometry_only(config, steps):
    """Child process: the geometry steps only (no torch)."""
    from eurelaps_lung.config import load_config
    from eurelaps_lung.runner import run_steps, start
    cfg = load_config(config, [])
    start(cfg, "geometry")
    import logging
    logging.getLogger("eurelaps_lung").info("building the model geometry: %s", " -> ".join(steps))
    run_steps(cfg, list(steps))


def main(argv):
    if argv and argv[0] == "--geometry-only":
        ap = argparse.ArgumentParser()
        ap.add_argument("--config", required=True)
        ap.add_argument("--steps", nargs="+", required=True)
        a = ap.parse_args(argv[1:])
        geometry_only(a.config, a.steps)
        return 0

    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", nargs="+")
    known, _rest = ap.parse_known_args(argv)
    from eurelaps_lung.config import load_config
    cfg = load_config(known.config, [])
    have = (cfg.setup_dir / "problem.npz").is_file()
    steps = known.steps or ([] if have else list(GEOMETRY_STEPS))
    if getattr(cfg, "ct", None) is not None and not (cfg.output_dir / "ct" / "ct_values.json").is_file():
        steps = ["ct"] + [s for s in steps if s != "ct"]     # density and emphysema volume (no torch here)
    if steps:
        sys.stdout.flush()
        code = subprocess.call([sys.executable, str(Path(__file__).resolve()), "--geometry-only",
                                "--config", known.config, "--steps"] + list(steps))
        if code != 0:
            return code
    # the forward: no --steps (the geometry is there now), torch in this process only
    fwd = []
    skip = False
    for a in argv:
        if a == "--steps":
            skip = True
            continue
        if skip and a.startswith("--"):
            skip = False
        if not skip:
            fwd.append(a)
    from eurelaps_lung.predict import main as predict_main
    predict_main(fwd)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
