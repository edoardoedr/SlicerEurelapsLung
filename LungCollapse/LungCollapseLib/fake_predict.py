"""Fake prediction (Part 1): the same command line, log lines and output layout as
`python -m eurelaps_lung.predict`, without the package. The inflated lung is shrunk uniformly towards its
hilum until the volume matches the collapse %, so that the sequence, the slider, the table and the
coordinates (frame 0 must lie on the inflated lung) can be tried in Slicer.

    PythonSlicer fake_predict.py --config <work>/config.json --collapse 40 --frames 30 --run pred_40
"""

import argparse
import json
import logging
import math
import os
import sys
import time

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

log = logging.getLogger("eurelaps_lung")


def read_lps(path):
    """Read a legacy .vtk / .vtp surface and return it in LPS (the package works in LPS)."""
    r = vtk.vtkXMLPolyDataReader() if str(path).endswith(".vtp") else vtk.vtkPolyDataReader()
    r.SetFileName(str(path))
    r.Update()
    pd = vtk.vtkPolyData()
    pd.DeepCopy(r.GetOutput())
    space = "LPS"
    arr = pd.GetFieldData().GetAbstractArray("SPACE") if pd.GetFieldData() else None
    if arr is not None:
        space = str(arr.GetVariantValue(0).ToString()).upper()
    elif str(path).endswith(".vtk"):
        with open(path, "rb") as f:
            if b"SPACE=RAS" in f.read(512).upper():
                space = "RAS"
    if space == "RAS":
        pts = vtk_to_numpy(pd.GetPoints().GetData()).copy()
        pts[:, :2] *= -1
        pd.GetPoints().SetData(numpy_to_vtk(pts, deep=True))
    return pd


def volume(pd):
    m = vtk.vtkMassProperties()
    m.SetInputData(pd)
    m.Update()
    return m.GetVolume()


def write_lps(pd, path):
    out = vtk.vtkPolyData()
    out.DeepCopy(pd)
    s = vtk.vtkStringArray()
    s.SetName("SPACE")
    s.InsertNextValue("LPS")
    out.GetFieldData().AddArray(s)
    w = vtk.vtkXMLPolyDataWriter()
    w.SetFileName(str(path))
    w.SetInputData(out)
    w.Write()


def point_array(pd, name, values):
    a = numpy_to_vtk(np.ascontiguousarray(values, dtype=np.float64), deep=True)
    a.SetName(name)
    pd.GetPointData().AddArray(a)


def distances(implicit, X):
    """|signed distance| of the points X to the implicit surface (vectorised)."""
    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(np.ascontiguousarray(X, dtype=np.float64), deep=True))
    out = vtk.vtkDoubleArray()
    implicit.FunctionValue(pts.GetData(), out)
    return np.abs(vtk_to_numpy(out))


def ring(center, normal, radius, n=48):
    normal = np.asarray(normal, float) / np.linalg.norm(normal)
    a = np.cross(normal, [1.0, 0.0, 0.0])
    if np.linalg.norm(a) < 1e-3:
        a = np.cross(normal, [0.0, 1.0, 0.0])
    a /= np.linalg.norm(a)
    b = np.cross(normal, a)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return center + radius * (np.outer(np.cos(t), a) + np.outer(np.sin(t), b))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--collapse")
    ap.add_argument("--run", default="predict")
    ap.add_argument("--frames", type=int)
    ap.add_argument("--steps", nargs="+")
    ap.add_argument("--set", nargs="+", default=[])
    ap.add_argument("--delay", type=float, default=float(os.environ.get("LUNGCOLLAPSE_FAKE_DELAY", "0.15")))
    ap.add_argument("--fail", action="store_true", help="exit with an error after the hilum (to test)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    t_start = time.time()
    config_path = os.path.abspath(args.config)
    cfg = json.loads(open(config_path).read())
    base = os.path.dirname(config_path)
    data_dir = os.path.join(base, cfg.get("data_dir", "."))
    out = os.path.join(base, cfg.get("output_dir", "out"))
    run_dir = os.path.join(out, "runs", args.run)
    seq_dir = os.path.join(run_dir, "sequence")
    for d in (os.path.join(out, "logs"), seq_dir, os.path.join(out, "hilum"), os.path.join(out, "fem", "setup")):
        os.makedirs(d, exist_ok=True)
    frames = args.frames if args.frames is not None else cfg.get("collapse", {}).get("sequence_frames", 30)
    (open(os.path.join(out, "logs", "config_used_predict.json"), "w")
     .write(json.dumps(dict(cfg, run_name=args.run, _fake=True, _overrides=args.set), indent=2)))
    log.info("patient %s, config %s (FAKE prediction: no FEM)", cfg.get("patient"), config_path)

    inputs = cfg["inputs"]
    lung = read_lps(os.path.join(data_dir, inputs["inflated"]))
    collapsed = read_lps(os.path.join(data_dir, inputs["collapsed"])) if inputs.get("collapsed") else None
    airways = read_lps(os.path.join(data_dir, inputs["structures"]["airways"]))
    X0 = vtk_to_numpy(lung.GetPoints().GetData()).astype(float)
    V0 = volume(lung)

    problem = os.path.join(out, "fem", "setup", "problem.npz")
    need = list(args.steps or ([] if os.path.isfile(problem) else ["geometry", "hilum", "fem_setup"]))
    # hilum: the lung point nearest to the airways' centroid
    A = vtk_to_numpy(airways.GetPoints().GetData()).astype(float)
    loc = vtk.vtkPointLocator()
    loc.SetDataSet(lung)
    loc.BuildLocator()
    H = X0[loc.FindClosestPoint(A.mean(0))]
    if need:
        log.info("building the model geometry: %s", " -> ".join(need))
    for step in need:
        log.info("=== %s ===", step)
        time.sleep(4 * args.delay)
        if step == "hilum":
            if args.fail:
                log.error("=== hilum FAILED === (fake failure asked with --fail)")
                sys.exit(1)
            normal = A.mean(0) - X0.mean(0)
            names = ("airways", "arteries", "veins")
            pts, lines, cell_struct = [], [], []
            for k, name in enumerate(names):
                c = H + 8.0 * (k - 1) * np.array([0.0, 0.0, 1.0])
                r = ring(c, normal, 6.0 + 2 * k)
                start = len(pts)
                pts.extend(r)
                for i in range(len(r)):
                    lines.append((start + i, start + (i + 1) % len(r)))
                    cell_struct.append(k)
                log.info("%s: 1 ring(s), centre LPS [%.1f, %.1f, %.1f]", name, *c)
            pd = vtk.vtkPolyData()
            p = vtk.vtkPoints()
            p.SetData(numpy_to_vtk(np.asarray(pts), deep=True))
            pd.SetPoints(p)
            ca = vtk.vtkCellArray()
            for a, b in lines:
                ca.InsertNextCell(2)
                ca.InsertCellPoint(a)
                ca.InsertCellPoint(b)
            pd.SetLines(ca)
            sa = numpy_to_vtk(np.asarray(cell_struct, dtype=np.int32), deep=True)
            sa.SetName("Structure")
            pd.GetCellData().AddArray(sa)
            write_lps(pd, os.path.join(out, "hilum", "rings.vtp"))
            mk = {"@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/"
                             "Resources/Schema/markups-schema-v1.0.3.json#",
                  "markups": [{"type": "Fiducial", "name": "hilum", "coordinateSystem": "LPS",
                               "coordinateUnits": "mm", "locked": True,
                               "controlPoints": [{"id": "1", "label": "hilum", "position": [float(v) for v in H],
                                                  "positionStatus": "defined", "locked": True,
                                                  "visibility": True}],
                               "display": {"glyphType": "Sphere3D", "useGlyphScale": False, "glyphSize": 20.0,
                                           "opacity": 0.6, "color": [1.0, 0.8, 0.0],
                                           "selectedColor": [1.0, 0.8, 0.0]}}]}
            open(os.path.join(out, "hilum_anchor.mrk.json"), "w").write(json.dumps(mk, indent=2))
            log.warning("a ring centre is > 50 mm from the hilum: check the rings in hilum/rings.vtp (FAKE warning)")
        if step == "fem_setup":
            np.savez(problem, fake=np.array([1]))
        log.info("=== %s done in %.1f s ===", step, 4 * args.delay)

    if args.collapse == "target":
        if collapsed is None:
            sys.exit("collapse 'target' needs inputs.collapsed (the collapsed lung)")
        pct = 100.0 * (1.0 - volume(collapsed) / V0)
        log.info("collapse %% of the collapsed lung (inputs.collapsed): %.2f %%", pct)
    else:
        pct = float(args.collapse) if args.collapse is not None else 50.0
    ratio = 1.0 - pct / 100.0
    log.info("=== collapse ===")
    E = 10.0 ** (3.5380 - 0.012883 * pct)
    for i, q0 in enumerate((0.40, 0.48, 0.45)):
        time.sleep(3 * args.delay)
        log.info("solve %d: q0 %.4f -> volume ratio %.4f (target %.4f)", i + 1, q0, ratio * (1 + 0.02 * (2 - i)), ratio)

    log.info("sequence: %d frames", frames + 1)
    target_dist = None
    if collapsed is not None:
        target_dist = vtk.vtkImplicitPolyDataDistance()
        target_dist.SetInput(collapsed)
    for d in os.listdir(seq_dir):
        if d.startswith("frame_"):
            os.remove(os.path.join(seq_dir, d))
    dmax = 0.0
    last = None
    for i in range(frames + 1):
        lam = i / max(frames, 1)
        s = (1.0 - lam * (1.0 - ratio)) ** (1.0 / 3.0)
        X = H + (X0 - H) * s
        U = X - X0
        pd = vtk.vtkPolyData()
        pd.DeepCopy(lung)
        pd.GetPoints().SetData(numpy_to_vtk(X, deep=True))
        mag = np.linalg.norm(U, axis=1)
        dmax = max(dmax, float(mag.max()))
        point_array(pd, "Displacement_mm", U)
        point_array(pd, "Disp_mag_mm", mag)
        if target_dist is not None:
            point_array(pd, "Error_mm", distances(target_dist, X))
        pa = numpy_to_vtk(np.full(pd.GetNumberOfCells(), 1000.0 * lam * E / 1000.0), deep=True)
        pa.SetName("Pressure_Pa")
        pd.GetCellData().AddArray(pa)
        write_lps(pd, os.path.join(seq_dir, "frame_%03d.vtp" % i))
        last = pd
        if i % 5 == 0 or i == frames:
            log.info("frame %3d/%d  load %5.1f%%  (%.0f s)", i, frames, 100 * lam, time.time() - t_start)
        time.sleep(args.delay / 3)
    write_lps(last, os.path.join(run_dir, "surface.vtp"))
    if collapsed is not None:
        write_lps(collapsed, os.path.join(seq_dir, "target.vtp"))
    if os.path.isfile(os.path.join(out, "hilum_anchor.mrk.json")):
        open(os.path.join(seq_dir, "hilum_anchor.mrk.json"), "w").write(
            open(os.path.join(out, "hilum_anchor.mrk.json")).read())

    summary = dict(kind="collapse", patient=cfg.get("patient"), run_name=args.run, solver="fake",
                   physics=dict(E_Pa=E, q0=0.45, p0_Pa=0.45 * E, nu=0.3, tissue_density_kg_m3=200.0,
                                E_rule=dict(rule="collapse", a=3.5380, b=-0.012883, collapse_percent=pct)),
                   volume_ratio=ratio, collapse_percent=pct, solves=3, failed_solves=0,
                   elapsed_min=(time.time() - t_start) / 60.0, wall=True, wall_max_penetration_mm=0.8,
                   fake=True)
    if collapsed is not None:
        err = distances(target_dist, vtk_to_numpy(last.GetPoints().GetData()))
        summary.update(assd_mm=float(err.mean()), mean_err_mm=float(err.mean()),
                       median_err_mm=float(np.median(err)), p95_err_mm=float(np.percentile(err, 95)),
                       max_err_mm=float(err.max()), target_volume_ratio=volume(collapsed) / V0)
    open(os.path.join(run_dir, "summary.json"), "w").write(json.dumps(summary, indent=2))
    log.info("wrote %s/ (surface.vtp, summary.json); sequence written to %s (max displacement %.0f mm)",
             run_dir, seq_dir, math.ceil(dmax))


if __name__ == "__main__":
    main()
