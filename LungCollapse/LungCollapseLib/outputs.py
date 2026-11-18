"""Outputs of a run into the scene (SLICER_EXTENSION.md, section 6.6): the collapse sequence, the final
model, the collapsed lung, the hilum and its rings, the results table. Everything is loaded into memory
(nothing stays linked to the files), so the work folder can be deleted afterwards."""

import json
import os
from pathlib import Path

import vtk

import slicer

from .inputs import read_surface_file

ATTRIBUTE = "LungCollapse.Output"
COLORS = "vtkMRMLColorTableNodeFileColdToHotRainbow.txt"
COLOR_ARRAYS = {"displacement": "Disp_mag_mm", "error": "Error_mm"}


def _tag(node, role):
    node.SetAttribute(ATTRIBUTE, role)
    # not linked to the files of the (maybe temporary) work folder
    sn = node.GetStorageNode() if hasattr(node, "GetStorageNode") else None
    if sn is not None:
        node.SetAndObserveStorageNodeID(None)
        slicer.mrmlScene.RemoveNode(sn)
    return node


def output_nodes(role=None):
    out = []
    scene = slicer.mrmlScene
    for i in range(scene.GetNumberOfNodes()):
        n = scene.GetNthNode(i)
        r = n.GetAttribute(ATTRIBUTE) if n is not None else None
        if r and (role is None or r == role):
            out.append(n)
    return out


def remove_outputs():
    """Remove the nodes of an earlier run (sequence, browser, models, markups, table)."""
    for n in output_nodes():
        if slicer.mrmlScene.IsNodePresent(n):
            slicer.mrmlScene.RemoveNode(n)


def _load_model(path, name, role):
    node = slicer.util.loadModel(str(path))     # SPACE=LPS in the file: Slicer converts to RAS
    node.SetName(name)
    return _tag(node, role)


def add_polydata_model(polydata, name, role, color=(0.9, 0.9, 0.9), opacity=1.0, visible=True):
    node = slicer.modules.models.logic().AddModel(polydata)
    node.SetName(name)
    d = node.GetDisplayNode()
    d.SetColor(*color)
    d.SetOpacity(opacity)
    d.SetVisibility(visible)
    d.SetVisibility2D(True)
    return _tag(node, role)


def _color_by(displayNode, array, vmax):
    displayNode.SetActiveScalarName(array)
    displayNode.SetAndObserveColorNodeID(COLORS)
    displayNode.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseManualScalarRange)
    displayNode.SetScalarRange(0.0, max(vmax, 1e-3))
    displayNode.SetScalarVisibility(True)


def _array_max(polydata, name):
    a = polydata.GetPointData().GetArray(name)
    return a.GetRange()[1] if a is not None else 0.0


def load_run(out_dir, run_name, prefix="LungCollapse", color_by="displacement"):
    """Load out_dir/runs/<run_name>/ into the scene. Returns {role: node}."""
    run_dir = Path(out_dir) / "runs" / run_name
    seq_dir = run_dir / "sequence"
    nodes = {}
    frames = sorted(seq_dir.glob("frame_*.vtp"))

    # sequence of the frames (index = load %)
    if frames:
        seq = _tag(slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceNode", prefix + "_sequence"), "sequence")
        seq.SetIndexName("load")
        seq.SetIndexUnit("%")
        seq.SetIndexType(slicer.vtkMRMLSequenceNode.NumericIndex)
        maxima = {"displacement": 0.0, "error": 0.0}
        n = len(frames)
        for i, f in enumerate(frames):
            pd, _space = read_surface_file(str(f), "auto", clean=False)   # SPACE=LPS in the file -> RAS
            maxima["displacement"] = max(maxima["displacement"], _array_max(pd, "Disp_mag_mm"))
            maxima["error"] = max(maxima["error"], _array_max(pd, "Error_mm"))
            node = slicer.vtkMRMLModelNode()
            node.SetAndObservePolyData(pd)
            seq.SetDataNodeAtValue(node, "%.1f" % (100.0 * i / max(n - 1, 1)))
        browser = _tag(slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceBrowserNode", prefix + "_browser"),
                       "browser")
        browser.AddSynchronizedSequenceNode(seq)
        browser.SetIndexDisplayFormat("%.0f")
        browser.SetSelectedItemNumber(n - 1)
        slicer.modules.sequences.logic().UpdateProxyNodesFromSequences(browser)
        proxy = browser.GetProxyNode(seq)
        proxy.SetName(prefix + "_collapse")
        _tag(proxy, "proxy")
        proxy.CreateDefaultDisplayNodes()
        d = proxy.GetDisplayNode()
        d.SetColor(0.9, 0.5, 0.4)
        d.SetVisibility2D(True)
        array = COLOR_ARRAYS.get(color_by, "Disp_mag_mm")
        if color_by == "error" and maxima["error"] <= 0:
            color_by, array = "displacement", "Disp_mag_mm"
        _color_by(d, array, maxima[color_by])
        browser.SetPlaybackRateFps(8.0)
        browser.SetPlaybackItemSkippingEnabled(False)
        browser.SetPlaybackLooped(False)
        nodes.update(sequence=seq, browser=browser, proxy=proxy)
        nodes["_maxima"] = maxima

    # final collapsed model
    if (run_dir / "surface.vtp").is_file():
        m = _load_model(run_dir / "surface.vtp", prefix + "_predicted", "predicted")
        m.GetDisplayNode().SetColor(0.85, 0.35, 0.3)
        m.GetDisplayNode().SetVisibility(False)          # the sequence shows it; the final model is for saving
        nodes["predicted"] = m

    # the real collapsed lung (validation)
    if (seq_dir / "target.vtp").is_file():
        t = _load_model(seq_dir / "target.vtp", prefix + "_collapsed_real", "target")
        t.GetDisplayNode().SetRepresentation(slicer.vtkMRMLDisplayNode.WireframeRepresentation)
        t.GetDisplayNode().SetColor(0.6, 0.6, 0.6)
        t.GetDisplayNode().SetVisibility2D(True)
        nodes["target"] = t

    # hilum and rings
    hil = seq_dir / "hilum_anchor.mrk.json"
    if not hil.is_file():
        hil = Path(out_dir) / "hilum_anchor.mrk.json"
    if hil.is_file():
        before = {n.GetID() for n in slicer.util.getNodesByClass("vtkMRMLMarkupsNode")}
        slicer.util.loadMarkups(str(hil))
        marks = [n for n in slicer.util.getNodesByClass("vtkMRMLMarkupsNode") if n.GetID() not in before]
        for mk in marks:
            mk.SetName(prefix + "_" + mk.GetName())
            mk.SetLocked(True)
            _tag(mk, "hilum")
        nodes["hilum"] = marks
    rings = Path(out_dir) / "hilum" / "rings.vtp"
    if rings.is_file():
        r = _load_model(rings, prefix + "_hilum_rings", "rings")
        r.GetDisplayNode().SetColor(1.0, 0.85, 0.1)
        r.GetDisplayNode().SetLineWidth(3)
        r.GetDisplayNode().SetVisibility2D(True)
        nodes["rings"] = r
    return nodes


def set_color_by(proxy, color_by, maxima):
    if proxy is None or proxy.GetDisplayNode() is None:
        return
    array = COLOR_ARRAYS[color_by]
    _color_by(proxy.GetDisplayNode(), array, maxima.get(color_by, 0.0))


# --------------------------------------------------------------------------------------------------
# results table

def summary_rows(summary, inflated_ml=None, asked=None):
    """[(quantity, value)] for the results table (section 7)."""
    ph = summary.get("physics", {})
    rows = []
    ratio = summary.get("volume_ratio")
    if inflated_ml:
        rows.append(("Inflated volume (mL)", "%.0f" % inflated_ml))
        if ratio is not None:
            rows.append(("Collapsed volume (mL)", "%.0f" % (ratio * inflated_ml)))
    if asked is not None:
        rows.append(("Collapse asked (%)", asked if isinstance(asked, str) else "%.1f" % asked))
    if summary.get("collapse_percent") is not None:
        rows.append(("Collapse reached (%)", "%.1f" % summary["collapse_percent"]))
    if ph.get("E_Pa") is not None:
        rows.append(("E (Pa)", "%.0f" % ph["E_Pa"]))
    if ph.get("p0_Pa") is not None:
        rows.append(("p0 (Pa)", "%.0f" % ph["p0_Pa"]))
    if summary.get("solves") is not None:
        rows.append(("Solves", "%d" % summary["solves"]))
    if summary.get("elapsed_min") is not None:
        rows.append(("Time (min)", "%.2f" % summary["elapsed_min"]))
    if summary.get("wall_max_penetration_mm") is not None:
        rows.append(("Wall penetration (mm)", "%.2f" % summary["wall_max_penetration_mm"]))
    errs = (("ASSD (mm)", "assd_mm"), ("Mean error (mm)", "mean_err_mm"), ("Median error (mm)", "median_err_mm"),
            ("95th percentile error (mm)", "p95_err_mm"), ("Max error (mm)", "max_err_mm"))
    for label, key in errs:
        if summary.get(key) is not None:
            rows.append((label, "%.2f" % summary[key]))
    if summary.get("target_volume_ratio") is not None:
        rows.append(("Collapse of the real collapsed lung (%)", "%.1f" % (100 * (1 - summary["target_volume_ratio"]))))
    if summary.get("fake"):
        rows.append(("Note", "FAKE run (no FEM): Part 1 interface test"))
    return rows


def results_table(rows, name="LungCollapse_results"):
    table = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", name)
    _tag(table, "table")
    q, v = vtk.vtkStringArray(), vtk.vtkStringArray()
    q.SetName("Quantity")
    v.SetName("Value")
    for a, b in rows:
        q.InsertNextValue(a)
        v.InsertNextValue(b)
    table.AddColumn(q)
    table.AddColumn(v)
    table.SetUseColumnTitleAsColumnHeader(True)
    table.SetLocked(True)
    return table


def read_summary(out_dir, run_name):
    p = Path(out_dir) / "runs" / run_name / "summary.json"
    return json.loads(p.read_text()) if p.is_file() else {}
