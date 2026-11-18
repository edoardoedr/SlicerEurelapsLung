"""Input side of Lung Collapse: segmentation -> closed surfaces in RAS (SLICER_EXTENSION.md, section 3).

- loading segmentation / labelmap / surface files into the scene;
- guessing the side and the segments from their names (or centroids);
- merging the lobes of one side (union + closing) and exporting closed surfaces (vtkPolyData, RAS, world);
- writing them as legacy .vtk tagged SPACE=RAS;
- light checks before the run.

Only Slicer, VTK, numpy and scipy: no eurelaps_lung here.
"""

import hashlib
import logging
import os
import re

import numpy as np
import vtk

import slicer

STRUCTURES = ("airways", "arteries", "veins")

_STRUCTURE_PATTERNS = {
    "airways": re.compile(r"airway|bronch|trachea|vie.?aeree", re.I),
    "arteries": re.compile(r"arter", re.I),
    "veins": re.compile(r"vein|vena|vene", re.I),
}
_LUNG = re.compile(r"lung|lobe|polmon|lobo", re.I)
_NOT_LUNG = re.compile(r"airway|bronch|trachea|arter|vein|vena|vene|vessel|vas[ic]|nodul|tumou?r|wall|lesion", re.I)
_COLLAPSED = re.compile(r"collaps|atelect|pneumo", re.I)
_LEFT = re.compile(r"(^|[^a-z])(left|l|lt|sx|sin|sinistr[oa]?)([^a-z]|$)", re.I)
_RIGHT = re.compile(r"(^|[^a-z])(right|r|rt|dx|destr[oa]?)([^a-z]|$)", re.I)

SEGMENTATION_FILE_FILTER = "Segmentations (*.seg.nrrd *.nrrd *.nii *.nii.gz *.vtk *.vtp *.stl *.obj)"
SURFACE_EXTENSIONS = (".vtk", ".vtp", ".stl", ".obj", ".ply")

# Slicer's closed-surface conversion parameters (the defaults), written to the config record
CONVERSION_PARAMETERS = {"Smoothing factor": "0.5", "Decimation factor": "0.0", "Compute surface normals": "1"}


# --------------------------------------------------------------------------------------------------
# names and guesses

def name_side(name):
    """'left' / 'right' from a segment name, else None."""
    clean = name.replace("-", "_").replace(" ", "_")
    left, right = bool(_LEFT.search(clean)), bool(_RIGHT.search(clean))
    if left == right:
        return None
    return "left" if left else "right"


def segments(segmentationNode):
    """[(segment id, name)] in the segmentation's order."""
    if segmentationNode is None:
        return []
    seg = segmentationNode.GetSegmentation()
    return [(seg.GetNthSegmentID(i), seg.GetNthSegment(i).GetName()) for i in range(seg.GetNumberOfSegments())]


def is_lung_name(name):
    return bool(_LUNG.search(name)) and not _NOT_LUNG.search(name)


def is_collapsed_name(name):
    return bool(_COLLAPSED.search(name))


def segment_side(segmentationNode, segmentId, name=None):
    """Side of a segment: from its name, else from the x of its centre (RAS: x > 0 is the patient's right)."""
    name = name if name is not None else segmentationNode.GetSegmentation().GetSegment(segmentId).GetName()
    side = name_side(name)
    if side is not None:
        return side
    try:
        c = segmentationNode.GetSegmentCenterRAS(segmentId)
    except Exception:
        c = None
    if c is None:
        return None
    return "right" if c[0] > 0 else "left"


def guess(segmentationNode, collapsedNode=None):
    """Pre-selection from the segment names (section 7):

    {"side": "left"|"right"|None, "lungs": {"left": [ids], "right": [ids]},
     "airways": id|None, "arteries": id|None, "veins": id|None, "collapsed": id|None (in collapsedNode,
     default the same segmentation)}"""
    out = {"side": None, "lungs": {"left": [], "right": []}, "collapsed": None}
    out.update({s: None for s in STRUCTURES})
    collapsed_side = None
    whole = {"left": False, "right": False}
    for sid, name in segments(segmentationNode):
        for structure, pattern in _STRUCTURE_PATTERNS.items():
            if out[structure] is None and pattern.search(name):
                out[structure] = sid
        if is_collapsed_name(name):
            if collapsedNode is None or collapsedNode is segmentationNode:
                out["collapsed"] = out["collapsed"] or sid
            collapsed_side = collapsed_side or name_side(name)
            continue
        if is_lung_name(name):
            side = segment_side(segmentationNode, sid, name)
            if side:
                out["lungs"][side].append(sid)
                if not re.search(r"lobe|lobo", name, re.I):
                    whole[side] = True
    if collapsedNode is not None and collapsedNode is not segmentationNode:
        segs = segments(collapsedNode)
        named = [sid for sid, name in segs if is_collapsed_name(name)]
        if named or segs:
            out["collapsed"] = named[0] if named else segs[0][0]
        if named:
            collapsed_side = collapsed_side or name_side(dict(segs)[named[0]])
    # the side: the one of a collapsed segment, else the only side present, else the side segmented as
    # one whole lung (the other given as lobes), else left
    if collapsed_side:
        out["side"] = collapsed_side
    elif bool(out["lungs"]["left"]) != bool(out["lungs"]["right"]):
        out["side"] = "left" if out["lungs"]["left"] else "right"
    elif whole["left"] != whole["right"]:
        out["side"] = "left" if whole["left"] else "right"
    elif out["lungs"]["left"]:
        out["side"] = "left"
    return out


# --------------------------------------------------------------------------------------------------
# loading files into the scene

def _is_segmentation_nrrd(path):
    try:
        with open(path, "rb") as f:
            head = f.read(65536).decode("latin-1", errors="ignore")
        return "Segment0_" in head
    except OSError:
        return False


def load_segmentation_file(path, name=None):
    """Load a .seg.nrrd, a labelmap (.nrrd / .nii / .nii.gz) or a surface (.vtk/.vtp/.stl/.obj) into a
    vtkMRMLSegmentationNode. Returns (segmentationNode, [nodes added to the scene])."""
    path = str(path)
    if not os.path.isfile(path):
        raise ValueError("file not found: %s" % path)
    low = path.lower()
    added = []
    if low.endswith(".seg.nrrd") or (low.endswith(".nrrd") and _is_segmentation_nrrd(path)):
        segNode = slicer.util.loadSegmentation(path)
        added.append(segNode)
    elif low.endswith((".nrrd", ".nii", ".nii.gz", ".nhdr", ".mha", ".mhd")):
        labelNode = slicer.util.loadLabelVolume(path)
        added.append(labelNode)
        segNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode",
                                                     name or (labelNode.GetName() + "_segmentation"))
        segNode.CreateDefaultDisplayNodes()
        segNode.SetReferenceImageGeometryParameterFromVolumeNode(labelNode)
        if not slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelNode, segNode):
            raise ValueError("could not import the labelmap %s" % path)
        added.append(segNode)
        labelNode.GetDisplayNode() and labelNode.GetDisplayNode().SetVisibility(False)
        slicer.util.setSliceViewerLayers(label=None)
    elif low.endswith(SURFACE_EXTENSIONS):
        modelNode = slicer.util.loadModel(path)
        added.append(modelNode)
        segNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode",
                                                     name or (modelNode.GetName() + "_segmentation"))
        segNode.CreateDefaultDisplayNodes()
        if not slicer.modules.segmentations.logic().ImportModelToSegmentationNode(modelNode, segNode):
            raise ValueError("could not import the surface %s" % path)
        added.append(segNode)
        modelNode.GetDisplayNode() and modelNode.GetDisplayNode().SetVisibility(False)
    else:
        raise ValueError("unsupported file type: %s" % os.path.basename(path))
    if name:
        segNode.SetName(name)
    return segNode, added


def load_volume_file(path):
    node = slicer.util.loadVolume(str(path))
    return node, [node]


# --------------------------------------------------------------------------------------------------
# segments -> closed surfaces (RAS, world)

def _to_world(polydata, transformableNode):
    """Copy of `polydata` (in the node's local coordinates) moved to world coordinates (parent transforms)."""
    out = vtk.vtkPolyData()
    out.DeepCopy(polydata)
    tnode = transformableNode.GetParentTransformNode()
    if tnode is None:
        return out
    t = vtk.vtkGeneralTransform()
    tnode.GetTransformToWorld(t)
    f = vtk.vtkTransformPolyDataFilter()
    f.SetTransform(t)
    f.SetInputData(out)
    f.Update()
    res = vtk.vtkPolyData()
    res.DeepCopy(f.GetOutput())
    return res


def _clean_triangles(polydata):
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(polydata)
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(tri.GetOutputPort())
    clean.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(clean.GetOutput())
    return out


def segment_surface(segmentationNode, segmentId):
    """Closed surface of one segment (vtkPolyData copy, RAS world)."""
    segmentationNode.CreateClosedSurfaceRepresentation()
    pd = segmentationNode.GetClosedSurfaceInternalRepresentation(segmentId)
    if pd is None or pd.GetNumberOfPoints() == 0:
        name = segmentationNode.GetSegmentation().GetSegment(segmentId).GetName()
        raise ValueError("segment '%s' is empty" % name)
    return _clean_triangles(_to_world(pd, segmentationNode))


def _ball(radius_mm, spacing_kji):
    r = [max(int(np.ceil(radius_mm / s)), 0) for s in spacing_kji]
    zz, yy, xx = np.ogrid[-r[0]:r[0] + 1, -r[1]:r[1] + 1, -r[2]:r[2] + 1]
    d = (zz * spacing_kji[0]) ** 2 + (yy * spacing_kji[1]) ** 2 + (xx * spacing_kji[2]) ** 2
    return d <= radius_mm ** 2 + 1e-9, r


def merged_surface(segmentationNode, segmentIds, closing_mm=1.5, referenceVolumeNode=None):
    """Union of several segments (the lobes of one side) with a closing of `closing_mm` to fill the
    fissure gaps, as one closed surface (vtkPolyData, RAS world). One segment: its own closed surface."""
    segmentIds = list(segmentIds)
    if not segmentIds:
        raise ValueError("no lung segment selected")
    if len(segmentIds) == 1:
        return segment_surface(segmentationNode, segmentIds[0])
    import scipy.ndimage as ndi

    scene = slicer.mrmlScene
    labelNode = scene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "LungCollapse_tmp_merge")
    tmpSeg = None
    try:
        ids = vtk.vtkStringArray()
        for sid in segmentIds:
            ids.InsertNextValue(sid)
        logic = slicer.modules.segmentations.logic()
        if not logic.ExportSegmentsToLabelmapNode(segmentationNode, ids, labelNode, referenceVolumeNode):
            raise ValueError("could not export the lung segments to a labelmap")
        labelNode.SetAndObserveTransformNodeID(None)      # stay in the segmentation's local coordinates
        arr = slicer.util.arrayFromVolume(labelNode) > 0  # KJI
        spacing_kji = labelNode.GetSpacing()[::-1]
        if closing_mm > 0:
            ball, r = _ball(closing_mm, spacing_kji)
            pad = [(ri + 1, ri + 1) for ri in r]
            big = np.pad(arr, pad)
            big = ndi.binary_closing(big, structure=ball)
            arr = big[pad[0][0]:-pad[0][1], pad[1][0]:-pad[1][1], pad[2][0]:-pad[2][1]]
        arr = ndi.binary_fill_holes(arr)
        slicer.util.updateVolumeFromArray(labelNode, arr.astype(np.uint8))
        tmpSeg = scene.AddNewNodeByClass("vtkMRMLSegmentationNode", "LungCollapse_tmp_merged")
        if not logic.ImportLabelmapToSegmentationNode(labelNode, tmpSeg):
            raise ValueError("could not import the merged lung")
        tmpSeg.CreateClosedSurfaceRepresentation()
        sid = tmpSeg.GetSegmentation().GetNthSegmentID(0)
        pd = tmpSeg.GetClosedSurfaceInternalRepresentation(sid)
        if pd is None or pd.GetNumberOfPoints() == 0:
            raise ValueError("the merged lung is empty")
        local = vtk.vtkPolyData()
        local.DeepCopy(pd)
    finally:
        scene.RemoveNode(labelNode)
        if tmpSeg is not None:
            scene.RemoveNode(tmpSeg)
    # back to world with the transform of the original segmentation
    return _clean_triangles(_to_world(local, segmentationNode))


def read_surface_file(path, space="auto", clean=True):
    """A ready surface file -> (vtkPolyData in RAS, space found). space 'auto' = from the file tag
    (legacy header SPACE= or the SPACE field data), LPS when there is none (Slicer's default)."""
    path = str(path)
    low = path.lower()
    if low.endswith(".vtp"):
        r = vtk.vtkXMLPolyDataReader()
    elif low.endswith(".vtk"):
        r = vtk.vtkPolyDataReader()
    elif low.endswith(".stl"):
        r = vtk.vtkSTLReader()
    elif low.endswith(".obj"):
        r = vtk.vtkOBJReader()
    elif low.endswith(".ply"):
        r = vtk.vtkPLYReader()
    else:
        raise ValueError("unsupported surface file: %s" % path)
    r.SetFileName(path)
    r.Update()
    pd = vtk.vtkPolyData()
    pd.DeepCopy(r.GetOutput())
    if pd.GetNumberOfPoints() == 0:
        raise ValueError("empty surface: %s" % path)
    found = file_space(pd, path)
    src = found if space == "auto" else space
    src = src or "LPS"
    if src == "LPS":
        pd = lps_ras(pd)
    if not clean:
        out = vtk.vtkPolyData()
        out.DeepCopy(pd)
        return out, found
    return _clean_triangles(pd), found


def lps_ras(polydata):
    """Copy with x, y negated (LPS <-> RAS): the points and the 3-component point / cell vectors."""
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
    out = vtk.vtkPolyData()
    out.DeepCopy(polydata)
    pts = vtk_to_numpy(out.GetPoints().GetData()).astype(np.float64)
    pts[:, :2] *= -1.0
    out.GetPoints().SetData(numpy_to_vtk(pts, deep=True))
    for data in (out.GetPointData(), out.GetCellData()):
        for i in range(data.GetNumberOfArrays()):
            a = data.GetArray(i)
            if a is not None and a.GetNumberOfComponents() == 3 and "normal" not in (a.GetName() or "").lower():
                v = vtk_to_numpy(a)
                v[:, :2] *= -1.0
                a.Modified()
    normals = out.GetPointData().GetNormals()
    if normals is not None:
        v = vtk_to_numpy(normals)
        v[:, :2] *= -1.0
        normals.Modified()
    return out


def file_space(polydata, path):
    fd = polydata.GetFieldData()
    arr = fd.GetAbstractArray("SPACE") if fd is not None else None
    if arr is not None and arr.GetNumberOfValues() > 0:
        v = str(arr.GetVariantValue(0).ToString()).upper()
        if v in ("LPS", "RAS"):
            return v
    if str(path).lower().endswith(".vtk"):
        with open(path, "rb") as f:
            head = f.read(512).decode("latin-1", errors="ignore").upper()
        for s in ("LPS", "RAS"):
            if "SPACE=" + s in head:
                return s
    return None


def write_vtk_ras(polydata, path):
    """Legacy .vtk with SPACE=RAS in the header and in the field data (what eurelaps_lung reads)."""
    pd = vtk.vtkPolyData()
    pd.ShallowCopy(polydata)
    fd = vtk.vtkFieldData()
    space = vtk.vtkStringArray()
    space.SetName("SPACE")
    space.InsertNextValue("RAS")
    fd.AddArray(space)
    pd.SetFieldData(fd)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    w = vtk.vtkPolyDataWriter()
    w.SetFileName(str(path))
    w.SetInputData(pd)
    w.SetHeader("LungCollapse (3D Slicer). SPACE=RAS")
    w.SetFileTypeToBinary()
    if not w.Write():
        raise OSError("could not write %s" % path)
    return path


# --------------------------------------------------------------------------------------------------
# numbers and checks

def volume_ml(polydata):
    mp = vtk.vtkMassProperties()
    mp.SetInputData(polydata)
    mp.Update()
    return mp.GetVolume() / 1000.0


def n_components(polydata):
    c = vtk.vtkPolyDataConnectivityFilter()
    c.SetInputData(polydata)
    c.SetExtractionModeToAllRegions()
    c.Update()
    return c.GetNumberOfExtractedRegions()


def open_edges(polydata):
    fe = vtk.vtkFeatureEdges()
    fe.SetInputData(polydata)
    fe.BoundaryEdgesOn()
    fe.NonManifoldEdgesOn()
    fe.FeatureEdgesOff()
    fe.ManifoldEdgesOff()
    fe.Update()
    return fe.GetOutput().GetNumberOfCells()


def crosses(tree, lung):
    """True when the tree surface has points both inside and outside the lung surface (it pierces it:
    the hilum needs these rings)."""
    dist = vtk.vtkImplicitPolyDataDistance()
    dist.SetInput(lung)
    n = tree.GetNumberOfPoints()
    step = max(1, n // 5000)
    inside = outside = False
    for i in range(0, n, step):
        d = dist.EvaluateFunction(tree.GetPoint(i))
        inside |= d < 0
        outside |= d > 0
        if inside and outside:
            return True
    return False


def check_surfaces(surfaces):
    """Light checks before the run (section 3). surfaces: {name: vtkPolyData}. Returns a list of warnings."""
    warnings = []
    lung = surfaces.get("inflated")
    for name, pd in surfaces.items():
        if pd is None or name not in ("inflated", "collapsed"):      # trees may be open / in pieces
            continue
        k = n_components(pd)
        if k != 1:
            warnings.append("%s: %d connected components (one is expected)" % (name, k))
        e = open_edges(pd)
        if e:
            warnings.append("%s: not closed (%d open or non-manifold edges)" % (name, e))
    if lung is not None:
        v = volume_ml(lung)
        if not 1000.0 <= v <= 4000.0:
            warnings.append("inflated lung volume %.0f mL is outside the plausible range 1-4 L" % v)
        for s in STRUCTURES:
            if surfaces.get(s) is not None and not crosses(surfaces[s], lung):
                warnings.append("%s do not cross the lung surface: the hilum needs their rings" % s)
    return warnings


def surfaces_hash(surfaces, extra=""):
    """Hash of the exported surfaces (points) and of a text (the geometry keys of the config)."""
    from vtk.util.numpy_support import vtk_to_numpy
    h = hashlib.sha1()
    for name in sorted(surfaces):
        pd = surfaces[name]
        h.update(name.encode())
        if pd is not None and pd.GetNumberOfPoints():
            pts = np.ascontiguousarray(vtk_to_numpy(pd.GetPoints().GetData()), dtype=np.float64)
            h.update(np.round(pts, 4).tobytes())
    h.update(extra.encode())
    return h.hexdigest()


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "lung"


log = logging.getLogger(__name__)
