"""Lung Collapse: predicts the collapsed lung of a patient loaded in 3D Slicer and shows the collapse as a
sequence (eurelaps_lung.predict, run in a separate process). Design: SLICER_EXTENSION.md."""

import enum
import importlib.util
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Annotated

import qt
import vtk

import slicer
from slicer.i18n import tr as _
from slicer.parameterNodeWrapper import WithinRange, parameterNodeWrapper
from slicer.ScriptedLoadableModule import (ScriptedLoadableModule, ScriptedLoadableModuleLogic,
                                           ScriptedLoadableModuleTest, ScriptedLoadableModuleWidget)
from slicer.util import VTKObservationMixin

from slicer import vtkMRMLScalarVolumeNode, vtkMRMLSegmentationNode

from LungCollapseLib import dependencies as lcdeps
from LungCollapseLib import inputs as lcinputs
from LungCollapseLib import outputs as lcoutputs
from LungCollapseLib import process as lcprocess
from LungCollapseLib import workfolder as lcwork

PACKAGE = "eurelaps_lung"
PACKAGE_VERSION = "0.1.1"
STRUCTURES = lcinputs.STRUCTURES
LUNGS = ("lungLeft", "lungRight")
SEG_INPUTS = LUNGS + STRUCTURES + ("collapsed",)       # segmentation inputs (scene: segments; file: files)
MULTI_INPUTS = LUNGS + ("collapsed",)                   # several segments / files, merged
INPUT_LABELS = {"ct": "CT", "lungLeft": "left lung", "lungRight": "right lung", "airways": "airways",
                "arteries": "arteries", "veins": "veins", "collapsed": "collapsed lung"}
INPUT_COLORS = {"lungLeft": (0.85, 0.55, 0.45), "lungRight": (0.85, 0.55, 0.45), "airways": (0.9, 0.9, 0.9),
                "arteries": (0.8, 0.2, 0.2), "veins": (0.2, 0.3, 0.85), "collapsed": (0.5, 0.5, 0.5)}
LUNG_CLOSING_MM = 1.5   # closing after merging several segments of a lung (fills the fissure gaps)
GEOMETRY_STEPS = ("geometry", "hilum", "fem_setup")


# The rules (E from the collapse %, q0 = p0 / E from the collapse %) come from the package: predict.rules(), the
# file eurelaps_lung/data/rules.json (or $EURELAPS_RULES), rewritten by main_calibrate.py --make-default after new
# fits. The values below are only the fallback without the package (the fake backend of the tests).
FALLBACK_RULES = {"rule": "E", "a": 3.5380, "b": -0.012883, "n": 13, "residual_sd": 0.158,
                  "collapse_range": [23.1, 86.3], "positions": ["supine", "supine"], "source": "built-in fallback",
                  "q0": {"a": -0.9351, "b": 0.010662, "residual_sd": 0.053, "r": 0.964, "n": 13}}
_RULES_CACHE = {}


def rules():
    """The prediction rules in use (cached; reload_rules() after changing the package or $EURELAPS_RULES)."""
    if "r" not in _RULES_CACHE:
        r = None
        try:
            if importlib.util.find_spec(PACKAGE) is not None:
                from eurelaps_lung import predict
                if hasattr(predict, "rules"):              # eurelaps_lung >= 0.1.1
                    r = predict.rules()
        except Exception as e:
            logging.warning("Lung Collapse: rules of %s not read (%s): built-in fallback", PACKAGE, e)
        if r is None or r.get("rule") != "E" or not r.get("q0"):
            r = FALLBACK_RULES
        _RULES_CACHE["r"] = r
    return _RULES_CACHE["r"]


def reload_rules():
    _RULES_CACHE.clear()
    return rules()


def E_from_collapse(percent):
    r = rules()
    return 10.0 ** (r["a"] + r["b"] * percent)


def collapse_from_E(E):
    r = rules()
    return (math.log10(max(E, 1e-9)) - r["a"]) / r["b"]


def q0_from_collapse(percent):
    q = rules()["q0"]
    return 10.0 ** (q["a"] + q["b"] * percent)


def collapse_from_q0(q0):
    q = rules()["q0"]
    return (math.log10(max(q0, 1e-9)) - q["a"]) / q["b"]


def rule_info():
    r = rules()
    return dict(patients=r.get("n"), collapse_percent_range=tuple(r.get("collapse_range", (23.0, 86.0))),
                residual_sd_log10=r.get("residual_sd"), positions=tuple(r.get("positions", ("supine", "supine"))),
                source=r.get("source"))


#
# LungCollapse
#


class LungCollapse(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Lung Collapse")
        self.parent.categories = [_("Chest")]
        self.parent.dependencies = ["Segmentations", "Sequences", "Markups", "Models", "Tables"]
        self.parent.contributors = ["Edoardo De Rose (University of Calabria)"]
        self.parent.helpText = _("""
Predicts the collapsed lung (pneumothorax) of the patient in the scene with the finite element model of
<i>eurelaps_lung</i>, and shows the collapse as a sequence over the CT.<br>
Inputs: a segmentation with the lung (or its lobes), the airways, the pulmonary arteries and veins.
Optionally, the real collapsed lung, to measure the error of the prediction.
""")
        self.parent.acknowledgementText = _("""
EURELAPS project. The prediction model and its calibration: eurelaps_lung.
""")


#
# Parameter node
#


class Position(enum.Enum):
    SUPINE = "supine"
    LATERAL = "lateral"
    PRONE = "prone"

    def label(self):
        return {"supine": _("Supine (as the CT)"), "lateral": _("Lateral decubitus, healthy side down"),
                "prone": _("Prone")}[self.value]


class CtPosition(enum.Enum):
    SUPINE = "supine"
    PRONE = "prone"
    LEFT_LATERAL = "left_lateral"
    RIGHT_LATERAL = "right_lateral"

    def label(self):
        return {"supine": _("Supine"), "prone": _("Prone"), "left_lateral": _("Left lateral"),
                "right_lateral": _("Right lateral")}[self.value]


class Device(enum.Enum):
    CPU = "cpu"
    CUDA = "cuda"

    def label(self):
        return {"cpu": _("CPU"), "cuda": _("CUDA")}[self.value]


@parameterNodeWrapper
class LungCollapseParameterNode:
    # inputs: per input a source ("scene" | "file"), the scene node (+ segment IDs, comma separated) and the
    # file path(s) (";" separated). Kept by the widget's own code (radios, segment selectors, path edits).
    ctSource: str = "scene"
    ctVolume: vtkMRMLScalarVolumeNode
    ctPath: str = ""
    lungLeftSource: str = "scene"
    lungLeftSegmentation: vtkMRMLSegmentationNode
    lungLeftSegmentIds: str = ""
    lungLeftPath: str = ""
    lungRightSource: str = "scene"
    lungRightSegmentation: vtkMRMLSegmentationNode
    lungRightSegmentIds: str = ""
    lungRightPath: str = ""
    airwaysSource: str = "scene"
    airwaysSegmentation: vtkMRMLSegmentationNode
    airwaysSegmentIds: str = ""
    airwaysPath: str = ""
    arteriesSource: str = "scene"
    arteriesSegmentation: vtkMRMLSegmentationNode
    arteriesSegmentIds: str = ""
    arteriesPath: str = ""
    veinsSource: str = "scene"
    veinsSegmentation: vtkMRMLSegmentationNode
    veinsSegmentIds: str = ""
    veinsPath: str = ""
    compareCollapsed: bool = False
    collapsedSource: str = "scene"
    collapsedSegmentation: vtkMRMLSegmentationNode
    collapsedSegmentIds: str = ""
    collapsedPath: str = ""
    side: str = "left"
    # collapse
    collapsePercent: Annotated[float, WithinRange(0.0, 95.0)] = 50.0
    useTargetCollapse: bool = False
    driveBy: str = "percent"               # "percent": collapse % given, p0 estimated; "pressure": the reverse
    linkE: bool = True                     # E follows the collapse % (and back) through the calibrated rule
    p0Value: Annotated[float, WithinRange(1.0, 50000.0)] = 330.0
    eValue: Annotated[float, WithinRange(10.0, 100000.0)] = 1004.0
    position: Position = Position.SUPINE
    # lung tissue: emphysema yes / no is the user's reading of the CT ("" = not answered yet); the limit of
    # compression (barrier) uses the CT's density and emphysema volume, measured in the run
    emphysema: str = ""
    barrier: bool = False
    frames: Annotated[int, WithinRange(0, 60)] = 30
    # advanced
    nu: Annotated[float, WithinRange(0.0, 0.49)] = 0.30
    ctPosition: CtPosition = CtPosition.SUPINE
    wallMargin: Annotated[float, WithinRange(0.0, 30.0)] = 5.0
    device: Device = Device.CPU
    rebuildGeometry: bool = False
    # run
    outputDir: str = ""


#
# Widget
#


def _set_row_visible(widget, visible):
    """Show / hide a QFormLayout field together with its label."""
    widget.setVisible(visible)
    parent = widget.parentWidget()
    layout = parent.layout() if parent is not None else None
    if isinstance(layout, qt.QFormLayout):
        label = layout.labelForField(widget)
        if label is not None:
            label.setVisible(visible)


class LungCollapseWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None) -> None:
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None
        self._updating = False
        self._guess = None
        self._inflatedMl = None
        self._collapsedMl = None
        self._targetPercent = None
        self._lastPercent = None
        self._lastE = None
        self._results = None
        self._lastConfig = None
        self._volumeTimer = None
        self._deviceChecked = False

    # ------------------------------------------------------------------ setup
    def setup(self) -> None:
        ScriptedLoadableModuleWidget.setup(self)
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/LungCollapse.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)
        self.logic = LungCollapseLogic()
        ui = self.ui

        # inputs: one row per input, Scene / File
        imageFilters = ["Images (*.nrrd *.nii *.nii.gz *.mha *.mhd *.nhdr)", "All files (*)"]
        ui.ctPathEdit.nameFilters = imageFilters
        for n in STRUCTURES:
            getattr(ui, n + "PathEdit").nameFilters = [lcinputs.SEGMENTATION_FILE_FILTER, "All files (*)"]
        for n in ("ctPathEdit", "airwaysPathEdit", "arteriesPathEdit", "veinsPathEdit", "outputDirPathLineEdit"):
            getattr(ui, n).settingKey = "LungCollapse/" + n
        for n in SEG_INPUTS:
            sel = getattr(ui, n + "SegmentSelector")
            sel.setMRMLScene(slicer.mrmlScene)
            sel.segmentationNodeSelectorVisible = True
            sel.multiSelection = n in MULTI_INPUTS
            sel.noneEnabled = True

        # sequence browser: play and seek widgets
        self.playWidget = slicer.qMRMLSequenceBrowserPlayWidget()
        self.seekWidget = slicer.qMRMLSequenceBrowserSeekWidget()
        ui.sequencePlaceholder.layout().addWidget(self.seekWidget)
        ui.sequencePlaceholder.layout().addWidget(self.playWidget)
        ui.sequenceWidget.setVisible(False)
        ui.resultsTableView.setVisible(False)
        ui.progressBar.setVisible(False)
        ui.resultsTableView.setMRMLScene(slicer.mrmlScene)

        # connections
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
        for n in ("ct",) + SEG_INPUTS:
            getattr(ui, n + "FileRadio").toggled.connect(lambda isFile, n=n: self.onSourceToggled(n, isFile))
        ui.ctNodeSelector.currentNodeChanged.connect(self.onCtNodeChanged)
        ui.ctPathEdit.currentPathChanged.connect(lambda path: self.onPathChanged("ct", path))
        for n in SEG_INPUTS:
            sel = getattr(ui, n + "SegmentSelector")
            sel.connect("currentNodeChanged(vtkMRMLNode*)", lambda node, n=n: self.onSegmentationNodeChanged(n, node))
            if n in MULTI_INPUTS:
                sel.connect("segmentSelectionChanged(QStringList)", lambda ids, n=n: self.onSegmentsChanged(n))
                getattr(ui, n + "PathsEdit").editingFinished.connect(lambda n=n: self.onPathsEdited(n))
                getattr(ui, n + "BrowseButton").clicked.connect(lambda checked=False, n=n: self.onBrowse(n))
            else:
                sel.connect("currentSegmentChanged(QString)", lambda sid, n=n: self.onSegmentsChanged(n))
                getattr(ui, n + "PathEdit").currentPathChanged.connect(lambda path, n=n: self.onPathChanged(n, path))
        ui.sideLeftRadio.toggled.connect(self.onSideChanged)
        ui.compareCheckBox.toggled.connect(lambda *args: self.inputsChanged())
        ui.drivePressureRadio.toggled.connect(self.onDriveChanged)
        ui.emphysemaYesRadio.toggled.connect(lambda on: self.onEmphysemaChanged("yes", on))
        ui.emphysemaNoRadio.toggled.connect(lambda on: self.onEmphysemaChanged("no", on))
        ui.sideRightRadio.toggled.connect(self.onSideChanged)
        ui.collapseVolumeSpinBox.valueChanged.connect(self.onCollapseVolumeChanged)
        ui.outputDirPathLineEdit.currentPathChanged.connect(self.onOutputDirChanged)
        ui.advancedCollapsibleButton.contentsCollapsed.connect(self.onAdvancedToggled)
        ui.applyButton.clicked.connect(self.onApply)
        ui.cancelButton.clicked.connect(self.onCancel)
        ui.saveOutputsButton.clicked.connect(self.onSaveOutputs)
        ui.showConfigButton.clicked.connect(self.onShowConfig)
        ui.colorByComboBox.currentIndexChanged.connect(self.onColorBy)
        for cb in (ui.showCtCheckBox, ui.showInflatedCheckBox, ui.showPredictionCheckBox, ui.showTargetCheckBox,
                   ui.showHilumCheckBox):
            cb.toggled.connect(self.updateVisibility)

        self._volumeTimer = qt.QTimer()
        self._volumeTimer.setSingleShot(True)
        self._volumeTimer.setInterval(400)
        self._volumeTimer.timeout.connect(self.updateInflatedVolume)

        self.initializeParameterNode()

    def cleanup(self) -> None:
        self.removeObservers()
        if self.logic:
            self.logic.cancel()
            self.logic.workFolders.discard()

    def enter(self) -> None:
        self.initializeParameterNode()

    def exit(self) -> None:
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGui)

    def onSceneStartClose(self, caller, event) -> None:
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        self._results = None
        self._guess = None
        self._inflatedMl = None
        self.ui.sequenceWidget.setVisible(False)
        self.ui.resultsTableView.setVisible(False)
        self.ui.saveOutputsButton.enabled = False
        if self.parent.isEntered:
            self.initializeParameterNode()

    # ------------------------------------------------------------------ parameter node
    def initializeParameterNode(self) -> None:
        self.setParameterNode(self.logic.getParameterNode())
        p = self._parameterNode
        if p.ctSource == "scene" and p.ctVolume is None:
            first = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if first is not None and first.GetClassName() == "vtkMRMLScalarVolumeNode":
                p.ctVolume = first
        self.restoreFromParameterNode()
        if not any(getattr(p, n + "Segmentation") for n in SEG_INPUTS):
            first = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSegmentationNode")
            if first is not None and not first.GetAttribute("LungCollapse.Input"):
                self.autofill(first)

    def setParameterNode(self, parameterNode) -> None:
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGui)
        self._parameterNode = parameterNode
        if self._parameterNode:
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGui)
            self.updateGui()

    def restoreFromParameterNode(self):
        """The widgets that the parameter node does not connect by itself."""
        p, ui = self._parameterNode, self.ui
        if p is None:
            return
        self._updating = True
        try:
            ui.ctNodeSelector.setCurrentNode(p.ctVolume)
            ui.ctPathEdit.currentPath = p.ctPath
            for n in ("ct",) + SEG_INPUTS:
                isFile = getattr(p, n + "Source") == "file"
                getattr(ui, n + ("FileRadio" if isFile else "SceneRadio")).checked = True
                self._showStackPage(n, isFile)
            for n in SEG_INPUTS:
                self._setSelector(n, getattr(p, n + "Segmentation"), self._ids(n))
                if n in MULTI_INPUTS:
                    getattr(ui, n + "PathsEdit").text = getattr(p, n + "Path")
                else:
                    getattr(ui, n + "PathEdit").currentPath = getattr(p, n + "Path")
            (ui.sideLeftRadio if p.side == "left" else ui.sideRightRadio).checked = True
            (ui.drivePressureRadio if p.driveBy == "pressure" else ui.drivePercentRadio).checked = True
            for radio, value in ((ui.emphysemaYesRadio, "yes"), (ui.emphysemaNoRadio, "no")):
                radio.autoExclusive = False                   # none checked until the user answers
                radio.checked = p.emphysema == value
                radio.autoExclusive = True
            ui.outputDirPathLineEdit.currentPath = p.outputDir
        finally:
            self._updating = False
        self._volumeTimer.start()
        self.updateGui()

    # ------------------------------------------------------------------ inputs
    def _ids(self, n):
        return [i for i in getattr(self._parameterNode, n + "SegmentIds").split(",") if i]

    def _setSelector(self, n, node, ids):
        sel = getattr(self.ui, n + "SegmentSelector")
        sel.setCurrentNode(node)
        if n in MULTI_INPUTS:
            sel.setSelectedSegmentIDs(ids)
        else:
            sel.setCurrentSegmentID(ids[0] if ids else "")

    def _showStackPage(self, n, isFile):
        stack = getattr(self.ui, n + "Stack")
        stack.currentIndex = 1 if isFile else 0
        for i in range(stack.count):                      # size the stack to the visible page only
            policy = qt.QSizePolicy.Preferred if i == stack.currentIndex else qt.QSizePolicy.Ignored
            stack.widget(i).setSizePolicy(qt.QSizePolicy.Preferred, policy)

    def source(self, n):
        return getattr(self._parameterNode, n + "Source") if self._parameterNode else "scene"

    def onSourceToggled(self, n, isFile):
        self._showStackPage(n, isFile)
        if self._updating or self._parameterNode is None:
            return
        setattr(self._parameterNode, n + "Source", "file" if isFile else "scene")
        if isFile:
            self.loadFileInput(n)
        self.inputsChanged()

    def onCtNodeChanged(self, node):
        if self._updating or self._parameterNode is None:
            return
        self._parameterNode.ctVolume = node
        if node is not None and self.source("ct") == "scene":
            slicer.util.setSliceViewerLayers(background=node, fit=True)

    def onSegmentationNodeChanged(self, n, node):
        """A segmentation chosen in one row: its segments are pre-selected by name in that row, and in the
        rows still empty (same segmentation)."""
        if self._updating or self._parameterNode is None:
            return
        if node is not None:
            self.autofill(node, row=n)
        else:
            self.onSegmentsChanged(n)

    def autofill(self, node, row=None):
        """Pre-selection from the segment names (or centroids): the lungs of both sides, the three trees, the
        collapsed lung and the side to collapse. `row` gets it in any case; the other scene rows only when empty."""
        p = self._parameterNode
        g = lcinputs.guess(node)
        wanted = {"lungLeft": g["lungs"]["left"], "lungRight": g["lungs"]["right"], "collapsed":
                  [g["collapsed"]] if g["collapsed"] else []}
        wanted.update({s: [g[s]] if g[s] else [] for s in STRUCTURES})
        self._updating = True
        try:
            for n in SEG_INPUTS:
                if self.source(n) != "scene":
                    continue
                if n == row or (getattr(p, n + "Segmentation") is None and wanted[n]):
                    self._setSelector(n, node, wanted[n])
                    if n == row and not wanted[n] and n not in MULTI_INPUTS and node.GetSegmentation().GetNumberOfSegments():
                        getattr(self.ui, n + "SegmentSelector").setCurrentSegmentID(
                            node.GetSegmentation().GetNthSegmentID(0))
            if g["collapsed"] and self.source("collapsed") == "scene" and (
                    row is None or row != "collapsed") and not p.compareCollapsed:
                p.compareCollapsed = True
            if g["side"]:
                p.side = g["side"]
                (self.ui.sideLeftRadio if g["side"] == "left" else self.ui.sideRightRadio).checked = True
        finally:
            self._updating = False
        self.ui.sideNoteLabel.text = _("(from the segment names)") if g["side"] else ""
        for n in SEG_INPUTS:
            self._syncFromSelector(n)
        self.inputsChanged()

    def _syncFromSelector(self, n):
        p = self._parameterNode
        sel = getattr(self.ui, n + "SegmentSelector")
        ids = list(sel.selectedSegmentIDs()) if n in MULTI_INPUTS else (
            [sel.currentSegmentID()] if sel.currentSegmentID() else [])
        node = sel.currentNode()
        if getattr(p, n + "Segmentation") is not node:
            setattr(p, n + "Segmentation", node)
        text = ",".join(ids)
        if getattr(p, n + "SegmentIds") != text:
            setattr(p, n + "SegmentIds", text)

    def onSegmentsChanged(self, n):
        if self._updating or self._parameterNode is None:
            return
        self._syncFromSelector(n)
        self.inputsChanged()

    def onPathChanged(self, n, path):
        """CT and trees from a file: loaded and shown as soon as the path points to a file."""
        if self._updating or self._parameterNode is None:
            return
        setattr(self._parameterNode, n + "Path", path or "")
        if self.source(n) == "file":
            self.loadFileInput(n)
        self.inputsChanged()

    def onPathsEdited(self, n):
        if self._updating or self._parameterNode is None:
            return
        text = getattr(self.ui, n + "PathsEdit").text.strip()
        if text != getattr(self._parameterNode, n + "Path"):
            setattr(self._parameterNode, n + "Path", text)
            self.loadFileInput(n)
            self.inputsChanged()

    def onBrowse(self, n):
        paths = qt.QFileDialog.getOpenFileNames(
            slicer.util.mainWindow(), _("{what}: one or more files (merged)").format(what=INPUT_LABELS[n].capitalize()),
            "", lcinputs.SEGMENTATION_FILE_FILTER + ";;All files (*)")
        if paths:
            getattr(self.ui, n + "PathsEdit").text = "; ".join(paths)
            self.onPathsEdited(n)

    def _paths(self, n):
        return [x.strip() for x in getattr(self._parameterNode, n + "Path").split(";") if x.strip()]

    def loadFileInput(self, n):
        """Load the file(s) of an input into the scene (once per path; a new path replaces the old node)."""
        paths = self._paths(n)
        if not paths or not all(os.path.isfile(x) for x in paths):
            return None
        with slicer.util.tryWithErrorDisplay(_("Failed to load the {what}.").format(what=INPUT_LABELS[n]),
                                             waitCursor=True):
            node = self.logic.loadFileInput(n, paths)
            lm = slicer.app.layoutManager()
            if n != "ct" and lm is not None and lm.threeDWidget(0) is not None:
                lm.threeDWidget(0).threeDView().resetFocalPoint()
            if n == "ct":
                slicer.util.setSliceViewerLayers(background=node, fit=True)
            elif n in LUNGS and self._parameterNode is not None:
                other = "lungRight" if n == "lungLeft" else "lungLeft"
                if not self.logic.inputAvailable(self._parameterNode, other):
                    self._setSide("left" if n == "lungLeft" else "right")
                elif n == "lungLeft" or n == "lungRight":
                    side = lcinputs.name_side(" ".join(os.path.basename(x) for x in self._paths("collapsed")))
                    if side:
                        self._setSide(side)
            elif n == "collapsed":
                side = lcinputs.name_side(" ".join(os.path.basename(x) for x in paths))
                if side and self.logic.inputAvailable(self._parameterNode, "lung" + side.capitalize()):
                    self._setSide(side)
            return node
        return None

    def _setSide(self, side):
        self._parameterNode.side = side
        self._updating = True
        try:
            (self.ui.sideLeftRadio if side == "left" else self.ui.sideRightRadio).checked = True
        finally:
            self._updating = False

    def onSideChanged(self, *args):
        if self._updating or self._parameterNode is None:
            return
        side = "left" if self.ui.sideLeftRadio.checked else "right"
        if side != self._parameterNode.side:
            self._parameterNode.side = side
            self.inputsChanged()

    def inputsChanged(self):
        self._inflatedMl = None
        self._volumeTimer.start()
        self.updateGui()

    def updateInflatedVolume(self):
        """Volumes of the lung to collapse and of the collapsed lung (sum of their segments' closed surfaces):
        the mL <-> % link, and the collapse % of the collapsed lung (the default when it is given)."""
        p = self._parameterNode
        self._inflatedMl = self._collapsedMl = None
        hadTarget = self._targetPercent is not None
        self._targetPercent = None

        def volume(name):
            resolved = self.logic.resolveInput(p, name, load=False)
            if resolved is None:
                return None
            node, ids = resolved
            return sum(lcinputs.volume_ml(lcinputs.segment_surface(node, sid)) for sid in ids)

        try:
            if p is not None:
                with slicer.util.WaitCursor():
                    self._inflatedMl = volume("lung" + p.side.capitalize())
                    if p.compareCollapsed:
                        self._collapsedMl = volume("collapsed")
        except Exception as e:
            logging.warning("Lung Collapse: volumes not computed: %s", e)
        if self._inflatedMl and self._collapsedMl:
            self._targetPercent = min(max(100.0 * (1.0 - self._collapsedMl / self._inflatedMl), 0.0), 95.0)
            if not hadTarget and not p.useTargetCollapse:
                p.useTargetCollapse = True               # a collapsed lung given: its collapse % is the default
        self.updateGui()

    def onDriveChanged(self, pressure):
        if self._updating or self._parameterNode is None:
            return
        self._parameterNode.driveBy = "pressure" if pressure else "percent"
        if pressure and not self._parameterNode.barrier:
            self._parameterNode.barrier = True               # a surgical pressure needs the limit of compression

    def onEmphysemaChanged(self, value, on):
        if self._updating or self._parameterNode is None or not on:
            return
        self._parameterNode.emphysema = value

    def syncCollapseAndE(self):
        """Collapse % <-> E through the calibrated rule (when linked); the collapsed lung's % when used."""
        p = self._parameterNode
        if p.driveBy == "pressure":
            # p0 and E given: the collapse % shown is the estimate from q0 = p0 / E
            pct = round(min(max(collapse_from_q0(p.p0Value / max(p.eValue, 1e-9)), 0.0), 95.0), 1)
            if abs(p.collapsePercent - pct) > 0.05:
                p.collapsePercent = pct
            self._lastPercent, self._lastE, self._lastLink = p.collapsePercent, p.eValue, p.linkE
            return
        if p.useTargetCollapse and self._targetPercent is not None and abs(p.collapsePercent - self._targetPercent) > 0.05:
            p.collapsePercent = round(self._targetPercent, 1)
        relinked = p.linkE and not getattr(self, "_lastLink", True)
        self._lastLink = p.linkE
        if p.linkE:
            if relinked or self._lastPercent is None or abs(p.collapsePercent - self._lastPercent) > 1e-6 or self._lastE is None:
                E = round(E_from_collapse(p.collapsePercent))
                self._lastPercent, self._lastE = p.collapsePercent, E
                if abs(p.eValue - E) > 0.5:
                    p.eValue = E
            elif abs(p.eValue - self._lastE) > 0.5:          # E edited: the collapse % follows
                pct = round(min(max(collapse_from_E(p.eValue), 0.0), 95.0), 1)
                self._lastE = p.eValue
                self._lastPercent = pct
                p.collapsePercent = pct
        else:
            self._lastPercent, self._lastE = p.collapsePercent, p.eValue
        p0 = round(q0_from_collapse(p.collapsePercent) * p.eValue)    # the suggested p0
        if abs(p.p0Value - p0) > 0.5:
            p.p0Value = p0

    def onCollapseVolumeChanged(self, value):
        if self._updating or not self._inflatedMl or self._parameterNode is None:
            return
        pct = 100.0 * (1.0 - value / self._inflatedMl)
        self._parameterNode.collapsePercent = min(max(pct, 0.0), 95.0)

    def onAdvancedToggled(self, collapsed):
        if collapsed or self._deviceChecked:
            return
        self._deviceChecked = True
        cuda = False
        try:
            if importlib.util.find_spec("torch") is not None:
                import torch
                cuda = bool(torch.cuda.is_available())
        except Exception:
            cuda = False
        model = self.ui.deviceComboBox.model()
        item = model.item(1) if hasattr(model, "item") else None
        if item is not None:
            item.setEnabled(cuda)
        if not cuda and self._parameterNode is not None and self._parameterNode.device == Device.CUDA:
            self._parameterNode.device = Device.CPU

    def onOutputDirChanged(self, path):
        if self._parameterNode is not None and not self._updating:
            self._parameterNode.outputDir = path

    # ------------------------------------------------------------------ state of the GUI
    @staticmethod
    def _setInput(spin, editable, estimate=False):
        """A spin box as an input (editable) or as a value that follows the others (read-only, no arrows)."""
        spin.readOnly = not editable
        spin.buttonSymbols = qt.QAbstractSpinBox.UpDownArrows if editable else qt.QAbstractSpinBox.NoButtons
        spin.prefix = "≈ " if estimate else ""
        spin.styleSheet = "" if editable else "QAbstractSpinBox { background: transparent; border: none; font-weight: bold; }"
        spin.focusPolicy = qt.Qt.StrongFocus if editable else qt.Qt.NoFocus

    def updateCollapseSection(self):
        """Collapse: what is given (editable) and what follows from it (read-only), for the two ways of driving
        the collapse: by the collapse % (E from the rule, p0 estimated) or by the pressure p0 (and E)."""
        p, ui = self._parameterNode, self.ui
        pressure = p.driveBy == "pressure"
        info = rule_info()
        lo, hi = info["collapse_percent_range"]

        # order of the groups: what is given first
        lay = ui.collapseGroupBox.parentWidget().layout()
        first, second = (ui.forcesGroupBox, ui.collapseGroupBox) if pressure else (ui.collapseGroupBox, ui.forcesGroupBox)
        if lay.indexOf(first) > lay.indexOf(second):
            lay.removeWidget(first)
            lay.insertWidget(lay.indexOf(second), first)
        ui.collapseGroupBox.title = _("Expected collapse (result of the run)") if pressure else _("Collapse")
        ui.forcesGroupBox.title = _("Pressure and stiffness") if pressure else _("Stiffness and pressure")

        # collapse group: the slider and the volume when the % is given; the estimate when p0 is given
        ui.percentRowsWidget.setVisible(not pressure)
        ui.estimateRowsWidget.setVisible(pressure)
        if self._targetPercent is not None:
            ui.useTargetCollapseCheckBox.text = _("Use the collapse % of the collapsed lung ({v:.1f} %)").format(
                v=self._targetPercent)
        else:
            ui.useTargetCollapseCheckBox.text = _("Use the collapse % of the collapsed lung")
        ui.collapsePercentSlider.enabled = not p.useTargetCollapse
        self._updating = True
        try:
            if self._inflatedMl:
                ui.collapseVolumeSpinBox.enabled = not p.useTargetCollapse
                ui.collapseVolumeSpinBox.maximum = self._inflatedMl
                ui.collapseVolumeSpinBox.value = (1.0 - p.collapsePercent / 100.0) * self._inflatedMl
                ui.inflatedOfLabel.text = _("of {v:.0f} mL inflated").format(v=self._inflatedMl)
            else:
                ui.collapseVolumeSpinBox.enabled = False
                ui.inflatedOfLabel.text = ""
        finally:
            self._updating = False
        lastPct = (self._results or {}).get("collapse_percent")
        if pressure:
            vol = (_(" (≈ {v:.0f} mL)").format(v=(1 - p.collapsePercent / 100) * self._inflatedMl)
                   if self._inflatedMl else "")
            last = _("last run: {v:.1f} %").format(v=lastPct) if lastPct is not None else ""
            ui.collapseEstimateLabel.text = ("<b>≈ %.1f %%</b>%s<br><span style='color:gray'>%s%s</span>" % (
                p.collapsePercent, vol, _("estimate from q0 = p0 / E; the run gives the real value"),
                ("<br>" + last) if last else ""))
        if p.useTargetCollapse and not pressure:
            ui.collapseRangeLabel.text = _("The collapse % of the collapsed lung.")
            ui.collapseRangeLabel.styleSheet = "color: gray"
        elif lo <= p.collapsePercent <= hi:
            ui.collapseRangeLabel.text = _("The rules are calibrated on {lo:.0f}-{hi:.0f} % ({n} patients).").format(
                lo=lo, hi=hi, n=info.get("patients", "?"))
            ui.collapseRangeLabel.styleSheet = "color: gray"
        else:
            ui.collapseRangeLabel.text = _("Extrapolation: the rules are calibrated on {lo:.0f}-{hi:.0f} %.").format(
                lo=lo, hi=hi)
            ui.collapseRangeLabel.styleSheet = "color: #d07000; font-weight: bold"
        ui.collapseGroupBox.layout().activate()

        # stiffness and pressure: inputs or values that follow
        eEditable = pressure or not p.linkE
        self._setInput(ui.eSpinBox, eEditable)
        self._setInput(ui.p0SpinBox, pressure, estimate=not pressure)
        ui.linkECheckBox.setVisible(not pressure)
        lo_E, hi_E = E_from_collapse(95.0), E_from_collapse(0.0)
        ui.eSpinBox.minimum = math.floor(lo_E) if (p.linkE and not pressure) else 10.0
        ui.eSpinBox.maximum = math.ceil(hi_E) if (p.linkE and not pressure) else 100000.0
        ui.p0CmLabel.text = "= %.2f cmH₂O" % (p.p0Value / 98.0665)
        last = (self._results or {}).get("p0_Pa")
        if pressure:
            ui.estimateLabel.text = _("Both given. E cannot come from the rule here: the rule needs the collapse %.")
        else:
            ui.estimateLabel.text = (
                (_("E follows the collapse % through the rule.") if p.linkE else _("E given (not from the rule)."))
                + " " + _("p0 is an estimate from the cohort fits: the run finds the exact value{last}.").format(
                    last=_(" (last run: {v:.0f} Pa)").format(v=last) if last else ""))
        ui.estimateLabel.styleSheet = "color: gray"

        # lung tissue
        if p.barrier:
            note = (_("The CT gives the lung density and the emphysema volume (measured in the run, once per CT)")
                    + (_("; emphysema 'No': its volume is not used.") if p.emphysema == "no" else "."))
            style = "color: gray"
        elif pressure:
            note = _("A surgical pressure (e.g. 12 mmHg = 1600 Pa) needs the limit of compression: without it the "
                     "lung can shrink below its own tissue volume.")
            style = "color: #d07000"
        else:
            note = _("Off: the collapse % sets the volume. Emphysema yes / no is recorded in the results.")
            style = "color: gray"
        last = (self._results or {}).get("compression")
        if last and last.get("phi") is not None:
            note += "<br>" + _("Last run: smallest volume {phi:.0f} % of the inflated lung (collapse at most "
                               "{mx:.0f} %).").format(phi=100 * last["phi"], mx=100 * (1 - last["phi"]))
        ui.tissueNoteLabel.text = note
        ui.tissueNoteLabel.styleSheet = style

        # position and sequence
        if p.position == Position.SUPINE:
            ui.positionNoteLabel.text = _("Calibrated on supine CTs.")
            ui.positionNoteLabel.styleSheet = "color: gray"
        else:
            healthy = "right" if p.side == "left" else "left"
            extra = (_(" ({healthy} side down)").format(healthy=healthy) if p.position == Position.LATERAL else "")
            ui.positionNoteLabel.text = _("Calibrated on supine CTs: the rules are not validated in this "
                                          "position{extra}.").format(extra=extra)
            ui.positionNoteLabel.styleSheet = "color: #d07000"
        ui.framesNoteLabel.styleSheet = "color: gray"

    def missingInputs(self):
        """What is missing before Apply (empty list = ready)."""
        p = self._parameterNode
        if p is None:
            return [_("parameter node")]
        missing = []
        lungs = [n for n in LUNGS if self.logic.inputAvailable(p, n)]
        if not lungs:
            missing.append(_("a lung (left or right)"))
        elif "lung" + p.side.capitalize() not in lungs:
            missing.append(_("the {side} lung (the side to collapse)").format(side=p.side))
        for n in STRUCTURES:
            if not self.logic.inputAvailable(p, n):
                missing.append(_(INPUT_LABELS[n]))
        if p.compareCollapsed and not self.logic.inputAvailable(p, "collapsed"):
            missing.append(_("collapsed lung"))
        if p.emphysema not in ("yes", "no"):
            missing.append(_("emphysema yes / no (Collapse section)"))
        if p.barrier and not self.logic.inputAvailable(p, "ct"):
            missing.append(_("the CT (for the limit of compression)"))
        return missing

    def updateGui(self, caller=None, event=None):
        p, ui = self._parameterNode, self.ui
        if p is None:
            return
        running = self.logic.running()

        # side: only the lungs given
        left, right = (self.logic.inputAvailable(p, n) for n in LUNGS)
        ui.sideLeftRadio.enabled = left or not right
        ui.sideRightRadio.enabled = right or not left
        ui.collapsedRow.setVisible(p.compareCollapsed)
        ui.collapsedLabel.setVisible(p.compareCollapsed)
        ui.compareNoteLabel.setVisible(p.compareCollapsed)
        anyFile = any(self.source(n) == "file" for n in ("ct",) + SEG_INPUTS)
        ui.removeLoadedCheckBox.setVisible(anyFile)
        compare = p.compareCollapsed and self.logic.inputAvailable(p, "collapsed")
        ui.useTargetCollapseCheckBox.enabled = compare
        if not compare and p.useTargetCollapse:
            p.useTargetCollapse = False

        # collapse % <-> E (rule) and the collapsed lung's %
        if not self._updating:
            self._updating = True
            try:
                self.syncCollapseAndE()
            finally:
                self._updating = False
        self.updateCollapseSection()

        # run
        missing = self.missingInputs()
        ui.inputStatusLabel.text = (_("Missing: ") + ", ".join(missing)) if missing else ""
        ui.inputStatusLabel.styleSheet = "color: #b03030" if missing else ""
        ui.applyButton.enabled = not missing and not running
        ui.applyButton.toolTip = (_("Missing: ") + ", ".join(missing)) if missing else _("Run the prediction")
        ui.cancelButton.enabled = running
        ui.saveOutputsButton.enabled = (not running and self.logic.workFolders.temp is not None
                                        and self._results is not None and self._results.get("is_temp", False))
        ui.showConfigButton.enabled = self._lastConfig is not None and os.path.isfile(self._lastConfig)
        for w in (ui.inputsCollapsibleButton, ui.collapseCollapsibleButton, ui.advancedCollapsibleButton,
                  ui.outputDirPathLineEdit):
            w.enabled = not running

    # ------------------------------------------------------------------ run
    def collectParams(self):
        """The UI values (named as the config keys) + surfaces + output_dir: what LungCollapseLogic.run gets."""
        p = self._parameterNode
        params = {
            "side": p.side,
            "collapse_percent": (None if p.driveBy == "pressure" else
                                 "target" if p.useTargetCollapse else float(p.collapsePercent)),
            "physics.collapse.p0_Pa": float(p.p0Value) if p.driveBy == "pressure" else None,
            "physics.gravity.collapse": self.logic.gravity_collapse(p.position.value, p.side),
            "physics.gravity.reference": p.ctPosition.value,
            "collapse.sequence_frames": int(p.frames),
            "physics.E_Pa": ({"rule": "collapse"} if p.linkE and p.driveBy != "pressure" else float(p.eValue)),
            "physics.material.nu": float(p.nu),
            "fem_setup.wall.tol_mm": float(p.wallMargin),
            "solver.device": p.device.value,
            "rebuild_geometry": bool(p.rebuildGeometry),
            "physics.emphysema": p.emphysema == "yes",
            "physics.barrier": bool(p.barrier),
            "output_dir": p.outputDir or None,
        }
        roles = {"inflated": "lung" + p.side.capitalize(), "airways": "airways", "arteries": "arteries",
                 "veins": "veins"}
        if p.compareCollapsed:
            roles["collapsed"] = "collapsed"
            other = "lungRight" if p.side == "left" else "lungLeft"
            if self.logic.inputAvailable(p, other):
                roles["contralateral"] = other                # emphysema measured there (pneumothorax CT)
        params["inputs"] = {}
        for role, n in roles.items():
            resolved = self.logic.resolveInput(p, n, load=True)
            if resolved is None:
                raise ValueError(_("Missing input: {what}").format(what=INPUT_LABELS[n]))
            params["inputs"][role] = resolved
        params["reference_volume"] = self.logic.resolveCT(p)
        params["ct_key"] = params["reference_volume"].GetID() if params["reference_volume"] is not None else None
        params["name"] = lcinputs.safe_name(self.logic.inputName(p, roles["inflated"]))
        params["inputs_key"] = "|".join(
            [p.side] + ["%s:%s:%s" % (r, node.GetID(), ",".join(ids)) for r, (node, ids) in params["inputs"].items()])
        return params

    def onApply(self):
        ui = self.ui
        if self.logic.use_package():
            with slicer.util.tryWithErrorDisplay(_("Failed to install the dependencies.")):
                if not lcdeps.ensure(ask=True):
                    return
                reload_rules()
        with slicer.util.tryWithErrorDisplay(_("Failed to start the prediction."), waitCursor=True):
            params = self.collectParams()
            ui.logTextEdit.clear()
            ui.warningsLabel.text = ""
            ui.stageLabel.text = _("Converting the segments to surfaces...")
            slicer.app.processEvents()
            params["surfaces"] = self.logic.exportSurfaces(params)
            self._warnings = []
            ui.progressBar.setVisible(True)
            ui.progressBar.value = 0
            self.logic.run(params, self.onProgress, self.onDone)
        self.updateGui()

    def onCancel(self):
        self.logic.cancel()
        self.ui.stageLabel.text = _("Cancelling...")

    def onProgress(self, info):
        ui = self.ui
        line = info.get("line")
        if line is not None:
            ui.logTextEdit.appendPlainText(line)
        if info.get("warning"):
            self._warnings.append(info["warning"])
            ui.warningsLabel.text = "\n".join("⚠ " + w for w in self._warnings[-6:])
            ui.warningsLabel.styleSheet = "color: #d07000"
        if info.get("percent") is not None:
            ui.progressBar.value = int(info["percent"])
        if info.get("stage"):
            ui.stageLabel.text = info["stage"]

    def onDone(self, result):
        ui = self.ui
        ui.progressBar.setVisible(False)
        self._lastConfig = result.get("config")
        if not result.get("ok"):
            ui.stageLabel.text = result.get("message", _("Failed."))
            ui.stageLabel.styleSheet = "color: #b03030"
            if not result.get("cancelled"):
                ui.logGroupBox.collapsed = False
                slicer.util.errorDisplay(result.get("message", _("The prediction failed.")),
                                         detailedText="\n".join(result.get("tail", [])))
            self.updateGui()
            return
        ui.stageLabel.styleSheet = ""
        with slicer.util.tryWithErrorDisplay(_("Failed to load the results."), waitCursor=True):
            self._results = self.logic.loadResults(result, colorBy=self.colorBy())
            self._results["is_temp"] = result.get("is_temp", False)
            self._results["p0_Pa"] = result.get("summary", {}).get("physics", {}).get("p0_Pa")
            self._results["collapse_percent"] = result.get("summary", {}).get("collapse_percent")
            self._results["compression"] = result.get("summary", {}).get("compression")
            ui.stageLabel.text = _("Done in {t:.0f} s: {where}").format(
                t=result.get("seconds", 0), where=(_("temporary folder") if result.get("is_temp")
                                                   else str(result.get("work"))))
            self.showResults()
            p = self._parameterNode
            for n in SEG_INPUTS:                          # the 3D view shows the results; 2D keeps the inputs
                resolved = self.logic.resolveInput(p, n, load=False)
                if resolved is not None and resolved[0].GetDisplayNode() is not None:
                    resolved[0].GetDisplayNode().SetVisibility3D(False)
            if self.ui.removeLoadedCheckBox.checked:
                self.logic.removeLoadedInputs(keep=("ct",))
        self.updateGui()

    def colorBy(self):
        return "error" if self.ui.colorByComboBox.currentIndex == 1 else "displacement"

    def showResults(self):
        ui, r = self.ui, self._results
        if r is None:
            return
        table = r.get("table")
        ui.resultsTableView.setMRMLTableNode(table)
        ui.resultsTableView.setVisible(table is not None)
        if table is not None:
            ui.resultsTableView.resizeColumnsToContents()
            rows = table.GetNumberOfRows()
            ui.resultsTableView.setMinimumHeight(min(28 + 24 * rows, 420))
        browser = r.get("browser")
        self.playWidget.setMRMLSequenceBrowserNode(browser)
        self.seekWidget.setMRMLSequenceBrowserNode(browser)
        ui.sequenceWidget.setVisible(browser is not None)
        has_error = r.get("_maxima", {}).get("error", 0) > 0
        model = ui.colorByComboBox.model()
        if hasattr(model, "item") and model.item(1) is not None:
            model.item(1).setEnabled(has_error)
        if browser is not None:
            slicer.modules.sequences.setToolBarActiveBrowserNode(browser)
        self.updateVisibility()
        lm = slicer.app.layoutManager()
        if lm is not None and lm.threeDWidget(0) is not None:
            lm.threeDWidget(0).threeDView().resetFocalPoint()

    def onColorBy(self, index):
        r = self._results
        if r is not None:
            lcoutputs.set_color_by(r.get("proxy"), self.colorBy(), r.get("_maxima", {}))

    def updateVisibility(self, *args):
        ui, p = self.ui, self._parameterNode
        if p is not None and p.ctVolume is not None and slicer.app.layoutManager() is not None:
            slicer.util.setSliceViewerLayers(background=p.ctVolume if ui.showCtCheckBox.checked else None)
        r = self._results or {}

        def show(node, on):
            if node is not None and slicer.mrmlScene.IsNodePresent(node) and node.GetDisplayNode() is not None:
                node.GetDisplayNode().SetVisibility(on)

        show(r.get("inflated"), ui.showInflatedCheckBox.checked)
        show(r.get("proxy"), ui.showPredictionCheckBox.checked)
        show(r.get("target"), ui.showTargetCheckBox.checked)
        show(r.get("rings"), ui.showHilumCheckBox.checked)
        for mk in r.get("hilum", []):
            show(mk, ui.showHilumCheckBox.checked)

    def onSaveOutputs(self):
        dest = qt.QFileDialog.getExistingDirectory(slicer.util.mainWindow(), _("Save the outputs in"))
        if not dest:
            return
        with slicer.util.tryWithErrorDisplay(_("Failed to save the outputs."), waitCursor=True):
            name = (self._results or {}).get("name", "LungCollapse")
            copy = self.logic.workFolders.save(dest, name)
            slicer.util.infoDisplay(_("Outputs saved in {p}").format(p=copy))

    def onShowConfig(self):
        if not self._lastConfig or not os.path.isfile(self._lastConfig):
            return
        used = Path(self._lastConfig).parent / "out" / "logs" / "config_used_predict.json"
        path = used if used.is_file() else Path(self._lastConfig)
        d = qt.QDialog(slicer.util.mainWindow())
        d.setWindowTitle(str(path))
        d.resize(700, 700)
        lay = qt.QVBoxLayout(d)
        t = qt.QPlainTextEdit()
        t.setReadOnly(True)
        t.setPlainText(path.read_text())
        lay.addWidget(t)
        d.exec_() if hasattr(d, "exec_") else d.exec()


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


#
# Logic
#


class LungCollapseLogic(ScriptedLoadableModuleLogic):
    """Can be used without the GUI:

        logic = LungCollapseLogic()
        params = {...}                     # see LungCollapseWidget.collectParams
        params["surfaces"] = logic.exportSurfaces(params)
        logic.run(params, progress_callback, done_callback)   # asynchronous (a separate process)
        nodes = logic.loadResults(result)                     # in done_callback
    """

    def __init__(self) -> None:
        ScriptedLoadableModuleLogic.__init__(self)
        self.workFolders = lcwork.WorkFolders(Path(slicer.app.temporaryPath) / "LungCollapse")
        self.workFolders.remove_stale()
        self._inputs = {}         # input name -> (paths, node) of the inputs loaded from files
        self.runner = None
        self.backend = os.environ.get("LUNGCOLLAPSE_BACKEND", "auto")   # auto | fake | package
        self._closeObserver = slicer.mrmlScene.AddObserver(slicer.mrmlScene.EndCloseEvent, self._onSceneClosed)
        slicer.app.connect("aboutToQuit()", self._onQuit)

    def getParameterNode(self):
        return LungCollapseParameterNode(super().getParameterNode())

    def _onSceneClosed(self, caller=None, event=None):
        self.cancel()
        self.workFolders.discard()
        self._inputs = {}

    def _onQuit(self):
        self.cancel()
        self.workFolders.discard()

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def gravity_collapse(position, side):
        """UI position -> physics.gravity.collapse: lateral decubitus on the healthy side."""
        if position == "lateral":
            return "right_lateral" if side == "left" else "left_lateral"
        return position

    def running(self):
        return self.runner is not None and self.runner.running()

    def cancel(self):
        if self.runner is not None:
            self.runner.cancel()

    def use_package(self):
        """The real prediction, unless LUNGCOLLAPSE_BACKEND=fake (the Part 1 stand-in, for tests)."""
        return self.backend != "fake"

    # ------------------------------------------------------------------ inputs
    def loadFileInput(self, name, paths):
        """Node of an input given as file(s): loaded once, replaced when the paths change. Several files of a
        segmentation input are merged into one segmentation (all their segments are used)."""
        paths = tuple(str(x) for x in paths)
        old = self._inputs.get(name)
        if old is not None and slicer.mrmlScene.IsNodePresent(old[1]):
            if old[0] == paths:
                return old[1]
            slicer.mrmlScene.RemoveNode(old[1])
        if name == "ct":
            node = slicer.util.loadVolume(paths[0])
        else:
            node = self._loadSegmentations(name, paths)
            color = INPUT_COLORS.get(name)
            seg = node.GetSegmentation()
            for i in range(seg.GetNumberOfSegments()):
                if color:
                    seg.GetNthSegment(i).SetColor(*color)
            node.CreateClosedSurfaceRepresentation()
        node.SetAttribute("LungCollapse.Input", name)
        self._inputs[name] = (paths, node)
        return node

    @staticmethod
    def _loadSegmentations(name, paths):
        nodeName = "%s (%s)" % (INPUT_LABELS[name], ", ".join(os.path.basename(x).split(".")[0] for x in paths))
        loaded = []
        try:
            for path in paths:
                seg, added = lcinputs.load_segmentation_file(path)
                for n in added:                          # labelmaps / models used for the import
                    if n is not seg:
                        slicer.mrmlScene.RemoveNode(n)
                loaded.append((path, seg))
            if len(loaded) == 1:
                loaded[0][1].SetName(nodeName)
                return loaded[0][1]
            merged = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", nodeName)
            merged.CreateDefaultDisplayNodes()
            for path, seg in loaded:
                segs = seg.GetSegmentation()
                if segs.GetNumberOfSegments() == 1:
                    segs.GetNthSegment(0).SetName(os.path.basename(path).split(".")[0])
                for i in range(segs.GetNumberOfSegments()):
                    merged.GetSegmentation().CopySegmentFromSegmentation(segs, segs.GetNthSegmentID(i))
            return merged
        finally:
            if len(loaded) > 1:
                for _path, seg in loaded:
                    slicer.mrmlScene.RemoveNode(seg)

    def _fileNode(self, name, paths):
        old = self._inputs.get(name)
        if old is not None and old[0] == tuple(paths) and slicer.mrmlScene.IsNodePresent(old[1]):
            return old[1]
        return None

    @staticmethod
    def _paths(p, name):
        return [x.strip() for x in getattr(p, name + "Path").split(";") if x.strip()]

    def inputAvailable(self, p, name):
        """True when the input is set: scene node (+ segments), or existing file(s)."""
        if getattr(p, name + "Source") == "file":
            paths = self._paths(p, name)
            return bool(paths) and all(os.path.isfile(x) for x in paths)
        if name == "ct":
            return p.ctVolume is not None
        node = getattr(p, name + "Segmentation")
        ids = [i for i in getattr(p, name + "SegmentIds").split(",") if i]
        return node is not None and bool(ids) and all(node.GetSegmentation().GetSegment(i) for i in ids)

    def resolveInput(self, p, name, load=True):
        """(segmentation node, [segment IDs]) of a segmentation input, or None when not set. From files: all
        the segments of the loaded file(s); load=False does not load files that are not loaded yet."""
        if not self.inputAvailable(p, name):
            return None
        if getattr(p, name + "Source") == "file":
            paths = self._paths(p, name)
            node = self._fileNode(name, paths) or (self.loadFileInput(name, paths) if load else None)
            if node is None:
                return None
            seg = node.GetSegmentation()
            return node, [seg.GetNthSegmentID(i) for i in range(seg.GetNumberOfSegments())]
        return getattr(p, name + "Segmentation"), [i for i in getattr(p, name + "SegmentIds").split(",") if i]

    def resolveCT(self, p):
        if not self.inputAvailable(p, "ct"):
            return None
        if p.ctSource == "file":
            paths = self._paths(p, "ct")
            return self._fileNode("ct", paths) or self.loadFileInput("ct", paths)
        return p.ctVolume

    def inputName(self, p, name):
        """A name for the work folder: the segmentation's, or the first file's."""
        if getattr(p, name + "Source") == "file":
            paths = self._paths(p, name)
            return os.path.basename(paths[0]).split(".")[0] if paths else "lung"
        node = getattr(p, name + "Segmentation")
        return node.GetName() if node is not None else "lung"

    def removeLoadedInputs(self, keep=()):
        """Remove from the scene the inputs loaded from files (except the names in keep)."""
        for name, (paths, node) in list(self._inputs.items()):
            if name in keep:
                continue
            if slicer.mrmlScene.IsNodePresent(node):
                slicer.mrmlScene.RemoveNode(node)
            del self._inputs[name]

    # ------------------------------------------------------------------ surfaces
    def exportSurfaces(self, params):
        """{role: vtkPolyData in RAS (world)} from params["inputs"] = {role: (segmentation node, [segment IDs])}
        (section 3): several segments are merged (a lung with a closing, to fill the fissure gaps)."""
        out = {"collapsed": None}
        ref = params.get("reference_volume")
        for role, (node, ids) in params["inputs"].items():
            closing = LUNG_CLOSING_MM if role in ("inflated", "collapsed", "contralateral") else 0.0
            out[role] = lcinputs.merged_surface(node, ids, closing, ref)
        return out

    # ------------------------------------------------------------------ config
    def baseConfig(self):
        """The package defaults (predict.default_config()); without the package a minimal stand-in."""
        if self.use_package():
            from eurelaps_lung import predict
            return predict.default_config()
        return {"_note": "eurelaps_lung not installed: minimal config for the fake run (Part 1)",
                "patient": "patient", "data_dir": ".", "output_dir": "out", "run_name": "predict",
                "inputs": {}, "geometry": {}, "hilum": {"lung": "inflated"},
                "fem_setup": {"wall": {"tol_mm": 5.0}},
                "physics": {"material": {"name": "neo_hookean", "nu": 0.3}, "E_Pa": {"rule": "collapse"},
                            "gravity": {"reference": "supine", "collapse": "supine"},
                            "collapse": {"volume_ratio": 0.5}},
                "solver": {"device": "cpu", "linear_solver": "scipy"}, "collapse": {"sequence_frames": 30}}

    def buildConfig(self, params, names):
        cfg = self.baseConfig()
        cfg["patient"] = "%s_%s" % (params["name"], params["side"])
        cfg["data_dir"] = "inputs"
        cfg["output_dir"] = "out"
        cfg["inputs"] = {"inflated": names["inflated"], "collapsed": names.get("collapsed"),
                         "structures": {s: names[s] for s in STRUCTURES}}
        if names.get("ct"):
            cfg["inputs"]["ct"] = names["ct"]
        if names.get("contralateral"):
            cfg["inputs"]["contralateral"] = [names["contralateral"]]
        ph = cfg.setdefault("physics", {})
        pct = params["collapse_percent"]
        if params.get("physics.collapse.p0_Pa") is not None:          # p0 given: the collapse % is the result
            ph["collapse"] = {"p0_Pa": float(params["physics.collapse.p0_Pa"])}
        elif pct != "target":
            ph["collapse"] = {"volume_ratio": round(1.0 - float(pct) / 100.0, 6)}
        ph["gravity"] = {"reference": params["physics.gravity.reference"],
                         "collapse": params["physics.gravity.collapse"]}
        ph["E_Pa"] = params["physics.E_Pa"]
        ph.setdefault("material", {})["nu"] = params["physics.material.nu"]
        ph["emphysema"] = bool(params.get("physics.emphysema", False))
        if params.get("physics.barrier"):
            ph["barrier"] = {"phi": "ct"}
        cfg.setdefault("fem_setup", {}).setdefault("wall", {})["tol_mm"] = params["fem_setup.wall.tol_mm"]
        solver = cfg.setdefault("solver", {})
        solver["device"] = params["solver.device"]
        if params["solver.device"] == "cuda":
            solver["linear_solver"] = None
        cfg.setdefault("collapse", {})["sequence_frames"] = params["collapse.sequence_frames"]
        return cfg

    @staticmethod
    def geometryKeys(cfg):
        inputs = {k: v for k, v in (cfg.get("inputs") or {}).items() if k not in ("ct", "contralateral")}
        return json.dumps(dict({k: cfg.get(k) for k in ("geometry", "hilum", "fem_setup")}, inputs=inputs),
                          sort_keys=True)

    @staticmethod
    def slicerRecord(params):
        rec = {"side": params.get("side"), "lung_closing_mm": LUNG_CLOSING_MM,
               "conversion_parameters": lcinputs.CONVERSION_PARAMETERS,
               "slicer_version": slicer.app.applicationVersion, "date": time.strftime("%Y-%m-%d %H:%M:%S"),
               "inputs": {}}
        for role, (node, ids) in params.get("inputs", {}).items():
            seg = node.GetSegmentation()
            rec["inputs"][role] = {"segmentation": node.GetName(),
                                   "segments": [seg.GetSegment(i).GetName() for i in ids if seg.GetSegment(i)]}
            try:
                rec["conversion_parameters"] = {k: seg.GetConversionParameter(k) for k in lcinputs.CONVERSION_PARAMETERS}
            except Exception:
                pass
        return rec

    # ------------------------------------------------------------------ run
    def command(self, work, params, run_name, steps):
        """(program, args, env) of the prediction process."""
        py = lcprocess.python_slicer()
        pct = params["collapse_percent"]
        args = ["--config", str(Path(work) / "config.json")]
        if pct is not None:
            args += ["--collapse", "target" if pct == "target" else "%.4f" % float(pct)]
        elif params.get("physics.collapse.p0_Pa") is not None:      # driven by the pressure: p0 and E given
            args += ["--p0", "%.6g" % params["physics.collapse.p0_Pa"], "--E", "%.6g" % float(params["physics.E_Pa"])]
        args += ["--frames", str(params["collapse.sequence_frames"]), "--run", run_name]
        if steps:
            args += ["--steps"] + list(steps)
        env = {"EURELAPS_PYTHON": py}
        if self.use_package():
            launcher = Path(__file__).parent / "LungCollapseLib" / "run_predict.py"
            return py, [str(launcher)] + args, env
        script = Path(__file__).parent / "LungCollapseLib" / "fake_predict.py"
        if params.get("fake_fail"):
            args.append("--fail")
        return py, [str(script)] + args, env

    def run(self, params, progress_callback=None, done_callback=None):
        """Start the prediction (asynchronous). params: the UI values named as the config keys, plus
        `surfaces` ({inflated, airways, arteries, veins, collapsed}: vtkPolyData in RAS or a file path) and
        `output_dir` (None = temporary). progress_callback(dict(line, percent, stage, warning));
        done_callback(dict(ok, work, out_dir, run_name, summary, ...))."""
        if self.running():
            raise RuntimeError(_("A prediction is already running."))
        progress_callback = progress_callback or (lambda info: None)
        done_callback = done_callback or (lambda result: None)
        t0 = time.time()
        surfaces = dict(params["surfaces"])
        for k, v in list(surfaces.items()):
            if isinstance(v, (str, Path)) and v:
                surfaces[k], _found = lcinputs.read_surface_file(str(v), "auto")
        if params["collapse_percent"] == "target" and surfaces.get("collapsed") is None:
            raise ValueError(_("'Use the collapse % of the collapsed lung' needs the collapsed lung."))
        surfaces.setdefault("contralateral", None)
        for k in ("inflated",) + STRUCTURES:
            if surfaces.get(k) is None:
                raise ValueError(_("Missing surface: {k}").format(k=k))

        # work folder and inputs/*.vtk
        work, is_temp = self.workFolders.get(params.get("output_dir"), params["name"], params["side"],
                                             params.get("inputs_key"))
        names = {"inflated": "lung_inflated.vtk", "airways": "airways.vtk", "arteries": "arteries.vtk",
                 "veins": "veins.vtk"}
        if surfaces.get("collapsed") is not None:
            names["collapsed"] = "lung_collapsed.vtk"
            if surfaces.get("contralateral") is not None:
                names["contralateral"] = "lung_contralateral.vtk"
        for k, n in names.items():
            lcinputs.write_vtk_ras(surfaces[k], str(Path(work) / "inputs" / n))
        warnings = lcinputs.check_surfaces({k: surfaces.get(k) for k in names if k != "contralateral"})
        if params.get("physics.barrier"):
            names["ct"] = self.exportCT(work, params, progress_callback)
        for w in warnings:
            progress_callback({"line": "WARNING input check: " + w, "warning": w})

        # config
        cfg = self.buildConfig(params, names)
        cfg["_slicer"] = self.slicerRecord(params)
        config_path = Path(work) / "config.json"
        config_path.write_text(json.dumps(cfg, indent=2))

        # geometry: rebuilt when the inputs or the geometry keys changed, or when asked
        ghash = lcinputs.surfaces_hash({k: surfaces[k] for k in names if k not in ("ct", "contralateral")},
                                       self.geometryKeys(cfg))
        steps = []
        if params.get("rebuild_geometry") or lcwork.stored_hash(work) != ghash or not lcwork.has_geometry(work):
            steps = list(GEOMETRY_STEPS)
        pct = params["collapse_percent"]
        if pct is None:
            run_name = "pred_p0_%d" % round(params["physics.collapse.p0_Pa"])
        else:
            run_name = "pred_target" if pct == "target" else ("pred_%g" % round(float(pct), 1)).replace(".", "p")
        program, args, env = self.command(work, params, run_name, steps)
        progress = lcprocess.Progress()
        inflated_ml = lcinputs.volume_ml(surfaces["inflated"])
        logging.info("Lung Collapse: %s %s", program, " ".join(args))
        progress_callback({"line": "$ %s %s" % (os.path.basename(program), " ".join(args)),
                           "stage": _("Starting") + (" (" + _("geometry is rebuilt") + ")" if steps else
                                                     " (" + _("geometry reused") + ")"), "percent": 0})
        result = {"work": str(work), "out_dir": str(Path(work) / "out"), "run_name": run_name, "is_temp": is_temp,
                  "config": str(config_path), "inflated_ml": inflated_ml, "asked": pct,
                  "p0_given": params.get("physics.collapse.p0_Pa"), "name": params["name"],
                  "side": params["side"], "warnings": list(warnings), "inflated_surface": surfaces["inflated"],
                  "steps": steps, "fake": not self.use_package()}

        def on_line(line):
            info = {"line": line}
            if progress.feed(line):
                info.update(percent=progress.percent, stage=progress.text())
            if "WARNING" in line:
                msg = line.split("WARNING", 1)[1].strip(" :")
                msg = msg.split(": ", 1)[1] if msg.startswith("eurelaps_lung") and ": " in msg else msg
                info["warning"] = msg
                result["warnings"].append(msg)
            progress_callback(info)

        def on_finished(code, cancelled):
            result["seconds"] = time.time() - t0
            result["tail"] = self.runner.lines[-40:] if self.runner else []
            self.runner = None
            if cancelled:
                result.update(ok=False, cancelled=True, message=_("Cancelled."))
            elif code != 0:
                result.update(ok=False, message=_("The prediction failed (exit code {c}).").format(c=code))
            elif not (Path(result["out_dir"]) / "runs" / run_name / "summary.json").is_file():
                result.update(ok=False, message=_("The prediction ended without a summary.json."))
            else:
                if steps:
                    lcwork.store_hash(work, ghash)
                result.update(ok=True, summary=lcoutputs.read_summary(result["out_dir"], run_name))
            done_callback(result)

        self.runner = lcprocess.ProcessRunner(program, args, cwd=str(work), env=env, on_line=on_line,
                                              on_finished=on_finished)
        self.runner.start()
        return result

    def exportCT(self, work, params, progress_callback=lambda info: None):
        """The CT into work/inputs/ct.nrrd for the `ct` step of the package (density, emphysema volume), once
        per CT: a new CT (or new lung surfaces) drops the measured values out/ct/ct_values.json."""
        vol = params.get("reference_volume")
        if vol is None:
            raise ValueError(_("The limit of compression needs the CT (Inputs)."))
        inputs = Path(work) / "inputs"
        path, keyfile = inputs / "ct.nrrd", inputs / "ct_key.txt"
        key = "%s|%s" % (params.get("ct_key"), params.get("inputs_key"))
        values = Path(work) / "out" / "ct" / "ct_values.json"
        if path.is_file() and keyfile.is_file() and keyfile.read_text() == key:
            return path.name
        progress_callback({"line": "exporting the CT for the density / emphysema measure", "stage": _("Exporting the CT...")})
        inputs.mkdir(parents=True, exist_ok=True)
        export = getattr(slicer.util, "exportNode", None)
        if export is not None:
            export(vol, str(path))
        else:
            slicer.util.saveNode(vol, str(path))
        keyfile.write_text(key)
        if values.is_file():
            values.unlink()
        return path.name

    def runSync(self, params, timeout_s=3600, progress_callback=None):
        """run() and wait (for tests and scripts). Returns the result dict."""
        box = {}
        self.run(params, progress_callback, lambda r: box.setdefault("r", r))
        t0 = time.time()
        while "r" not in box:
            slicer.app.processEvents()
            time.sleep(0.02)
            if time.time() - t0 > timeout_s:
                self.cancel()
                t1 = time.time()
                while "r" not in box and time.time() - t1 < 10:
                    slicer.app.processEvents()
                    time.sleep(0.02)
                break
        return box.get("r", {"ok": False, "message": "timeout"})

    # ------------------------------------------------------------------ outputs
    def loadResults(self, result, colorBy="displacement"):
        """Outputs of a finished run into the scene; returns {role: node}."""
        prefix = "LungCollapse_%s_%s" % (result.get("name", "lung"), result.get("side", ""))
        lcoutputs.remove_outputs()
        nodes = lcoutputs.load_run(result["out_dir"], result["run_name"], prefix=prefix, color_by=colorBy)
        if result.get("inflated_surface") is not None:
            nodes["inflated"] = lcoutputs.add_polydata_model(result["inflated_surface"], prefix + "_inflated",
                                                             "inflated", color=(0.55, 0.75, 0.95), opacity=0.25)
        asked = result.get("asked")
        if asked is None and result.get("p0_given") is not None:
            asked = "p0 given: %.0f Pa" % result["p0_given"]
        rows = lcoutputs.summary_rows(result.get("summary", {}), result.get("inflated_ml"), asked)
        nodes["table"] = lcoutputs.results_table(rows, prefix + "_results")
        nodes["name"] = "%s_%s" % (result.get("name"), result.get("side"))
        # the segmentation stays in 2D, the 3D view shows the results
        return nodes


#
# Test
#


def _sphere_polydata(center, radii, res=48):
    s = vtk.vtkSphereSource()
    s.SetThetaResolution(res)
    s.SetPhiResolution(res)
    s.SetRadius(1.0)
    t = vtk.vtkTransform()
    t.Translate(*center)
    t.Scale(*radii)
    f = vtk.vtkTransformPolyDataFilter()
    f.SetTransform(t)
    f.SetInputConnection(s.GetOutputPort())
    f.Update()
    return f.GetOutput()


def _tube_polydata(p0, p1, radius):
    line = vtk.vtkLineSource()
    line.SetPoint1(*p0)
    line.SetPoint2(*p1)
    tube = vtk.vtkTubeFilter()
    tube.SetInputConnection(line.GetOutputPort())
    tube.SetRadius(radius)
    tube.SetNumberOfSides(24)
    tube.CappingOn()
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(tube.GetOutputPort())
    tri.Update()
    return tri.GetOutput()


def make_synthetic_segmentation(name="LungCollapseTest"):
    """A synthetic chest: left lung (one segment), right lung as two lobes, the three trees crossing the
    left lung surface medially, and a collapsed left lung. RAS: x > 0 is the patient's right."""
    seg = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", name)
    seg.CreateDefaultDisplayNodes()
    pieces = [
        ("lung_left", _sphere_polydata((-80, 0, 0), (45, 60, 100)), (0.8, 0.6, 0.5)),
        ("lung_upper_lobe_right", _sphere_polydata((80, 0, 45), (45, 60, 52)), (0.5, 0.7, 0.5)),
        ("lung_lower_lobe_right", _sphere_polydata((80, 0, -45), (45, 60, 52)), (0.5, 0.5, 0.8)),
        ("airways", _tube_polydata((0, 0, 60), (-60, 0, 0), 9), (0.9, 0.9, 0.9)),
        ("arteries", _tube_polydata((0, 15, 40), (-60, 15, 0), 7), (0.8, 0.2, 0.2)),
        ("veins", _tube_polydata((0, -15, 30), (-60, -15, -5), 7), (0.2, 0.2, 0.8)),
        ("lung_left_collapsed", _sphere_polydata((-65, 5, 5), (30, 42, 70)), (0.6, 0.6, 0.6)),
    ]
    for n, pd, color in pieces:
        seg.AddSegmentFromClosedSurfaceRepresentation(pd, n, color)
    return seg


class LungCollapseTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_guess()
        self.setUp()
        self.test_fake_run()
        self.setUp()
        self.test_widget_inputs()

    def test_guess(self):
        seg = make_synthetic_segmentation()
        g = lcinputs.guess(seg)
        name = lambda sid: seg.GetSegmentation().GetSegment(sid).GetName()
        self.assertEqual(g["side"], "left")
        self.assertEqual([name(s) for s in g["lungs"]["left"]], ["lung_left"])
        self.assertEqual(sorted(name(s) for s in g["lungs"]["right"]),
                         ["lung_lower_lobe_right", "lung_upper_lobe_right"])
        self.assertEqual(name(g["airways"]), "airways")
        self.assertEqual(name(g["collapsed"]), "lung_left_collapsed")
        self.assertEqual(lcinputs.name_side("lung_lower_lobe_left"), "left")
        self.assertEqual(lcinputs.name_side("Polmone dx"), "right")
        self.assertIsNone(lcinputs.name_side("lung"))
        # merged lobes: one closed component, volume ~ the two lobes
        merged = lcinputs.merged_surface(seg, g["lungs"]["right"], 1.5)
        self.assertEqual(lcinputs.n_components(merged), 1)
        v_lobes = sum(lcinputs.volume_ml(lcinputs.segment_surface(seg, s)) for s in g["lungs"]["right"])
        self.assertAlmostEqual(lcinputs.volume_ml(merged) / v_lobes, 1.0, delta=0.15)
        self.delayDisplay("guess and merge: ok")

    def test_fake_run(self):
        import numpy as np
        from vtk.util.numpy_support import vtk_to_numpy
        seg = make_synthetic_segmentation()
        logic = LungCollapseLogic()
        logic.backend = "fake"
        g = lcinputs.guess(seg)
        params = {"side": "left", "collapse_percent": 40.0,
                  "physics.gravity.collapse": "supine", "physics.gravity.reference": "supine",
                  "collapse.sequence_frames": 6, "physics.E_Pa": {"rule": "collapse"}, "physics.material.nu": 0.3,
                  "fem_setup.wall.tol_mm": 5.0, "solver.device": "cpu", "rebuild_geometry": False,
                  "output_dir": None, "reference_volume": None, "name": "test", "inputs_key": "test",
                  "inputs": {"inflated": (seg, g["lungs"]["left"]), "collapsed": (seg, [g["collapsed"]])}}
        params["inputs"].update({s: (seg, [g[s]]) for s in STRUCTURES})
        os.environ["LUNGCOLLAPSE_FAKE_DELAY"] = "0.01"
        params["surfaces"] = logic.exportSurfaces(params)
        result = logic.runSync(params, timeout_s=120)
        self.assertTrue(result.get("ok"), result.get("message", "") + "\n" + "\n".join(result.get("tail", [])))
        self.assertEqual(result["steps"], list(GEOMETRY_STEPS))
        work = Path(result["work"])
        self.assertTrue(work.is_dir() and result["is_temp"])
        nodes = logic.loadResults(result)
        self.assertEqual(nodes["sequence"].GetNumberOfDataNodes(), 7)
        # frame 0 lies on the inflated lung (coordinates LPS -> RAS right)
        nodes["browser"].SetSelectedItemNumber(0)
        X0 = vtk_to_numpy(nodes["proxy"].GetPolyData().GetPoints().GetData())
        Xi = vtk_to_numpy(params["surfaces"]["inflated"].GetPoints().GetData())
        self.assertLess(np.abs(X0.mean(0) - Xi.mean(0)).max(), 1.0)
        self.assertGreater(nodes["table"].GetNumberOfRows(), 5)
        # p0 given (and E): the collapse % is the result
        p0params = dict(params, collapse_percent=None, **{"physics.collapse.p0_Pa": 330.0, "physics.E_Pa": 800.0})
        r0 = logic.runSync(p0params, timeout_s=120)
        self.assertTrue(r0.get("ok"), r0.get("message"))
        self.assertEqual(r0["run_name"], "pred_p0_330")
        cfg = json.loads(Path(r0["config"]).read_text())
        self.assertEqual(cfg["physics"]["collapse"], {"p0_Pa": 330.0})
        self.assertAlmostEqual(r0["summary"]["collapse_percent"], collapse_from_q0(330.0 / 800.0), delta=0.1)
        # a second run with another collapse %: geometry reused, same temporary folder
        params["collapse_percent"] = 60.0
        result2 = logic.runSync(params, timeout_s=120)
        self.assertTrue(result2.get("ok"))
        self.assertEqual(result2["steps"], [])
        self.assertEqual(result2["work"], result["work"])
        # the temporary folder is deleted when the scene is closed
        slicer.mrmlScene.Clear()
        self.assertFalse(work.exists())
        self.delayDisplay("fake run: ok")

    def test_widget_inputs(self):
        """The Inputs rows: scene pre-selection, a lung from several files (merged), the checks before Apply."""
        import tempfile
        seg = make_synthetic_segmentation()
        slicer.util.selectModule("LungCollapse")
        w = slicer.modules.LungCollapseWidget
        logic, ui = w.logic, w.ui
        w.ui.lungLeftSegmentSelector.setCurrentNode(None)
        p = w._parameterNode
        for n in SEG_INPUTS:
            setattr(p, n + "Segmentation", None)
            setattr(p, n + "SegmentIds", "")
        w.restoreFromParameterNode()
        # scene: choosing the segmentation in one row fills the others from the segment names
        ui.airwaysSegmentSelector.setCurrentNode(seg)
        name = lambda n: [seg.GetSegmentation().GetSegment(i).GetName() for i in w._ids(n)]
        self.assertEqual(name("lungLeft"), ["lung_left"])
        self.assertEqual(sorted(name("lungRight")), ["lung_lower_lobe_right", "lung_upper_lobe_right"])
        self.assertEqual(name("veins"), ["veins"])
        self.assertEqual(name("collapsed"), ["lung_left_collapsed"])
        self.assertTrue(p.compareCollapsed)
        self.assertEqual(p.side, "left")
        p.emphysema, p.barrier = "", False
        self.assertIn(_("emphysema yes / no (Collapse section)"), w.missingInputs())
        ui.emphysemaNoRadio.checked = True                   # the user's answer
        self.assertEqual(p.emphysema, "no")
        self.assertEqual(w.missingInputs(), [])
        # right lung from two files (its lobes): merged into one segmentation, all segments used
        tmp = tempfile.mkdtemp(prefix="LungCollapseTest_", dir=slicer.app.temporaryPath)
        paths = []
        for sid in w._ids("lungRight"):
            pd = lcinputs.segment_surface(seg, sid)
            path = os.path.join(tmp, seg.GetSegmentation().GetSegment(sid).GetName() + ".vtk")
            model = slicer.modules.models.logic().AddModel(pd)
            slicer.util.saveNode(model, path)
            slicer.mrmlScene.RemoveNode(model)
            paths.append(path)
        ui.lungRightFileRadio.checked = True
        ui.lungRightPathsEdit.text = "; ".join(paths)
        w.onPathsEdited("lungRight")
        node, ids = logic.resolveInput(p, "lungRight")
        self.assertEqual(len(ids), 2)
        self.assertEqual(node.GetAttribute("LungCollapse.Input"), "lungRight")
        ui.sideRightRadio.checked = True
        self.assertEqual(p.side, "right")
        # collapse % <-> E through the rule; the collapsed lung's % as the default
        w.updateInflatedVolume()
        self.assertIsNotNone(w._targetPercent)
        self.assertTrue(p.useTargetCollapse)
        self.assertAlmostEqual(p.collapsePercent, round(w._targetPercent, 1), delta=0.06)
        p.useTargetCollapse = False
        p.collapsePercent = 40.0
        self.assertAlmostEqual(p.eValue, round(E_from_collapse(40.0)), delta=0.6)
        p.eValue = 1500.0
        self.assertAlmostEqual(p.collapsePercent, round(collapse_from_E(1500.0), 1), delta=0.06)
        p.linkE = False
        p.eValue = 3000.0
        self.assertAlmostEqual(p.collapsePercent, round(collapse_from_E(1500.0), 1), delta=0.06)
        p.linkE = True
        self.assertAlmostEqual(p.eValue, round(E_from_collapse(p.collapsePercent)), delta=0.6)
        self.assertAlmostEqual(p.p0Value, round(q0_from_collapse(p.collapsePercent) * p.eValue), delta=0.6)
        # driven by p0: the collapse % is the estimate from q0 = p0 / E
        ui.drivePressureRadio.checked = True
        p.p0Value = 400.0
        p.eValue = 900.0
        self.assertAlmostEqual(p.collapsePercent, round(collapse_from_q0(400.0 / 900.0), 1), delta=0.06)
        self.assertTrue(ui.percentRowsWidget.isHidden())
        self.assertFalse(ui.p0SpinBox.readOnly)
        self.assertTrue(p.barrier)                           # a pressure switches the limit of compression on
        if p.ctVolume is None:
            self.assertIn(_("the CT (for the limit of compression)"), w.missingInputs())
        ui.drivePercentRadio.checked = True
        p.barrier = False
        self.assertEqual(w.missingInputs(), [])
        merged = lcinputs.merged_surface(node, ids, LUNG_CLOSING_MM)
        self.assertEqual(lcinputs.n_components(merged), 1)
        # no left lung: the right one is enough; no lung at all: missing
        ui.lungLeftSegmentSelector.setCurrentNode(None)
        self.assertEqual(w.missingInputs(), [])
        ui.lungRightPathsEdit.text = ""
        w.onPathsEdited("lungRight")
        self.assertIn(_("a lung (left or right)"), w.missingInputs())
        logic.removeLoadedInputs()
        self.delayDisplay("inputs: ok")
