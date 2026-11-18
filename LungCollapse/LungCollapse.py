"""Lung Collapse: predicts the collapsed lung of a patient loaded in 3D Slicer and shows the collapse as a
sequence (eurelaps_lung.predict, run in a separate process). Design: SLICER_EXTENSION.md."""

import enum
import importlib.util
import json
import logging
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
PACKAGE_VERSION = "0.1.0"
STRUCTURES = lcinputs.STRUCTURES
GEOMETRY_STEPS = ("geometry", "hilum", "fem_setup")
# what the E rule was fitted on (eurelaps_lung.config.E_RULE_INFO); the package's own copy wins when installed
E_RULE_INFO = dict(patients=13, collapse_percent_range=(23.0, 86.0), residual_sd_log10=0.158,
                   positions=("supine", "supine"))


def rule_info():
    try:
        if importlib.util.find_spec(PACKAGE) is not None:
            from eurelaps_lung.config import E_RULE_INFO as info
            return info
    except Exception:
        pass
    return E_RULE_INFO


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


class Stiffness(enum.Enum):
    RULE = "rule"
    FIXED = "fixed"

    def label(self):
        return {"rule": _("From the collapse % (calibrated rule)"), "fixed": _("Fixed E")}[self.value]


class Device(enum.Enum):
    CPU = "cpu"
    CUDA = "cuda"

    def label(self):
        return {"cpu": _("CPU"), "cuda": _("CUDA")}[self.value]


@parameterNodeWrapper
class LungCollapseParameterNode:
    # inputs (the str ones are kept by the widget's own code)
    inputSource: str = "scene"
    ctVolume: vtkMRMLScalarVolumeNode
    segmentation: vtkMRMLSegmentationNode
    side: str = "left"
    lungSegmentIds: str = ""
    airwaysSegmentId: str = ""
    arteriesSegmentId: str = ""
    veinsSegmentId: str = ""
    compareCollapsed: bool = False
    collapsedSegmentation: vtkMRMLSegmentationNode
    collapsedSegmentId: str = ""
    # collapse
    collapsePercent: Annotated[float, WithinRange(0.0, 95.0)] = 50.0
    useTargetCollapse: bool = False
    position: Position = Position.SUPINE
    frames: Annotated[int, WithinRange(0, 60)] = 30
    # advanced
    stiffness: Stiffness = Stiffness.RULE
    fixedE: Annotated[float, WithinRange(10.0, 100000.0)] = 1000.0
    nu: Annotated[float, WithinRange(0.0, 0.49)] = 0.30
    ctPosition: CtPosition = CtPosition.SUPINE
    wallMargin: Annotated[float, WithinRange(0.0, 30.0)] = 5.0
    closingMm: Annotated[float, WithinRange(0.0, 5.0)] = 1.5
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

        # file filters
        ui.segmentationPathLineEdit.nameFilters = [lcinputs.SEGMENTATION_FILE_FILTER, "All files (*)"]
        ui.collapsedPathLineEdit.nameFilters = [lcinputs.SEGMENTATION_FILE_FILTER, "All files (*)"]
        ui.ctPathLineEdit.nameFilters = ["Images (*.nrrd *.nii *.nii.gz *.mha *.mhd *.nhdr)", "All files (*)"]
        for w in (ui.lungSurfacePath, ui.airwaysSurfacePath, ui.arteriesSurfacePath, ui.veinsSurfacePath,
                  ui.collapsedSurfacePath):
            w.nameFilters = ["Surfaces (*.vtk *.vtp *.stl *.obj *.ply)", "All files (*)"]
        for name in ("ctPathLineEdit", "segmentationPathLineEdit", "collapsedPathLineEdit", "outputDirPathLineEdit",
                     "lungSurfacePath", "airwaysSurfacePath", "arteriesSurfacePath", "veinsSurfacePath",
                     "collapsedSurfacePath"):
            getattr(ui, name).settingKey = "LungCollapse/" + name

        # sequence browser: play and seek widgets
        self.playWidget = slicer.qMRMLSequenceBrowserPlayWidget()
        self.seekWidget = slicer.qMRMLSequenceBrowserSeekWidget()
        ui.sequencePlaceholder.layout().addWidget(self.seekWidget)
        ui.sequencePlaceholder.layout().addWidget(self.playWidget)
        ui.sequenceWidget.setVisible(False)
        ui.resultsTableView.setVisible(False)
        ui.progressBar.setVisible(False)
        ui.resultsTableView.setMRMLScene(slicer.mrmlScene)

        # segment selectors: the node comes from the segmentation combo
        for sel in self._segmentSelectors():
            sel.setMRMLScene(slicer.mrmlScene)
            sel.segmentationNodeSelectorVisible = False
        ui.lungSegmentSelector.multiSelection = True

        # connections
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
        for radio in (ui.sourceSceneRadio, ui.sourceFilesRadio, ui.sourceSurfacesRadio):
            radio.toggled.connect(self.onSourceChanged)
        ui.sideLeftRadio.toggled.connect(self.onSideChanged)
        ui.segmentationSelector.currentNodeChanged.connect(self.onSegmentationChanged)
        ui.collapsedSegmentationSelector.currentNodeChanged.connect(self.onCollapsedSegmentationChanged)
        ui.lungSegmentSelector.connect("segmentSelectionChanged(QStringList)", self.onSegmentsChanged)
        for sel in (ui.airwaysSegmentSelector, ui.arteriesSegmentSelector, ui.veinsSegmentSelector,
                    ui.collapsedSegmentSelector):
            sel.connect("currentSegmentChanged(QString)", self.onSegmentsChanged)
        ui.loadFilesButton.clicked.connect(self.onLoadFiles)
        ui.loadCollapsedButton.clicked.connect(self.onLoadCollapsed)
        for w in (ui.lungSurfacePath, ui.airwaysSurfacePath, ui.arteriesSurfacePath, ui.veinsSurfacePath,
                  ui.collapsedSurfacePath):
            w.currentPathChanged.connect(lambda *_: self.updateGui())
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
        self.onSourceChanged()

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

    def _segmentSelectors(self):
        ui = self.ui
        return (ui.lungSegmentSelector, ui.airwaysSegmentSelector, ui.arteriesSegmentSelector,
                ui.veinsSegmentSelector, ui.collapsedSegmentSelector)

    # ------------------------------------------------------------------ parameter node
    def initializeParameterNode(self) -> None:
        self.setParameterNode(self.logic.getParameterNode())
        p = self._parameterNode
        if p.segmentation is None and p.inputSource == "scene":
            first = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSegmentationNode")
            if first is not None:
                p.segmentation = first
        if p.ctVolume is None and p.inputSource == "scene":
            first = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if first is not None and first.GetClassName() == "vtkMRMLScalarVolumeNode":
                p.ctVolume = first
        self.restoreFromParameterNode()

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
        p = self._parameterNode
        if p is None:
            return
        ui = self.ui
        self._updating = True
        try:
            {"scene": ui.sourceSceneRadio, "files": ui.sourceFilesRadio,
             "surfaces": ui.sourceSurfacesRadio}.get(p.inputSource, ui.sourceSceneRadio).checked = True
            (ui.sideLeftRadio if p.side == "left" else ui.sideRightRadio).checked = True
            ui.outputDirPathLineEdit.currentPath = p.outputDir
            self._setSegmentNodes()
            if p.lungSegmentIds:
                ui.lungSegmentSelector.setSelectedSegmentIDs(p.lungSegmentIds.split(","))
            for s in STRUCTURES:
                sid = getattr(p, s + "SegmentId")
                if sid:
                    getattr(ui, s + "SegmentSelector").setCurrentSegmentID(sid)
            if p.collapsedSegmentId:
                ui.collapsedSegmentSelector.setCurrentSegmentID(p.collapsedSegmentId)
        finally:
            self._updating = False
        self.onSourceChanged()
        self._volumeTimer.start()

    def _setSegmentNodes(self):
        p = self._parameterNode
        for sel in self._segmentSelectors()[:4]:
            sel.setCurrentNode(p.segmentation)
        coll = p.collapsedSegmentation if p.collapsedSegmentation is not None else p.segmentation
        self.ui.collapsedSegmentSelector.setCurrentNode(coll)

    # ------------------------------------------------------------------ inputs
    def source(self):
        ui = self.ui
        if ui.sourceFilesRadio.checked:
            return "files"
        if ui.sourceSurfacesRadio.checked:
            return "surfaces"
        return "scene"

    def onSourceChanged(self, *args):
        ui = self.ui
        src = self.source()
        if self._parameterNode and not self._updating:
            self._parameterNode.inputSource = src
        ui.sceneInputsWidget.setVisible(src == "scene")
        ui.filesInputsWidget.setVisible(src == "files")
        ui.surfacesInputsWidget.setVisible(src == "surfaces")
        for w in (ui.lungSegmentSelector, ui.airwaysSegmentSelector, ui.arteriesSegmentSelector,
                  ui.veinsSegmentSelector):
            _set_row_visible(w, src != "surfaces")
        _set_row_visible(ui.inflatedVolumeLabel, True)
        ui.compareCheckBox.setVisible(src != "surfaces")
        ui.removeLoadedCheckBox.setVisible(src == "files")
        self.updateGui()

    def onSegmentationChanged(self, node):
        if self._updating or self._parameterNode is None:
            return
        self._setSegmentNodes()
        self.preselect()

    def onCollapsedSegmentationChanged(self, node):
        if self._updating or self._parameterNode is None:
            return
        self._setSegmentNodes()
        self.preselectCollapsed()

    def preselect(self):
        """Side and segments pre-selected from the segment names (or centroids)."""
        p = self._parameterNode
        seg = p.segmentation
        if seg is None:
            self._guess = None
            return
        coll = p.collapsedSegmentation if p.compareCollapsed else None
        self._guess = g = lcinputs.guess(seg, coll)
        self._updating = True
        try:
            if g["side"]:
                p.side = g["side"]
                (self.ui.sideLeftRadio if g["side"] == "left" else self.ui.sideRightRadio).checked = True
            for s in STRUCTURES:
                getattr(self.ui, s + "SegmentSelector").setCurrentSegmentID(g[s] or "")
            self.ui.lungSegmentSelector.setSelectedSegmentIDs(g["lungs"].get(p.side, []))
            if g["collapsed"] and (p.collapsedSegmentation is None or p.collapsedSegmentation is seg):
                self.ui.collapsedSegmentSelector.setCurrentSegmentID(g["collapsed"])
                if not p.compareCollapsed:
                    p.compareCollapsed = True
        finally:
            self._updating = False
        self.ui.sideNoteLabel.text = _("(from the segment names)") if g["side"] else ""
        self.onSegmentsChanged()

    def preselectCollapsed(self):
        p = self._parameterNode
        node = p.collapsedSegmentation if p.collapsedSegmentation is not None else p.segmentation
        if node is None:
            return
        g = lcinputs.guess(node, node)
        if g["collapsed"]:
            self.ui.collapsedSegmentSelector.setCurrentSegmentID(g["collapsed"])
        elif node is not p.segmentation and node.GetSegmentation().GetNumberOfSegments() == 1:
            self.ui.collapsedSegmentSelector.setCurrentSegmentID(node.GetSegmentation().GetNthSegmentID(0))
        self.onSegmentsChanged()

    def onSideChanged(self, *args):
        if self._updating or self._parameterNode is None:
            return
        side = "left" if self.ui.sideLeftRadio.checked else "right"
        self._parameterNode.side = side
        if self.source() != "surfaces":
            g = self._guess
            if g is None and self._parameterNode.segmentation is not None:
                g = self._guess = lcinputs.guess(self._parameterNode.segmentation)
            if g is not None:
                self.ui.lungSegmentSelector.setSelectedSegmentIDs(g["lungs"].get(side, []))
        self.updateGui()

    def onSegmentsChanged(self, *args):
        if self._updating or self._parameterNode is None:
            return
        p, ui = self._parameterNode, self.ui
        with slicer.util.NodeModify(p.parameterNode) if hasattr(slicer.util, "NodeModify") else _nullcontext():
            p.lungSegmentIds = ",".join(ui.lungSegmentSelector.selectedSegmentIDs())
            for s in STRUCTURES:
                setattr(p, s + "SegmentId", getattr(ui, s + "SegmentSelector").currentSegmentID() or "")
            p.collapsedSegmentId = ui.collapsedSegmentSelector.currentSegmentID() or ""
        self._inflatedMl = None
        self._volumeTimer.start()
        self.updateGui()

    def updateInflatedVolume(self):
        """Volume of the selected lung segments (sum of their closed surfaces), for the mL <-> % link."""
        p = self._parameterNode
        self._inflatedMl = None
        try:
            if self.source() == "surfaces":
                path = self.ui.lungSurfacePath.currentPath
                if path and os.path.isfile(path):
                    pd, _space = lcinputs.read_surface_file(path)
                    self._inflatedMl = lcinputs.volume_ml(pd)
            elif p is not None and p.segmentation is not None and p.lungSegmentIds:
                with slicer.util.WaitCursor():
                    self._inflatedMl = sum(lcinputs.volume_ml(lcinputs.segment_surface(p.segmentation, sid))
                                           for sid in p.lungSegmentIds.split(","))
        except Exception as e:
            logging.warning("Lung Collapse: inflated volume not computed: %s", e)
        self.updateGui()

    def onLoadFiles(self):
        ui, p = self.ui, self._parameterNode
        seg_path = ui.segmentationPathLineEdit.currentPath
        ct_path = ui.ctPathLineEdit.currentPath
        with slicer.util.tryWithErrorDisplay(_("Failed to load the files."), waitCursor=True):
            if not seg_path:
                raise ValueError(_("Choose a segmentation file."))
            ct = None
            if ct_path:
                ct, added = lcinputs.load_volume_file(ct_path)
                self.logic.loadedNodes.extend(added)
            seg, added = lcinputs.load_segmentation_file(seg_path)
            self.logic.loadedNodes.extend(added)
            if ct is not None:
                seg.SetReferenceImageGeometryParameterFromVolumeNode(ct)
                slicer.util.setSliceViewerLayers(background=ct, fit=True)
            ui.segmentationPathLineEdit.addCurrentPathToHistory()
            self._updating = True
            try:
                p.ctVolume = ct
                p.segmentation = seg
            finally:
                self._updating = False
            self._setSegmentNodes()
            self.preselect()

    def onLoadCollapsed(self):
        ui, p = self.ui, self._parameterNode
        path = ui.collapsedPathLineEdit.currentPath
        with slicer.util.tryWithErrorDisplay(_("Failed to load the collapsed lung."), waitCursor=True):
            if not path:
                raise ValueError(_("Choose the file of the collapsed lung."))
            seg, added = lcinputs.load_segmentation_file(path)
            self.logic.loadedNodes.extend(added)
            self._updating = True
            try:
                p.collapsedSegmentation = seg
            finally:
                self._updating = False
            self._setSegmentNodes()
            self.preselectCollapsed()

    # ------------------------------------------------------------------ collapse
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
    def missingInputs(self):
        """What is missing before Apply (empty list = ready)."""
        p, ui = self._parameterNode, self.ui
        if p is None:
            return [_("parameter node")]
        missing = []
        if self.source() == "surfaces":
            for label, w in ((_("lung surface"), ui.lungSurfacePath), (_("airways surface"), ui.airwaysSurfacePath),
                             (_("arteries surface"), ui.arteriesSurfacePath), (_("veins surface"), ui.veinsSurfacePath)):
                if not (w.currentPath and os.path.isfile(w.currentPath)):
                    missing.append(label)
            if p.useTargetCollapse and not (ui.collapsedSurfacePath.currentPath
                                            and os.path.isfile(ui.collapsedSurfacePath.currentPath)):
                missing.append(_("collapsed lung surface"))
            return missing
        if p.segmentation is None:
            return [_("segmentation") if self.source() == "scene" else _("segmentation file (Load)")]
        if not p.lungSegmentIds:
            missing.append(_("lung segment"))
        for s, label in zip(STRUCTURES, (_("airways"), _("arteries"), _("veins"))):
            if not getattr(p, s + "SegmentId"):
                missing.append(label)
        if p.compareCollapsed and not p.collapsedSegmentId:
            missing.append(_("collapsed lung segment"))
        return missing

    def updateGui(self, caller=None, event=None):
        p, ui = self._parameterNode, self.ui
        if p is None:
            return
        running = self.logic.running()
        src = self.source()
        compare = p.compareCollapsed if src != "surfaces" else bool(ui.collapsedSurfacePath.currentPath)

        # compare widgets
        ui.compareWidget.setVisible(src != "surfaces" and p.compareCollapsed)
        _set_row_visible(ui.collapsedSegmentationSelector, src == "scene")
        ui.collapsedPathLineEdit.parentWidget()
        _set_row_visible(ui.collapsedFileLabel, False)
        ui.collapsedFileLabel.setVisible(src == "files")
        ui.collapsedPathLineEdit.setVisible(src == "files")
        ui.loadCollapsedButton.setVisible(src == "files")
        ui.useTargetCollapseCheckBox.enabled = compare
        if not compare and p.useTargetCollapse:
            p.useTargetCollapse = False

        # collapse % <-> volume
        self._updating = True
        try:
            if self._inflatedMl:
                ui.inflatedVolumeLabel.text = "%.0f mL" % self._inflatedMl
                ui.collapseVolumeSpinBox.enabled = not p.useTargetCollapse
                ui.collapseVolumeSpinBox.maximum = self._inflatedMl
                ui.collapseVolumeSpinBox.value = (1.0 - p.collapsePercent / 100.0) * self._inflatedMl
            else:
                ui.inflatedVolumeLabel.text = "-"
                ui.collapseVolumeSpinBox.enabled = False
        finally:
            self._updating = False
        ui.collapsePercentSlider.enabled = not p.useTargetCollapse
        info = rule_info()
        lo, hi = info["collapse_percent_range"]
        if p.useTargetCollapse:
            ui.collapseRangeLabel.text = _("The collapse % is taken from the collapsed lung.")
            ui.collapseRangeLabel.styleSheet = ""
        elif lo <= p.collapsePercent <= hi:
            ui.collapseRangeLabel.text = _("Calibrated on {lo:.0f}-{hi:.0f} % ({n} patients).").format(
                lo=lo, hi=hi, n=info.get("patients", "?"))
            ui.collapseRangeLabel.styleSheet = ""
        else:
            ui.collapseRangeLabel.text = _("Extrapolation: the stiffness rule was calibrated on {lo:.0f}-{hi:.0f} %.").format(
                lo=lo, hi=hi)
            ui.collapseRangeLabel.styleSheet = "color: #d07000; font-weight: bold"
        if p.position == Position.SUPINE:
            ui.positionNoteLabel.text = _("Calibrated on supine CTs.")
            ui.positionNoteLabel.styleSheet = ""
        else:
            healthy = "right" if p.side == "left" else "left"
            extra = (_(" ({healthy} side down)").format(healthy=healthy) if p.position == Position.LATERAL else "")
            ui.positionNoteLabel.text = _("Calibrated on supine CTs: the stiffness rule is not validated in this "
                                          "position{extra}.").format(extra=extra)
            ui.positionNoteLabel.styleSheet = "color: #d07000"
        ui.fixedESpinBox.enabled = p.stiffness == Stiffness.FIXED

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
        p, ui = self._parameterNode, self.ui
        src = self.source()
        params = {
            "source": src,
            "side": p.side,
            "collapse_percent": "target" if p.useTargetCollapse else float(p.collapsePercent),
            "physics.gravity.collapse": self.logic.gravity_collapse(p.position.value, p.side),
            "physics.gravity.reference": p.ctPosition.value,
            "collapse.sequence_frames": int(p.frames),
            "physics.E_Pa": {"rule": "collapse"} if p.stiffness == Stiffness.RULE else float(p.fixedE),
            "physics.material.nu": float(p.nu),
            "fem_setup.wall.tol_mm": float(p.wallMargin),
            "solver.device": p.device.value,
            "rebuild_geometry": bool(p.rebuildGeometry),
            "closing_mm": float(p.closingMm),
            "output_dir": p.outputDir or None,
        }
        if src == "surfaces":
            space = ui.surfaceSpaceComboBox.currentText
            params["surfaces"] = {
                "inflated": ui.lungSurfacePath.currentPath, "airways": ui.airwaysSurfacePath.currentPath,
                "arteries": ui.arteriesSurfacePath.currentPath, "veins": ui.veinsSurfacePath.currentPath,
                "collapsed": ui.collapsedSurfacePath.currentPath or None}
            params["surface_space"] = space
            params["name"] = lcinputs.safe_name(Path(ui.lungSurfacePath.currentPath).name.split(".")[0])
            params["inputs_key"] = "surfaces:" + "|".join(str(v) for v in params["surfaces"].values())
        else:
            params["segmentation"] = p.segmentation
            params["lung_segments"] = p.lungSegmentIds.split(",")
            params["structures"] = {s: getattr(p, s + "SegmentId") for s in STRUCTURES}
            if p.compareCollapsed and p.collapsedSegmentId:
                params["collapsed_segmentation"] = p.collapsedSegmentation or p.segmentation
                params["collapsed_segment"] = p.collapsedSegmentId
            params["reference_volume"] = p.ctVolume
            params["name"] = lcinputs.safe_name(p.segmentation.GetName())
            params["inputs_key"] = "|".join([p.segmentation.GetID(), p.side, p.lungSegmentIds] +
                                            [getattr(p, s + "SegmentId") for s in STRUCTURES] +
                                            [str(params.get("collapsed_segment"))])
        return params

    def onApply(self):
        ui = self.ui
        if self.logic.use_package():
            with slicer.util.tryWithErrorDisplay(_("Failed to install the dependencies.")):
                if not lcdeps.ensure(ask=True):
                    return
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
            ui.stageLabel.text = _("Done in {t:.0f} s: {where}").format(
                t=result.get("seconds", 0), where=(_("temporary folder") if result.get("is_temp")
                                                   else str(result.get("work"))))
            self.showResults()
            seg = self._parameterNode.segmentation
            if seg is not None and seg.GetDisplayNode() is not None:
                seg.GetDisplayNode().SetVisibility3D(False)     # the 3D view shows the results; 2D keeps it
            if self.ui.removeLoadedCheckBox.checked and self.source() == "files":
                self.logic.removeLoadedInputs(keep=[self._parameterNode.ctVolume])
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
        self.loadedNodes = []
        self.runner = None
        self.backend = os.environ.get("LUNGCOLLAPSE_BACKEND", "auto")   # auto | fake | package
        self._closeObserver = slicer.mrmlScene.AddObserver(slicer.mrmlScene.EndCloseEvent, self._onSceneClosed)
        slicer.app.connect("aboutToQuit()", self._onQuit)

    def getParameterNode(self):
        return LungCollapseParameterNode(super().getParameterNode())

    def _onSceneClosed(self, caller=None, event=None):
        self.cancel()
        self.workFolders.discard()
        self.loadedNodes = []

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

    def removeLoadedInputs(self, keep=()):
        keep_ids = {n.GetID() for n in keep if n is not None}
        for n in self.loadedNodes:
            if n is not None and slicer.mrmlScene.IsNodePresent(n) and n.GetID() not in keep_ids:
                slicer.mrmlScene.RemoveNode(n)
        self.loadedNodes = []

    # ------------------------------------------------------------------ surfaces
    def exportSurfaces(self, params):
        """{structure: vtkPolyData in RAS (world)} from the segmentation (section 3), or from surface files."""
        if params.get("source") == "surfaces":
            out = {}
            space = params.get("surface_space", "LPS")
            for k, path in params["surfaces"].items():
                if not path:
                    out[k] = None
                    continue
                pd, found = lcinputs.read_surface_file(path, "auto")
                if found is None and space == "RAS":
                    pd, _f = lcinputs.read_surface_file(path, "RAS")
                out[k] = pd
            return out
        seg = params["segmentation"]
        out = {"inflated": lcinputs.merged_surface(seg, params["lung_segments"], params.get("closing_mm", 1.5),
                                                   params.get("reference_volume"))}
        for s, sid in params["structures"].items():
            out[s] = lcinputs.segment_surface(seg, sid)
        out["collapsed"] = None
        if params.get("collapsed_segment"):
            out["collapsed"] = lcinputs.segment_surface(params["collapsed_segmentation"], params["collapsed_segment"])
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
        ph = cfg.setdefault("physics", {})
        pct = params["collapse_percent"]
        if pct != "target":
            ph["collapse"] = {"volume_ratio": round(1.0 - float(pct) / 100.0, 6)}
        ph["gravity"] = {"reference": params["physics.gravity.reference"],
                         "collapse": params["physics.gravity.collapse"]}
        ph["E_Pa"] = params["physics.E_Pa"]
        ph.setdefault("material", {})["nu"] = params["physics.material.nu"]
        cfg.setdefault("fem_setup", {}).setdefault("wall", {})["tol_mm"] = params["fem_setup.wall.tol_mm"]
        solver = cfg.setdefault("solver", {})
        solver["device"] = params["solver.device"]
        if params["solver.device"] == "cuda":
            solver["linear_solver"] = None
        cfg.setdefault("collapse", {})["sequence_frames"] = params["collapse.sequence_frames"]
        return cfg

    @staticmethod
    def geometryKeys(cfg):
        return json.dumps({k: cfg.get(k) for k in ("geometry", "hilum", "fem_setup", "inputs")}, sort_keys=True)

    @staticmethod
    def slicerRecord(params):
        rec = {"source": params.get("source"), "side": params.get("side"),
               "closing_mm": params.get("closing_mm"), "conversion_parameters": lcinputs.CONVERSION_PARAMETERS,
               "slicer_version": slicer.app.applicationVersion, "date": time.strftime("%Y-%m-%d %H:%M:%S")}
        seg = params.get("segmentation")
        if seg is not None:
            s = seg.GetSegmentation()
            rec["segmentation"] = seg.GetName()
            rec["lung_segments"] = [s.GetSegment(i).GetName() for i in params["lung_segments"] if s.GetSegment(i)]
            rec["structures"] = {k: s.GetSegment(v).GetName() for k, v in params["structures"].items()
                                 if s.GetSegment(v)}
            cs = params.get("collapsed_segmentation")
            if cs is not None and cs.GetSegmentation().GetSegment(params["collapsed_segment"]):
                rec["collapsed_segment"] = cs.GetSegmentation().GetSegment(params["collapsed_segment"]).GetName()
            try:
                rec["conversion_parameters"] = {
                    k: seg.GetSegmentation().GetConversionParameter(k) for k in lcinputs.CONVERSION_PARAMETERS}
            except Exception:
                pass
        elif params.get("source") == "surfaces":
            rec["surface_files"] = {k: v for k, v in params.get("surfaces_paths", {}).items()}
        return rec

    # ------------------------------------------------------------------ run
    def command(self, work, params, run_name, steps):
        """(program, args, env) of the prediction process."""
        py = lcprocess.python_slicer()
        pct = params["collapse_percent"]
        args = ["--config", str(Path(work) / "config.json"),
                "--collapse", "target" if pct == "target" else "%.4f" % float(pct),
                "--frames", str(params["collapse.sequence_frames"]), "--run", run_name]
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
        for k, n in names.items():
            lcinputs.write_vtk_ras(surfaces[k], str(Path(work) / "inputs" / n))
        warnings = lcinputs.check_surfaces({k: surfaces.get(k) for k in names})
        for w in warnings:
            progress_callback({"line": "WARNING input check: " + w, "warning": w})

        # config
        cfg = self.buildConfig(params, names)
        cfg["_slicer"] = self.slicerRecord(params)
        config_path = Path(work) / "config.json"
        config_path.write_text(json.dumps(cfg, indent=2))

        # geometry: rebuilt when the inputs or the geometry keys changed, or when asked
        ghash = lcinputs.surfaces_hash({k: surfaces[k] for k in names}, self.geometryKeys(cfg))
        steps = []
        if params.get("rebuild_geometry") or lcwork.stored_hash(work) != ghash or not lcwork.has_geometry(work):
            steps = list(GEOMETRY_STEPS)
        pct = params["collapse_percent"]
        run_name = "pred_target" if pct == "target" else ("pred_%g" % round(float(pct), 1)).replace(".", "p")
        program, args, env = self.command(work, params, run_name, steps)
        progress = lcprocess.Progress()
        inflated_ml = lcinputs.volume_ml(surfaces["inflated"])
        logging.info("Lung Collapse: %s %s", program, " ".join(args))
        progress_callback({"line": "$ %s %s" % (os.path.basename(program), " ".join(args)),
                           "stage": _("Starting") + (" (" + _("geometry is rebuilt") + ")" if steps else
                                                     " (" + _("geometry reused") + ")"), "percent": 0})
        result = {"work": str(work), "out_dir": str(Path(work) / "out"), "run_name": run_name, "is_temp": is_temp,
                  "config": str(config_path), "inflated_ml": inflated_ml, "asked": pct, "name": params["name"],
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
        rows = lcoutputs.summary_rows(result.get("summary", {}), result.get("inflated_ml"), result.get("asked"))
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
        params = {"source": "scene", "side": "left", "collapse_percent": 40.0,
                  "physics.gravity.collapse": "supine", "physics.gravity.reference": "supine",
                  "collapse.sequence_frames": 6, "physics.E_Pa": {"rule": "collapse"}, "physics.material.nu": 0.3,
                  "fem_setup.wall.tol_mm": 5.0, "solver.device": "cpu", "rebuild_geometry": False,
                  "closing_mm": 1.5, "output_dir": None, "segmentation": seg,
                  "lung_segments": g["lungs"]["left"], "structures": {s: g[s] for s in STRUCTURES},
                  "collapsed_segmentation": seg, "collapsed_segment": g["collapsed"], "reference_volume": None,
                  "name": "test", "inputs_key": "test"}
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
