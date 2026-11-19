"""Generates LungCollapse/Resources/UI/LungCollapse.ui:  python3 tools/make_ui.py LungCollapse/Resources/UI/LungCollapse.ui"""
import sys
from xml.sax.saxutils import escape

MRML = []


def prop(name, value, stdset=True):
    s = "" if stdset else ' stdset="0"'
    if isinstance(value, bool):
        v = "<bool>%s</bool>" % ("true" if value else "false")
    elif isinstance(value, int):
        v = "<number>%d</number>" % value
    elif isinstance(value, float):
        v = "<double>%s</double>" % repr(value)
    elif isinstance(value, list):
        v = "<stringlist>%s</stringlist>" % "".join("<string>%s</string>" % escape(x) for x in value)
    elif isinstance(value, tuple) and value[0] == "enum":
        v = "<enum>%s</enum>" % value[1]
    elif isinstance(value, tuple) and value[0] == "set":
        v = "<set>%s</set>" % value[1]
    else:
        v = "<string>%s</string>" % escape(value)
    return '<property name="%s"%s>%s</property>' % (name, s, v)


def W(cls, name, layout=None, param=None, **props):
    if cls.startswith("qMRML"):
        MRML.append(name)
    out = ['<widget class="%s" name="%s">' % (cls, name)]
    for k, v in props.items():
        if k == "toolTip" or k == "text" or k == "title" or k == "placeholderText" or k == "suffix":
            out.append(prop(k, v))
        else:
            out.append(prop(k, v))
    if param:
        out.append(prop("SlicerParameterName", param, stdset=False))
    if layout:
        out.append(layout)
    out.append("</widget>")
    return "\n".join(out)


_n = [0]


def _lname(kind):
    _n[0] += 1
    return "%s_%d" % (kind, _n[0])


def item(x, **attrs):
    a = "".join(' %s="%s"' % kv for kv in attrs.items())
    return "<item%s>%s</item>" % (a, x)


def VBox(*children, margins=None):
    m = ""
    if margins is not None:
        m = "".join(prop(k, margins) for k in ("leftMargin", "topMargin", "rightMargin", "bottomMargin"))
    return '<layout class="QVBoxLayout" name="%s">%s%s</layout>' % (_lname("vbox"), m, "".join(item(c) for c in children))


def HBox(*children, margins=None):
    m = ""
    if margins is not None:
        m = "".join(prop(k, margins) for k in ("leftMargin", "topMargin", "rightMargin", "bottomMargin"))
    return '<layout class="QHBoxLayout" name="%s">%s%s</layout>' % (_lname("hbox"), m, "".join(item(c) for c in children))


def Form(*rows, margins=None):
    """rows: (label text or None, widget/layout xml)"""
    m = ""
    if margins is not None:
        m = "".join(prop(k, margins) for k in ("leftMargin", "topMargin", "rightMargin", "bottomMargin"))
    out = []
    for r, (label, w) in enumerate(rows):
        if label is None:
            out.append(item(w, row=r, column=0, colspan=2))
        else:
            lname = label[1] if isinstance(label, tuple) else _lname("label")
            ltext = label[0] if isinstance(label, tuple) else label
            out.append(item(W("QLabel", lname, text=ltext), row=r, column=0))
            out.append(item(w, row=r, column=1))
    return '<layout class="QFormLayout" name="%s">%s%s</layout>' % (_lname("form"), m, "".join(out))


def Label(name, text, wrap=True, **kw):
    return W("QLabel", name, text=text, wordWrap=wrap, **kw)


def Spacer():
    return ('<spacer name="%s"><property name="orientation"><enum>Qt::Vertical</enum></property>'
            '<property name="sizeHint" stdset="0"><size><width>20</width><height>40</height></size></property></spacer>'
            % _lname("spacer"))


def HSpacer():
    return ('<spacer name="%s"><property name="orientation"><enum>Qt::Horizontal</enum></property>'
            '<property name="sizeHint" stdset="0"><size><width>40</width><height>20</height></size></property></spacer>'
            % _lname("hspacer"))


def NodeCombo(name, types, none=True, param=None, toolTip=""):
    return W("qMRMLNodeComboBox", name, param=param, nodeTypes=types, noneEnabled=none, addEnabled=False,
             removeEnabled=False, renameEnabled=False, showChildNodeTypes=False, toolTip=toolTip)


def SegSel(name, multi=False, toolTip=""):
    return W("qMRMLSegmentSelectorWidget", name, multiSelection=multi, segmentationNodeSelectorVisible=False,
             noneEnabled=not multi, toolTip=toolTip)


def Path(name, filters, toolTip=""):
    # filters: ctkPathLineEdit::Filters set
    return W("ctkPathLineEdit", name, filters=("set", filters), toolTip=toolTip)


def Combo(name, items, param=None, toolTip=""):
    its = "".join(item(prop("text", t)) for t in items)
    return ('<widget class="QComboBox" name="%s">%s%s%s</widget>'
            % (name, prop("toolTip", toolTip), prop("SlicerParameterName", param, stdset=False) if param else "", its))


def Collapsible(name, text, layout, collapsed=False):
    return W("ctkCollapsibleButton", name, layout=layout, text=text, collapsed=collapsed)


def Group(name, title, layout, collapsed=False):
    return W("ctkCollapsibleGroupBox", name, layout=layout, title=title, collapsed=collapsed)


FILE_FILTERS = "ctkPathLineEdit::Files|ctkPathLineEdit::Readable"
DIR_FILTERS = "ctkPathLineEdit::Dirs|ctkPathLineEdit::Drives|ctkPathLineEdit::NoDot|ctkPathLineEdit::NoDotDot"

# ------------------------------------------------------------------ Inputs
def Stack(name, *pages):
    return '<widget class="QStackedWidget" name="%s"><property name="currentIndex"><number>0</number></property>%s</widget>' % (
        name, "".join(pages))


SEG_FILES = "Segmentation, labelmap or surface (.seg.nrrd, .nrrd, .nii, .nii.gz, .vtk, .vtp, .stl)"


def SourceRow(n, scene, file):
    """One input: Scene / File radios (exclusive inside this row's widget) over a stack (scene page, file page)."""
    return W("QWidget", n + "Row", layout=VBox(
        HBox(W("QRadioButton", n + "SceneRadio", text="Scene", checked=True),
             W("QRadioButton", n + "FileRadio", text="File"), HSpacer(), margins=0),
        Stack(n + "Stack", scene, file), margins=0))


def MultiFiles(n, tip):
    return W("QWidget", n + "FilesPage", layout=HBox(
        W("QLineEdit", n + "PathsEdit", placeholderText="one or more files, separated by ;", toolTip=tip),
        W("QToolButton", n + "BrowseButton", text="...", toolTip="Choose one or more files"), margins=0))


def SegScene(n, multi, tip):
    return W("qMRMLSegmentSelectorWidget", n + "SegmentSelector", multiSelection=multi, noneEnabled=True,
             segmentationNodeSelectorVisible=True, toolTip=tip)


LUNG_TIP = ("One segment, or several (e.g. the lobes): they are merged into one lung")
rows = [
    ("CT:", SourceRow("ct", NodeCombo("ctNodeSelector", ["vtkMRMLScalarVolumeNode"],
                                      toolTip="For display, as the reference of the segmentation, and for the limit of compression (lung density and emphysema volume are measured on it)"),
                      Path("ctPathEdit", FILE_FILTERS, "Optional CT image (.nrrd, .nii, .nii.gz, ...)"))),
    ("Left lung:", SourceRow("lungLeft", SegScene("lungLeft", True, LUNG_TIP), MultiFiles("lungLeft", LUNG_TIP))),
    ("Right lung:", SourceRow("lungRight", SegScene("lungRight", True, LUNG_TIP), MultiFiles("lungRight", LUNG_TIP))),
    ("Airways:", SourceRow("airways", SegScene("airways", False, ""), Path("airwaysPathEdit", FILE_FILTERS, SEG_FILES))),
    ("Arteries:", SourceRow("arteries", SegScene("arteries", False, ""),
                            Path("arteriesPathEdit", FILE_FILTERS, SEG_FILES))),
    ("Veins:", SourceRow("veins", SegScene("veins", False, ""), Path("veinsPathEdit", FILE_FILTERS, SEG_FILES))),
    ("Lung to collapse:", W("QWidget", "sideWidget", layout=HBox(
        W("QRadioButton", "sideLeftRadio", text="Left", checked=True),
        W("QRadioButton", "sideRightRadio", text="Right"), HSpacer(), Label("sideNoteLabel", "", wrap=False),
        margins=0))),
    ("Inflated volume:", Label("inflatedVolumeLabel", "-")),
    (None, W("QCheckBox", "compareCheckBox", text="Compare with a collapsed lung (validation)",
             param="compareCollapsed")),
    (("Collapsed lung:", "collapsedLabel"),
     SourceRow("collapsed", SegScene("collapsed", True, "The collapsed lung (several segments are merged)"),
               MultiFiles("collapsed", "The collapsed lung (several files are merged)"))),
    (None, Label("compareNoteLabel", "Same coordinates as the inflated lung (same scan, or registered "
                                     "beforehand in Slicer): the extension does not register CTs.")),
]
inputs = Collapsible("inputsCollapsibleButton", "Inputs", VBox(
    Form(*rows, margins=0),
    W("QCheckBox", "removeLoadedCheckBox", text="Remove the inputs loaded from files after the run",
      toolTip="Remove from the scene the files loaded by this module once the run is done"),
    Label("inputStatusLabel", "")))

# ------------------------------------------------------------------ Collapse
def GroupBox(name, title, layout):
    return W("QGroupBox", name, layout=layout, title=title)


collapse = Collapsible("collapseCollapsibleButton", "Collapse", VBox(
    Form(("Drive the collapse by:", W("QWidget", "driveWidget", layout=HBox(
        W("QRadioButton", "drivePercentRadio", text="Collapse %", checked=True,
          toolTip="Give the collapse %: the run searches the pressure p0 that gives that volume"),
        W("QRadioButton", "drivePressureRadio", text="Pressure p0",
          toolTip="Give the pressure p0 and E: the collapse % is the result of the run"),
        HSpacer(), margins=0))), margins=0),
    GroupBox("collapseGroupBox", "Collapse", VBox(
        W("QWidget", "percentRowsWidget", layout=Form(
            (("Collapse:", "collapsePercentLabel"),
             W("ctkSliderWidget", "collapsePercentSlider", param="collapsePercent", minimum=0.0, maximum=95.0,
               singleStep=1.0, pageStep=5.0, decimals=1, value=50.0, suffix=" %",
               toolTip="100 x (1 - V collapsed / V inflated)")),
            (("Collapsed volume:", "collapseVolumeLabel"),
             HBox(W("QDoubleSpinBox", "collapseVolumeSpinBox", decimals=0, maximum=10000.0, suffix=" mL", enabled=False,
                    toolTip="The same as a volume, once the inflated volume is known"),
                  Label("inflatedOfLabel", "", wrap=False), HSpacer())),
            (None, W("QCheckBox", "useTargetCollapseCheckBox", param="useTargetCollapse",
                     text="Use the collapse % of the collapsed lung", enabled=False)),
            margins=0)),
        W("QWidget", "estimateRowsWidget", layout=Form(
            (("Expected collapse:", "collapseEstimateTitle"), Label("collapseEstimateLabel", "")),
            margins=0)),
        Label("collapseRangeLabel", ""),
    )),
    GroupBox("forcesGroupBox", "Stiffness and pressure", Form(
        ("Stiffness E:", HBox(W("QDoubleSpinBox", "eSpinBox", param="eValue", decimals=0, minimum=10.0,
                                maximum=100000.0, singleStep=50.0, value=1000.0, suffix=" Pa",
                                toolTip="Young's modulus of the lung"),
                              W("QCheckBox", "linkECheckBox", param="linkE", checked=True,
                                text="from the collapse % (calibrated rule)",
                                toolTip="E follows the collapse % (and back): log10 E = a + b x collapse %"),
                              HSpacer())),
        ("Pressure p0:", HBox(W("QDoubleSpinBox", "p0SpinBox", param="p0Value", decimals=0, minimum=1.0,
                                maximum=50000.0, singleStep=10.0, value=330.0, suffix=" Pa",
                                toolTip="Pressure released at the collapse (p0 = q0 E)"),
                              Label("p0CmLabel", "", wrap=False), HSpacer())),
        (None, Label("estimateLabel", "")),
    )),
    GroupBox("tissueGroupBox", "Lung tissue", Form(
        ("Emphysema:", W("QWidget", "emphysemaWidget", layout=HBox(
            W("QRadioButton", "emphysemaYesRadio", text="Yes",
              toolTip="The CT shows emphysema: its trapped air limits the collapse (volume measured on the CT)"),
            W("QRadioButton", "emphysemaNoRadio", text="No",
              toolTip="No emphysema to consider: only the tissue limits the collapse"),
            HSpacer(), margins=0))),
        (None, W("QCheckBox", "barrierCheckBox", param="barrier",
                 text="Limit of compression (tissue + air trapped by emphysema, from the CT)",
                 toolTip="The lung cannot lose more volume than the air that can leave. Needed for a surgical "
                         "pressure (12 mmHg = 1600 Pa); without it the cohort-like collapses are unchanged")),
        (None, Label("tissueNoteLabel", "")),
    )),
    GroupBox("sequenceGroupBox", "Position and sequence", Form(
        ("Position during the collapse:", Combo("positionComboBox", [], param="position")),
        (None, Label("positionNoteLabel", "")),
        ("Sequence:", HBox(W("QSpinBox", "framesSpinBox", param="frames", minimum=0, maximum=60, value=30,
                             suffix=" frames", toolTip="Frames of the collapse sequence (0 = only the final shape)"),
                           HSpacer())),
        (None, Label("framesNoteLabel", "From the inflated lung (load 0 %) to the collapsed one (load 100 %); "
                                        "0 = final shape only.")),
    )),
))

# ------------------------------------------------------------------ Advanced
advanced = Collapsible("advancedCollapsibleButton", "Advanced", Form(
    ("Poisson's ratio:", HBox(W("QDoubleSpinBox", "nuSpinBox", param="nu", decimals=2, minimum=0.0, maximum=0.49,
                                singleStep=0.01, value=0.3), HSpacer())),
    ("CT position:", Combo("ctPositionComboBox", [], param="ctPosition")),
    ("Wall margin:", HBox(W("QDoubleSpinBox", "wallMarginSpinBox", param="wallMargin", decimals=1, minimum=0.0,
                            maximum=30.0, singleStep=0.5, value=5.0, suffix=" mm"), HSpacer())),
    ("Device:", Combo("deviceComboBox", [], param="device")),
    (None, W("QCheckBox", "rebuildGeometryCheckBox", param="rebuildGeometry",
             text="Rebuild the geometry (surface, hilum, mesh)")),
    (None, HBox(W("QPushButton", "showConfigButton", text="Show config", toolTip="Open the config used"),
                HSpacer())),
), collapsed=True)

# ------------------------------------------------------------------ Run and outputs
results = Collapsible("outputsCollapsibleButton", "Run and outputs", VBox(
    Form(("Output folder:", Path("outputDirPathLineEdit", DIR_FILTERS,
                                 "Keeps every file of the run. Empty = temporary folder")),
         (None, Label("outputNoteLabel", "Empty = temporary folder, deleted when the scene is closed.")),
         margins=0),
    HBox(W("QPushButton", "applyButton", text="Apply", enabled=False),
         W("QPushButton", "cancelButton", text="Cancel", enabled=False),
         W("QPushButton", "saveOutputsButton", text="Save outputs...", enabled=False,
           toolTip="Copy the temporary work folder to a folder of your choice")),
    W("QProgressBar", "progressBar", value=0, visible=False),
    Label("stageLabel", ""),
    Label("warningsLabel", ""),
    Group("logGroupBox", "Log", VBox(W("QPlainTextEdit", "logTextEdit", readOnly=True,
                                       maximumBlockCount=2000)), collapsed=True),
    W("qMRMLTableView", "resultsTableView", visible=False),
    W("QWidget", "sequenceWidget", layout=VBox(
        W("QWidget", "sequencePlaceholder", layout=VBox(margins=0)),
        HBox(W("QLabel", "colorByLabel", text="Colour:"), Combo("colorByComboBox",
                                                                ["Displacement (mm)", "Error vs collapsed lung (mm)"]),
             HSpacer()),
        margins=0), visible=False),
    Group("visibilityGroupBox", "Show", HBox(
        W("QCheckBox", "showCtCheckBox", text="CT", checked=True),
        W("QCheckBox", "showInflatedCheckBox", text="Inflated lung", checked=True),
        W("QCheckBox", "showPredictionCheckBox", text="Prediction", checked=True),
        W("QCheckBox", "showTargetCheckBox", text="Real collapsed", checked=True),
        W("QCheckBox", "showHilumCheckBox", text="Hilum and rings", checked=True)), collapsed=False),
))

root = W("qMRMLWidget", "LungCollapse", layout=VBox(inputs, collapse, advanced, results, Spacer()))

CUSTOM = [("qMRMLWidget", "QWidget", "qMRMLWidget.h", 1), ("ctkCollapsibleButton", "QWidget", "ctkCollapsibleButton.h", 1),
          ("ctkCollapsibleGroupBox", "QGroupBox", "ctkCollapsibleGroupBox.h", 1),
          ("ctkSliderWidget", "QWidget", "ctkSliderWidget.h", 0), ("ctkPathLineEdit", "QWidget", "ctkPathLineEdit.h", 0),
          ("qMRMLNodeComboBox", "QWidget", "qMRMLNodeComboBox.h", 0),
          ("qMRMLSegmentSelectorWidget", "qMRMLWidget", "qMRMLSegmentSelectorWidget.h", 0),
          ("qMRMLTableView", "QTableView", "qMRMLTableView.h", 0)]
cw = "".join("<customwidget><class>%s</class><extends>%s</extends><header>%s</header>%s</customwidget>"
             % (c, e, h, "<container>1</container>" if cont else "") for c, e, h, cont in CUSTOM)
conns = "".join(
    "<connection><sender>LungCollapse</sender><signal>mrmlSceneChanged(vtkMRMLScene*)</signal>"
    "<receiver>%s</receiver><slot>setMRMLScene(vtkMRMLScene*)</slot>"
    "<hints><hint type=\"sourcelabel\"><x>10</x><y>10</y></hint><hint type=\"destinationlabel\"><x>10</x><y>10</y></hint></hints>"
    "</connection>" % n for n in MRML)
xmltext = ('<?xml version="1.0" encoding="UTF-8"?>\n<ui version="4.0">\n<class>LungCollapse</class>\n%s\n'
       '<customwidgets>%s</customwidgets>\n<resources/>\n<connections>%s</connections>\n</ui>\n' % (root, cw, conns))

from xml.dom import minidom
pretty = minidom.parseString(xmltext).toprettyxml(indent=" ")
pretty = "\n".join(l for l in pretty.splitlines() if l.strip())
pretty = pretty.replace('<?xml version="1.0" ?>', '<?xml version="1.0" encoding="UTF-8"?>')
open(sys.argv[1], "w").write(pretty + "\n")
print("ok", len(MRML), "mrml widgets")
