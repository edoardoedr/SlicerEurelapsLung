# SlicerLungCollapse

A 3D Slicer extension that predicts the collapsed lung (pneumothorax) of a patient loaded in Slicer and shows the collapse as a sequence over the CT. The prediction is the finite element model of [`eurelaps_lung`](https://github.com/edoardoedr/lung_collapse), with its calibrated defaults. The design is in [SLICER_EXTENSION.md](SLICER_EXTENSION.md).

Module **Lung Collapse**, category *Chest*. Tested on Slicer 5.13 (Apple Silicon, arm64) and 5.12 (x86_64).

## Use

1. **Inputs.** Pick a segmentation already in the scene, or load one from files: a `.seg.nrrd`, or a labelmap `.nrrd`, `.nii` or `.nii.gz`. The segmentation needs:
   - the lungs, as one segment per lung or as lobes;
   - the airways;
   - the pulmonary arteries;
   - the pulmonary veins.

   The side, the lung segments (lobes are merged) and the three trees are pre-selected from the segment names. Optionally, add the real collapsed lung, in the same coordinates, to measure the error.
2. **Collapse.** Drive the collapse by one of two quantities; the other one is then a result.
   - **Collapse %:** set it, or take it from the collapsed lung (the default when one is given). E follows it through the calibrated rule, unless unlinked. p0 is shown as the rules' estimate, and the run finds the exact value.
   - **Pressure p0:** set p0 and E. The collapse % is shown as the rules' estimate, and the run gives the real one.

   **Lung tissue.** Answer **Emphysema: Yes / No**: your reading of the CT, required before *Apply*. With **Limit of compression** on, the lung cannot lose more volume than the air that can leave: the run measures the lung density and the emphysema volume (LAA%−950) on the CT and stops the lung above φ = density / 1050 (+ 0.54 × emphysema volume when *Yes*; with *No* the emphysema volume is not used). It needs the CT, and it is switched on by itself when the collapse is driven by a pressure: a surgical pressure (12 mmHg = 1600 Pa) has no physical equilibrium without it. With it off, the collapses driven by the collapse % are unchanged.

   Also set the position during the collapse and the number of frames.
3. **Apply.**
   - The first run builds the model geometry: surface, hilum, Netgen mesh and wall. That takes about 30 s.
   - A new collapse % on the same inputs reuses the geometry and only re-solves.
4. **Outputs.**
   - The sequence, coloured by displacement or by the error against the collapsed lung, with play and seek.
   - The inflated lung, the real collapsed lung (wireframe), the hilum and its rings.
   - A results table: volumes, collapse % reached, E, p0, emphysema (your answer), lung density and emphysema volume from the CT, the smallest volume allowed and how much of the lung reached it, solves, time, wall penetration, and the errors (ASSD, mean, median, 95th percentile, max).

   Without an output folder, the files go to a temporary folder that is deleted when the scene is closed. *Save outputs…* copies it.

On karl01 (Apple M2, Slicer 5.13 arm64), a whole run with the geometry, 40 % collapse and 10 frames takes 45 s.

## Installation

Add the `LungCollapse` folder to *Edit → Application settings → Modules → Additional module paths*, or build the extension. The first *Apply* offers to install:

- **torch**, from the *PyTorch* extension (Extensions Manager). On a Mac, use the **Apple Silicon** build of Slicer. The Intel build gets only torch 2.2.2, and the PyTorch extension then downgrades Slicer's numpy to 1.x.
- **eurelaps_lung** and its prediction dependencies (pyacvd, netgen-mesher), with pip, from `eurelaps-lung @ git+https://github.com/edoardoedr/lung_collapse.git@v0.2.0`. Another source can be set with the environment variable `LUNGCOLLAPSE_PACKAGE_SPEC` or the setting `LungCollapse/PackageSpec`, for example a local folder or a wheel.

Slicer's VTK is used; no second VTK is installed. The extension checks this.

**OpenMP.** pykdtree, which pyacvd uses, and torch each bundle their own OpenMP runtime, and in one process they abort or deadlock. The prediction therefore runs in two processes ([run_predict.py](LungCollapse/LungCollapseLib/run_predict.py)): the geometry in one, the forward run in the other.

## Layout

| File | What it does |
|---|---|
| `LungCollapse/LungCollapse.py` | Widget, parameter node, Logic (`run(params, progress_callback, done_callback)`), self-test |
| `LungCollapse/Resources/UI/LungCollapse.ui` | the interface |
| `LungCollapseLib/inputs.py` | loading files, segment guessing, lobe merge (union + closing), closed surfaces in RAS, checks |
| `LungCollapseLib/workfolder.py` | output or temporary work folder, deletion, save, geometry hash |
| `LungCollapseLib/process.py` | the `QProcess`, progress from the log lines |
| `LungCollapseLib/run_predict.py` | process entry point of the real prediction |
| `LungCollapseLib/fake_predict.py` | a stand-in with the same outputs, without the package (`LUNGCOLLAPSE_BACKEND=fake`, used by the self-test) |
| `LungCollapseLib/outputs.py` | sequence, models, markups and results table into the scene |
| `LungCollapseLib/dependencies.py` | first-use installation |

Self-test: *Developer mode → Reload and Test*, or `LungCollapseTest().runTest()`. It uses a synthetic chest and the fake backend.

## The rules

The calibrated rules come from the package (`eurelaps_lung.predict.rules()`, the file `eurelaps_lung/data/rules.json`):

- **E rule:** log₁₀ E = a + b × collapse %.
- **q0 rule, for the estimates:** log₁₀ q0 = a + b × collapse %, with q0 = p0 / E.

After new fits, `main_calibrate.py --make-default` rewrites the file, and the package gets a new version that this extension pins (`PACKAGE_VERSION`). To try another rule file without a new version, set `EURELAPS_RULES=FILE` in Slicer's environment.

## Model and limits

- The stiffness comes from the collapse % through a rule calibrated on 13 patients, collapse 23–86 %, supine CT to supine collapse. Outside that range it is an extrapolation, and the module says so.
- The lateral decubitus position enters as a body force. The rule was not validated in that position.
- The hilum is taken from the inflated lung. When a ring centre is more than 50 mm from the hilum, a warning is shown: check the rings.
- The airways, arteries and veins are all required.
- The limit of compression uses one φ for the whole lung; the trapped share of the emphysematous air (0.54) is a cohort estimate made at the end of a pneumothorax, an upper value for a deflation with suction. The pressure-driven mode with the limit is not validated against surgical CTs.

## Dependencies and licenses

| Package | License | Source |
|---|---|---|
| numpy, scipy, VTK | BSD | Slicer |
| torch (CPU) | BSD | PyTorch extension |
| eurelaps-lung | to be chosen | pip, pinned tag |
| netgen-mesher (+ netgen-occt) | LGPL-2.1 | pip |
| pyacvd | MIT | pip |
| pykdtree | LGPL-3.0+ | pip |
| pyvista | MIT | pip (through pyacvd) |

There is no GPL code on the prediction path. The LGPL packages are installed separately with pip; they are neither bundled nor modified.
