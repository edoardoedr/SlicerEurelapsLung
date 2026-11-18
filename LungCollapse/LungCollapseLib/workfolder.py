"""The work folder of a run (SLICER_EXTENSION.md, section 6).

- output folder chosen: <output folder>/<name>_<side>/, kept;
- none: a temporary folder LungCollapse_* under Slicer's temporary path, kept while the inputs stay the
  same (the geometry is reused), deleted when the inputs change, the scene is closed or Slicer exits;
  leftovers older than a day (after a crash) are removed when the module starts.

Layout:  inputs/*.vtk  config.json  geometry_hash.txt  out/ (the package's output_dir)
"""

import os
import shutil
import tempfile
import time
from pathlib import Path

PREFIX = "LungCollapse_"


class WorkFolders:
    def __init__(self, temp_root):
        self.temp_root = Path(temp_root)
        self.temp = None          # current temporary work folder
        self.temp_key = None      # the inputs it was made for

    # ------------------------------------------------------------------
    def remove_stale(self, max_age_s=86400.0):
        """Remove LungCollapse_* folders older than max_age_s (left by a crash)."""
        removed = []
        if not self.temp_root.is_dir():
            return removed
        now = time.time()
        for p in self.temp_root.glob(PREFIX + "*"):
            try:
                if p.is_dir() and p != self.temp and now - p.stat().st_mtime > max_age_s:
                    shutil.rmtree(p, ignore_errors=True)
                    removed.append(p)
            except OSError:
                pass
        return removed

    def get(self, output_dir, name, side, key):
        """(work folder, is_temporary) for these inputs. key identifies the inputs (a new key drops the
        current temporary folder)."""
        if output_dir:
            work = Path(output_dir) / ("%s_%s" % (name, side))
            work.mkdir(parents=True, exist_ok=True)
            return work, False
        if self.temp is not None and self.temp.is_dir() and self.temp_key == key:
            return self.temp, True
        self.discard()
        self.temp_root.mkdir(parents=True, exist_ok=True)
        self.temp = Path(tempfile.mkdtemp(prefix=PREFIX, dir=str(self.temp_root)))
        self.temp_key = key
        return self.temp, True

    def inputs_changed(self, key):
        """Drop the temporary folder when the inputs are no longer the ones it was made for."""
        if self.temp is not None and key != self.temp_key:
            self.discard()

    def discard(self):
        """Delete the current temporary work folder."""
        if self.temp is not None:
            shutil.rmtree(self.temp, ignore_errors=True)
        self.temp = None
        self.temp_key = None

    def save(self, dest_dir, name):
        """Copy the current temporary work folder into dest_dir/<name>; returns the copy."""
        if self.temp is None or not self.temp.is_dir():
            raise ValueError("no temporary work folder to save")
        dest = Path(dest_dir) / name
        if dest.exists():
            i = 2
            while Path("%s_%d" % (dest, i)).exists():
                i += 1
            dest = Path("%s_%d" % (dest, i))
        shutil.copytree(self.temp, dest)
        return dest


def stored_hash(work):
    f = Path(work) / "geometry_hash.txt"
    return f.read_text().strip() if f.is_file() else None


def store_hash(work, value):
    (Path(work) / "geometry_hash.txt").write_text(value + "\n")


def has_geometry(work):
    return (Path(work) / "out" / "fem" / "setup" / "problem.npz").is_file()


def touch(path):
    try:
        os.utime(path, None)
    except OSError:
        pass
