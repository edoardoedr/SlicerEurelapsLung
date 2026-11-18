"""The prediction runs in a separate process (SLICER_EXTENSION.md, section 6): the GUI stays responsive,
Cancel kills it, a crash of a native library does not take Slicer down. Progress from the log lines."""

import os
import re
import shutil

import qt

import slicer

STAGES = (
    # (text in the line, stage number, name)
    ("=== geometry ===", 1, "surface"),
    ("=== hilum ===", 2, "hilum"),
    ("=== fem_setup ===", 3, "mesh and wall"),
    ("=== collapse ===", 4, "forward"),
)
N_STAGES = 5
_FRAME = re.compile(r"frame\s+(\d+)\s*/\s*(\d+)")
_SOLVE = re.compile(r"q0\b.*volume|volume.*\bq0\b|solve\s+\d+", re.I)


def python_slicer():
    """Path of PythonSlicer (Slicer's Python with Slicer's environment)."""
    home = slicer.app.slicerHome
    for p in (os.path.join(home, "bin", "PythonSlicer"), os.path.join(home, "bin", "PythonSlicer.exe")):
        if os.path.isfile(p):
            return p
    return shutil.which("PythonSlicer") or "PythonSlicer"


class Progress:
    """Log line -> (percent, stage text). Stage k of 5 spans 20 %; solves and frames advance inside it."""

    def __init__(self):
        self.stage, self.name, self.percent, self.solves = 0, "starting", 0.0, 0

    def feed(self, line):
        """Returns True when the progress changed."""
        for text, k, name in STAGES:
            if text in line:
                self._set(k, name, 0.0)
                return True
        m = _FRAME.search(line)
        if m and ("load" in line or "frame" in line):
            i, n = int(m.group(1)), max(int(m.group(2)), 1)
            self._set(5, "frames %d/%d" % (i, n), i / n)
            return True
        if ("frames" in line or "sequence" in line) and self.stage == 4 and "written" not in line:
            self._set(5, "frames", 0.0)
            return True
        if self.stage == 4 and _SOLVE.search(line):
            self.solves += 1
            self._set(4, "forward (solve %d)" % self.solves, min(0.9, self.solves / 8.0))
            return True
        return False

    def _set(self, k, name, frac):
        self.stage, self.name = k, name
        self.percent = max(self.percent, 100.0 * (k - 1 + frac) / N_STAGES)

    def text(self):
        return "%d/%d %s" % (self.stage, N_STAGES, self.name) if self.stage else self.name


class ProcessRunner:
    """A qt.QProcess with merged output read line by line."""

    def __init__(self, program, args, cwd=None, env=None, on_line=None, on_finished=None):
        self.program, self.args, self.cwd, self.env = program, list(args), cwd, dict(env or {})
        self.on_line, self.on_finished = on_line, on_finished
        self.proc = None
        self.cancelled = False
        self._buffer = ""
        self.lines = []

    def start(self):
        self.proc = qt.QProcess()
        self.proc.setProcessChannelMode(qt.QProcess.MergedChannels)
        if self.cwd:
            self.proc.setWorkingDirectory(str(self.cwd))
        env = qt.QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        for k, v in self.env.items():
            env.insert(k, str(v))
        self.proc.setProcessEnvironment(env)
        self.proc.connect("readyReadStandardOutput()", self._read)
        self.proc.connect("finished(int,QProcess::ExitStatus)", self._finished)
        self.proc.connect("errorOccurred(QProcess::ProcessError)", self._error)
        self.proc.start(self.program, self.args)

    def running(self):
        return self.proc is not None and self.proc.state() != qt.QProcess.NotRunning

    def cancel(self):
        if self.running():
            self.cancelled = True
            self.proc.kill()

    def _emit(self, line):
        self.lines.append(line)
        if self.on_line:
            self.on_line(line)

    def _read(self):
        data = self.proc.readAllStandardOutput().data()
        self._buffer += bytes(data).decode("utf-8", errors="replace")
        *done, self._buffer = self._buffer.split("\n")
        for line in done:
            self._emit(line.rstrip("\r"))

    def _error(self, err):
        if err == qt.QProcess.FailedToStart:
            self._emit("ERROR could not start %s" % self.program)
            if self.on_finished:
                self.on_finished(-1, self.cancelled)

    def _finished(self, code, status):
        self._read()
        if self._buffer:
            self._emit(self._buffer)
            self._buffer = ""
        if status != qt.QProcess.NormalExit and code == 0:
            code = -1
        if self.on_finished:
            self.on_finished(code, self.cancelled)
