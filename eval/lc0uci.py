"""Minimal UCI driver for lc0. Every model runs in stock lc0; lc0 never gets
SyzygyPath, only the scorer reads the tables.

Parses the final `info ... nodes N` line (trap 6: `go nodes N` overshoots by
up to a minibatch, so the actual count is what gets recorded) and, with
VerboseMoveStats, the per-move P and the root's raw V/WL/D.
"""
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

MOVE_LINE = re.compile(
    r"info string (\S+)\s+\(\s*(\d+)\s*\)\s+N:\s+(\d+).*?\(P:\s+([\d.]+)%\)"
    r".*?\(WL:\s+([-\d.]+|-\.-+)\).*?\(D:\s+([-\d.]+|-\.-+)\)"
    r".*?\(Q:\s+([-\d.]+)\).*?\(V:\s+([-\d.]+|-\.-+)\)")
NODE_LINE = re.compile(
    r"info string node\s+\(\s*(\d+)\s*\)\s+N:\s+(\d+).*?\(WL:\s+([-\d.]+)\)"
    r"\s+\(D:\s+([-\d.]+)\)\s+\(M:\s+([-\d.]+)\).*?\(V:\s+([-\d.]+|-\.-+)\)")
WARN_PAT = re.compile(r"warn|fp32|fall ?back|error", re.I)


def _f(x):
    try:
        return float(x)
    except ValueError:
        return None


def chess960_to_std(uci, fen=None):
    """lc0 prints castling as king-takes-rook in verbose stats; map back."""
    m = {"e1h1": "e1g1", "e1a1": "e1c1", "e8h8": "e8g8", "e8a8": "e8c8"}
    if uci in m and fen is not None:
        import chess
        b = chess.Board(fen)
        mv = chess.Move.from_uci(uci)
        if b.piece_type_at(mv.from_square) == chess.KING:
            return m[uci]
    return uci


class Lc0:
    def __init__(self, exe, weights, backend="onnx-trt", backend_opts=None,
                 options=None, log=None, extra_args=None, name=None):
        self.exe = str(exe)
        self.name = name or Path(weights).name
        # Absolute weights path: lc0 resolves relative paths against its cwd.
        args = [self.exe, f"--weights={Path(weights).resolve()}", f"--backend={backend}"]
        if backend_opts:
            args.append(f"--backend-opts={backend_opts}")
        args += list(extra_args or [])
        self.args = args
        self.log_path = log
        self._log = open(log, "a", encoding="utf-8") if log else None
        self.p = subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1, encoding="utf-8", errors="replace")
        self.q = queue.Queue()
        self.warnings = []
        threading.Thread(target=self._pump, args=(self.p.stdout, False), daemon=True).start()
        threading.Thread(target=self._pump, args=(self.p.stderr, True), daemon=True).start()
        self.send("uci")
        self.wait_for("uciok", timeout=120)
        for k, v in (options or {}).items():
            self.setoption(k, v)
        self.ready(timeout=3600)  # first start may build a TensorRT engine

    # ------------------------------------------------------------ plumbing
    def _pump(self, stream, is_err):
        for line in stream:
            line = line.rstrip("\n")
            if self._log:
                self._log.write(("E " if is_err else "O ") + line + "\n")
            if is_err:
                if WARN_PAT.search(line):
                    self.warnings.append(line)
                    print(f"[{self.name}] {line}", file=sys.stderr)
            else:
                self.q.put(line)
        self.q.put(None)

    def send(self, cmd):
        if self.p.poll() is not None:
            raise RuntimeError(f"lc0 exited ({self.p.returncode}); see {self.log_path}")
        self.p.stdin.write(cmd + "\n")
        self.p.stdin.flush()

    def readline(self, timeout):
        try:
            line = self.q.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"lc0 silent for {timeout}s")
        if line is None:
            raise RuntimeError(f"lc0 exited ({self.p.poll()}); see {self.log_path}")
        return line

    def wait_for(self, prefix, timeout=600):
        lines = []
        t0 = time.time()
        while True:
            left = timeout - (time.time() - t0)
            line = self.readline(max(left, 0.01))
            lines.append(line)
            if line.startswith(prefix):
                return lines

    def ready(self, timeout=600):
        self.send("isready")
        self.wait_for("readyok", timeout)

    def setoption(self, name, value):
        if isinstance(value, bool):
            value = "true" if value else "false"
        self.send(f"setoption name {name} value {value}")

    def newgame(self):
        self.send("ucinewgame")
        self.ready()

    def quit(self):
        try:
            self.send("quit")
            self.p.wait(timeout=30)
        except Exception:
            self.p.kill()
        if self._log:
            self._log.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.quit()

    # ------------------------------------------------------------ search
    def go(self, fen=None, moves=None, nodes=None, movetime=None, clock=None,
           timeout=3600):
        """One search. clock = dict(wtime,btime,winc,binc) in ms."""
        pos = f"position fen {fen}" if fen else "position startpos"
        if moves:
            pos += " moves " + " ".join(moves)
        self.send(pos)
        cmd = "go"
        if nodes is not None:
            cmd += f" nodes {int(nodes)}"
        if movetime is not None:
            cmd += f" movetime {int(movetime)}"
        if clock:
            cmd += " " + " ".join(f"{k} {int(v)}" for k, v in clock.items())
        t0 = time.perf_counter()
        self.send(cmd)
        lines = self.wait_for("bestmove", timeout)
        wall = time.perf_counter() - t0
        return self._parse(lines, wall, fen)

    def _parse(self, lines, wall, fen):
        out = {"wall": wall, "moves": {}, "nodes": None, "time_ms": None,
               "score_cp": None, "mate": None, "depth": None}
        for line in lines:
            if line.startswith("bestmove"):
                parts = line.split()
                out["bestmove"] = chess960_to_std(parts[1], fen)
            elif line.startswith("info string node"):
                m = NODE_LINE.search(line)
                if m:
                    out["root_n"] = int(m.group(2))
                    out["wl"] = _f(m.group(3))
                    out["d"] = _f(m.group(4))
                    out["m"] = _f(m.group(5))
                    out["v"] = _f(m.group(6))
            elif line.startswith("info string"):
                m = MOVE_LINE.search(line)
                if m:
                    mv = chess960_to_std(m.group(1), fen)
                    out["moves"][mv] = {"n": int(m.group(3)),
                                        "p": float(m.group(4)) / 100.0,
                                        "q": _f(m.group(7))}
            elif line.startswith("info") and " nodes " in line:
                tok = line.split()
                for key, k2 in (("nodes", "nodes"), ("time", "time_ms"),
                                ("depth", "depth")):
                    if key in tok:
                        out[k2] = int(tok[tok.index(key) + 1])
                if "score" in tok:
                    i = tok.index("score")
                    if tok[i + 1] == "cp":
                        out["score_cp"], out["mate"] = int(tok[i + 2]), None
                    elif tok[i + 1] == "mate":
                        out["mate"], out["score_cp"] = int(tok[i + 2]), None
        return out


def engine_from_args(a, weights, name, log_dir, threads=None, minibatch=None,
                     extra_options=None):
    """Build an Lc0 from the common CLI flags (see add_engine_args)."""
    opts = {"Threads": threads if threads is not None else a.threads,
            "MinibatchSize": minibatch if minibatch is not None else a.minibatch,
            "VerboseMoveStats": True}
    if a.nncache is not None:
        opts["NNCacheSize"] = a.nncache
    opts.update(extra_options or {})
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    return Lc0(a.lc0, weights, backend=a.backend, backend_opts=a.backend_opts,
               options=opts, log=str(Path(log_dir) / f"{name}.lc0.log"), name=name)


def add_engine_args(ap, threads=1):
    ap.add_argument("--lc0", default=os.environ.get("LC0", "lc0"))
    ap.add_argument("--weights", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--backend", default="onnx-trt")
    ap.add_argument("--backend-opts", default="",
                    help="e.g. optimize=6 (fp16) or optimize=8 (int8)")
    ap.add_argument("--threads", type=int, default=threads)
    ap.add_argument("--minibatch", type=int, default=256)
    ap.add_argument("--nncache", type=int, default=None)
