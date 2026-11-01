"""Small execution safeguards shared by the takeover runners."""
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def identity(path):
    p = Path(path).resolve()
    with p.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    return {"path": str(p), "bytes": p.stat().st_size, "sha256": digest}


def atomic_json(path, value):
    """Atomic JSON commit, using an OS temporary file outside the checkout.

    TEMP and the destination must share a volume. For another drive, configure
    a same-volume OS TEMP directory outside the checkout before execution.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, allow_nan=False) + "\n"
    handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False,
                                         prefix="lc0-commit-", suffix=".json")
    tmp = Path(handle.name)
    try:
        with handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if os.stat(tmp).st_dev != os.stat(path.parent).st_dev:
            raise RuntimeError("Configure same-volume OS TEMP outside checkout for atomic commits")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def strict_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    records = []
    for number, line in enumerate(p.read_text(encoding="utf-8-sig").splitlines(), 1):
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL {p}:{number}; preserve and review") from exc
    return records


def positive(value, label):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"Missing/nonpositive {label}: {value!r}")
    return value


def checked_result(result, label):
    if result.returncode != 0:
        raise RuntimeError(f"{label} exited {result.returncode}; inspect preserved log")


def scoped_environment(config, base=None):
    """TensorRT DLL resolution for child processes only; no global PATH edit."""
    env = dict(os.environ if base is None else base)
    lib = Path(config["trtexec"]).resolve().parent.parent / "lib"
    lc0dir = Path(config["lc0"]).resolve().parent
    if not (lib / "nvinfer_plugin_10.dll").is_file():
        raise FileNotFoundError(lib / "nvinfer_plugin_10.dll")
    env["PATH"] = os.pathsep.join([str(lib), str(lc0dir), env.get("PATH", "")])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUTF8"] = "1"
    return env


def require_session(path):
    """Require a recent operator record; this is not proof of a quiet GPU."""
    session = read_json(path)
    if session.get("exclusive_authorized") is not True or session.get("work_saved") is not True:
        raise RuntimeError("Exclusive session must be explicitly authorized after work is saved")
    if session.get("ac_power_confirmed") is not True or session.get("other_work_paused") is not True:
        raise RuntimeError("Confirm AC power and paused other work before GPU execution")
    if not 0 <= time.time() - session.get("created_unix", 0) <= 12 * 3600:
        raise RuntimeError("Exclusive-session confirmation is stale; recheck conditions")
    if Path(session.get("root", "")).resolve() != ROOT:
        raise RuntimeError("Exclusive session belongs to another checkout")
    return session


@contextlib.contextmanager
def single_workload(lock_path):
    """Refuse concurrent controller jobs. A stale lock requires explicit review."""
    p = Path(lock_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"pid": os.getpid(), "created_unix": time.time()}, f)
        yield
    finally:
        p.unlink()


def preserve_command(command, log, *, env=None, cwd=ROOT):
    """Run a structured argv, without a shell, preserving failed output too."""
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"argv": list(map(str, command)), "cwd": str(cwd)}) + "\n")
        stream.flush()
        return subprocess.run(list(map(str, command)), cwd=cwd, env=env,
                              stdout=stream, stderr=subprocess.STDOUT)
