"""Shared helpers: ONNX I/O, lc0 tensor names, jsonl, small stats."""
import json
import math
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Tensor names produced by `lc0 leela2onnx` (checked against v0.33).
INPUT = "/input/planes"
OUT_POLICY = "/output/policy"
OUT_WDL = "/output/wdl"
OUT_MLH = "/output/mlh"

WEIGHT_OPS = ("MatMul", "Gemm", "Conv")


# ------------------------------------------------------------------ jsonl
def read_jsonl(path):
    out = []
    p = Path(path)
    if not p.exists():
        return out
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # A run killed mid-write leaves a torn last line; skip it.
                continue
    return out


def append_jsonl(path, rec):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)


# ------------------------------------------------------------------ onnx
def load_onnx(path, load_data=True):
    import onnx
    return onnx.load(str(path), load_external_data=load_data)


def save_onnx(model, path):
    """Save, switching to external data above ~1.8 GB (protobuf's 2 GB cap).
    Saves a copy so the caller's in-memory model keeps inline tensors."""
    import onnx
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    size = model.ByteSize()
    if size < 1.8e9:
        onnx.save(model, str(path))
        return
    m = onnx.ModelProto()
    m.CopyFrom(model)
    data = path.name + ".data"
    if (path.parent / data).exists():
        (path.parent / data).unlink()
    onnx.save(m, str(path), save_as_external_data=True,
              all_tensors_to_one_file=True, location=data, size_threshold=1024)


def initializers(model):
    return {i.name: i for i in model.graph.initializer}


def producers(model):
    return {o: n for n in model.graph.node for o in n.output}


def consumers(model):
    c = {}
    for n in model.graph.node:
        for i in n.input:
            c.setdefault(i, []).append(n)
    return c


def weight_input(node, inits):
    """Index of the constant weight input of a MatMul/Gemm/Conv, or None.
    Follows nothing: leela2onnx feeds weights straight from initializers
    (possibly through a Cast for fp16 graphs, which we do not quantize)."""
    if node.op_type not in WEIGHT_OPS:
        return None
    if node.op_type in ("Conv",):
        return 1 if len(node.input) > 1 and node.input[1] in inits else None
    if node.op_type == "Gemm":
        return 1 if node.input[1] in inits else None
    # MatMul: weight is normally the second input.
    if node.input[1] in inits:
        return 1
    if node.input[0] in inits:
        return 0
    return None


def weight_nodes(model):
    inits = initializers(model)
    return [n for n in model.graph.node if weight_input(n, inits) is not None]


def act_act_matmuls(model):
    inits = initializers(model)
    return [n for n in model.graph.node
            if n.op_type == "MatMul" and n.input[0] not in inits
            and n.input[1] not in inits]


def ort_session(model_or_path, cpu=False, threads=0):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads:
        so.intra_op_num_threads = threads
    so.log_severity_level = 3
    provs = ["CPUExecutionProvider"]
    if not cpu:
        avail = ort.get_available_providers()
        if "CUDAExecutionProvider" in avail:
            provs = ["CUDAExecutionProvider"] + provs
    src = model_or_path
    if not isinstance(src, (str, Path)):
        if src.ByteSize() > 1.8e9:
            raise RuntimeError("model >1.8GB: save to disk and pass the path")
        src = src.SerializeToString()
    else:
        src = str(src)
    return ort.InferenceSession(src, so, providers=provs)


# ------------------------------------------------------------------ stats
def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def bootstrap_ci(values, stat=None, iters=2000, seed=0, alpha=0.05):
    import numpy as np
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(iters, len(v)))
    s = v[idx].mean(axis=1) if stat is None else np.array([stat(v[i]) for i in idx])
    return (float(np.quantile(s, alpha / 2)), float(np.quantile(s, 1 - alpha / 2)))


def die(msg):
    print("ERROR:", msg, file=sys.stderr)
    sys.exit(1)


def seeded(seed):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
