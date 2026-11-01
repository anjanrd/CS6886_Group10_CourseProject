"""Shared 2:4 helpers. TensorRT's sparse kernels need 2 zeros in every group
of 4 consecutive weights along the reduction (input-channel) dimension:
K for a MatMul weight [K, N], C for a Conv weight [O, C, kh, kw]."""
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import initializers, weight_input, weight_nodes  # noqa: E402


def targets(model, skip=None, min_k=16):
    """{initializer name: [nodes]} of prunable weights (shared ones grouped)."""
    inits = initializers(model)
    out = {}
    for n in weight_nodes(model):
        if skip and re.search(skip, n.name):
            continue
        wi = weight_input(n, inits)
        if n.op_type == "MatMul" and wi != 1:
            continue
        if n.op_type == "Gemm":
            attrs = {a.name: a for a in n.attribute}
            if "transB" in attrs and attrs["transB"].i:
                continue
        from onnx import numpy_helper
        w = numpy_helper.to_array(inits[n.input[wi]])
        k = w.shape[1] if n.op_type == "Conv" else w.shape[0]
        if k % 4 or k < min_k:
            continue
        out.setdefault(n.input[wi], []).append(n)
    return out


def to_kn(w, op):
    """View any supported weight as [K, M] with K the reduction dim."""
    if op == "Conv":  # [O, C, kh, kw] -> [C, O*kh*kw]
        o, c, kh, kw = w.shape
        return w.transpose(1, 0, 2, 3).reshape(c, o * kh * kw)
    return w


def from_kn(m, op, shape):
    if op == "Conv":
        o, c, kh, kw = shape
        return m.reshape(c, o, kh, kw).transpose(1, 0, 2, 3)
    return m


def magnitude_mask(kn):
    """Keep the 2 largest |w| of every 4 along K. kn: [K, M]."""
    k, m = kn.shape
    g = np.abs(kn).reshape(k // 4, 4, m)
    order = np.argsort(g, axis=1)          # ascending
    mask = np.ones_like(g, dtype=bool)
    np.put_along_axis(mask, order[:, :2, :], False, axis=1)
    return mask.reshape(k, m)


def check_24(kn):
    k, m = kn.shape
    nz = (kn.reshape(k // 4, 4, m) != 0).sum(axis=1)
    return bool((nz <= 2).all())
