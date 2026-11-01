#!/usr/bin/env python3
"""W8A8 QDQ ONNX for TensorRT explicit quantization.

Ladder knobs: --granularity per-tensor|per-channel (weights), --method
minmax|percentile|entropy (activations), --smoothquant ALPHA, and
--act-act-matmul (also quantize Q·Kᵀ and attn·V, which have no weights).

Only weight-carrying MatMul/Gemm/Conv are quantized by default; everything
else stays in the engine's float precision. All Q/DQ are symmetric int8
(zero point 0), which is what TensorRT accepts.

Trap 2: ONNX Runtime's entropy calibration is unusable here (defaults leave
one candidate threshold, i.e. max calibration; more bins make its first
candidate a no-merge slice and it clips everything to 1/8 of max). So ORT
always calibrates minmax (it places Q/DQ and sets weight scales), and
percentile / entropy activation thresholds are computed here from |x|
histograms (TensorRT's legacy entropy method, 2048 bins, zero bin excluded)
and written into the activation scale initializers.

SmoothQuant: for each group of weight matmuls reading the same activation X,
s_j = max|X_j|^a / max|W_j|^(1-a); W rows are multiplied by s and an explicit
Mul(X, 1/s) is inserted on the branch feeding that group. Folding 1/s into the
preceding LayerNorm is not possible in lc0's graphs: the LN output also feeds
the residual skip. Groups whose weight initializer is shared with matmuls
outside the group (smolgen's global weight) are left alone.

    python quant/quantize.py --onnx models/bt3_fp32.onnx --calib data/calib.npz \
        --out models/s4.onnx --granularity per-channel --method percentile \
        --percentile 99.99 --smoothquant 0.5 --outlier-report reports/outliers.txt
"""
import argparse
import json
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (INPUT, act_act_matmuls, consumers, initializers,  # noqa: E402
                    load_onnx, ort_session, save_onnx, weight_input, weight_nodes)

BATCH = 32


# ------------------------------------------------------------------ data
def load_planes(npz, n=0):
    d = np.load(npz)
    x = d["planes"].astype(np.float32)
    if n:
        x = x[:n]
    return x


def batches(x, bs=BATCH):
    for i in range(0, len(x), bs):
        yield x[i:i + bs]


def with_outputs(model, names):
    m = onnx.ModelProto()
    m.CopyFrom(model)
    have = {o.name for o in m.graph.output}
    for n in names:
        if n not in have:
            m.graph.output.append(helper.make_tensor_value_info(n, onnx.TensorProto.FLOAT, None))
    return m


def run_collect(model, planes, names, fn, cpu=True):
    """Run `model` over planes, calling fn(name, array) per batch."""
    names = list(dict.fromkeys(names))
    chunk = 64  # cap extra outputs per session to bound memory
    for c in range(0, len(names), chunk):
        sub = names[c:c + chunk]
        s = _session(with_outputs(model, sub), cpu)
        for xb in batches(planes):
            outs = s.run(sub, {INPUT: xb})
            for n, o in zip(sub, outs):
                fn(n, o)


def _session(model, cpu):
    if model.ByteSize() < 1.8e9:
        return ort_session(model, cpu=cpu)
    tmp = Path(tempfile.mkdtemp()) / "m.onnx"
    save_onnx(model, tmp)
    return ort_session(tmp, cpu=cpu)


# ------------------------------------------------------------------ stats
def channel_amax(model, planes, tensors, axis_of):
    amax = {}

    def fn(n, o):
        ax = axis_of[n] % o.ndim
        red = tuple(i for i in range(o.ndim) if i != ax)
        v = np.abs(o).max(axis=red)
        amax[n] = v if n not in amax else np.maximum(amax[n], v)

    run_collect(model, planes, tensors, fn)
    return amax


def abs_histograms(model, planes, tensors, bins=2048):
    amax = {}
    run_collect(model, planes, tensors,
                lambda n, o: amax.__setitem__(n, max(amax.get(n, 0.0), float(np.abs(o).max()))))
    hist = {t: np.zeros(bins, dtype=np.float64) for t in tensors}

    def fn(n, o):
        if amax[n] > 0:
            h, _ = np.histogram(np.abs(o), bins=bins, range=(0, amax[n]))
            hist[n] += h

    run_collect(model, planes, tensors, fn)
    return amax, hist


def percentile_amax(hist, amax, pct):
    c = np.cumsum(hist)
    if c[-1] == 0:
        return amax
    i = int(np.searchsorted(c, c[-1] * pct / 100.0))
    return min(amax, (i + 1) * amax / len(hist))


def entropy_amax(hist, amax, nq=128, start=128):
    """TensorRT's legacy entropy calibration, matching pytorch-quantization's
    _compute_amax_entropy for a symmetric int8 tensor (|x| histogram, 128
    levels, stride 1), including its zero-spike suppression (bin 0 set to
    bin 1: ReLU/mish outputs pile up at zero, which is exact in int8)."""
    h = hist.astype(np.float64).copy()
    if len(h) > 1:
        h[0] = h[1]
    nb = len(h)
    if h.sum() == 0:
        return amax
    best_kl, best_i = np.inf, nb
    for i in range(start, nb + 1):
        hb = h[:i]
        nz = hb != 0
        g = (np.arange(i) * nq) // i                 # = digitize(range(i), linspace(0,i,nq+1)) - 1
        qb = np.bincount(g, weights=hb, minlength=nq)
        cnt = np.bincount(g, weights=nz.astype(np.float64), minlength=nq)
        per = np.divide(qb, cnt, out=np.zeros_like(qb), where=cnt > 0)
        q = per[g] * nz
        p = hb.copy()
        p[-1] += h[i:].sum()
        ps, qs = p / p.sum(), q / q.sum()
        mask = ps > 0
        if (qs[mask] == 0).any():
            continue                                 # KL = inf
        kl = float(np.sum(ps[mask] * np.log(ps[mask] / qs[mask])))
        if kl < best_kl:
            best_kl, best_i = kl, i
    return best_i * amax / nb


# ------------------------------------------------------------------ smoothquant
def smoothquant(model, planes, alpha, targets):
    """Returns (new_model, report). targets: weight-matmul nodes eligible."""
    inits = initializers(model)
    cons = consumers(model)
    by_input = {}
    for n in targets:
        if n.op_type not in ("MatMul", "Gemm") or weight_input(n, inits) != 1:
            continue
        if n.op_type == "Gemm":
            attrs = {a.name: helper.get_attribute_value(a) for a in n.attribute}
            if attrs.get("transA", 0) or attrs.get("transB", 0):
                continue
        by_input.setdefault(n.input[0], []).append(n)
    w_users = {}
    for n in model.graph.node:
        for i in n.input:
            if i in inits:
                w_users.setdefault(i, set()).add(n.name)
    groups = {}
    for x, nodes in by_input.items():
        names = {n.name for n in nodes}
        if any(w_users[n.input[1]] - names for n in nodes):
            continue
        if len({n.input[1] for n in nodes}) != len(nodes):
            continue
        groups[x] = nodes
    if not groups:
        return model, {}
    amax = channel_amax(model, planes, list(groups), {x: -1 for x in groups})
    m = onnx.ModelProto()
    m.CopyFrom(model)
    inits = initializers(m)
    node_by_name = {n.name: n for n in m.graph.node}
    report = {}
    new_nodes = []
    for x, nodes in groups.items():
        ax = amax[x].astype(np.float64)
        ws = [numpy_helper.to_array(inits[n.input[1]]).astype(np.float64) for n in nodes]
        if any(w.ndim != 2 or w.shape[0] != len(ax) for w in ws):
            continue
        wmax = np.max(np.stack([np.abs(w).max(axis=1) for w in ws]), axis=0)
        s = np.where((ax > 0) & (wmax > 0),
                     np.power(np.maximum(ax, 1e-8), alpha) /
                     np.power(np.maximum(wmax, 1e-8), 1 - alpha), 1.0)
        s = np.clip(s, 1e-4, 1e4)
        for n, w in zip(nodes, ws):
            init = inits[n.input[1]]
            init.CopyFrom(numpy_helper.from_array((w * s[:, None]).astype(np.float32), init.name))
        sname = f"{x}/smooth_inv"
        m.graph.initializer.append(numpy_helper.from_array((1.0 / s).astype(np.float32), sname))
        out = f"{x}/smoothed"
        new_nodes.append((x, helper.make_node("Mul", [x, sname], [out], name=f"{x}/smooth")))
        for n in nodes:
            nn = node_by_name[n.name]
            nn.input[0] = out
        before = ax.max() / max(np.median(ax), 1e-12)
        after_ch = ax / s
        report[x] = {"consumers": [n.name for n in nodes], "outlier_before": float(before),
                     "outlier_after": float(after_ch.max() / max(np.median(after_ch), 1e-12))}
    # Insert each Mul right after its producer to keep the graph sorted.
    prod_pos = {o: i for i, n in enumerate(m.graph.node) for o in n.output}
    nodes = list(m.graph.node)
    for x, node in sorted(new_nodes, key=lambda t: -prod_pos.get(t[0], -1)):
        nodes.insert(prod_pos.get(x, -1) + 1, node)
    del m.graph.node[:]
    m.graph.node.extend(nodes)
    onnx.checker.check_model(m) if m.ByteSize() < 1.8e9 else None
    return m, report


# ------------------------------------------------------------------ outliers
def outlier_report(model, planes, nodes, path, smooth_rep=None):
    inits = initializers(model)
    xs = {}
    for n in nodes:
        if n.op_type in ("MatMul", "Gemm") and weight_input(n, inits) == 1:
            xs.setdefault(n.input[0], []).append(n.name)
        elif n.op_type == "Conv":
            xs.setdefault(n.input[0], []).append(n.name)
    axis = {x: (1 if any(nn.op_type == "Conv" for nn in nodes if nn.input[0] == x) else -1)
            for x in xs}
    amax = channel_amax(model, planes, list(xs), axis)
    rows = []
    for x, a in amax.items():
        # Median over live channels: binary input planes have many all-zero
        # channels, which would make the ratio meaningless.
        live = a[a > 0]
        med = float(np.median(live)) if len(live) else 0.0
        rows.append((float(a.max()) / max(med, 1e-12), x, float(a.max()), med, xs[x]))
    rows.sort(reverse=True)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("# outlier_ratio = worst input channel |x|max / median channel |x|max\n")
        f.write("# tens-to-hundreds on attention/FFN inputs: SmoothQuant has work to do\n")
        f.write(f"{'ratio':>9} {'max':>10} {'median':>10}  tensor -> consumers\n")
        for r, x, mx, med, c in rows:
            f.write(f"{r:9.1f} {mx:10.4g} {med:10.4g}  {x} -> {','.join(c)}\n")
        if smooth_rep:
            f.write("\n# after SmoothQuant (per group)\n")
            for x, v in sorted(smooth_rep.items(), key=lambda kv: -kv[1]["outlier_before"]):
                f.write(f"{v['outlier_before']:9.1f} -> {v['outlier_after']:7.1f}  {x}\n")
    return rows


# ------------------------------------------------------------------ quantize
class Reader:
    def __init__(self, planes):
        self.it = iter(list(batches(planes)))

    def get_next(self):
        b = next(self.it, None)
        return None if b is None else {INPUT: b}

    def rewind(self):
        pass


def ort_quantize(model, planes, nodes, per_channel, tmpdir):
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod,
                                          QuantFormat, QuantType, quantize_static)

    class R(CalibrationDataReader, Reader):
        def __init__(self, p):
            Reader.__init__(self, p)

        def get_next(self):
            return Reader.get_next(self)

    src = Path(tmpdir) / "src.onnx"
    dst = Path(tmpdir) / "q.onnx"
    save_onnx(model, src)
    quantize_static(
        str(src), str(dst), R(planes),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
        per_channel=per_channel,
        calibrate_method=CalibrationMethod.MinMax,
        op_types_to_quantize=["MatMul", "Gemm", "Conv"],
        nodes_to_quantize=[n.name for n in nodes],
        use_external_data_format=model.ByteSize() > 1.8e9,
        extra_options={"ActivationSymmetric": True, "WeightSymmetric": True,
                       "QDQOpTypePerChannelSupportToAxis": {"MatMul": 1},
                       "CalibMovingAverage": False})
    return onnx.load(str(dst), load_external_data=True)


def activation_scales(qmodel):
    """activation tensor -> scale initializer name, for scalar activation Q."""
    inits = initializers(qmodel)
    out = {}
    for n in qmodel.graph.node:
        if n.op_type == "QuantizeLinear" and n.input[0] not in inits \
                and n.input[1] in inits \
                and numpy_helper.to_array(inits[n.input[1]]).size == 1:
            out.setdefault(n.input[0], n.input[1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--calib-n", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--granularity", choices=["per-tensor", "per-channel"], default="per-channel")
    ap.add_argument("--method", choices=["minmax", "percentile", "entropy"], default="minmax")
    ap.add_argument("--percentile", type=float, default=99.99)
    ap.add_argument("--smoothquant", type=float, default=None, metavar="ALPHA")
    ap.add_argument("--act-act-matmul", action="store_true",
                    help="also quantize Q·Kᵀ / attn·V / policy Q·Kᵀ")
    ap.add_argument("--exclude", default=None, help="regex on node names to keep float")
    ap.add_argument("--outlier-report", default=None)
    a = ap.parse_args()
    t0 = time.time()

    model = load_onnx(a.onnx)
    planes = load_planes(a.calib, a.calib_n)
    print(f"{len(planes)} calibration positions from {a.calib}", file=sys.stderr)
    targets = weight_nodes(model)
    if a.exclude:
        targets = [n for n in targets if not re.search(a.exclude, n.name)]

    smooth_rep = {}
    if a.smoothquant is not None:
        model, smooth_rep = smoothquant(model, planes, a.smoothquant, targets)
        print(f"SmoothQuant alpha={a.smoothquant}: {len(smooth_rep)} groups smoothed",
              file=sys.stderr)
        names = {n.name for n in targets}
        targets = [n for n in model.graph.node if n.name in names]
    if a.outlier_report:
        # Outliers are a property of the original net; report on it, plus the
        # post-smoothing ratios if SmoothQuant ran.
        orig = load_onnx(a.onnx)
        on = {n.name for n in targets}
        outlier_report(orig, planes, [n for n in orig.graph.node if n.name in on],
                       a.outlier_report, smooth_rep)
        print(f"outlier report -> {a.outlier_report}", file=sys.stderr)

    qnodes = list(targets)
    aa = []
    if a.act_act_matmul:
        aa = [n for n in act_act_matmuls(model)
              if not (a.exclude and re.search(a.exclude, n.name))]
        qnodes += aa
    with tempfile.TemporaryDirectory() as td:
        q = ort_quantize(model, planes, qnodes, a.granularity == "per-channel", td)

    scales = activation_scales(q)
    clip = {}
    if a.method != "minmax":
        # The input planes are binary (plus a few scalars): minmax is exact
        # for them and clipping them only destroys information.
        tensors = sorted(t for t in scales if t in {o for n in model.graph.node for o in n.output})
        amax, hist = abs_histograms(model, planes, tensors)
        inits = initializers(q)
        for t in tensors:
            if amax[t] == 0:
                continue
            thr = (percentile_amax(hist[t], amax[t], a.percentile) if a.method == "percentile"
                   else entropy_amax(hist[t], amax[t]))
            init = inits[scales[t]]
            old = numpy_helper.to_array(init)
            init.CopyFrom(numpy_helper.from_array((old * (thr / amax[t])).astype(old.dtype), init.name))
            h = hist[t]
            above = h[int(np.floor(thr / amax[t] * len(h))):].sum() / max(h.sum(), 1)
            clip[t] = {"ratio": float(thr / amax[t]), "clipped_frac": float(above)}
        r = np.array([v["ratio"] for v in clip.values()]) if clip else np.array([1.0])
        print(f"{a.method}: median range ratio {np.median(r):.3f}, min {r.min():.3f} "
              f"over {len(clip)} activations", file=sys.stderr)

    save_onnx(q, a.out)
    n_q = sum(1 for n in q.graph.node if n.op_type == "QuantizeLinear")
    side = {"source": str(a.onnx), "calib": str(a.calib), "n_calib": int(len(planes)),
            "granularity": a.granularity, "method": a.method,
            "percentile": a.percentile if a.method == "percentile" else None,
            "smoothquant": a.smoothquant, "act_act_matmul": a.act_act_matmul,
            "exclude": a.exclude, "n_weight_nodes": len(targets), "n_act_act": len(aa),
            "n_quantize_linear": n_q, "smoothed_groups": smooth_rep,
            "activation_clip": clip, "seconds": round(time.time() - t0, 1)}
    with open(Path(a.out).with_suffix(".json"), "w") as f:
        json.dump(side, f, indent=1)
    print(f"wrote {a.out}: {len(targets)} weight nodes + {len(aa)} act-act, "
          f"{n_q} QuantizeLinear ({time.time() - t0:.0f}s)", file=sys.stderr)


if __name__ == "__main__":
    main()
