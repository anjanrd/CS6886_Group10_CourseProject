#!/usr/bin/env python3
"""Per-layer roofline from a trtexec build (hw/trt_layers.py build).

FLOPs and bytes come from the ONNX graph at the profiled batch size; times
from trtexec's per-layer profile; TensorRT layer names are mapped back to
ONNX nodes (fused layers are named "a + b + ...", Myelin regions
"{ForeignNode[first...last]}", which is attributed to every ONNX node
between first and last in graph order).

Bytes are a model, not a measurement: weights at 1 B (INT8, weight has a
QuantizeLinear) or 2 B (FP16), activations read and written once at 2 B
(1 B when the input is quantized). It ignores cache reuse across fused ops.

Peaks: measure them (`peak` subcommand: big GEMMs through trtexec) rather
than trusting spec sheets. Spec-sheet RTX 4060 numbers for reference:
272 GB/s; tensor FP16 ~60 TFLOPS with FP16 accumulate (half that with FP32
accumulate on GeForce parts); INT8 ~121 TOPS dense, ~242 sparse.

    python hw/roofline.py peak --trtexec trtexec --out reports/trt/peak.json
    python hw/roofline.py layers --onnx models/s4.onnx --prefix reports/trt/s4 \
        --batch 256 --peaks reports/trt/peak.json --out reports/roofline/s4
"""
import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import INPUT, initializers  # noqa: E402

SPEC_4060 = {"bw_gbs": 272.0, "fp16_tflops": 60.0, "int8_tops": 121.0}


def shapes_at_batch(model, batch):
    m = onnx.ModelProto()
    m.CopyFrom(model)
    for i in m.graph.input:
        if i.name == INPUT:
            i.type.tensor_type.shape.dim[0].dim_value = batch
    for o in m.graph.output:
        if o.type.tensor_type.shape.dim:
            o.type.tensor_type.shape.dim[0].dim_value = batch
    m = shape_inference.infer_shapes(m, data_prop=True)
    shp = {}
    for v in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output):
        dims = [d.dim_value if d.HasField("dim_value") else None for d in v.type.tensor_type.shape.dim]
        if dims and all(d is not None for d in dims):
            shp[v.name] = dims
    for t in m.graph.initializer:
        shp[t.name] = list(t.dims)
    return shp


def node_costs(model, batch):
    inits = initializers(model)
    shp = shapes_at_batch(model, batch)
    prod = {o: n for n in model.graph.node for o in n.output}
    quantized_inputs = set()  # tensors that are the output of a DequantizeLinear

    def src(t):
        """Follow DequantizeLinear back to the int8 source."""
        n = prod.get(t)
        if n is not None and n.op_type == "DequantizeLinear":
            quantized_inputs.add(t)
            return n.input[0]
        return t

    costs = {}
    for n in model.graph.node:
        if n.op_type not in ("MatMul", "Gemm", "Conv"):
            continue
        ins = [src(i) for i in n.input[:2]]
        a_shape = shp.get(n.input[0]) or shp.get(ins[0])
        b_shape = shp.get(n.input[1]) or shp.get(ins[1])
        o_shape = shp.get(n.output[0])
        if not (a_shape and b_shape and o_shape):
            continue
        if n.op_type == "Conv":
            o, c, kh, kw = b_shape
            flops = 2 * int(np.prod(o_shape)) * c * kh * kw
        else:
            k = a_shape[-1]
            flops = 2 * int(np.prod(o_shape)) * k
        def nbytes(t, shape):  # noqa: E306
            return int(np.prod(shape)) * (1 if t in quantized_inputs else 2)
        by = nbytes(n.input[0], a_shape) + nbytes(n.input[1], b_shape) + 2 * int(np.prod(o_shape))
        weight = any(i in inits for i in ins)
        costs[n.name] = {"flops": flops, "bytes": by, "op": n.op_type,
                         "int8": n.input[0] in quantized_inputs and n.input[1] in quantized_inputs,
                         "weight": weight}
    return costs


def map_layers(model, trt_names):
    order = [n.name for n in model.graph.node]
    pos = {nm: i for i, nm in enumerate(order)}
    out_to_node = {o: n.name for n in model.graph.node for o in n.output}
    mapping = {}
    for name in trt_names:
        hits = set()
        m = re.search(r"ForeignNode\[(.+?)\.\.\.(.+?)\]", name)
        if m:
            a, b = m.group(1), m.group(2)
            a = out_to_node.get(a, a)
            b = out_to_node.get(b, b)
            if a in pos and b in pos:
                lo, hi = sorted((pos[a], pos[b]))
                hits.update(order[lo:hi + 1])
        for tok in re.split(r"\s+\+\s+|,\s*", name.strip("{}")):
            tok = re.sub(r"\s*\[.*\]$", "", tok.strip())
            if tok in pos:
                hits.add(tok)
            elif tok in out_to_node:
                hits.add(out_to_node[tok])
        mapping[name] = hits
    return mapping


def cmd_layers(a):
    model = onnx.load(a.onnx)
    costs = node_costs(model, a.batch)
    info = json.loads(Path(f"{a.prefix}_layers.json").read_text())
    layers = info.get("Layers", [])
    prof = {}
    for r in json.loads(Path(f"{a.prefix}_profile.json").read_text()):
        if isinstance(r, dict) and "name" in r:
            prof[r["name"]] = float(r.get("averageMs", 0))
    names = [L.get("Name", "") for L in layers]
    mp = map_layers(model, names)
    peaks = dict(SPEC_4060)
    if a.peaks and Path(a.peaks).exists():
        peaks.update(json.loads(Path(a.peaks).read_text()))
    rows = []
    claimed = set()
    for L in layers:
        nm = L.get("Name", "")
        nodes = [n for n in mp.get(nm, ()) if n in costs and n not in claimed]
        claimed.update(nodes)
        if not nodes:
            continue
        fl = sum(costs[n]["flops"] for n in nodes)
        by = sum(costs[n]["bytes"] for n in nodes)
        ms = prof.get(nm, 0.0)
        is8 = all(costs[n]["int8"] for n in nodes)
        peak = (peaks["int8_tops"] if is8 else peaks["fp16_tflops"]) * 1e12
        ai = fl / max(by, 1)
        bound = min(peak, ai * peaks["bw_gbs"] * 1e9)
        ach = fl / (ms * 1e-3) if ms > 0 else float("nan")
        rows.append({"layer": nm, "onnx_nodes": len(nodes), "precision": "int8" if is8 else "fp16",
                     "gflop": fl / 1e9, "mbytes": by / 1e6, "ai_flop_per_byte": ai, "ms": ms,
                     "achieved_tflops": ach / 1e12, "roof_tflops": bound / 1e12,
                     "pct_of_roof": 100 * ach / bound if ms > 0 else float("nan"),
                     "bound": "memory" if ai * peaks["bw_gbs"] * 1e9 < peak else "compute"})
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{out}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["layer"])
        w.writeheader()
        w.writerows(rows)
    tot_ms = sum(prof.values())
    gemm_ms = sum(r["ms"] for r in rows)
    tot_fl = sum(c["flops"] for c in costs.values())
    unmapped = sorted(set(costs) - claimed)
    lines = [f"# Roofline: {a.onnx} at batch {a.batch}", "",
             f"peaks: {peaks}",
             f"engine time {tot_ms:.3f} ms/batch; mapped GEMM/conv layers {gemm_ms:.3f} ms "
             f"({100 * gemm_ms / max(tot_ms, 1e-9):.0f}%)",
             f"graph GEMM work {tot_fl / 1e9:.1f} GFLOP/batch -> {tot_fl / max(tot_ms, 1e-9) / 1e9:.1f} TFLOPS "
             f"over the whole engine time",
             f"{len(unmapped)} ONNX GEMM nodes not matched to a TensorRT layer name"
             + (f" (e.g. {unmapped[:3]})" if unmapped else ""), "",
             "| ms | prec | GFLOP | AI | achieved TF | roof TF | % roof | bound | layer |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: -r["ms"])[:40]:
        nm = r["layer"] if len(r["layer"]) < 80 else r["layer"][:77] + "..."
        lines.append(f"| {r['ms']:.4f} | {r['precision']} | {r['gflop']:.2f} | {r['ai_flop_per_byte']:.0f} | "
                     f"{r['achieved_tflops']:.2f} | {r['roof_tflops']:.2f} | {r['pct_of_roof']:.0f} | "
                     f"{r['bound']} | `{nm}` |")
    Path(f"{out}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:6]))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        x = np.logspace(0, 4, 200)
        for lab, pk, ls in (("FP16 roof", peaks["fp16_tflops"], "-"), ("INT8 roof", peaks["int8_tops"], "--")):
            ax.plot(x, np.minimum(pk, x * peaks["bw_gbs"] / 1e3), ls, color="#444", lw=1, label=lab)
        for prec, mk, col in (("fp16", "o", "#1f6fb4"), ("int8", "s", "#d9541e")):
            pts = [r for r in rows if r["precision"] == prec and r["ms"] > 0]
            if pts:
                ax.scatter([r["ai_flop_per_byte"] for r in pts], [r["achieved_tflops"] for r in pts],
                           s=[10 + 400 * r["ms"] / max(gemm_ms, 1e-9) for r in pts],
                           marker=mk, color=col, alpha=0.7, label=f"{prec} layers")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("arithmetic intensity (FLOP/byte, modelled)")
        ax.set_ylabel("achieved TFLOPS")
        ax.set_title(f"{Path(a.onnx).stem}, batch {a.batch} (marker area ~ time)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{out}.png", dpi=150)
    except Exception as e:  # plotting is optional
        print(f"(no plot: {e})", file=sys.stderr)


def gemm_onnx(path, m, k, n, int8):
    rng = np.random.default_rng(0)
    w = rng.normal(0, 0.02, (k, n)).astype(np.float32)
    nodes, inits = [], [numpy_helper.from_array(w, "W")]
    x, wt = "X", "W"
    if int8:
        inits += [numpy_helper.from_array(np.array(0.02, np.float32), "sx"),
                  numpy_helper.from_array(np.array(0, np.int8), "zx"),
                  numpy_helper.from_array(np.full(n, 0.0005, np.float32), "sw"),
                  numpy_helper.from_array(np.zeros(n, np.int8), "zw")]
        nodes += [helper.make_node("QuantizeLinear", ["X", "sx", "zx"], ["Xq"]),
                  helper.make_node("DequantizeLinear", ["Xq", "sx", "zx"], ["Xd"]),
                  helper.make_node("QuantizeLinear", ["W", "sw", "zw"], ["Wq"], axis=1),
                  helper.make_node("DequantizeLinear", ["Wq", "sw", "zw"], ["Wd"], axis=1)]
        x, wt = "Xd", "Wd"
    nodes.append(helper.make_node("MatMul", [x, wt], ["Y"]))
    g = helper.make_graph(nodes, "gemm", [helper.make_tensor_value_info("X", TensorProto.FLOAT, [m, k])],
                          [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [m, n])], initializer=inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8  # what leela2onnx emits; newer IRs trip older parsers
    onnx.save(m, path)


def cmd_peak(a):
    res = {}
    d = Path(tempfile.mkdtemp())
    m, k, n = a.m, a.k, a.n
    flops = 2 * m * k * n
    for tag, int8 in (("fp16", False), ("int8", True)):
        p = d / f"gemm_{tag}.onnx"
        gemm_onnx(str(p), m, k, n, int8)
        args = [a.trtexec, f"--onnx={p}", "--fp16", "--useCudaGraph", "--noDataTransfers",
                "--builderOptimizationLevel=5"] + (["--int8"] if int8 else [])
        r = subprocess.run(args, capture_output=True, text=True)
        txt = r.stdout + r.stderr
        mm = re.search(r"GPU Compute Time: min = [\d.]+ ms, max = [\d.]+ ms, mean = ([\d.]+) ms", txt)
        if not mm:
            sys.exit(f"trtexec failed for {tag}:\n{txt[-2000:]}")
        ms = float(mm.group(1))
        res[f"{tag}_measured_ms"] = ms
        res["fp16_tflops" if tag == "fp16" else "int8_tops"] = flops / (ms * 1e-3) / 1e12
    # bandwidth: a large memory-bound GEMV-ish matmul
    p = d / "bw.onnx"
    gemm_onnx(str(p), 8, 16384, 16384, False)
    r = subprocess.run([a.trtexec, f"--onnx={p}", "--fp16", "--useCudaGraph", "--noDataTransfers"],
                       capture_output=True, text=True)
    mm = re.search(r"GPU Compute Time: min = [\d.]+ ms, max = [\d.]+ ms, mean = ([\d.]+) ms", r.stdout + r.stderr)
    if mm:
        res["bw_gbs"] = 16384 * 16384 * 2 / (float(mm.group(1)) * 1e-3) / 1e9
    res["gemm_shape"] = [m, k, n]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    l = sub.add_parser("layers")
    l.add_argument("--onnx", required=True)
    l.add_argument("--prefix", required=True, help="trt_layers.py build --out prefix")
    l.add_argument("--batch", type=int, default=256)
    l.add_argument("--peaks", default=None)
    l.add_argument("--out", required=True)
    p = sub.add_parser("peak")
    p.add_argument("--trtexec", default="trtexec")
    p.add_argument("--m", type=int, default=8192)
    p.add_argument("--k", type=int, default=4096)
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"layers": cmd_layers, "peak": cmd_peak}[a.cmd](a)


if __name__ == "__main__":
    main()
