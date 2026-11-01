#!/usr/bin/env python3
"""Do the matmuls really run in INT8 (and sparse)? Two independent checks.

1. build: run trtexec on an ONNX with lc0's settings (FP16 [+INT8], builder
   optimization level 5, the same dynamic batch profile lc0 uses), export
   per-layer info and a per-layer profile, and summarise which GEMM/conv
   layers run in which precision and how much of the time they take.
       python hw/trt_layers.py build --trtexec trtexec --onnx models/s4.onnx --int8 --out reports/trt/s4
       python hw/trt_layers.py build ... --sparsity     # S0/S1 speed ceiling
2. kernels: what lc0 actually ran, from an Nsight Systems capture:
       nsys profile -o reports/trt/s4_lc0 --force-overwrite true lc0 benchmark --weights=models/s4.pb.gz --backend=onnx-trt --backend-opts=optimize=8 --nodes=20000 --num-positions=3
       nsys stats --report cuda_gpu_kern_sum --format csv -o reports/trt/s4_lc0 reports/trt/s4_lc0.nsys-rep
       python hw/trt_layers.py kernels --csv reports/trt/s4_lc0_cuda_gpu_kern_sum.csv
   Kernel names are classified by the precision tokens TensorRT/cuBLAS put
   in them (i8/imma/int8, h/hmma/f16, bf16, sparse). Read the top names
   yourself too: the classifier is a heuristic.

lc0's onnx-trt maps optimize>=6 to fp16, >=7 to bf16 (only when lc0 was
built against onnxruntime >= 1.23), >=8 to int8 (trap 4). The Windows
v0.33.0-rc0 package bundles onnxruntime 1.22.1, so optimize=8 there means
FP16 + INT8, no BF16; `build` mirrors that unless --bf16.
"""
import argparse
import csv
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

INPUT = "/input/planes"
GEMM_TYPES = re.compile(r"gemm|matrix|matmul|conv|fully|cask", re.I)


def classify_text(s):
    """Primary precision of a kernel/tactic name (int8 > bf16 > fp16 > fp32,
    since e.g. `gemm_i8f16_i8i32` has int8 operands and an fp16 output),
    plus 'sparse' when the name says so."""
    s = s.lower()
    tags = set()
    if re.search(r"int8|i8|imma|s8|qgemm", s):
        tags.add("int8")
    elif re.search(r"bf16|bfloat", s):
        tags.add("bf16")
    elif re.search(r"fp16|f16|half|hmma|h16|_h_", s):
        tags.add("fp16")
    elif re.search(r"fp32|f32|float|sgemm|tf32", s):
        tags.add("fp32")
    if re.search(r"sparse|_sp_|sptensor|spmma", s):
        tags.add("sparse")
    return tags


def layer_precision(layer):
    """Precision of a layer from its I/O formats, else its tactic name."""
    fmts = []
    for io in layer.get("Inputs", []) + layer.get("Outputs", []):
        f = io.get("Format/Datatype") or io.get("Format") or io.get("Datatype") or ""
        fmts.append(f)
    ins = [(io.get("Format/Datatype") or "") for io in layer.get("Inputs", [])]
    if any(re.search(r"int8", f, re.I) for f in ins):
        p = "int8"
    elif any(re.search(r"bf16|bfloat", f, re.I) for f in ins):
        p = "bf16"
    elif any(re.search(r"half|fp16", f, re.I) for f in ins):
        p = "fp16"
    elif any(re.search(r"float|fp32", f, re.I) for f in ins):
        p = "fp32"
    else:
        t = classify_text(str(layer.get("TacticName", "")) + " " + str(layer.get("Name", "")))
        p = next((x for x in ("int8", "bf16", "fp16", "fp32") if x in t), "?")
    sparse = "sparse" in classify_text(str(layer.get("TacticName", "")))
    return p, sparse


def load_profile(path):
    if not path or not Path(path).exists():
        return {}
    d = json.loads(Path(path).read_text())
    out = {}
    for r in d:
        if isinstance(r, dict) and "name" in r:
            out[r["name"]] = float(r.get("averageMs", r.get("timeMs", 0)))
    return out


def summarize_layers(info_path, profile_path=None):
    info = json.loads(Path(info_path).read_text())
    layers = info.get("Layers", info if isinstance(info, list) else [])
    if layers and isinstance(layers[0], str):
        raise SystemExit("layer info has names only: rebuild with --profilingVerbosity=detailed")
    prof = load_profile(profile_path)
    tot = defaultdict(float)
    rows = []
    for L in layers:
        name = L.get("Name", "?")
        ltype = str(L.get("LayerType", L.get("ParameterType", "")))
        p, sp = layer_precision(L)
        ms = prof.get(name, 0.0)
        is_gemm = bool(GEMM_TYPES.search(ltype)) or "ForeignNode" in name
        rows.append({"name": name, "type": ltype, "precision": p, "sparse": sp,
                     "ms": ms, "gemm": is_gemm, "tactic": L.get("TacticName", "")})
        tot["all"] += ms
        if is_gemm:
            tot["gemm"] += ms
            tot[f"gemm_{p}"] += ms
            if sp:
                tot["gemm_sparse"] += ms
    return rows, dict(tot)


def report_layers(rows, tot, out):
    g = [r for r in rows if r["gemm"]]
    cnt = defaultdict(int)
    for r in g:
        cnt[r["precision"] + ("+sparse" if r["sparse"] else "")] += 1
    lines = ["# TensorRT layer precision", "",
             f"GEMM/conv-like layers: {len(g)} of {len(rows)} engine layers",
             "count by precision: " + ", ".join(f"{k}={v}" for k, v in sorted(cnt.items()))]
    if tot.get("all"):
        lines.append(f"time: all {tot['all']:.3f} ms, gemm-like {tot.get('gemm', 0):.3f} ms; "
                     + ", ".join(f"{k[5:]} {100 * v / max(tot.get('gemm', 1e-9), 1e-9):.1f}%"
                                 for k, v in sorted(tot.items()) if k.startswith("gemm_")))
    foreign = [r for r in rows if "ForeignNode" in r["name"]]
    if foreign:
        lines.append(f"\n{len(foreign)} Myelin ForeignNode layer(s): fused regions whose "
                     "per-op precision is not visible here; use `kernels` on an nsys capture.")
    lines += ["", "| ms | precision | sparse | type | layer |", "|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: -r["ms"])[:60]:
        nm = r["name"] if len(r["name"]) < 110 else r["name"][:107] + "..."
        lines.append(f"| {r['ms']:.4f} | {r['precision']} | {'y' if r['sparse'] else ''} | "
                     f"{r['type']} | `{nm}` |")
    Path(out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:6]))


def cmd_build(a):
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shp = lambda b: f"{INPUT}:{b}x112x8x8"  # noqa: E731
    args = [a.trtexec, f"--onnx={a.onnx}", "--fp16",
            f"--builderOptimizationLevel={a.opt_level}",
            f"--minShapes={shp(a.min_batch)}", f"--optShapes={shp(a.opt_batch)}",
            f"--maxShapes={shp(a.max_batch)}", f"--shapes={shp(a.batch)}",
            "--profilingVerbosity=detailed",
            f"--exportLayerInfo={out}_layers.json", f"--exportProfile={out}_profile.json",
            f"--exportTimes={out}_times.json", "--separateProfileRun",
            f"--saveEngine={out}.engine", f"--timingCacheFile={out.parent / 'timing.cache'}",
            "--useCudaGraph", "--noDataTransfers"]
    if a.int8:
        args.append("--int8")
    if a.bf16:
        args.append("--bf16")
    if a.sparsity:
        args.append("--sparsity=enable")
    if a.layer_norm_fp32:
        # lc0 sets trt_layer_norm_fp32_fallback=1; trtexec has no direct switch.
        args.append("--layerPrecisions=*/ln*:fp32")
        args.append("--precisionConstraints=prefer")
    print(" ".join(args), file=sys.stderr)
    log = Path(f"{out}_trtexec.log")
    with open(log, "w") as f:
        r = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT)
    txt = log.read_text(errors="replace")
    m = re.search(r"Throughput:\s*([\d.]+)\s*qps", txt)
    lat = re.search(r"GPU Compute Time: min = [\d.]+ ms, max = [\d.]+ ms, mean = ([\d.]+) ms", txt)
    summ = {"onnx": a.onnx, "int8": a.int8, "sparsity": a.sparsity, "batch": a.batch,
            "returncode": r.returncode,
            "qps": float(m.group(1)) if m else None,
            "gpu_ms_mean": float(lat.group(1)) if lat else None}
    if summ["qps"]:
        summ["positions_per_s"] = summ["qps"] * a.batch
    if a.sparsity:
        summ["sparse_layers_reported"] = len(re.findall(r"[Ss]parsity.*?(?:eligible|enabled)", txt))
    Path(f"{out}_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ), file=sys.stderr)
    if r.returncode != 0:
        sys.exit(f"trtexec failed; see {log}")
    rows, tot = summarize_layers(f"{out}_layers.json", f"{out}_profile.json")
    report_layers(rows, tot, f"{out}_layers.md")


def cmd_layers(a):
    rows, tot = summarize_layers(a.info, a.profile)
    report_layers(rows, tot, a.out)


def cmd_kernels(a):
    rows = []
    with open(a.csv, newline="", encoding="utf-8", errors="replace") as f:
        txt = f.read()
    # nsys stats may print a header before the CSV
    i = txt.find('"Time')
    i = i if i >= 0 else txt.find("Time")
    rd = csv.DictReader(txt[i:].splitlines())
    for r in rd:
        name = r.get("Name") or r.get("Kernel Name") or ""
        t = r.get("Total Time (ns)") or r.get("Total Time") or "0"
        try:
            ns = float(str(t).replace(",", ""))
        except ValueError:
            continue
        rows.append((ns, name, classify_text(name), int(float(str(r.get("Instances", "0")).replace(",", "") or 0))))
    tot = sum(r[0] for r in rows) or 1
    by = defaultdict(float)
    gemm_by = defaultdict(float)
    for ns, name, tags, _ in rows:
        key = "+".join(sorted(tags)) or "other"
        by[key] += ns
        if re.search(r"gemm|mma|conv|xmma|cutlass|matmul", name, re.I):
            gemm_by[key] += ns
    gt = sum(gemm_by.values()) or 1
    lines = [f"# GPU kernels in {a.csv}", "",
             "all kernel time by tag: " + ", ".join(f"{k} {100 * v / tot:.1f}%" for k, v in sorted(by.items(), key=lambda kv: -kv[1])),
             "GEMM-like kernel time by tag: " + ", ".join(f"{k} {100 * v / gt:.1f}%" for k, v in sorted(gemm_by.items(), key=lambda kv: -kv[1])),
             "", "| % time | instances | tags | kernel |", "|---|---|---|---|"]
    for ns, name, tags, n in sorted(rows, reverse=True)[:40]:
        nm = name if len(name) < 120 else name[:117] + "..."
        lines.append(f"| {100 * ns / tot:.2f} | {n} | {'+'.join(sorted(tags))} | `{nm}` |")
    out = a.out or str(Path(a.csv).with_suffix(".md"))
    Path(out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:4]))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--trtexec", default="trtexec")
    b.add_argument("--onnx", required=True)
    b.add_argument("--out", required=True, help="path prefix for outputs")
    b.add_argument("--int8", action="store_true")
    b.add_argument("--bf16", action="store_true")
    b.add_argument("--sparsity", action="store_true")
    b.add_argument("--layer-norm-fp32", action="store_true",
                   help="try to mimic lc0's trt_layer_norm_fp32_fallback (layer-name glob "
                        "support in --layerPrecisions varies by TensorRT version)")
    b.add_argument("--opt-level", type=int, default=5)
    b.add_argument("--batch", type=int, default=256)
    b.add_argument("--min-batch", type=int, default=4)
    b.add_argument("--opt-batch", type=int, default=256, help="lc0 uses max_batch/4 = 256")
    b.add_argument("--max-batch", type=int, default=1024)
    l = sub.add_parser("layers", help="summarise existing trtexec exports")
    l.add_argument("--info", required=True)
    l.add_argument("--profile", default=None)
    l.add_argument("--out", required=True)
    k = sub.add_parser("kernels")
    k.add_argument("--csv", required=True)
    k.add_argument("--out", default=None)
    a = ap.parse_args()
    {"build": cmd_build, "layers": cmd_layers, "kernels": cmd_kernels}[a.cmd](a)


if __name__ == "__main__":
    main()
