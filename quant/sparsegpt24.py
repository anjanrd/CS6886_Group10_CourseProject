#!/usr/bin/env python3
"""S1: SparseGPT 2:4 (Frantar & Alistarh, 2023) on every weight-carrying
MatMul/Gemm, no fine-tuning. Output is an FP32 ONNX with the same 2:4
pattern as S0, so speed is identical; only accuracy differs.

Per layer, with H = Xᵀ X from that layer's calibration inputs, columns are
processed left to right in blocks; in each group of 4 input channels the 2
weights with the smallest w² / [H⁻¹]_jj² are zeroed and the OBS update
spreads their error over the not-yet-processed columns.

Sequential (default): layers are pruned in dependency stages and each stage's
inputs are captured from the model with all earlier stages already pruned,
as in the paper. --one-shot captures everything from the dense model in one
pass (faster, slightly worse). Weights shared by several matmuls (smolgen's
global weight) get one H summed over all their consumers, from the dense
model, in the first stage. Conv weights (none in BT3) fall back to magnitude.

    python quant/sparsegpt24.py --onnx models/bt3_fp32.onnx --calib data/calib.npz \
        --out models/s1sp_fp32.onnx
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from onnx import numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import initializers, load_onnx, save_onnx  # noqa: E402
from quantize import load_planes, run_collect  # noqa: E402
from sparse_common import check_24, from_kn, magnitude_mask, targets, to_kn  # noqa: E402


def sparsegpt_24(w_kn, H, blocksize=128, percdamp=0.01):
    """w_kn: [K, N] (y = x @ W). Returns pruned [K, N]."""
    W = w_kn.T.astype(np.float64).copy()          # rows = outputs, cols = inputs
    K = W.shape[1]
    H = H.astype(np.float64).copy()
    dead = np.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H[np.diag_indices(K)] += percdamp * np.mean(np.diag(H))
    Hinv = np.linalg.cholesky(np.linalg.inv(H)).T  # upper Cholesky of H^-1
    bs = max(4, blocksize - blocksize % 4)
    for i1 in range(0, K, bs):
        i2 = min(i1 + bs, K)
        W1 = W[:, i1:i2].copy()
        Q1 = np.zeros_like(W1)
        E1 = np.zeros_like(W1)
        Hi1 = Hinv[i1:i2, i1:i2]
        dg = np.diag(Hi1)
        mask1 = np.zeros(W1.shape, dtype=bool)
        for i in range(i2 - i1):
            if i % 4 == 0:
                tmp = W1[:, i:i + 4] ** 2 / (dg[i:i + 4][None, :] ** 2)
                idx = np.argsort(tmp, axis=1)[:, :2]
                sub = mask1[:, i:i + 4]
                np.put_along_axis(sub, idx, True, axis=1)
            w = W1[:, i]
            q = w.copy()
            q[mask1[:, i]] = 0
            Q1[:, i] = q
            err = (w - q) / Hi1[i, i]
            W1[:, i:] -= np.outer(err, Hi1[i, i:])
            E1[:, i] = err
        W[:, i1:i2] = Q1
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return W.T.astype(np.float32)


def recon_err(w, wp, H):
    d = (w - wp).astype(np.float64)
    num = np.einsum("kn,kj,jn->", d, H, d)
    den = np.einsum("kn,kj,jn->", w.astype(np.float64), H, w.astype(np.float64))
    return float(num / max(den, 1e-30))


def upstream_targets(model, tmap):
    """tensor -> set of target initializers that influence it."""
    node_w = {}
    for w, nodes in tmap.items():
        for n in nodes:
            node_w[n.name] = w
    up = {}
    for n in model.graph.node:  # ONNX graphs are topologically sorted
        s = set()
        for i in n.input:
            s |= up.get(i, set())
        if n.name in node_w:
            s = s | {node_w[n.name]}
        for o in n.output:
            up[o] = s
    return up


def hessians(model, planes, need):
    """need: {weight: [input tensor names]} -> {weight: (H, rows)}"""
    tensors = sorted({t for ts in need.values() for t in ts})
    acc = {t: None for t in tensors}
    rows = {t: 0 for t in tensors}

    def fn(n, o):
        x = o.reshape(-1, o.shape[-1]).astype(np.float64)
        h = x.T @ x
        acc[n] = h if acc[n] is None else acc[n] + h
        rows[n] += x.shape[0]

    run_collect(model, planes, tensors, fn)
    out = {}
    for w, ts in need.items():
        H = sum(acc[t] for t in ts)
        r = sum(rows[t] for t in ts)
        out[w] = (2 * H / max(r, 1), r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--calib-n", type=int, default=0,
                    help="0 = all; smolgen's dense1 gets ONE row per position, so it needs many")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip", default=None)
    ap.add_argument("--one-shot", action="store_true")
    ap.add_argument("--blocksize", type=int, default=128)
    ap.add_argument("--damp", type=float, default=0.01)
    a = ap.parse_args()
    t0 = time.time()
    m = load_onnx(a.onnx)
    planes = load_planes(a.calib, a.calib_n)
    tmap = targets(m, a.skip)
    inits = initializers(m)
    rep = {}

    def prune(w, H, rows):
        nodes = tmap[w]
        op = nodes[0].op_type
        W = numpy_helper.to_array(inits[w]).astype(np.float32)
        kn = to_kn(W, op)
        mag = np.where(magnitude_mask(kn), kn, 0).astype(np.float32)
        if op == "Conv" or H is None:
            p = mag
            meth = "magnitude"
        else:
            if rows < 2 * kn.shape[0]:
                print(f"  warning: {w}: {rows} calibration rows for K={kn.shape[0]} inputs; "
                      "H is poorly conditioned, so SparseGPT may overfit this layer "
                      "(more calibration positions help)", file=sys.stderr)
            p = sparsegpt_24(kn, H, a.blocksize, a.damp)
            meth = "sparsegpt"
        assert check_24(p), w
        inits[w].CopyFrom(numpy_helper.from_array(from_kn(p, op, W.shape).astype(np.float32), w))
        r = {"nodes": [n.name for n in nodes], "method": meth, "rows": rows}
        if H is not None:
            r["recon_err"] = recon_err(kn, p, H)
            r["recon_err_magnitude"] = recon_err(kn, mag, H)
        rep[w] = r

    mm = {w: ns for w, ns in tmap.items() if ns[0].op_type != "Conv"}
    for w in set(tmap) - set(mm):
        prune(w, None, 0)
    shared = [w for w, ns in mm.items() if len(ns) > 1]
    single = [w for w, ns in mm.items() if len(ns) == 1]
    need = {w: [n.input[0] for n in mm[w]] for w in mm}

    if a.one_shot:
        Hs = hessians(m, planes, need)
        for w in mm:
            prune(w, *Hs[w])
        stages = 1
    else:
        done = set(tmap) - set(mm)   # convs were pruned (magnitude) above
        stages = 0
        if shared:
            Hs = hessians(m, planes, {w: need[w] for w in shared})
            for w in shared:
                prune(w, *Hs[w])
            done |= set(shared)
            stages += 1
        pending = list(single)
        while pending:
            up = upstream_targets(m, tmap)
            ready = [w for w in pending
                     if all(up.get(t, set()) <= done for t in need[w])]
            if not ready:
                raise RuntimeError("dependency cycle among targets: " + ", ".join(pending[:5]))
            Hs = hessians(m, planes, {w: need[w] for w in ready})
            for w in ready:
                prune(w, *Hs[w])
            done |= set(ready)
            pending = [w for w in pending if w not in done]
            stages += 1
            print(f"  stage {stages}: {len(ready)} layers, {len(pending)} left "
                  f"({time.time() - t0:.0f}s)", file=sys.stderr)
    save_onnx(m, a.out)
    errs = [(v["recon_err"], v["recon_err_magnitude"]) for v in rep.values() if "recon_err" in v]
    side = {"method": "sparsegpt 2:4", "one_shot": a.one_shot, "stages": stages,
            "n_calib": int(len(planes)), "blocksize": a.blocksize, "damp": a.damp,
            "skip": a.skip, "layers": rep, "seconds": round(time.time() - t0, 1)}
    with open(Path(a.out).with_suffix(".json"), "w") as f:
        json.dump(side, f, indent=1)
    if errs:
        e = np.array(errs)
        print(f"pruned {len(rep)} weights in {stages} stages; median layer reconstruction "
              f"error {np.median(e[:, 0]):.4f} (magnitude would be {np.median(e[:, 1]):.4f}) "
              f"-> {a.out} ({time.time() - t0:.0f}s)", file=sys.stderr)


if __name__ == "__main__":
    main()
