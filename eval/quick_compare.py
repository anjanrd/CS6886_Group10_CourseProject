#!/usr/bin/env python3
"""Seconds-fast ONNX Runtime comparison of models against a reference on an
.npz of planes (from make_calib.py). For choosing ladder knobs on the dev
split; NOT a reported result (ORT's int8 kernels are not TensorRT's).

Reports policy KL(ref||model) over legal moves, top-1 agreement, mean |ΔV|
(V = W - L), and, for tablebase positions, raw-policy tablebase accuracy.

    python eval/quick_compare.py --ref models/fp32.onnx --npz data/dev.npz \
        --tbset data/tbset.jsonl models/s1.onnx models/s2.onnx ...
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import INPUT, OUT_POLICY, OUT_WDL, ort_session, read_jsonl  # noqa: E402


def run(path, planes, cpu, bs=64):
    s = ort_session(path, cpu=cpu)
    pol, wdl = [], []
    for i in range(0, len(planes), bs):
        p, w = s.run([OUT_POLICY, OUT_WDL], {INPUT: planes[i:i + bs]})[:2]
        pol.append(p)
        wdl.append(w)
    return np.concatenate(pol), np.concatenate(wdl)


def legal_softmax(pol, legal):
    out = []
    for p, idx in zip(pol, legal):
        idx = idx[idx >= 0]
        lg = p[idx].astype(np.float64)
        e = np.exp(lg - lg.max())
        out.append(e / e.sum())
    return out


def compare(ref, mod, wref, wmod, ucis, tb, tb_ids):
    kl, agree, dv, acc, n_tb = [], 0, [], 0, 0
    for i, (p, q) in enumerate(zip(ref, mod)):
        kl.append(float(np.sum(p * (np.log(np.maximum(p, 1e-12)) - np.log(np.maximum(q, 1e-12))))))
        agree += int(np.argmax(p) == np.argmax(q))
        dv.append(abs((wref[i][0] - wref[i][2]) - (wmod[i][0] - wmod[i][2])))
        if tb is not None and tb_ids[i] in tb:
            n_tb += 1
            acc += int(ucis[i][int(np.argmax(q))] in tb[tb_ids[i]])
    return {"kl": float(np.mean(kl)), "top1": agree / len(ref), "dv": float(np.mean(dv)),
            "tb_acc": acc / n_tb if n_tb else float("nan"), "n_tb": n_tb}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--tbset", default=None)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", default=None, help="also write the table here")
    ap.add_argument("models", nargs="+")
    a = ap.parse_args()
    d = np.load(a.npz)
    planes = d["planes"].astype(np.float32)
    legal = d["legal_idx"]
    ucis = [str(u).split() for u in d["legal_uci"]]
    tb = None
    if a.tbset:
        tb = {r["id"]: set(r["correct"]) for r in read_jsonl(a.tbset)}
    tb_ids = d["tb_id"]
    rp, rw = run(a.ref, planes, a.cpu)
    ref = legal_softmax(rp, legal)
    rows = [("ref " + Path(a.ref).name, compare(ref, ref, rw, rw, ucis, tb, tb_ids))]
    for m in a.models:
        mp, mw = run(m, planes, a.cpu)
        rows.append((Path(m).name, compare(ref, legal_softmax(mp, legal), rw, mw, ucis, tb, tb_ids)))
    lines = [f"{len(planes)} positions from {a.npz} (ORT, not TensorRT)",
             f"{'model':28s} {'KL':>9s} {'top1':>6s} {'|dV|':>7s} {'tb_acc':>7s}"]
    for name, r in rows:
        lines.append(f"{name:28s} {r['kl']:9.2e} {r['top1']:6.3f} {r['dv']:7.4f} {r['tb_acc']:7.3f}")
    print("\n".join(lines))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
