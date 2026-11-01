#!/usr/bin/env python3
"""Score Studies A/B/C against the tablebase labels.

Per model:
  Study A  raw tablebase accuracy (+95% Wilson CI), DTZ-optimal rate,
           KL(ref || model) over legal moves, top-1 agreement, mean |ΔV|.
           KL is approximate below ~1e-4 (trap 7: lc0 prints P to 0.01%).
  Study B  accuracy at each node budget with paired counts vs the reference
           (−worse / +better) and an exact McNemar p, mean actual nodes,
           and node-equivalents: FP16 nodes needed to reach the model's
           accuracy ÷ the model's nodes (1.00 = matches FP16; 0 = below
           FP16's raw net; inf = above FP16's largest budget).
  Study C  accuracy at each movetime, paired vs reference, mean nodes and
           the nodes ratio vs the reference (the realised speed-up).

Rows are scored on positions that the model AND the reference finished at
every budget (the `n` column); the header warns when rows differ in n.
The hard subset (reference's raw top move wrong, or under 80% raw policy
mass on correct moves) is scored the same way into <out>/hard/.

    python eval/score.py --tbset data/tbset.jsonl --results results --ref fp16 \
        --models fp16,s1,s2,s3,s4 --out reports
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import read_jsonl, wilson  # noqa: E402


def load(results, name):
    d = Path(results) / name
    A = {r["id"]: r for r in read_jsonl(d / "study_a.jsonl")}
    B, C = {}, {}
    for r in read_jsonl(d / "study_b.jsonl"):
        B.setdefault(r["budget"], {})[r["id"]] = r
    for r in read_jsonl(d / "study_c.jsonl"):
        C.setdefault(r["movetime"], {})[r["id"]] = r
    return A, B, C


def mcnemar_p(worse, better):
    n = worse + better
    if n == 0:
        return 1.0
    k = min(worse, better)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def kl(p, q):
    keys = sorted(set(p) | set(q))
    a = np.array([max(p.get(k, 0.0), 1e-6) for k in keys])
    b = np.array([max(q.get(k, 0.0), 1e-6) for k in keys])
    a /= a.sum()
    b /= b.sum()
    return float(np.sum(a * np.log(a / b)))


def ne_curve(ref_points):
    """ref_points: [(mean_nodes, acc)] incl. raw (1 node). Monotone envelope."""
    pts = sorted(ref_points)
    xs = [math.log(max(n, 1)) for n, _ in pts]
    ys, best = [], -1
    for _, acc in pts:
        best = max(best, acc)
        ys.append(best)
    return xs, ys


def nodes_for(acc, xs, ys, hint):
    """FP16 nodes needed to reach `acc` on the envelope (xs = log nodes).
    Where the envelope is flat at `acc` the inverse is an interval; take the
    point of it nearest the model's own node count (`hint`), so a model that
    matches a saturated FP16 gets 1.00, not an artefact of the flat stretch."""
    eps = 1e-12
    if acc < ys[0] - eps:
        return 0.0
    if acc > ys[-1] + eps:
        return float("inf")

    def interp(j, target):
        if ys[j] == ys[j - 1]:
            return xs[j - 1]
        t = (target - ys[j - 1]) / (ys[j] - ys[j - 1])
        return xs[j - 1] + t * (xs[j] - xs[j - 1])

    j = next(k for k in range(len(ys)) if ys[k] >= acc - eps)
    lo = xs[0] if j == 0 else interp(j, acc)
    j2 = next((k for k in range(len(ys)) if ys[k] > acc + eps), None)
    hi = float("inf") if j2 is None else (xs[0] if j2 == 0 else interp(j2, acc))
    x = min(max(math.log(max(hint, 1)), lo), hi)
    return math.exp(x)


def fmt_ci(k, n):
    lo, hi = wilson(k, n)
    return f"{k / n:.4f} [{lo:.3f},{hi:.3f}]" if n else "n/a"


def score(tb, data, ref, models, ids_filter=None):
    """Returns (markdown lines, json dict)."""
    rA, rB, rC = data[ref]
    budgets = sorted(rB)
    movetimes = sorted(rC)
    out = {"ref": ref, "models": {}}
    L = []

    # ---------------- Study A
    L.append("## Study A: raw network (budget 0)\n")
    L.append("| model | n | tb accuracy [95% CI] | DTZ-optimal | KL vs ref | top-1 agree | mean abs dV |")
    L.append("|---|---|---|---|---|---|---|")
    ns = set()
    for m in models:
        A = data[m][0]
        ids = [i for i in A if i in rA and i in tb and (ids_filter is None or i in ids_filter)]
        n = len(ids)
        if not n:
            continue
        ns.add(n)
        acc = sum(A[i]["best"] in tb[i]["correct"] for i in ids)
        dtz = sum(A[i]["best"] in tb[i]["dtz_opt"] for i in ids)
        kls = [kl(rA[i]["p"], A[i]["p"]) for i in ids]
        t1 = sum(max(A[i]["p"], key=A[i]["p"].get) == max(rA[i]["p"], key=rA[i]["p"].get) for i in ids)
        dv = [abs((A[i]["v"] or 0) - (rA[i]["v"] or 0)) for i in ids
              if A[i].get("v") is not None and rA[i].get("v") is not None]
        L.append(f"| {m} | {n} | {fmt_ci(acc, n)} | {dtz / n:.4f} | {np.mean(kls):.2e} | "
                 f"{t1 / n:.4f} | {np.mean(dv) if dv else float('nan'):.4f} |")
        out["models"].setdefault(m, {})["A"] = {"n": n, "acc": acc / n, "dtz_opt": dtz / n,
                                               "kl": float(np.mean(kls)), "top1": t1 / n,
                                               "dv": float(np.mean(dv)) if dv else None}
    if len(ns) > 1:
        L.insert(1, f"> WARNING: rows have different n {sorted(ns)} (interrupted run?)\n")

    # ---------------- Study B
    if budgets:
        L.append("\n## Study B: fixed node budgets\n")
        L.append("Accuracy (paired vs ref: −worse/+better, McNemar p), mean actual nodes.\n")
        L.append("| model | n | raw | " + " | ".join(f"{b} nodes" for b in budgets) + " |")
        L.append("|---|---|---|" + "---|" * len(budgets))
        ref_ids = None
        ne_rows = []
        for m in models:
            A, B, _ = data[m]
            ids = set(i for i in rA if i in tb and (ids_filter is None or i in ids_filter))
            ids &= set(A)
            for b in budgets:
                ids &= set(B.get(b, {})) & set(rB[b])
            ids = sorted(ids)
            if not ids:
                continue
            if m == ref:
                ref_ids = ids
            cells, mres = [], {}
            raw = sum(A[i]["best"] in tb[i]["correct"] for i in ids) / len(ids)
            pts = []
            for b in budgets:
                ok = [B[b][i]["best"] in tb[i]["correct"] for i in ids]
                rok = [rB[b][i]["best"] in tb[i]["correct"] for i in ids]
                worse = sum(r and not o for o, r in zip(ok, rok))
                better = sum(o and not r for o, r in zip(ok, rok))
                nodes = float(np.mean([B[b][i]["nodes"] or b for i in ids]))
                acc = sum(ok) / len(ids)
                pts.append((nodes, acc))
                s = f"{acc:.4f}"
                if m != ref:
                    s += f" (−{worse}/+{better}, p={mcnemar_p(worse, better):.2g})"
                s += f" @{nodes:.0f}"
                cells.append(s)
                mres[b] = {"acc": acc, "worse": worse, "better": better,
                           "p": mcnemar_p(worse, better), "nodes": nodes}
            L.append(f"| {m} | {len(ids)} | {raw:.4f} | " + " | ".join(cells) + " |")
            out["models"].setdefault(m, {})["B"] = {"n": len(ids), "raw": raw, "budgets": mres}
            ne_rows.append((m, ids, raw, pts))
        # node-equivalents, each against FP16 on the same ids
        L.append("\n### Node-equivalents (FP16 nodes needed ÷ model nodes)\n")
        L.append("| model | " + " | ".join(f"{b}" for b in budgets) + " |")
        L.append("|---|" + "---|" * len(budgets))
        for m, ids, raw, pts in ne_rows:
            rraw = sum(rA[i]["best"] in tb[i]["correct"] for i in ids) / len(ids)
            rpts = [(1.0, rraw)]
            for b in budgets:
                rpts.append((float(np.mean([rB[b][i]["nodes"] or b for i in ids])),
                             sum(rB[b][i]["best"] in tb[i]["correct"] for i in ids) / len(ids)))
            xs, ys = ne_curve(rpts)
            cells = []
            for (nodes, acc) in pts:
                need = nodes_for(acc, xs, ys, nodes)
                cells.append("inf" if need == float("inf") else f"{need / nodes:.2f}")
            L.append(f"| {m} | " + " | ".join(cells) + " |")
            out["models"][m].setdefault("B", {})["node_equiv"] = cells

    # ---------------- Study C
    if movetimes:
        L.append("\n## Study C: fixed time per move\n")
        L.append("| model | " + " | ".join(f"{t} ms" for t in movetimes) + " |")
        L.append("|---|" + "---|" * len(movetimes))
        for m in models:
            C = data[m][2]
            cells, res = [], {}
            for t in movetimes:
                ids = sorted(i for i in C.get(t, {}) if i in rC[t] and i in tb
                             and (ids_filter is None or i in ids_filter))
                if not ids:
                    cells.append("–")
                    continue
                ok = [C[t][i]["best"] in tb[i]["correct"] for i in ids]
                rok = [rC[t][i]["best"] in tb[i]["correct"] for i in ids]
                worse = sum(r and not o for o, r in zip(ok, rok))
                better = sum(o and not r for o, r in zip(ok, rok))
                nodes = float(np.mean([C[t][i]["nodes"] or 0 for i in ids]))
                rnodes = float(np.mean([rC[t][i]["nodes"] or 0 for i in ids]))
                acc = sum(ok) / len(ids)
                s = f"{acc:.4f} n={len(ids)}"
                if m != ref:
                    s += f" (−{worse}/+{better}, p={mcnemar_p(worse, better):.2g})"
                s += f" @{nodes:.0f} nodes (×{nodes / max(rnodes, 1e-9):.2f})"
                cells.append(s)
                res[t] = {"n": len(ids), "acc": acc, "worse": worse, "better": better,
                          "nodes": nodes, "nodes_ratio": nodes / max(rnodes, 1e-9)}
            L.append(f"| {m} | " + " | ".join(cells) + " |")
            out["models"].setdefault(m, {})["C"] = res
    return L, out


def by_class(tb, data, ref, models, keep):
    rA = {i: v for i, v in data[ref][0].items() if i in keep}
    classes = sorted({tb[i]["cls"] for i in rA if i in tb})
    L = ["\n## Raw accuracy by material class\n",
         "| class | n | " + " | ".join(models) + " |", "|---|---|" + "---|" * len(models)]
    for c in classes:
        ids = [i for i in rA if i in tb and tb[i]["cls"] == c]
        row = [f"{sum(data[m][0][i]['best'] in tb[i]['correct'] for i in ids if i in data[m][0]) / max(1, sum(i in data[m][0] for i in ids)):.3f}"
               for m in models]
        L.append(f"| {c} | {len(ids)} | " + " | ".join(row) + " |")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tbset", required=True)
    ap.add_argument("--results", default="results")
    ap.add_argument("--ref", default="fp16")
    ap.add_argument("--models", required=True, help="comma list; ref included or not")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--split", default="test",
                    help="score only positions of this tbset split (results files may hold several)")
    ap.add_argument("--tag", default=None, help="report name suffix (default: the split)")
    ap.add_argument("--hard-mass", type=float, default=0.8)
    a = ap.parse_args()
    a.tag = a.tag or a.split
    recs = read_jsonl(a.tbset)
    tb = {r["id"]: {"correct": set(r["correct"]), "dtz_opt": set(r["dtz_opt"]), "cls": r["cls"]}
          for r in recs}
    split_ids = {r["id"] for r in recs if r["split"] == a.split}
    models = [m for m in a.models.split(",") if m]
    if a.ref not in models:
        models = [a.ref] + models
    data = {m: load(a.results, m) for m in models}
    if not data[a.ref][0]:
        sys.exit(f"no Study A results for the reference {a.ref}")
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    L, js = score(tb, data, a.ref, models, ids_filter=split_ids)
    L = [f"# Scores ({a.tag}), reference = {a.ref}\n"] + L + by_class(tb, data, a.ref, models, split_ids)
    (out_dir / f"scores_{a.tag}.md").write_text("\n".join(L) + "\n")
    (out_dir / f"scores_{a.tag}.json").write_text(json.dumps(js, indent=1))

    rA = data[a.ref][0]
    hard = set()
    for i, r in rA.items():
        if i not in tb or i not in split_ids:
            continue
        mass = sum(p for mv, p in r["p"].items() if mv in tb[i]["correct"])
        if r["best"] not in tb[i]["correct"] or mass < a.hard_mass:
            hard.add(i)
    hd = out_dir / "hard"
    hd.mkdir(exist_ok=True)
    (hd / "ids.txt").write_text("\n".join(str(i) for i in sorted(hard)) + "\n")
    L2, js2 = score(tb, data, a.ref, models, ids_filter=hard)
    L2 = [f"# Hard subset ({a.tag}): {len(hard)} positions where {a.ref}'s raw top move is "
          f"wrong or <{a.hard_mass:.0%} raw policy mass is on correct moves\n"] + L2
    (hd / f"scores_{a.tag}.md").write_text("\n".join(L2) + "\n")
    (hd / f"scores_{a.tag}.json").write_text(json.dumps(js2, indent=1))
    print("\n".join(L))
    print(f"\nwrote {out_dir}/scores_{a.tag}.md and hard/ ({len(hard)} hard positions)")


if __name__ == "__main__":
    main()
