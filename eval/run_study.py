#!/usr/bin/env python3
"""Studies A, B and C on the tablebase set, through stock lc0 over UCI.

  A  (budget 0)       raw network: `go nodes 1`, PolicyTemperature=1, P of
                      every legal move + raw V/WL/D of the root.
  B  (--budgets)      fixed node budgets, SmartPruningFactor=0 so the search
                      really spends the budget. Actual nodes are recorded
                      (trap 6: `go nodes N` overshoots by up to a minibatch).
  C  (--movetimes)    fixed time per move with the engine's normal settings
                      (smart pruning on). A fresh engine per movetime so NN
                      cache entries from one pass cannot speed up the next.

Resumable: each (position, budget) is one jsonl line, finished ones are
skipped. --shard i/n splits the positions for parallel machines.

    python eval/run_study.py --name fp16 --weights models/fp16.pb.gz --backend-opts optimize=6 \
        --tbset data/tbset.jsonl --split test --budgets 0,100,400,800,1600
    python eval/run_study.py --name s4 --weights models/s4.pb.gz --backend-opts optimize=8 \
        --tbset data/tbset.jsonl --split test --movetimes 200,800 --threads 0
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import append_jsonl, read_jsonl, write_json  # noqa: E402
from lc0uci import Lc0  # noqa: E402


def positions(a):
    recs = [r for r in read_jsonl(a.tbset) if r["split"] == a.split]
    recs.sort(key=lambda r: r["id"])
    if a.classes:
        keep = set(a.classes.split(","))
        recs = [r for r in recs if r["cls"] in keep]
    if a.ids:
        keep = {int(x) for x in Path(a.ids).read_text().split()}
        recs = [r for r in recs if r["id"] in keep]
    if a.limit:
        # spread the limit over classes instead of taking the first classes
        by = {}
        for r in recs:
            by.setdefault(r["cls"], []).append(r)
        out, i = [], 0
        while len(out) < a.limit and any(by.values()):
            for c in list(by):
                if by[c]:
                    out.append(by[c].pop(0))
                    if len(out) >= a.limit:
                        break
            i += 1
        recs = sorted(out, key=lambda r: r["id"])
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        recs = [r for k, r in enumerate(recs) if k % n == i]
    return recs


def make_engine(a, log_dir, tag, study):
    opts = {"VerboseMoveStats": True, "MinibatchSize": a.minibatch}
    if a.nncache is not None:
        opts["NNCacheSize"] = a.nncache
    if study in ("A", "B"):
        opts["Threads"] = a.threads if a.threads else 1
    elif a.threads:
        opts["Threads"] = a.threads
    if study == "A":
        opts["PolicyTemperature"] = 1.0
    if study == "B":
        opts["SmartPruningFactor"] = 0
    if study == "C":
        # Trap 8: lc0 subtracts MoveOverheadMs (default 200 ms) from every
        # movetime, so `go movetime 200` would search for ~0 ms.
        opts["MoveOverheadMs"] = a.move_overhead
    for kv in a.option or []:
        k, v = kv.split("=", 1)
        opts[k] = v
    return Lc0(a.lc0, a.weights, backend=a.backend, backend_opts=a.backend_opts,
               options=opts, log=str(log_dir / f"{tag}.lc0.log"), name=f"{a.name}:{tag}")


class Progress:
    def __init__(self, total, label):
        self.total, self.label, self.t0, self.n = total, label, time.time(), 0

    def tick(self):
        self.n += 1
        if self.n % 25 == 0 or self.n == self.total:
            el = time.time() - self.t0
            eta = el / self.n * (self.total - self.n)
            print(f"  [{self.label}] {self.n}/{self.total}  {el / self.n:.2f}s/item  "
                  f"eta {eta / 60:.1f} min", file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lc0", default="lc0")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--backend", default="onnx-trt")
    ap.add_argument("--backend-opts", default="")
    ap.add_argument("--threads", type=int, default=0,
                    help="0: 1 for studies A/B, lc0's default for C")
    ap.add_argument("--minibatch", type=int, default=256)
    ap.add_argument("--nncache", type=int, default=None)
    ap.add_argument("--option", action="append", help="extra UCI option Name=Value")
    ap.add_argument("--tbset", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--classes", default=None)
    ap.add_argument("--ids", "--ids-file", dest="ids", default=None, help="file of position ids to run")
    ap.add_argument("--shard", default=None, help="i/n")
    ap.add_argument("--budgets", default="", help="e.g. 0,100,400,800,1600 (0 = Study A)")
    ap.add_argument("--movetimes", default="", help="ms per move, e.g. 200,800 (Study C)")
    ap.add_argument("--move-overhead", type=int, default=20,
                    help="MoveOverheadMs for Study C (lc0 default 200 eats short movetimes)")
    ap.add_argument("--out-dir", default="results")
    a = ap.parse_args()

    out = Path(a.out_dir) / a.name
    out.mkdir(parents=True, exist_ok=True)
    recs = positions(a)
    budgets = [int(x) for x in a.budgets.split(",") if x != ""]
    movetimes = [int(x) for x in a.movetimes.split(",") if x != ""]
    meta = {"name": a.name, "weights": a.weights, "backend": a.backend,
            "backend_opts": a.backend_opts, "threads": a.threads, "minibatch": a.minibatch,
            "split": a.split, "n_positions": len(recs), "move_overhead": a.move_overhead, "host": platform.node(),
            "started": time.strftime("%Y-%m-%d %H:%M:%S"), "warnings": []}
    print(f"{a.name}: {len(recs)} positions, budgets {budgets}, movetimes {movetimes}",
          file=sys.stderr)

    def done_keys(path, key):
        return {(r["id"], r[key]) for r in read_jsonl(path)}

    # ---------------- Study A
    if 0 in budgets:
        path = out / "study_a.jsonl"
        done = done_keys(path, "budget")
        todo = [r for r in recs if (r["id"], 0) not in done]
        if todo:
            e = make_engine(a, out, "A", "A")
            meta["warnings"] += e.warnings
            pr = Progress(len(todo), f"{a.name} A")
            for r in todo:
                e.newgame()
                res = e.go(fen=r["fen"], nodes=1)
                append_jsonl(path, {"id": r["id"], "budget": 0, "best": res["bestmove"],
                                    "p": {m: round(v["p"], 6) for m, v in res["moves"].items()},
                                    "wl": res.get("wl"), "d": res.get("d"), "v": res.get("v"),
                                    "m": res.get("m"), "nodes": res["nodes"]})
                pr.tick()
            e.quit()

    # ---------------- Study B
    bb = sorted(b for b in budgets if b > 0)
    if bb:
        path = out / "study_b.jsonl"
        done = done_keys(path, "budget")
        todo = [r for r in recs if any((r["id"], b) not in done for b in bb)]
        if todo:
            e = make_engine(a, out, "B", "B")
            meta["warnings"] += e.warnings
            pr = Progress(len(todo), f"{a.name} B")
            for r in todo:
                for b in bb:  # ascending; same-position cache reuse only saves time
                    if (r["id"], b) in done:
                        continue
                    e.newgame()
                    res = e.go(fen=r["fen"], nodes=b)
                    append_jsonl(path, {"id": r["id"], "budget": b, "best": res["bestmove"],
                                        "nodes": res["nodes"], "time_ms": res["time_ms"],
                                        "wall": round(res["wall"], 4),
                                        "score_cp": res["score_cp"], "depth": res["depth"]})
                pr.tick()
            e.quit()

    # ---------------- Study C
    if movetimes:
        path = out / "study_c.jsonl"
        done = done_keys(path, "movetime")
        for mt in movetimes:
            todo = [r for r in recs if (r["id"], mt) not in done]
            if not todo:
                continue
            e = make_engine(a, out, f"C{mt}", "C")
            meta["warnings"] += e.warnings
            e.go(fen=todo[0]["fen"], movetime=mt)  # warm-up, not recorded
            pr = Progress(len(todo), f"{a.name} C{mt}")
            for r in todo:
                e.newgame()
                res = e.go(fen=r["fen"], movetime=mt)
                append_jsonl(path, {"id": r["id"], "movetime": mt, "best": res["bestmove"],
                                    "nodes": res["nodes"], "time_ms": res["time_ms"],
                                    "wall": round(res["wall"], 4), "depth": res["depth"]})
                pr.tick()
            e.quit()

    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    hist = out / "meta.jsonl"
    append_jsonl(hist, meta)
    write_json(out / "meta.json", meta)
    if any("fp32" in w.lower() for w in meta["warnings"]):
        print("!!! lc0 warned about FP32 (trap 3): this run is not a valid FP16/INT8 "
              "measurement. Check the onnxruntime version.", file=sys.stderr)


if __name__ == "__main__":
    main()
