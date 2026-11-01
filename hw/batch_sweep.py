#!/usr/bin/env python3
"""MCTS batch-size sweep: search nps and GPU energy per node for each
MinibatchSize, for one or more models, through timed UCI searches
(hw/bench.py search_bench; `lc0 benchmark` crashes on V2 nets, trap 11).

By default the TensorRT engine is the same dynamic-shape engine every other
experiment uses (lc0 builds it for batches 4..1024, optimised at 256), so
only the search's batching changes. --fixed-batch instead passes
backend-opts batch=N, which builds one engine per batch size optimised for
exactly N (minutes each the first time; cached afterwards).

    python hw/batch_sweep.py --lc0 lc0 --sizes 32,64,128,256 \
        --model fp16=models/fp16.pb.gz:optimize=6 --model s4=models/s4.pb.gz:optimize=8
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from bench import search_bench  # noqa: E402
from energy import describe_env, idle_power  # noqa: E402
from match import parse_engine  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lc0", default="lc0")
    ap.add_argument("--model", action="append", required=True,
                    help="name=weights[:backend_opts][@backend]")
    ap.add_argument("--backend", default="onnx-trt")
    ap.add_argument("--sizes", default="32,64,128,256")
    ap.add_argument("--movetime", type=int, default=10000)
    ap.add_argument("--positions", type=int, default=6)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--fixed-batch", action="store_true")
    ap.add_argument("--idle-seconds", type=float, default=10)
    ap.add_argument("--out", default="reports/batch_sweep")
    a = ap.parse_args()
    sizes = [int(x) for x in a.sizes.split(",")]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    idle = None
    for spec in a.model:
        m = parse_engine(spec)
        for mb in sizes:
            opts = m["opts"]
            if a.fixed_batch:
                opts = ",".join(x for x in (opts, f"batch={mb}") if x)
            if idle is None:
                idle = idle_power(a.idle_seconds)
            b = search_bench(a.lc0, m["weights"], m["backend"] or a.backend, opts, a.movetime,
                             a.positions, idle, a.threads, mb,
                             log=str(out / f"{m['name']}_mb{mb}.lc0.log"), name=f"{m['name']}:{mb}")
            e = b["energy"]
            r = {"model": m["name"], "minibatch": mb, "fixed_batch": a.fixed_batch,
                 "nps": b.get("nps"), "nodes": b.get("nodes"), "seconds": e.get("seconds"),
                 "gpu_j_above_idle": e.get("gpu_j_above_idle"),
                 "mj_per_node": 1e3 * b["gpu_j_per_node"] if "gpu_j_per_node" in b else None,
                 "sm_mhz_mean": e.get("sm_mhz_mean"), "power_w_mean": e.get("power_w_mean"),
                 "temp_c_max": e.get("temp_c_max")}
            rows.append(r)
            print(f"{m['name']:10s} minibatch {mb:5d}: {r['nps']} nps, "
                  f"{r['mj_per_node'] if r['mj_per_node'] is not None else float('nan'):.3f} mJ/node",
                  file=sys.stderr, flush=True)
    with open(out / "sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out / "sweep.json").write_text(json.dumps({"env": describe_env(), "idle": idle, "rows": rows}, indent=1))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
        for name in dict.fromkeys(r["model"] for r in rows):
            rr = [r for r in rows if r["model"] == name]
            axs[0].plot([r["minibatch"] for r in rr], [r["nps"] or 0 for r in rr], "o-", label=name)
            if any(r["mj_per_node"] is not None for r in rr):
                axs[1].plot([r["minibatch"] for r in rr], [r["mj_per_node"] or float("nan") for r in rr],
                            "o-", label=name)
        for ax, yl in zip(axs, ("search nodes/s", "GPU mJ/node above idle")):
            ax.set_xscale("log", base=2)
            ax.set_xlabel("MinibatchSize")
            ax.set_ylabel(yl)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "sweep.png", dpi=150)
    except Exception as e:
        print(f"(no plot: {e})", file=sys.stderr)
    print(f"wrote {out}/sweep.csv", file=sys.stderr)


if __name__ == "__main__":
    main()
