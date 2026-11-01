#!/usr/bin/env python3
"""Throughput and energy for one model: `lc0 backendbench` (raw NN nps per
batch size) and timed UCI searches (search nps), with GPU energy from NVML
(CPU energy from RAPL on Linux) above idle, and clocks/temperature logged.

J/node = energy above idle during the searches ÷ nodes searched. Only the
searches are metered (engine load and TensorRT engine build are not).

Trap 11: `lc0 benchmark`'s built-in positions include FENs with a nonzero
halfmove clock, which crash lc0 v0.33 with V2 canonical nets (segfault on
position 4; same root cause as trap 5). So search speed is measured here
over UCI on SEARCH_FENS, a fixed list with halfmove clock 0.

    python hw/bench.py --lc0 lc0 --weights models/fp16.pb.gz --backend-opts optimize=6 --name fp16
    python hw/bench.py --lc0 lc0 --weights nets/BT3.pb.gz --backend cuda-fp16 --name native_cuda

Lock clocks first if you can (`nvidia-smi -lgc 1500,1500`, admin); laptops
and some desktop drivers refuse, so the SM clock is always logged instead.
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from energy import Meter, describe_env, idle_power, run_metered  # noqa: E402

# Openings, middlegames and endgames (Stockfish bench positions), halfmove 0.
SEARCH_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 10",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 11",
    "4rrk1/pp1n3p/3q2pQ/2p1pb2/2PP4/2P3N1/P2B2PP/4RRK1 b - - 0 19",
    "rq3rk1/ppp2ppp/1bnpb3/3N2B1/3NP3/7P/PPPQ1PP1/2KR3R w - - 0 14",
    "r1bq1r1k/1pp1n1pp/1p1p4/4p2Q/4Pp2/1BNP4/PPP2PPP/3R1RK1 w - - 0 14",
    "r3r1k1/2p2ppp/p1p1bn2/8/1q2P3/2NPQN2/PPP3PP/R4RK1 b - - 0 15",
    "r1bbk1nr/pp3p1p/2n5/1N4p1/2Np1B2/8/PPP2PPP/2KR1B1R w kq - 0 13",
    "r1bq1rk1/ppp1nppp/4n3/3p3Q/3P4/1BP1B3/PP1N2PP/R4RK1 w - - 0 16",
    "4r1k1/r1q2ppp/ppp2n2/4P3/5Rb1/1N1BQ3/PPP3PP/R5K1 w - - 0 17",
    "2rqkb1r/ppp2p2/2npb1p1/1N1Nn2p/2P1PP2/8/PP2B1PP/R1BQK2R b KQ - 0 11",
    "r1bq1r1k/b1p1npp1/p2p3p/1p6/3PP3/1B2NN2/PP3PPP/R2Q1RK1 w - - 0 16",
]


def search_bench(lc0, weights, backend, backend_opts, movetime, positions, idle=None,
                 threads=0, minibatch=0, log=None, name="bench"):
    """Timed `go movetime` searches over SEARCH_FENS[:positions]; energy is
    metered around the searches only. Returns a dict."""
    from lc0uci import Lc0
    opts = {"MoveOverheadMs": 20}
    if threads:
        opts["Threads"] = threads
    if minibatch:
        opts["MinibatchSize"] = minibatch
    e = Lc0(lc0, weights, backend=backend, backend_opts=backend_opts, options=opts,
            log=log, name=name)
    fens = SEARCH_FENS[:positions] if positions <= len(SEARCH_FENS) else \
        [SEARCH_FENS[i % len(SEARCH_FENS)] for i in range(positions)]
    try:
        e.go(fen=fens[0], movetime=min(movetime, 2000))  # warm-up, clocks up
        nodes, lc0_ms = 0, 0
        with Meter() as m:
            for f in fens:
                e.newgame()
                r = e.go(fen=f, movetime=movetime)
                nodes += r.get("nodes") or 0
                lc0_ms += r.get("time_ms") or 0
    finally:
        e.quit()
    en = dict(m.result)
    if idle:
        if "gpu_j" in en and "gpu_w" in idle:
            en["gpu_j_above_idle"] = en["gpu_j"] - idle["gpu_w"] * en["seconds"]
        if "cpu_j" in en and "cpu_w" in idle:
            en["cpu_j_above_idle"] = en["cpu_j"] - idle["cpu_w"] * en["seconds"]
    out = {"nodes": nodes, "time_ms": lc0_ms, "positions": len(fens), "movetime": movetime,
           "nps": round(1000 * nodes / lc0_ms) if lc0_ms else None, "energy": en,
           "warnings": e.warnings}
    if nodes and "gpu_j_above_idle" in en:
        out["gpu_j_per_node"] = en["gpu_j_above_idle"] / nodes
        out["gpu_j_per_node_total"] = en["gpu_j"] / nodes
    if nodes and "cpu_j_above_idle" in en:
        out["cpu_j_per_node"] = en["cpu_j_above_idle"] / nodes
    return out


def lc0_args(a):
    args = [f"--weights={a.weights}", f"--backend={a.backend}"]
    if a.backend_opts:
        args.append(f"--backend-opts={a.backend_opts}")
    return args


def parse_backendbench(txt):
    rows = []
    for line in txt.splitlines():
        m = re.match(r"\s*(\d+),\s*([\d.]+),\s*([\d.]+),", line)
        if m:
            rows.append({"batch": int(m.group(1)), "nps": float(m.group(2)),
                         "ms": float(m.group(3))})
    return rows


def parse_benchmark(txt):
    out = {}
    for key, pat in (("time_ms", r"Total time \(ms\)\s*:\s*(\d+)"),
                     ("nodes", r"Nodes searched\s*:\s*(\d+)"),
                     ("nps", r"Nodes/second\s*:\s*(\d+)")):
        m = re.search(pat, txt)
        if m:
            out[key] = int(m.group(1))
    return out


def warn_lines(txt):
    return [l for l in txt.splitlines() if re.search(r"warn|fp32|fall ?back", l, re.I)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lc0", default="lc0")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--backend", default="onnx-trt")
    ap.add_argument("--backend-opts", default="")
    ap.add_argument("--idle-seconds", type=float, default=10)
    ap.add_argument("--batches", type=int, default=200, help="backendbench batches per size")
    ap.add_argument("--start-batch", type=int, default=32)
    ap.add_argument("--max-batch", type=int, default=512)
    ap.add_argument("--batch-step", type=int, default=32)
    ap.add_argument("--movetime", type=int, default=10000, help="search ms per position")
    ap.add_argument("--positions", type=int, default=10)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--minibatch", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--skip-backendbench", action="store_true")
    ap.add_argument("--out-dir", default="reports/bench")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = {"name": a.name, "weights": a.weights, "backend": a.backend,
           "backend_opts": a.backend_opts, "env": describe_env(),
           "time": time.strftime("%Y-%m-%d %H:%M:%S")}

    # Warm-up: builds / loads the TensorRT engine so it is not timed.
    print(f"[{a.name}] warm-up (TensorRT engine build on first use can take minutes)",
          file=sys.stderr)
    w = subprocess.run([a.lc0, "backendbench", *lc0_args(a), "--batches=5",
                        "--start-batch-size=64", "--max-batch-size=64"],
                       capture_output=True, text=True)
    res["warnings"] = warn_lines(w.stdout + w.stderr)
    for l in res["warnings"]:
        print(f"[{a.name}] {l}", file=sys.stderr)
    if w.returncode != 0:
        sys.exit(f"lc0 failed:\n{w.stdout[-3000:]}\n{w.stderr[-3000:]}")

    time.sleep(3)
    res["idle"] = idle_power(a.idle_seconds)
    print(f"[{a.name}] idle {res['idle']}", file=sys.stderr)

    if not a.skip_backendbench:
        cmd = [a.lc0, "backendbench", *lc0_args(a), f"--batches={a.batches}",
               f"--start-batch-size={a.start_batch}", f"--max-batch-size={a.max_batch}",
               f"--batch-step={a.batch_step}"]
        p, e = run_metered(cmd, res["idle"])
        res["backendbench"] = {"rows": parse_backendbench(p.stdout + p.stderr), "energy": e}
        for r in res["backendbench"]["rows"]:
            print(f"  batch {r['batch']:5d}  {r['nps']:10.0f} nps", file=sys.stderr)

    runs = []
    for k in range(a.repeat):
        b = search_bench(a.lc0, a.weights, a.backend, a.backend_opts, a.movetime, a.positions,
                         res["idle"], a.threads, a.minibatch,
                         log=str(out / f"{a.name}.lc0.log"), name=a.name)
        e = b["energy"]
        runs.append(b)
        print(f"  search run {k + 1}: {b.get('nps')} nps, {b.get('nodes')} nodes, "
              f"GPU {b.get('gpu_j_per_node', float('nan')) * 1e3:.3f} mJ/node above idle, "
              f"SM {e.get('sm_mhz_mean', 0):.0f} MHz (min {e.get('sm_mhz_min', 0)}), "
              f"{e.get('temp_c_max', 0)} C max", file=sys.stderr)
    res["search"] = runs
    path = out / f"{a.name}.json"
    path.write_text(json.dumps(res, indent=1))
    print(f"wrote {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
