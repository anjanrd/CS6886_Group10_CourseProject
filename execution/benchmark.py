"""Checked one-trial NN/search measurement; only main() launches an engine.

Reuses imported positions, parser and search/energy implementation unchanged.
Each trial gets its own directory; the outer controller owns resumption.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "hw"))
sys.path.insert(0, str(ROOT / "eval"))

from hw.bench import SEARCH_FENS, parse_backendbench, warn_lines
from hw.energy import Meter, describe_env, idle_power
from eval.lc0uci import Lc0
from execution.support import atomic_json, checked_result, identity, positive, preserve_command, require_session


def validate_backend_rows(rows, expected_batch):
    if len(rows) != 1 or rows[0]["batch"] != expected_batch:
        raise ValueError(f"Expected exactly batch {expected_batch}; got {rows}")
    positive(rows[0].get("nps"), "NN positions/s")
    positive(rows[0].get("ms"), "NN batch latency")
    return rows[0]


def validate_search(result, expected_positions):
    for field in ("nodes", "time_ms", "nps"):
        positive(result.get(field), "search " + field)
    if result.get("positions") != expected_positions:
        raise ValueError("Search position count mismatch")
    energy = result.get("energy", {})
    for field in ("seconds", "gpu_j", "gpu_j_above_idle", "temp_c_max", "sm_mhz_mean"):
        positive(energy.get(field), "search energy/telemetry " + field)
    positive(result.get("gpu_j_per_node"), "GPU joules/node above idle")


def checked_search(a, idle, out):
    """Imported search-bench loop with per-position missing-data checks added."""
    opts={"MoveOverheadMs":20,"MinibatchSize":a.minibatch}
    if a.threads:
        opts["Threads"]=a.threads
    e=Lc0(a.lc0,a.weights,backend=a.backend,backend_opts=a.backend_opts,
          options=opts,log=str(out/"search.lc0.log"),name=a.name)
    positions=[]
    try:
        e.go(fen=SEARCH_FENS[0],movetime=min(a.movetime,2000))
        with Meter() as meter:
            for fen in SEARCH_FENS[:a.positions]:
                e.newgame()
                row=e.go(fen=fen,movetime=a.movetime)
                for field in ("nodes","time_ms","wall"):
                    positive(row.get(field),"per-position "+field)
                positions.append({"fen":fen,**{k:row[k] for k in ("nodes","time_ms","wall","bestmove")}})
    finally:
        e.quit()
    energy=dict(meter.result)
    positive(energy.get("gpu_j"),"NVML GPU energy")
    energy["gpu_j_above_idle"]=energy["gpu_j"]-idle["gpu_w"]*energy["seconds"]
    nodes=sum(r["nodes"] for r in positions)
    ms=sum(r["time_ms"] for r in positions)
    return {"nodes":nodes,"time_ms":ms,"nps":round(1000*nodes/ms),
            "wall_nps":nodes/energy["seconds"],"positions":len(positions),
            "movetime":a.movetime,"energy":energy,"warnings":e.warnings,
            "gpu_j_per_node":energy["gpu_j_above_idle"]/nodes,
            "gpu_j_per_node_total":energy["gpu_j"]/nodes,"per_position":positions}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lc0", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--backend-opts", required=True)
    ap.add_argument("--backend", default="onnx-trt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--positions", type=int, default=10)
    ap.add_argument("--movetime", type=int, default=10000)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--minibatch", type=int, default=64)
    ap.add_argument("--batches", type=int, default=200)
    ap.add_argument("--conditioning-seconds",type=float,default=30)
    a = ap.parse_args()
    require_session(a.session)
    if not 1 <= a.positions <= len(SEARCH_FENS):
        raise ValueError("Use the existing fixed benchmark positions")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    base = [a.lc0, "backendbench", f"--weights={Path(a.weights).resolve()}",
            f"--backend={a.backend}", f"--backend-opts={a.backend_opts}"]
    warm = preserve_command(base + ["--batches=5", "--start-batch-size=64", "--max-batch-size=64"],
                            out / "warmup.log")
    checked_result(warm, "lc0 engine build/warm-up")
    start=time.perf_counter();iteration=0
    while time.perf_counter()-start<a.conditioning_seconds:
        p=preserve_command(base+["--batches=200","--start-batch-size=64","--max-batch-size=64"],
                           out/f"conditioning-{iteration:03d}.log")
        checked_result(p,"common conditioning")
        iteration+=1
    conditioning=time.perf_counter()-start
    idle = idle_power(10)
    positive(idle.get("gpu_w"), "NVML idle power")
    network = []
    warnings = []
    for batch in (32, 64, 128, 256):
        log = out / f"backend-b{batch}.log"
        p = preserve_command(base + [f"--batches={a.batches}", f"--start-batch-size={batch}",
                                     f"--max-batch-size={batch}", "--batch-step=32"], log)
        checked_result(p, f"backendbench batch {batch}")
        txt = log.read_text(encoding="utf-8", errors="replace")
        network.append(validate_backend_rows(parse_backendbench(txt), batch))
        warnings += warn_lines(txt)
    # Fresh idle observation for the search trial; load/build stays outside its meter.
    idle_search = idle_power(10)
    positive(idle_search.get("gpu_w"), "NVML search idle power")
    search = checked_search(a,idle_search,out)
    validate_search(search, a.positions)
    result = {"name": a.name, "model": identity(a.weights), "backend": a.backend,
              "backend_opts": a.backend_opts, "env": describe_env(), "idle": idle,
              "idle_search": idle_search, "network": network, "search": search,
              "threads": a.threads, "minibatch": a.minibatch,
              "conditioning_wall_seconds":conditioning,
              "positions": SEARCH_FENS[:a.positions], "warnings": warnings,
              "precision_coverage": "not inferred from flags or throughput"}
    atomic_json(out / "result.json", result)
    print(json.dumps({"name": a.name, "nps": search["nps"], "positions": a.positions}))


if __name__ == "__main__":
    main()
