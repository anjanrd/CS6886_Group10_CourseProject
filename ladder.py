#!/usr/bin/env python3
"""One entry point for the whole pipeline. Works the same on Windows and Linux.

    python ladder.py <step> [--config config.json] [step options]

Steps, in order (each is resumable / skips work already done):
  tbset     build the Syzygy test set                              (CPU, once)
  calib     calibration + dev planes in lc0's exact encoding          (CPU, once)
  export    BT3 -> models/bt3_fp32.onnx -> models/fp16.pb.gz        (needs lc0)
  day1      gate C1 checks: wrapped FP16 == native lc0, INT8 loads, INT8 kernels
  select    dev-split grid for step 3 method and step 4 alpha       (ONNX Runtime)
  build     the ladder s1..s4 (+ s4aa, s0, s1sp, swaps), wrapped for lc0
  study     Studies A + B (overnight)                               (GPU)
  studyc    Study C fixed time per move                             (GPU)
  score     reports/scores_<split>.md (+ hard subset)
  match     time-parity match fp16 vs best                          (GPU, overnight)
  hw        bench + J/node, batch sweep, trtexec layer builds, roofline, peaks (GPU)

See README.md for the schedule and what each gate needs.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

DEFAULT = {
    "lc0": "lc0",
    "lc0_py": None,
    "trtexec": "trtexec",
    "nsys": "nsys",
    "bt3": "nets/BT3.pb.gz",
    "syzygy": None,
    "work": ".",
    "backend": "onnx-trt",
    "fp16_opts": "optimize=6",
    "int8_opts": "optimize=8",
    "native_backend": "onnx-trt",
    "native_opts": "optimize=6",
    "tbset": {"per_class": 400, "classes": None},
    "calib": {"random_games": 256, "n_tb": 256},
    "recipe": {"s3_method": "percentile", "s3_percentile": 99.99, "s4_alpha": 0.5},
    "best": "s4",
    "study": {"split": "test", "limit": 0, "budgets": "0,100,400,800,1600",
              "threads": 1, "minibatch": 256,
              "models": "fp16,s1,s2,s3,s4,s4aa"},
    "extras": {"s0_in_study_b": False, "swap": True, "s1sp": True},
    "studyc": {"movetimes": "200,800", "models": "fp16,BEST", "limit": 0},
    "match": {"tc": "10+0.1", "pairs": 250},
    "hw": {"batch_sizes": "32,64,128,256", "bench_movetime": 10000},
}


# ------------------------------------------------------------------ config
def load_config(path):
    cfg = json.loads(json.dumps(DEFAULT))
    config_dir = ROOT
    if path and Path(path).exists():
        config_path = Path(path).expanduser().resolve()
        config_dir = config_path.parent
        user = json.loads(config_path.read_text(encoding="utf-8-sig"))
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    elif path:
        sys.exit(f"config {path} not found (copy config.example.json)")
    # Resolve local paths independently of the original Desktop/LC0-B checkout.
    # Scientific settings are unchanged by this portability layer.
    w = Path(os.path.expandvars(cfg["work"])).expanduser()
    w = (config_dir / w).resolve() if not w.is_absolute() else w.resolve()
    cfg["_w"] = w
    for key in ("lc0", "trtexec", "nsys", "lc0_py", "syzygy", "bt3"):
        value = cfg.get(key)
        if not value:
            continue
        value = os.path.expandvars(value)
        # Bare executable names deliberately use PATH on either platform.
        if key in ("lc0", "trtexec", "nsys") and not any(c in value for c in ("/", "\\")):
            cfg[key] = value
            continue
        local_path = Path(value).expanduser()
        anchor = w if key == "bt3" else config_dir
        cfg[key] = str((anchor / local_path).resolve())
    for d in ("models", "data", "results", "reports", "reports/logs"):
        (w / d).mkdir(parents=True, exist_ok=True)
    return cfg


def P(cfg, *parts):
    return str(cfg["_w"].joinpath(*parts))


def run(cmd, log=None, check=True):
    cmd = [str(c) for c in cmd]
    print("$ " + " ".join(cmd), flush=True)
    t0 = time.time()
    if log:
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"\n$ {' '.join(cmd)}\n")
            f.flush()
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in p.stdout:
                sys.stdout.write(line)
                f.write(line)
            p.wait()
            rc = p.returncode
    else:
        rc = subprocess.call(cmd)
    if check and rc != 0:
        sys.exit(f"step failed ({rc}) after {time.time() - t0:.0f}s: {' '.join(cmd)}")
    return rc


def script(*parts):
    return str(ROOT.joinpath(*parts))


# ------------------------------------------------------------------ models
def recipe_flags(cfg, name):
    r = cfg["recipe"]
    s3 = ["--granularity", "per-channel", "--method", r["s3_method"]]
    if r["s3_method"] == "percentile":
        s3 += ["--percentile", str(r["s3_percentile"])]
    s4 = s3 + ["--smoothquant", str(r["s4_alpha"])]
    table = {"s1": ["--granularity", "per-tensor", "--method", "minmax"],
             "s2": ["--granularity", "per-channel", "--method", "minmax"],
             "s3": s3, "s4": s4}
    if name in table:
        return table[name]
    best = table[cfg["best"]] if cfg["best"] in table else s4
    if name == "s4aa":
        return best + ["--act-act-matmul"]
    if name in ("s0", "s1sp"):
        return best
    raise KeyError(name)


def model_opts(cfg, name):
    return cfg["fp16_opts"] if name == "fp16" else cfg["int8_opts"]


def model_weights(cfg, name):
    if name == "native":
        return cfg["bt3"]
    return P(cfg, "models", f"{name}.pb.gz")


def resolve(cfg, names):
    out = []
    for n in names.split(","):
        n = n.strip()
        if n == "BEST":
            n = cfg["best"]
        if n and n not in out:
            out.append(n)
    return out


# ------------------------------------------------------------------ steps
def step_tbset(cfg, a):
    out = P(cfg, "data", "tbset.jsonl")
    if Path(out).exists() and not a.force:
        print(f"{out} exists (--force to rebuild)")
        return
    if not cfg["syzygy"]:
        sys.exit("set syzygy in the config")
    cmd = [PY, script("tb", "build_tbset.py"), "--syzygy", cfg["syzygy"], "--out", out,
           "--per-class", cfg["tbset"]["per_class"]]
    if cfg["tbset"]["classes"]:
        cmd += ["--classes", cfg["tbset"]["classes"]]
    run(cmd, P(cfg, "reports", "logs", "tbset.log"))


def step_calib(cfg, a):
    """Python port of lc0's encoder by default; lc0's bindings if lc0_py is set."""
    tb = P(cfg, "data", "tbset.jsonl")
    c = cfg["calib"]
    log = P(cfg, "reports", "logs", "calib.log")
    enc = ["--lc0-py", cfg["lc0_py"]] if cfg["lc0_py"] else ["--lc0", cfg["lc0"]]
    if not Path(P(cfg, "data", "calib.npz")).exists() or a.force:
        run([PY, script("quant", "make_calib.py"), *enc, "--weights", cfg["bt3"],
             "--random-games", c["random_games"], "--tbset", tb, "--tb-split", "calib",
             "--n-tb", c["n_tb"], "--out", P(cfg, "data", "calib.npz")], log)
    if not Path(P(cfg, "data", "dev.npz")).exists() or a.force:
        run([PY, script("quant", "make_calib.py"), *enc, "--weights", cfg["bt3"],
             "--tbset", tb, "--tb-split", "dev", "--n-tb", 0, "--out", P(cfg, "data", "dev.npz")], log)


def check_npz_format(cfg):
    import numpy as np
    import re
    r = subprocess.run([cfg["lc0"], "describenet", f"--weights={cfg['bt3']}"],
                       capture_output=True, text=True)
    m = re.search(r"Input:\s+(\S+)", r.stdout + r.stderr)
    names = {"INPUT_CLASSICAL_112_PLANE": 1, "INPUT_112_WITH_CASTLING_PLANE": 2,
             "INPUT_112_WITH_CANONICALIZATION": 3, "INPUT_112_WITH_CANONICALIZATION_HECTOPLIES": 4,
             "INPUT_112_WITH_CANONICALIZATION_HECTOPLIES_ARMAGEDDON": 132,
             "INPUT_112_WITH_CANONICALIZATION_V2": 5, "INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON": 133}
    want = names.get(m.group(1)) if m else None
    for f in ("calib.npz", "dev.npz"):
        p = Path(P(cfg, "data", f))
        if not p.exists():
            sys.exit(f"missing {p}: run `python ladder.py calib` first")
        got = int(np.load(p)["input_format"])
        if want is not None and got != want:
            sys.exit(f"{p} was made for input format {got}, BT3 uses {want} ({m.group(1)}). "
                     "Remake it with a net of BT3's input format.")
    print(f"calibration planes match BT3's input format ({m.group(1) if m else '?'})")


def step_export(cfg, a):
    onnx_path = P(cfg, "models", "bt3_fp32.onnx")
    log = P(cfg, "reports", "logs", "export.log")
    if not Path(onnx_path).exists() or a.force:
        run([cfg["lc0"], "describenet", f"--weights={cfg['bt3']}"], log)
        run([cfg["lc0"], "leela2onnx", f"--input={cfg['bt3']}", f"--output={onnx_path}"], log)
    if not Path(model_weights(cfg, "fp16")).exists() or a.force:
        run([PY, script("quant", "wrap.py"), "--lc0", cfg["lc0"], "--orig", cfg["bt3"],
             "--onnx", onnx_path, "--out", model_weights(cfg, "fp16")], log)


def study_cmd(cfg, name, budgets, split, limit, extra=()):
    s = cfg["study"]
    backend = cfg["native_backend"] if name == "native" else cfg["backend"]
    opts = cfg["native_opts"] if name == "native" else model_opts(cfg, name)
    return [PY, script("eval", "run_study.py"), "--lc0", cfg["lc0"], "--name", name,
            "--weights", model_weights(cfg, name), "--backend", backend, "--backend-opts", opts,
            "--tbset", P(cfg, "data", "tbset.jsonl"), "--split", split, "--limit", limit,
            "--budgets", budgets, "--threads", s["threads"], "--minibatch", s["minibatch"],
            "--out-dir", P(cfg, "results"), *extra]


def build_model(cfg, name, log, force=False):
    """Make models/<name>.onnx and models/<name>.pb.gz."""
    src = P(cfg, "models", "bt3_fp32.onnx")
    calib = P(cfg, "data", "calib.npz")
    onnx_out = P(cfg, "models", f"{name}.onnx")
    pb = model_weights(cfg, name)
    if Path(pb).exists() and not force:
        return
    if name in ("s0", "s1sp"):
        dense = P(cfg, "models", f"{name}_fp32.onnx")
        if not Path(dense).exists() or force:
            if name == "s0":
                run([PY, script("quant", "prune24.py"), "--onnx", src, "--out", dense], log)
            else:
                run([PY, script("quant", "sparsegpt24.py"), "--onnx", src, "--calib", calib,
                     "--out", dense], log)
        src = dense
    if name.startswith("swap_"):
        best = P(cfg, "models", f"{cfg['best']}.onnx")
        fp = P(cfg, "models", "bt3_fp32.onnx")
        pol, val = (best, fp) if name == "swap_pol8" else (fp, best)
        run([PY, script("quant", "stitch_swap.py"), "--policy", pol, "--value", val,
             "--out", onnx_out], log)
    else:
        cmd = [PY, script("quant", "quantize.py"), "--onnx", src, "--calib", calib,
               "--out", onnx_out, *recipe_flags(cfg, name)]
        if name == "s4":
            cmd += ["--outlier-report", P(cfg, "reports", "outliers.txt")]
        run(cmd, log)
    run([PY, script("quant", "wrap.py"), "--lc0", cfg["lc0"], "--orig", cfg["bt3"],
         "--onnx", onnx_out, "--out", pb], log)


def step_day1(cfg, a):
    """Gate C1 checks on a few hundred dev positions."""
    log = P(cfg, "reports", "logs", "day1.log")
    check_npz_format(cfg)
    step_export(cfg, a)
    build_model(cfg, "s2", log)
    n = a.limit or 200
    for name in ("native", "fp16", "s2"):
        run(study_cmd(cfg, name, "0", "dev", n), log)
    run([PY, script("eval", "score.py"), "--tbset", P(cfg, "data", "tbset.jsonl"),
         "--results", P(cfg, "results"), "--ref", "native", "--models", "native,fp16,s2",
         "--out", P(cfg, "reports", "day1"), "--split", "dev"], log)
    js = json.loads(Path(P(cfg, "reports", "day1", "scores_dev.json")).read_text())["models"]
    fp, s2 = js.get("fp16", {}).get("A", {}), js.get("s2", {}).get("A", {})
    print("\n=== day-1 summary ===")
    print(f"1. wrapped FP16 vs native lc0: KL {fp.get('kl', float('nan')):.2e}, top-1 "
          f"{fp.get('top1', float('nan')):.3f}  -> {'OK' if fp.get('top1', 0) > 0.97 else 'CHECK: wrapped path differs'}")
    print(f"2. lc0 runs the INT8 file: {'OK' if s2 else 'FAILED'} (KL vs native {s2.get('kl', float('nan')):.2e}, "
          f"top-1 {s2.get('top1', float('nan')):.3f})")
    for nm in ("native", "fp16", "s2"):
        mp = Path(P(cfg, "results", nm, "meta.json"))
        if mp.exists():
            w = json.loads(mp.read_text()).get("warnings", [])
            if w:
                print(f"   lc0 warnings for {nm}: {w[:3]}")
    nsys = shutil.which(cfg["nsys"])
    rep = P(cfg, "reports", "trt", "s2_lc0")
    Path(rep).parent.mkdir(parents=True, exist_ok=True)
    bench = [cfg["lc0"], "benchmark", f"--weights={model_weights(cfg, 's2')}",
             f"--backend={cfg['backend']}", f"--backend-opts={cfg['int8_opts']}",
             "--nodes=20000", "--num-positions=3"]
    if nsys:
        run([nsys, "profile", "-o", rep, "--force-overwrite", "true", *bench], log, check=False)
        run([nsys, "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv", "-o", rep,
             rep + ".nsys-rep"], log, check=False)
        csvs = sorted(Path(rep).parent.glob(Path(rep).name + "*kern_sum*.csv"))
        if csvs:
            run([PY, script("hw", "trt_layers.py"), "kernels", "--csv", str(csvs[-1])], log, check=False)
            print("3. matmuls in INT8: see the 'GEMM-like kernel time by tag' line above "
                  "(most should be int8)")
    else:
        print("3. nsys not found. Run by hand:\n   nsys profile -o " + rep + " --force-overwrite true "
              + " ".join(bench) + f"\n   nsys stats --report cuda_gpu_kern_sum --format csv -o {rep} {rep}.nsys-rep"
              f"\n   python hw/trt_layers.py kernels --csv {rep}_cuda_gpu_kern_sum.csv")


def step_select(cfg, a):
    """Dev grid with ONNX Runtime: step 3 method and step 4 alpha."""
    log = P(cfg, "reports", "logs", "select.log")
    src = P(cfg, "models", "bt3_fp32.onnx")
    calib = P(cfg, "data", "calib.npz")
    d = Path(P(cfg, "models", "select"))
    d.mkdir(exist_ok=True)
    grid = {"s3_p999": ["--method", "percentile", "--percentile", "99.9"],
            "s3_p9999": ["--method", "percentile", "--percentile", "99.99"],
            "s3_p99999": ["--method", "percentile", "--percentile", "99.999"],
            "s3_entropy": ["--method", "entropy"]}
    for nm, fl in grid.items():
        if not (d / f"{nm}.onnx").exists():
            run([PY, script("quant", "quantize.py"), "--onnx", src, "--calib", calib,
                 "--out", str(d / f"{nm}.onnx"), "--granularity", "per-channel", *fl], log)
    run([PY, script("eval", "quick_compare.py"), "--ref", src, "--npz", P(cfg, "data", "dev.npz"),
         "--tbset", P(cfg, "data", "tbset.jsonl"), "--out", P(cfg, "reports", "select_s3_dev.txt"),
         *[str(d / f"{nm}.onnx") for nm in grid]], log)
    print("\nPick step 3 from the table above (lowest KL / highest tb_acc), set recipe.s3_method "
          "and s3_percentile in the config, then rerun `select --alpha` for step 4.")
    if a.alpha:
        r = cfg["recipe"]
        base = ["--granularity", "per-channel", "--method", r["s3_method"]]
        if r["s3_method"] == "percentile":
            base += ["--percentile", str(r["s3_percentile"])]
        files = []
        for al in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            f = d / f"s4_a{al}.onnx"
            if not f.exists():
                run([PY, script("quant", "quantize.py"), "--onnx", src, "--calib", calib,
                     "--out", str(f), *base, "--smoothquant", str(al)], log)
            files.append(str(f))
        run([PY, script("eval", "quick_compare.py"), "--ref", src, "--npz", P(cfg, "data", "dev.npz"),
             "--tbset", P(cfg, "data", "tbset.jsonl"), "--out", P(cfg, "reports", "select_s4_dev.txt"),
             *files], log)
        print("\nSet recipe.s4_alpha in the config from the table above. If no alpha beats "
              "step 3, set best to s3.")


def step_build(cfg, a):
    log = P(cfg, "reports", "logs", "build.log")
    check_npz_format(cfg)
    step_export(cfg, a)
    names = ["s1", "s2", "s3", "s4", "s4aa", "s0"]
    if cfg["extras"]["s1sp"]:
        names.append("s1sp")
    if cfg["extras"]["swap"]:
        names += ["swap_pol8", "swap_val8"]
    if a.only:
        names = resolve(cfg, a.only)
    for n in names:
        build_model(cfg, n, log, force=a.force)
    onnxs = [P(cfg, "models", f"{n}.onnx") for n in names if not n.startswith("swap_")]
    run([PY, script("eval", "quick_compare.py"), "--ref", P(cfg, "models", "bt3_fp32.onnx"),
         "--npz", P(cfg, "data", "dev.npz"), "--tbset", P(cfg, "data", "tbset.jsonl"),
         "--out", P(cfg, "reports", "quick_compare_dev.txt"), *onnxs], log)


def step_study(cfg, a):
    s = cfg["study"]
    names = resolve(cfg, a.models or s["models"])
    log = P(cfg, "reports", "logs", "study.log")
    extras = []
    if cfg["extras"]["swap"] and not a.models:
        extras += ["swap_pol8", "swap_val8"]
    if cfg["extras"]["s1sp"] and not a.models:
        extras.append("s1sp")
    for n in names + extras:
        if n != "fp16" and not Path(model_weights(cfg, n)).exists():
            print(f"skipping {n}: {model_weights(cfg, n)} not built")
            continue
        run(study_cmd(cfg, n, s["budgets"], a.split or s["split"], a.limit if a.limit is not None else s["limit"]), log)
    # S0: Study A always; Study B only if switched on after looking at Study A
    if Path(model_weights(cfg, "s0")).exists() and not a.models:
        b = s["budgets"] if cfg["extras"]["s0_in_study_b"] else "0"
        run(study_cmd(cfg, "s0", b, a.split or s["split"], a.limit if a.limit is not None else s["limit"]), log)


def step_studyc(cfg, a):
    c = cfg["studyc"]
    log = P(cfg, "reports", "logs", "studyc.log")
    for n in resolve(cfg, a.models or c["models"]):
        cmd = study_cmd(cfg, n, "", a.split or cfg["study"]["split"],
                        a.limit if a.limit is not None else c["limit"])
        # Study C uses lc0's default Threads, not the fixed-node setting
        i = cmd.index("--threads")
        cmd[i + 1] = 0
        run(cmd + ["--movetimes", c["movetimes"]], log)


def step_score(cfg, a):
    names = resolve(cfg, a.models or ",".join(
        [cfg["study"]["models"], "s0", "s1sp", "swap_pol8", "swap_val8"]))
    names = [n for n in names if Path(P(cfg, "results", n)).exists()]
    run([PY, script("eval", "score.py"), "--tbset", P(cfg, "data", "tbset.jsonl"),
         "--results", P(cfg, "results"), "--ref", "fp16", "--models", ",".join(names),
         "--out", P(cfg, "reports"), "--split", a.split or cfg["study"]["split"]])


def step_match(cfg, a):
    m = cfg["match"]
    best = cfg["best"]
    cmd = [PY, script("eval", "match.py"), "--lc0", cfg["lc0"], "--name", f"fp16_vs_{best}",
           "--a", f"fp16={model_weights(cfg, 'fp16')}:{cfg['fp16_opts']}",
           "--b", f"{best}={model_weights(cfg, best)}:{cfg['int8_opts']}",
           "--backend", cfg["backend"], "--tc", m["tc"], "--pairs", m["pairs"],
           "--out-dir", P(cfg, "results")]
    if cfg["syzygy"]:
        cmd += ["--syzygy", cfg["syzygy"]]
    if a.estimate:
        cmd += ["--estimate", a.estimate]
    run(cmd, P(cfg, "reports", "logs", "match.log"))


def step_hw(cfg, a):
    h = cfg["hw"]
    log = P(cfg, "reports", "logs", "hw.log")
    best = cfg["best"]
    parts = a.parts.split(",") if a.parts else ["bench", "sweep", "trt", "roofline"]
    if "bench" in parts:
        for n in ("fp16", best, "native"):
            backend = cfg["native_backend"] if n == "native" else cfg["backend"]
            opts = cfg["native_opts"] if n == "native" else model_opts(cfg, n)
            run([PY, script("hw", "bench.py"), "--lc0", cfg["lc0"], "--weights", model_weights(cfg, n),
                 "--name", n, "--backend", backend, "--backend-opts", opts,
                 "--movetime", h["bench_movetime"], "--out-dir", P(cfg, "reports", "bench")], log)
    if "sweep" in parts:
        run([PY, script("hw", "batch_sweep.py"), "--lc0", cfg["lc0"], "--backend", cfg["backend"],
             "--sizes", h["batch_sizes"],
             "--model", f"fp16={model_weights(cfg, 'fp16')}:{cfg['fp16_opts']}",
             "--model", f"{best}={model_weights(cfg, best)}:{cfg['int8_opts']}",
             "--out", P(cfg, "reports", "batch_sweep")], log)
    trt = Path(P(cfg, "reports", "trt"))
    trt.mkdir(parents=True, exist_ok=True)
    builds = [("fp16", P(cfg, "models", "bt3_fp32.onnx"), [], ), (best, P(cfg, "models", f"{best}.onnx"), ["--int8"])]
    for sp in ("s0", "s1sp"):
        if Path(P(cfg, "models", f"{sp}.onnx")).exists():
            builds.append((f"{sp}_sparse", P(cfg, "models", f"{sp}.onnx"), ["--int8", "--sparsity"]))
            builds.append((f"{sp}_dense", P(cfg, "models", f"{sp}.onnx"), ["--int8"]))
    if "trt" in parts:
        for tag, onnx_path, fl in builds:
            run([PY, script("hw", "trt_layers.py"), "build", "--trtexec", cfg["trtexec"],
                 "--onnx", onnx_path, "--out", str(trt / tag), *fl], log, check=False)
        run([PY, script("hw", "roofline.py"), "peak", "--trtexec", cfg["trtexec"],
             "--out", str(trt / "peak.json")], log, check=False)
    if "roofline" in parts:
        for tag, onnx_path, _ in builds[:2]:
            if Path(f"{trt / tag}_layers.json").exists():
                run([PY, script("hw", "roofline.py"), "layers", "--onnx", onnx_path,
                     "--prefix", str(trt / tag), "--peaks", str(trt / "peak.json"),
                     "--out", P(cfg, "reports", "roofline", tag)], log, check=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["tbset", "calib", "export", "day1", "select", "build", "study",
                                     "studyc", "score", "match", "hw"])
    ap.add_argument("--config", default=str(ROOT / "config.json"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default=None, help="build: comma list of models")
    ap.add_argument("--models", default=None, help="study/studyc/score: comma list (BEST = config best)")
    ap.add_argument("--split", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--alpha", action="store_true", help="select: also run the step-4 alpha grid")
    ap.add_argument("--estimate", type=int, default=0, help="match: time N games and extrapolate")
    ap.add_argument("--parts", default=None, help="hw: bench,sweep,trt,roofline")
    a = ap.parse_args()
    cfg = load_config(a.config)
    os.chdir(cfg["_w"])
    globals()[f"step_{a.step}"](cfg, a)


if __name__ == "__main__":
    main()
