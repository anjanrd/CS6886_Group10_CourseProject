#!/usr/bin/env python3
"""Wrap an ONNX model as an lc0 weights file (trap 1).

`lc0 onnx2leela` defaults to INPUT_CLASSICAL_112_PLANE. BT3 uses a canonical
input format, and with the default lc0 loads the file and silently feeds it
wrongly encoded planes. This copies every format from `lc0 describenet` of the
original net, passes the tensor names explicitly, and re-reads the result to
check the formats survived.

    python quant/wrap.py --lc0 lc0 --orig nets/BT3.pb.gz --onnx models/s2.onnx --out models/s2.pb.gz
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import INPUT, OUT_MLH, OUT_POLICY, OUT_WDL  # noqa: E402

FIELDS = {"Input": "input-format", "Policy": "policy-format",
          "Value": "value-format", "MLH": "moves-left-format"}


def describe(lc0, weights):
    r = subprocess.run([lc0, "describenet", f"--weights={weights}"],
                       capture_output=True, text=True)
    txt = r.stdout + r.stderr
    if r.returncode != 0:
        sys.exit(f"describenet failed on {weights}:\n{txt}")
    fmt = {}
    for line in txt.splitlines():
        m = re.match(r"\s*(Input|Policy|Value|MLH|Network|Data type|Input planes|"
                     r"Output WDL|Output Policy|Output MLH|Output value):\s+(\S+)", line)
        if m and m.group(1) not in fmt:
            fmt[m.group(1)] = m.group(2)
    return fmt, txt


def inline_external_data(onnx_path):
    import onnx
    m = onnx.load(str(onnx_path), load_external_data=True)
    has_ext = any(t.data_location == onnx.TensorProto.EXTERNAL for t in m.graph.initializer)
    if not has_ext and not Path(str(onnx_path) + ".data").exists():
        return onnx_path, None
    if m.ByteSize() > 2e9:
        sys.exit("model is >2GB inline; lc0 weights files cannot hold it")
    fd, tmp = tempfile.mkstemp(suffix=".onnx")
    os.close(fd)
    onnx.save(m, tmp)
    return tmp, tmp


def wrap(lc0, orig, onnx_path, out, names=None):
    fmt, _ = describe(lc0, orig)
    missing = [k for k in FIELDS if k not in fmt]
    if missing:
        sys.exit(f"could not read {missing} from describenet of {orig}")
    names = names or {}
    src, tmp = inline_external_data(onnx_path)
    args = [lc0, "onnx2leela", f"--input={src}", f"--output={out}"]
    for k, flag in FIELDS.items():
        args.append(f"--{flag}={fmt[k]}")
    args += [f"--onnx-input={names.get('input', INPUT)}",
             f"--onnx-output-policy={names.get('policy', OUT_POLICY)}",
             f"--onnx-output-wdl={names.get('wdl', OUT_WDL)}"]
    if fmt["MLH"] != "MOVES_LEFT_NONE":
        args.append(f"--onnx-output-mlh={names.get('mlh', OUT_MLH)}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(args, capture_output=True, text=True)
    if tmp:
        os.unlink(tmp)
    if r.returncode != 0 or not Path(out).exists():
        sys.exit(f"onnx2leela failed:\n{r.stdout}\n{r.stderr}")
    got, txt = describe(lc0, out)
    bad = {k: (fmt[k], got.get(k)) for k in FIELDS if got.get(k) != fmt[k]}
    if bad:
        Path(out).unlink()
        sys.exit(f"format mismatch after wrapping (orig, wrapped): {bad}")
    if got.get("Network") != "NETWORK_ONNX":
        sys.exit(f"wrapped file is not NETWORK_ONNX:\n{txt}")
    return fmt


def check_against_ort(lc0, weights, onnx_path, npz, n, backend, backend_opts):
    """lc0's raw policy (nodes=1) vs ONNX Runtime on the same ONNX."""
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
    from common import ort_session
    from lc0uci import Lc0
    d = np.load(npz)
    s = ort_session(onnx_path, cpu=True)
    pol, wdl = s.run([OUT_POLICY, OUT_WDL], {INPUT: d["planes"][:n]})[:2]
    e = Lc0(lc0, weights, backend=backend, backend_opts=backend_opts,
            options={"Threads": 1, "VerboseMoveStats": True, "PolicyTemperature": 1.0})
    kls, top1, dv = [], 0, []
    for i in range(n):
        r = e.go(fen=str(d["fen"][i]), nodes=1)
        idx = d["legal_idx"][i]
        idx = idx[idx >= 0]
        ucis = str(d["legal_uci"][i]).split()
        lg = pol[i][idx].astype(np.float64)
        p = np.exp(lg - lg.max())
        p /= p.sum()
        q = np.array([max(r["moves"][u]["p"], 1e-6) for u in ucis])
        q /= q.sum()
        kls.append(float(np.sum(p * (np.log(np.maximum(p, 1e-12)) - np.log(q)))))
        top1 += int(np.argmax(p) == np.argmax(q))
        dv.append(abs(float(wdl[i][0] - wdl[i][2]) - r["v"]))
    e.quit()
    return float(np.mean(kls)), top1 / n, float(np.mean(dv))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lc0", default=os.environ.get("LC0", "lc0"))
    ap.add_argument("--orig", required=True, help="original lc0 net (formats source)")
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-npz", default=None,
                    help="also compare lc0 vs ORT-CPU raw outputs on these planes")
    ap.add_argument("--check-n", type=int, default=50)
    ap.add_argument("--check-backend", default="onnx-cpu")
    ap.add_argument("--check-backend-opts", default="")
    a = ap.parse_args()
    fmt = wrap(a.lc0, a.orig, a.onnx, a.out)
    print(f"wrapped {a.onnx} -> {a.out}  "
          + " ".join(f"{k}={fmt[k]}" for k in FIELDS))
    if a.check_npz:
        kl, t1, dv = check_against_ort(a.lc0, a.out, a.onnx, a.check_npz, a.check_n,
                                       a.check_backend, a.check_backend_opts)
        print(f"lc0[{a.check_backend} {a.check_backend_opts}] vs ORT-CPU on "
              f"{a.check_n} positions: KL={kl:.2e} top1={t1:.3f} |dV|={dv:.4f}")


if __name__ == "__main__":
    main()
