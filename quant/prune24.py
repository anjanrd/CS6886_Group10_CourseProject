#!/usr/bin/env python3
"""S0: one-shot magnitude 2:4 pruning of every weight-carrying MatMul/Conv
(no fine-tuning). Output is an FP32 ONNX; the ladder then quantizes it with
the frozen PTQ recipe (zeros stay zeros under symmetric int8).

    python quant/prune24.py --onnx models/bt3_fp32.onnx --out models/s0_fp32.onnx
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from onnx import numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import initializers, load_onnx, save_onnx  # noqa: E402
from sparse_common import check_24, from_kn, magnitude_mask, targets, to_kn  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip", default=None, help="regex of node names to leave dense")
    a = ap.parse_args()
    m = load_onnx(a.onnx)
    inits = initializers(m)
    rep = {}
    for name, nodes in targets(m, a.skip).items():
        op = nodes[0].op_type
        w = numpy_helper.to_array(inits[name]).astype(np.float32)
        kn = to_kn(w, op)
        mask = magnitude_mask(kn)
        p = np.where(mask, kn, 0).astype(np.float32)
        assert check_24(p)
        inits[name].CopyFrom(numpy_helper.from_array(from_kn(p, op, w.shape), name))
        rep[name] = {"nodes": [n.name for n in nodes],
                     "rel_err": float(np.linalg.norm(kn - p) / max(np.linalg.norm(kn), 1e-12))}
    save_onnx(m, a.out)
    with open(Path(a.out).with_suffix(".json"), "w") as f:
        json.dump({"method": "magnitude 2:4", "skip": a.skip, "layers": rep}, f, indent=1)
    errs = [v["rel_err"] for v in rep.values()]
    print(f"pruned {len(rep)} weights 2:4 (magnitude); median rel. weight error "
          f"{np.median(errs):.3f} -> {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
