#!/usr/bin/env python3
"""Policy/value swap models for stock lc0: one ONNX that takes its policy
output from model P and its value (WDL) + MLH outputs from model V.

lc0 cannot mix two nets, but it runs any ONNX. This prefixes every name in
each graph, shares the input, keeps only the nodes each kept output needs
(so P's value head and V's policy head are dropped) and renames the outputs
back to lc0's names. Both bodies run, so the result is ~2× slower: use it
for accuracy studies (A/B) only, never for timing.

    python quant/stitch_swap.py --policy models/s4.onnx --value models/bt3_fp32.onnx \
        --out models/swap_int8pol.onnx
    python quant/wrap.py --orig nets/BT3.pb.gz --onnx models/swap_int8pol.onnx --out models/swap_int8pol.pb.gz
"""
import argparse
import sys
from pathlib import Path

import onnx
from onnx import helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import INPUT, OUT_MLH, OUT_POLICY, OUT_WDL, load_onnx, save_onnx  # noqa: E402


def needed_nodes(graph, outputs):
    prod = {o: n for n in graph.node for o in n.output}
    keep, stack, seen = [], list(outputs), set()
    while stack:
        t = stack.pop()
        if t in seen:
            continue
        seen.add(t)
        n = prod.get(t)
        if n is None:
            continue
        if id(n) not in {id(k) for k in keep}:
            keep.append(n)
        stack.extend(n.input)
        # subgraph-free ops only in lc0 graphs
    order = {id(n): i for i, n in enumerate(graph.node)}
    keep.sort(key=lambda n: order[id(n)])
    return keep


def prefixed(model, pfx, outputs):
    g = model.graph
    keep = needed_nodes(g, outputs)
    used = {i for n in keep for i in n.input}

    def ren(x):
        return x if x in ("", INPUT) else pfx + x

    nodes = []
    for n in keep:
        nn = onnx.NodeProto()
        nn.CopyFrom(n)
        nn.name = pfx + n.name
        del nn.input[:]
        nn.input.extend(ren(i) for i in n.input)
        del nn.output[:]
        nn.output.extend(ren(o) for o in n.output)
        nodes.append(nn)
    inits = []
    for t in g.initializer:
        if t.name in used:
            tt = onnx.TensorProto()
            tt.CopyFrom(t)
            tt.name = pfx + t.name
            inits.append(tt)
    vis = []
    for v in g.value_info:
        if v.name in used:
            vv = onnx.ValueInfoProto()
            vv.CopyFrom(v)
            vv.name = pfx + v.name
            vis.append(vv)
    return nodes, inits, vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="ONNX whose policy head is used")
    ap.add_argument("--value", required=True, help="ONNX whose WDL (and MLH) heads are used")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    P = load_onnx(a.policy)
    V = load_onnx(a.value)
    v_outs = [OUT_WDL] + ([OUT_MLH] if any(o.name == OUT_MLH for o in V.graph.output) else [])
    pn, pi, pv = prefixed(P, "P", [OUT_POLICY])
    vn, vi, vv = prefixed(V, "V", v_outs)
    # Rename the kept outputs back to lc0's names with Identity nodes.
    ident = [helper.make_node("Identity", ["P" + OUT_POLICY], [OUT_POLICY], name="/swap/policy")]
    ident += [helper.make_node("Identity", ["V" + o], [o], name=f"/swap{o}") for o in v_outs]
    outs = [o for o in P.graph.output if o.name == OUT_POLICY] + \
           [o for o in V.graph.output if o.name in v_outs]
    g = helper.make_graph(pn + vn + ident, "swap", list(P.graph.input), outs,
                          initializer=pi + vi, value_info=pv + vv)
    opsets = {}
    for m in (P, V):
        for o in m.opset_import:
            opsets[o.domain] = max(opsets.get(o.domain, 0), o.version)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid(d, v) for d, v in opsets.items()],
                          producer_name="leela-int8 stitch_swap")
    m.ir_version = max(P.ir_version, V.ir_version)
    if m.ByteSize() < 1.8e9:
        onnx.checker.check_model(m)
    save_onnx(m, a.out)
    print(f"policy from {a.policy}, value from {a.value} -> {a.out} "
          f"({len(pn)} + {len(vn)} nodes)", file=sys.stderr)


if __name__ == "__main__":
    main()
