#!/usr/bin/env python3
"""Calibration / evaluation planes in lc0's exact input encoding, so the
ONNX-side tools see exactly what lc0 feeds the net.

Default encoder: quant/lc0_encode.py, a pure-Python port of lc0's
EncodePositionForNN, verified bit for bit against lc0 for all input formats
(tests/test_encoder.py). Needs only `lc0 describenet` (or --input-format) to
learn the net's input format. Runs anywhere, including Windows.

Optional: --lc0-py uses lc0's own Python bindings instead (and is needed for
--policy-games). Build them with lc0-bindings.patch: stock bindings ignore
the canonical-format board transform in policy_indices(), which gives wrong
legal-move indices on every pawnless position.

    # 256 random-game positions + 256 tablebase calib positions
    python quant/make_calib.py --lc0 lc0 --weights nets/BT3.pb.gz \
        --random-games 256 --tbset data/tbset.jsonl --tb-split calib --n-tb 256 \
        --out data/calib.npz
    # dev split with labels, for quick_compare
    python quant/make_calib.py --lc0 lc0 --weights nets/BT3.pb.gz \
        --tbset data/tbset.jsonl --tb-split dev --n-tb 0 --out data/dev.npz
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chess  # noqa: E402

from common import read_jsonl  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lc0_encode import encode  # noqa: E402


FORMAT_NAMES = {"INPUT_CLASSICAL_112_PLANE": 1, "INPUT_112_WITH_CASTLING_PLANE": 2,
                "INPUT_112_WITH_CANONICALIZATION": 3,
                "INPUT_112_WITH_CANONICALIZATION_HECTOPLIES": 4,
                "INPUT_112_WITH_CANONICALIZATION_HECTOPLIES_ARMAGEDDON": 132,
                "INPUT_112_WITH_CANONICALIZATION_V2": 5,
                "INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON": 133}


def input_format_of(lc0, weights):
    import re
    import subprocess
    if not weights:
        sys.exit("need --weights (or --input-format)")
    r = subprocess.run([lc0, "describenet", f"--weights={weights}"], capture_output=True, text=True)
    m = re.search(r"Input:\s+(\S+)", r.stdout + r.stderr)
    if not m or m.group(1) not in FORMAT_NAMES:
        sys.exit(f"could not read the input format from `{lc0} describenet`:\n{r.stdout}{r.stderr}")
    return FORMAT_NAMES[m.group(1)]


def planes_of(inp):
    out = np.zeros((112, 64), dtype=np.float32)
    for p in range(112):
        m = inp.mask(p)
        if not m:
            continue
        v = inp.val(p)
        bits = np.array([(m >> i) & 1 for i in range(64)], dtype=np.float32)
        out[p] = bits * v
    return out.reshape(112, 8, 8)


def random_game(rng, min_plies, max_plies):
    b = chess.Board()
    n = rng.randint(min_plies, max_plies)
    moves = []
    for _ in range(n):
        legal = list(b.legal_moves)
        if not legal or b.is_game_over():
            break
        mv = rng.choice(legal)
        b.push(mv)
        moves.append(mv.uci())
    # Drop trailing plies until the position is not terminal.
    while moves and b.is_game_over():
        b.pop()
        moves.pop()
    return moves


def policy_game(rng, lb, backend, min_plies, max_plies, temp):
    b = chess.Board()
    n = rng.randint(min_plies, max_plies)
    moves = []
    for _ in range(n):
        if b.is_game_over():
            break
        g = lb.GameState(moves=moves)
        inp = g.as_input(backend)
        out = backend.evaluate(inp)[0]
        idx = g.policy_indices()
        p = np.array(out.p_softmax(*idx), dtype=np.float64)
        p = np.power(np.maximum(p, 1e-12), 1.0 / temp)
        p /= p.sum()
        uci = g.moves()[int(np.searchsorted(np.cumsum(p), rng.random()))]
        b.push_uci(uci)
        moves.append(uci)
    while moves and b.is_game_over():
        b.pop()
        moves.pop()
    return moves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lc0-py", default=None,
                    help="use lc0's Python bindings from this build dir instead of the Python encoder")
    ap.add_argument("--lc0", default="lc0", help="for `describenet` (input format)")
    ap.add_argument("--weights", default=None,
                    help="any net with the target input format (BT3 itself is fine)")
    ap.add_argument("--input-format", type=int, default=None,
                    help="lc0 InputFormat enum; skips describenet")
    ap.add_argument("--random-games", type=int, default=0)
    ap.add_argument("--policy-games", type=int, default=0,
                    help="games sampled from the net's own policy (slow on CPU)")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--min-plies", type=int, default=8)
    ap.add_argument("--max-plies", type=int, default=160)
    ap.add_argument("--tbset", default=None)
    ap.add_argument("--tb-split", default="calib")
    ap.add_argument("--n-tb", type=int, default=0, help="0 = all of that split")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    lb = enc = net = None
    if a.lc0_py:
        sys.path.insert(0, str(Path(a.lc0_py).resolve()))
        import backends as lb  # noqa: E402
        w = lb.Weights(a.weights)
        fmt = w.input_format()
        # The trivial backend reports the weights' input format and evaluates
        # nothing (it misreports a few formats; fall back to onnx-cpu then).
        enc = lb.Backend(weights=w, backend="trivial")
        if enc.capabilities().input_format() != fmt:
            enc = lb.Backend(weights=w, backend="onnx-cpu")
        net = lb.Backend(weights=w, backend="onnx-cpu") if a.policy_games else None
    else:
        if a.policy_games:
            sys.exit("--policy-games needs --lc0-py")
        fmt = a.input_format if a.input_format is not None else input_format_of(a.lc0, a.weights)

    rng = random.Random(a.seed)
    items = []  # (fen, moves, src, tb_id)
    for _ in range(a.random_games):
        mv = random_game(rng, a.min_plies, a.max_plies)
        items.append((None, mv, "random", -1))
    for i in range(a.policy_games):
        mv = policy_game(rng, lb, net, a.min_plies, a.max_plies, a.temp)
        items.append((None, mv, "policy", -1))
        if (i + 1) % 16 == 0:
            print(f"  policy games {i + 1}/{a.policy_games}", file=sys.stderr)
    if a.tbset:
        recs = [r for r in read_jsonl(a.tbset) if r["split"] == a.tb_split]
        rng.shuffle(recs)
        if a.n_tb:
            recs = recs[:a.n_tb]
        for r in recs:
            fen = r["fen"]
            f = fen.split()
            if f[4] != "0":  # trap 5
                f[4] = "0"
                fen = " ".join(f)
            items.append((fen, [], "tb", r["id"]))

    planes, legal, ucis, fens, srcs, ids, hist = [], [], [], [], [], [], []
    for fen, mv, src, tid in items:
        if lb is not None:
            g = lb.GameState(fen=fen, moves=mv) if fen else lb.GameState(moves=mv)
            inp = g.as_input(enc)          # sets the transform used below
            planes.append(planes_of(inp))
            legal.append(list(g.policy_indices()))
            ucis.append(" ".join(g.moves()))
        else:
            pl, idx, uc = encode(fen, mv, fmt)
            planes.append(pl)
            legal.append(idx)
            ucis.append(" ".join(uc))
        if fen is None:
            b = chess.Board()
            for m in mv:
                b.push_uci(m)
            fen = b.fen()
        fens.append(fen)
        srcs.append(src)
        ids.append(tid)
        hist.append(" ".join(mv))
    L = max((len(x) for x in legal), default=1)
    li = np.full((len(legal), L), -1, dtype=np.int32)
    for i, x in enumerate(legal):
        li[i, :len(x)] = x
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.out, planes=np.stack(planes) if planes else np.zeros((0, 112, 8, 8), np.float32),
        legal_idx=li, legal_uci=np.array(ucis), fen=np.array(fens),
        src=np.array(srcs), tb_id=np.array(ids, dtype=np.int64),
        history=np.array(hist), input_format=np.int64(fmt))
    print(f"wrote {len(items)} positions (input format {fmt}) to {a.out}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
