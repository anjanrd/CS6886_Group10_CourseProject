#!/usr/bin/env python3
"""Build the Syzygy test set: random legal endgame positions with the exact
tablebase outcome of every legal move.

A move is *correct* if it keeps the position's game-theoretic result
(win stays win, draw stays draw) under the 50-move rule: cursed wins and
blessed losses count as draws. Lost positions drop out (every move keeps a
loss) and so do positions where every legal move is correct. For won
positions we also record the DTZ-optimal moves as a secondary metric.

Every FEN has halfmove clock 0 (trap 5: V2 canonical input formats hang on
FEN-only positions with a nonzero clock).

    python tb/build_tbset.py --syzygy /path/to/syzygy --out data/tbset.jsonl --per-class 400
"""
import argparse
import json
import random
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import chess
import chess.syzygy

# 28 classes: 3 three-man, 13 four-man, 12 five-man. Chosen to have many
# positions where some moves throw the result away.
DEFAULT_CLASSES = [
    "KQvK", "KRvK", "KPvK",
    "KQvKR", "KQvKP", "KQvKQ", "KRvKB", "KRvKN", "KRvKP", "KBvKP", "KNvKP",
    "KPvKP", "KBNvK", "KBPvK", "KNPvK", "KPPvK",
    "KRPvKR", "KRPvKB", "KQPvKQ", "KBPvKB", "KBPvKN", "KPPvKP", "KQvKRP",
    "KRRvKR", "KQRvKQ", "KRBvKR", "KRNvKR", "KPPvKN",
]

PIECE = {"K": chess.KING, "Q": chess.QUEEN, "R": chess.ROOK, "B": chess.BISHOP,
         "N": chess.KNIGHT, "P": chess.PAWN}

_TB = None


def _init(path):
    global _TB
    _TB = chess.syzygy.open_tablebase(path)


def category(wdl):
    """Syzygy wdl (2,1,0,-1,-2) -> result under the 50-move rule."""
    return 1 if wdl == 2 else (-1 if wdl == -2 else 0)


def child_category(board, move):
    """Result for the side making `move`, after it."""
    b = board.copy(stack=False)
    b.push(move)
    if b.is_checkmate():
        return 1, b
    if b.is_stalemate() or b.is_insufficient_material() or len(b.piece_map()) == 2:
        return 0, b
    return -category(_TB.probe_wdl(b)), b


def random_position(cls, rng):
    white, black = cls.split("v")
    if rng.random() < 0.5:  # either colour can hold the stronger side
        white, black = black, white
    for _ in range(1000):
        b = chess.Board(None)
        squares = list(chess.SQUARES)
        rng.shuffle(squares)
        ok = True
        for color, pieces in ((chess.WHITE, white), (chess.BLACK, black)):
            for ch in pieces:
                pt = PIECE[ch]
                while squares:
                    sq = squares.pop()
                    if pt == chess.PAWN and chess.square_rank(sq) in (0, 7):
                        squares.insert(0, sq)
                        if all(chess.square_rank(s) in (0, 7) for s in squares):
                            ok = False
                            break
                        continue
                    b.set_piece_at(sq, chess.Piece(pt, color))
                    break
        if not ok:
            continue
        b.turn = rng.random() < 0.5
        b.castling_rights = chess.BB_EMPTY
        b.ep_square = None
        b.halfmove_clock = 0
        b.fullmove_number = 1
        if b.is_valid() and not b.is_game_over():
            return b
    return None


def label(board):
    root = category(_TB.probe_wdl(board))
    if root < 0:
        return None
    moves = {}
    for mv in board.legal_moves:
        c, child = child_category(board, mv)
        moves[mv.uci()] = {"res": c}
        if root == 1 and c == 1:
            if child.is_checkmate():
                d = 0
            elif board.is_zeroing(mv):
                d = 1
            else:
                d = 1 + abs(_TB.probe_dtz(child))
            moves[mv.uci()]["dtz"] = d
    correct = [m for m, v in moves.items() if v["res"] == root]
    if len(correct) == len(moves) or not correct:
        return None
    if root == 1:
        best = min(moves[m]["dtz"] for m in correct)
        dtz_opt = [m for m in correct if moves[m]["dtz"] == best]
    else:
        dtz_opt = list(correct)
    return {"root": root, "moves": moves, "correct": sorted(correct),
            "dtz_opt": sorted(dtz_opt)}


def gen_class(args):
    cls, n, seed, max_tries = args
    rng = random.Random(f"{seed}-{cls}")
    out, seen, tries = [], set(), 0
    while len(out) < n and tries < max_tries:
        tries += 1
        b = random_position(cls, rng)
        if b is None:
            continue
        fen = b.fen()
        if fen in seen:
            continue
        seen.add(fen)
        try:
            lab = label(b)
        except (KeyError, chess.syzygy.MissingTableError):
            return cls, None, "missing table"
        if lab is None:
            continue
        out.append({"fen": fen, "cls": cls, **lab})
    return cls, out, f"{len(out)} kept / {tries} tried"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--syzygy", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-class", type=int, default=400)
    ap.add_argument("--classes", default=None,
                    help="comma list, e.g. KQvKR,KRPvKR (default: 28 built in)")
    ap.add_argument("--splits", default="0.1,0.3,0.6",
                    help="calib,dev,test fractions per class")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--max-tries", type=int, default=200000)
    ap.add_argument("--allow-missing", action="store_true",
                    help="skip classes whose tables are missing instead of failing")
    a = ap.parse_args()

    classes = a.classes.split(",") if a.classes else DEFAULT_CLASSES
    fr = [float(x) for x in a.splits.split(",")]
    assert len(fr) == 3 and abs(sum(fr) - 1) < 1e-6, "--splits must sum to 1"
    jobs = [(c, a.per_class, a.seed, a.max_tries) for c in classes]
    workers = a.workers or None
    recs, missing = [], []
    with Pool(workers, initializer=_init, initargs=(a.syzygy,)) as pool:
        for cls, out, msg in pool.imap_unordered(gen_class, jobs):
            print(f"{cls:8s} {msg}", file=sys.stderr)
            if out is None:
                print(f"  skipped {cls}: tables missing under {a.syzygy}", file=sys.stderr)
                missing.append(cls)
                continue
            if len(out) < a.per_class:
                print(f"  only {len(out)} positions for {cls}", file=sys.stderr)
            rng = random.Random(f"split-{a.seed}-{cls}")
            rng.shuffle(out)
            n = len(out)
            n_cal, n_dev = round(fr[0] * n), round(fr[1] * n)
            for i, r in enumerate(out):
                r["split"] = "calib" if i < n_cal else ("dev" if i < n_cal + n_dev else "test")
            recs.extend(out)
    if missing and not a.allow_missing:
        sys.exit(f"tables missing for {missing}; get the full 3-4-5 set "
                 "or pass --allow-missing")
    order = {c: i for i, c in enumerate(classes)}
    recs.sort(key=lambda r: (order[r["cls"]], r["split"], r["fen"]))
    for i, r in enumerate(recs):
        r["id"] = i
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    sp = Counter(r["split"] for r in recs)
    res = Counter(r["root"] for r in recs)
    print(f"wrote {len(recs)} positions to {a.out}: {dict(sp)}; "
          f"won {res[1]}, drawn {res[0]}", file=sys.stderr)


if __name__ == "__main__":
    main()
