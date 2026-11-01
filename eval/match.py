#!/usr/bin/env python3
"""Time-parity match between two lc0 configurations (e.g. FP16 vs INT8) on
the same GPU, one game at a time, ponder off.

Games start from a built-in book of balanced openings; each opening is
played twice with colours swapped (a game pair). The clock is kept here, by
wall time per move, with increment; lc0 gets wtime/btime/winc/binc and its
own time manager decides. Adjudication: Syzygy (if --syzygy, at <= 5 men),
resign (both engines agree |score| >= 1000 cp for 4 plies each), draw (both
within 10 cp for 8 plies after ply 80), plus the usual rules via python-chess.

Elo is computed from game-pair (pentanomial) scores, which is the correct
error model for paired openings. Resumable: finished games are skipped.

    python eval/match.py --lc0 lc0 --name fp16_vs_s4 \
        --a fp16=models/fp16.pb.gz:optimize=6 --b s4=models/s4.pb.gz:optimize=8 \
        --tc 10+0.1 --pairs 250 --syzygy /path/to/syzygy
    python eval/match.py ... --estimate 2      # time 2 games and extrapolate
"""
import argparse
import io
import json
import math
import sys
import time
from pathlib import Path

import chess
import chess.pgn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import append_jsonl, read_jsonl, write_json  # noqa: E402
from lc0uci import Lc0  # noqa: E402

# Balanced, mainstream openings (SAN). Each is played with both colours.
OPENINGS = [
    "e4 e5 Nf3 Nc6 Bb5 a6", "e4 e5 Nf3 Nc6 Bc4 Bc5", "e4 e5 Nf3 Nc6 d4 exd4",
    "e4 e5 Nf3 Nf6", "e4 e5 Nc3 Nf6", "e4 e5 f4 exf4",
    "e4 c5 Nf3 d6 d4 cxd4", "e4 c5 Nf3 Nc6 d4 cxd4", "e4 c5 Nf3 e6 d4 cxd4",
    "e4 c5 Nc3 Nc6 g3", "e4 c5 c3 Nf6", "e4 c5 Nf3 d6 Bb5+",
    "e4 e6 d4 d5 Nc3 Nf6", "e4 e6 d4 d5 Nd2 c5", "e4 e6 d4 d5 e5 c5",
    "e4 c6 d4 d5 Nc3 dxe4", "e4 c6 d4 d5 e5 Bf5", "e4 d5 exd5 Qxd5 Nc3 Qa5",
    "e4 d6 d4 Nf6 Nc3 g6", "e4 g6 d4 Bg7 Nc3 d6", "e4 Nf6 e5 Nd5 d4 d6",
    "d4 d5 c4 e6 Nc3 Nf6", "d4 d5 c4 c6 Nf3 Nf6", "d4 d5 c4 dxc4 Nf3 Nf6",
    "d4 Nf6 c4 e6 Nc3 Bb4", "d4 Nf6 c4 e6 Nf3 b6", "d4 Nf6 c4 g6 Nc3 Bg7",
    "d4 Nf6 c4 g6 Nc3 d5", "d4 Nf6 c4 c5 d5 b5", "d4 Nf6 c4 c5 d5 e6",
    "d4 Nf6 Nf3 e6 Bg5 c5", "d4 d5 Bf4 Nf6 e3 c5", "d4 f5 g3 Nf6 Bg2 g6",
    "d4 Nf6 c4 e6 g3 d5", "d4 d5 Nf3 Nf6 e3 e6", "d4 e6 c4 Bb4+ Bd2 Qe7",
    "c4 e5 Nc3 Nf6 Nf3 Nc6", "c4 c5 Nc3 Nc6 g3 g6", "c4 Nf6 Nc3 e6 e4 d5",
    "c4 e6 Nf3 d5 g3 Nf6", "Nf3 d5 g3 Nf6 Bg2 c6", "Nf3 Nf6 c4 b6 g3 Bb7",
    "Nf3 c5 c4 Nc6 Nc3 e5", "g3 d5 Bg2 Nf6 Nf3 c6", "b3 e5 Bb2 Nc6 e3 d5",
    "e4 e5 Nf3 d6 d4 Nf6", "e4 c5 Nf3 g6 d4 cxd4", "d4 c5 d5 e5 e4 d6",
    "e4 b6 d4 Bb7 Bd3 e6", "d4 Nf6 Bg5 e6 e4 h6",
]


def opening_moves(i):
    b = chess.Board()
    out = []
    for san in OPENINGS[i % len(OPENINGS)].split():
        mv = b.parse_san(san)
        out.append(mv.uci())
        b.push(mv)
    return out


def parse_engine(spec):
    """name=path[:backend_opts][@backend]"""
    name, rest = spec.split("=", 1)
    backend = None
    if "@" in rest:
        rest, backend = rest.rsplit("@", 1)
    # Windows paths contain ':' after the drive letter; split on the LAST ':'
    # only if what follows looks like backend options.
    path, opts = rest, ""
    if ":" in rest:
        head, tail = rest.rsplit(":", 1)
        if "=" in tail or tail == "":
            path, opts = head, tail
    return {"name": name, "weights": path, "opts": opts, "backend": backend}


def parse_tc(tc):
    base, inc = (tc.split("+") + ["0"])[:2]
    return int(float(base) * 1000), int(float(inc) * 1000)


def elo_from_score(s):
    s = min(max(s, 1e-6), 1 - 1e-6)
    return -400 * math.log10(1 / s - 1) + 0.0  # avoid printing -0.0


def summarize(games, a, b):
    """Elo of A over B from pentanomial pair scores (+ trinomial counts)."""
    w = sum(1 for g in games if g["score_a"] == 1)
    d = sum(1 for g in games if g["score_a"] == 0.5)
    l = sum(1 for g in games if g["score_a"] == 0)
    pairs = {}
    for g in games:
        pairs.setdefault(g["pair"], []).append(g["score_a"])
    full = [sum(v) for v in pairs.values() if len(v) == 2]
    res = {"games": len(games), "a": a, "b": b, "w": w, "d": d, "l": l,
           "pairs_complete": len(full)}
    if games:
        s = (w + d / 2) / len(games)
        res["score"] = s
        res["elo"] = elo_from_score(s)
    if len(full) >= 2:
        penta = {k: sum(1 for x in full if x == k) for k in (0, 0.5, 1, 1.5, 2)}
        res["pentanomial"] = penta
        n = len(full)
        mu = sum(full) / (2 * n)
        var = sum((x / 2 - mu) ** 2 for x in full) / n
        se = math.sqrt(var / n) if var > 0 else 0.0
        res["score_pairs"] = mu
        res["elo_pairs"] = elo_from_score(mu)
        lo, hi = mu - 1.96 * se, mu + 1.96 * se
        res["elo_ci95"] = [elo_from_score(lo), elo_from_score(hi)]
        res["los"] = 0.5 * (1 + math.erf((mu - 0.5) / (se * math.sqrt(2)))) if se > 0 else None
    return res


class Game:
    def __init__(self, a):
        self.a = a
        self.tb = None
        if a.syzygy:
            import chess.syzygy
            self.tb = chess.syzygy.open_tablebase(a.syzygy)

    def tb_result(self, board):
        if self.tb is None or len(board.piece_map()) > self.a.tb_men:
            return None
        if board.castling_rights:
            return None
        try:
            wdl = self.tb.probe_wdl(board)
        except Exception:
            return None
        if wdl in (-1, 0, 1):
            return "1/2-1/2"
        stm_wins = wdl == 2
        white_wins = stm_wins == (board.turn == chess.WHITE)
        return "1-0" if white_wins else "0-1"

    def play(self, white, black, opening):
        a = self.a
        base, inc = parse_tc(a.tc)
        clock = {chess.WHITE: base, chess.BLACK: base}
        eng = {chess.WHITE: white, chess.BLACK: black}
        board = chess.Board()
        moves = list(opening)
        for m in moves:
            board.push_uci(m)
        for e in (white, black):
            e.newgame()
        stats = {chess.WHITE: {"nodes": [], "ms": []}, chess.BLACK: {"nodes": [], "ms": []}}
        scores = []  # (side, cp)
        result, reason = None, None
        while result is None:
            if board.is_game_over(claim_draw=True):
                result = board.result(claim_draw=True)
                o = board.outcome(claim_draw=True)
                reason = o.termination.name.lower() if o else "rules"
                break
            r = self.tb_result(board)
            if r:
                result, reason = r, "tablebase"
                break
            if board.ply() >= a.max_plies:
                result, reason = "1/2-1/2", "max_plies"
                break
            side = board.turn
            e = eng[side]
            t0 = time.perf_counter()
            res = e.go(moves=moves, clock={"wtime": clock[chess.WHITE], "btime": clock[chess.BLACK],
                                           "winc": inc, "binc": inc},
                       timeout=max(60, 3 * clock[side] / 1000 + 30))
            used = (time.perf_counter() - t0) * 1000
            clock[side] -= used
            if clock[side] < 0 and not a.no_flag:
                result = "0-1" if side == chess.WHITE else "1-0"
                reason = "time"
                break
            clock[side] += inc
            mv = chess.Move.from_uci(res["bestmove"])
            if mv not in board.legal_moves:
                result = "0-1" if side == chess.WHITE else "1-0"
                reason = f"illegal {res['bestmove']}"
                break
            stats[side]["nodes"].append(res.get("nodes") or 0)
            stats[side]["ms"].append(used)
            cp = res.get("score_cp")
            if res.get("mate") is not None:
                cp = 30000 if res["mate"] > 0 else -30000
            scores.append((side, cp))
            board.push(mv)
            moves.append(mv.uci())
            # resign / draw adjudication on the last plies of both engines
            if len(scores) >= 8 and all(c is not None for _, c in scores[-8:]):
                last = scores[-8:]
                # cp from white's view
                wv = [c if s == chess.WHITE else -c for s, c in last]
                if all(v >= a.resign_cp for v in wv):
                    result, reason = "1-0", "adjudicated_resign"
                elif all(v <= -a.resign_cp for v in wv):
                    result, reason = "0-1", "adjudicated_resign"
                elif board.ply() >= a.draw_after and all(abs(v) <= a.draw_cp for v in wv):
                    result, reason = "1/2-1/2", "adjudicated_draw"
        return board, moves, result, reason, stats, clock


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lc0", default="lc0")
    ap.add_argument("--name", required=True)
    ap.add_argument("--a", required=True, help="name=weights[:backend_opts][@backend]")
    ap.add_argument("--b", required=True)
    ap.add_argument("--backend", default="onnx-trt")
    ap.add_argument("--tc", default="10+0.1", help="base seconds + increment seconds")
    ap.add_argument("--pairs", type=int, default=250)
    ap.add_argument("--estimate", type=int, default=0, help="play N games, print time/game, stop")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--minibatch", type=int, default=0)
    ap.add_argument("--move-overhead", type=int, default=50)
    ap.add_argument("--syzygy", default=None, help="adjudication only; engines never see it")
    ap.add_argument("--tb-men", type=int, default=5)
    ap.add_argument("--resign-cp", type=int, default=1000)
    ap.add_argument("--draw-cp", type=int, default=10)
    ap.add_argument("--draw-after", type=int, default=80)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--no-flag", action="store_true", help="never forfeit on time")
    ap.add_argument("--out-dir", default="results")
    a = ap.parse_args()

    out = Path(a.out_dir) / f"match_{a.name}"
    out.mkdir(parents=True, exist_ok=True)
    ea, eb = parse_engine(a.a), parse_engine(a.b)
    games_path = out / "games.jsonl"
    done = {g["game"] for g in read_jsonl(games_path)}

    def start(spec, tag):
        opts = {"MoveOverheadMs": a.move_overhead, "Ponder": False}
        if a.threads:
            opts["Threads"] = a.threads
        if a.minibatch:
            opts["MinibatchSize"] = a.minibatch
        return Lc0(a.lc0, spec["weights"], backend=spec["backend"] or a.backend,
                   backend_opts=spec["opts"], options=opts,
                   log=str(out / f"{tag}.lc0.log"), name=spec["name"])

    A, B = start(ea, "A"), start(eb, "B")
    game = Game(a)
    n_games = a.estimate if a.estimate else 2 * a.pairs
    t_start = time.time()
    played = 0
    pgn_path = out / "games.pgn"
    try:
        for gi in range(n_games):
            if gi in done and not a.estimate:
                continue
            pair = gi // 2
            a_white = gi % 2 == 0
            white, black = (A, B) if a_white else (B, A)
            op = opening_moves(pair)
            t0 = time.time()
            board, moves, result, reason, stats, clock = game.play(white, black, op)
            secs = time.time() - t0
            score_white = {"1-0": 1, "0-1": 0}.get(result, 0.5)
            score_a = score_white if a_white else 1 - score_white
            na = stats[chess.WHITE if a_white else chess.BLACK]
            nb = stats[chess.BLACK if a_white else chess.WHITE]
            rec = {"game": gi, "pair": pair, "opening": pair % len(OPENINGS),
                   "white": ea["name"] if a_white else eb["name"],
                   "black": eb["name"] if a_white else ea["name"],
                   "result": result, "reason": reason, "score_a": score_a,
                   "plies": len(moves), "seconds": round(secs, 1),
                   "a_nodes_per_move": (sum(na["nodes"]) / len(na["nodes"])) if na["nodes"] else 0,
                   "b_nodes_per_move": (sum(nb["nodes"]) / len(nb["nodes"])) if nb["nodes"] else 0,
                   "a_ms_per_move": (sum(na["ms"]) / len(na["ms"])) if na["ms"] else 0,
                   "b_ms_per_move": (sum(nb["ms"]) / len(nb["ms"])) if nb["ms"] else 0,
                   "clock_left_ms": {"white": round(clock[chess.WHITE]), "black": round(clock[chess.BLACK])}}
            if not a.estimate:
                append_jsonl(games_path, rec)
                g = chess.pgn.Game.from_board(board)
                g.headers.update({"Event": a.name, "Round": str(gi + 1), "White": rec["white"],
                                  "Black": rec["black"], "Result": result,
                                  "TimeControl": a.tc, "Termination": reason})
                with open(pgn_path, "a", encoding="utf-8") as f:
                    print(g, file=f, end="\n\n")
            played += 1
            el = time.time() - t_start
            left = n_games - len(done) - played if not a.estimate else 0
            print(f"game {gi + 1}: {rec['white']} - {rec['black']} {result} ({reason}, "
                  f"{rec['plies']} plies, {secs:.0f}s; nodes/move A {rec['a_nodes_per_move']:.0f} "
                  f"B {rec['b_nodes_per_move']:.0f})  eta {el / played * left / 3600:.1f} h",
                  file=sys.stderr, flush=True)
            if not a.estimate and played % 10 == 0:
                write_json(out / "summary.json", summarize(read_jsonl(games_path), ea["name"], eb["name"]))
    finally:
        A.quit()
        B.quit()
    if a.estimate:
        per = (time.time() - t_start) / max(played, 1)
        print(f"~{per:.0f} s/game at {a.tc}: {2 * a.pairs} games ≈ {2 * a.pairs * per / 3600:.1f} h. "
              f"Pairs that fit 8 h: {int(8 * 3600 / per / 2)}", file=sys.stderr)
        return
    s = summarize(read_jsonl(games_path), ea["name"], eb["name"])
    write_json(out / "summary.json", s)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
