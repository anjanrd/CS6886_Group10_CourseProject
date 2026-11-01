"""Fixed-movetime extension; stock legal-game and adjudication semantics retained."""
import io
import math
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import chess
import chess.pgn
import numpy as np

from eval.match import Game
from execution.support import fingerprint, read_json
from execution.ultimate_contract import CONTRACT, freeze, finite_positive, atomic_json


def match_counters(nodes, engine_ms, wall_ms):
    # Forced/cached moves can print zero new visits or sub-millisecond time.
    # Keep those real counters; missing, negative and nonfinite values are faults.
    for label,value in (('nodes',nodes),('engine_ms',engine_ms)):
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
            raise ValueError('Invalid match '+label)
    if int(nodes)!=nodes:
        raise ValueError('Noninteger match nodes')
    finite_positive(wall_ms,'match wall_ms')


def play(white, black, opening, budget, syzygy, guard=lambda: None):
    settings = CONTRACT['match']
    args = SimpleNamespace(syzygy=str(syzygy), **{k: settings[k] for k in
                           ('tb_men', 'resign_cp', 'draw_cp', 'draw_after', 'max_plies')})
    rules = Game(args)
    board = chess.Board()
    moves = list(opening)
    for m in moves:
        board.push_uci(m)
    stats, scores = [], []
    try:
        for e in (white, black):
            e.newgame()
        result = reason = None
        while result is None:
            guard()
            if board.is_game_over(claim_draw=True):
                result = board.result(claim_draw=True)
                reason = board.outcome(claim_draw=True).termination.name.lower()
                break
            result = rules.tb_result(board)
            if result:
                reason = 'tablebase'
                break
            if board.ply() >= args.max_plies:
                result, reason = '1/2-1/2', 'max_plies'
                break
            side = board.turn
            e = white if side == chess.WHITE else black
            start = time.perf_counter()
            res = e.go(moves=moves, movetime=budget, timeout=30)
            used = (time.perf_counter() - start) * 1000
            move = chess.Move.from_uci(res['bestmove'])
            if move not in board.legal_moves:
                raise ValueError('Illegal match move: ' + res['bestmove'])
            match_counters(res.get('nodes'),res.get('time_ms'),used)
            stats.append({'ply': board.ply(), 'side': 'white' if side else 'black',
                          'wall_ms': used, 'engine_ms': res['time_ms'], 'nodes': res['nodes'],
                          'printed_nps': res.get('printed_nps'), 'best': move.uci(),
                          'score_cp':res.get('score_cp'),'mate':res.get('mate'),
                          'zero_new_visits':res['nodes']==0})
            cp = res.get('score_cp')
            if res.get('mate') is not None:
                cp = 30000 if res['mate'] > 0 else -30000
            scores.append((side, cp))
            board.push(move)
            moves.append(move.uci())
            if len(scores) >= 8 and all(c is not None for _, c in scores[-8:]):
                wv = [c if s else -c for s, c in scores[-8:]]
                if all(v >= args.resign_cp for v in wv):
                    result, reason = '1-0', 'adjudicated_resign'
                elif all(v <= -args.resign_cp for v in wv):
                    result, reason = '0-1', 'adjudicated_resign'
                elif board.ply() >= args.draw_after and all(abs(v) <= args.draw_cp for v in wv):
                    result, reason = '1/2-1/2', 'adjudicated_draw'
        return board, result, reason, stats
    finally:
        if rules.tb:
            rules.tb.close()


def assignments(pairs=250, batch=5):
    # Fixed batches of complete pairs; reverse first-color order for odd pairs.
    for first in range(0, pairs, batch):
        for budget in (100, 200):
            for pair in range(first, min(first + batch, pairs)):
                for color in ((0, 1) if pair % 2 == 0 else (1, 0)):
                    yield budget, pair, color


def validate_game(envelope, contract):
    if envelope.get('contract') != fingerprint(contract):
        raise ValueError('Match resume contract differs')
    row = envelope['record']
    budget, pair, color = row['budget_ms'], row['pair'], row['color']
    if budget not in (100, 200) or type(pair) is not int or not 0 <= pair < len(contract['openings']) or color not in (0, 1):
        raise ValueError('Invalid match key')
    white, black = ('fp16', 's4') if color == 0 else ('s4', 'fp16')
    if (row['white'], row['black']) != (white, black):
        raise ValueError('Wrong match colors')
    if row['result'] not in ('1-0', '0-1', '1/2-1/2'):
        raise ValueError('Game incomplete')
    sw = {'1-0': 1., '0-1': 0., '1/2-1/2': .5}[row['result']]
    if row['score_fp16'] != (sw if color == 0 else 1 - sw):
        raise ValueError('Result/score mismatch')
    stream = io.StringIO(envelope['pgn'])
    game = chess.pgn.read_game(stream)
    if game is None or game.errors or chess.pgn.read_game(stream) is not None:
        raise ValueError('PGN invalid')
    headers = {'White': white, 'Black': black, 'Result': row['result'],
               'Termination': row['reason'], 'TimeControl': '-', 'MoveTimeMs': str(budget)}
    if any(game.headers.get(k) != v for k, v in headers.items()):
        raise ValueError('PGN/record headers mismatch')
    moves = [m.uci() for m in game.mainline_moves()]
    opening = contract['openings'][pair]
    if moves[:6] != opening or len(moves) != row['plies']:
        raise ValueError('PGN opening/ply mismatch')
    if len(row['moves']) != len(moves) - 6:
        raise ValueError('Missing match timing records')
    for ply, stats in enumerate(row['moves'], 6):
        if stats['ply'] != ply or stats['best'] != moves[ply] or stats['side'] != ('white' if ply % 2 == 0 else 'black'):
            raise ValueError('Match move/timing identity mismatch')
        match_counters(stats.get('nodes'),stats.get('engine_ms'),stats.get('wall_ms'))
    return row


def make_envelope(board, result, reason, stats, budget, pair, color, seconds, contract):
    white, black = ('fp16', 's4') if color == 0 else ('s4', 'fp16')
    sw = {'1-0': 1., '0-1': 0., '1/2-1/2': .5}[result]
    row = {'budget_ms': budget, 'pair': pair, 'color': color, 'white': white, 'black': black,
           'result': result, 'reason': reason, 'plies': board.ply(), 'seconds': seconds,
           'score_fp16': sw if color == 0 else 1-sw, 'moves': stats}
    pgn = chess.pgn.Game.from_board(board)
    pgn.headers.update({'Event': 'BT3 fixed-movetime', 'White': white, 'Black': black,
                        'Round': str(pair + 1), 'Result': result, 'Termination': reason,
                        'TimeControl': '-', 'MoveTimeMs': str(budget)})
    envelope = {'contract': fingerprint(contract), 'record': row, 'pgn': str(pgn)}
    validate_game(envelope, contract)
    return envelope


def journal(folder, contract):
    rows, keys = [], set()
    for p in sorted(Path(folder).glob('game-*.json')):
        e = read_json(p)
        row = validate_game(e, contract)
        key = (row['budget_ms'], row['pair'], row['color'])
        if key in keys or p.name != f'game-{key[0]}-{key[1]:03d}-{key[2]}.json':
            raise ValueError('Duplicate/wrongly named match envelope')
        rows.append(e)
        keys.add(key)
    return rows


def timing_gate(envelopes):
    summaries, failures = [], []
    for budget in (100, 200):
        for model in ('fp16', 's4'):
            values = [s['wall_ms'] for e in envelopes if e['record']['budget_ms'] == budget
                      for s in e['record']['moves'] if e['record'][s['side']] == model]
            if not values:
                raise ValueError('Pilot lacks model timing coverage')
            mean, median, p95 = statistics.mean(values), statistics.median(values), float(np.percentile(values, 95))
            row = {'model': model, 'budget_ms': budget, 'n': len(values), 'mean_ms': mean,
                   'median_ms': median, 'p95_ms': p95}
            summaries.append(row)
            if mean > budget + 10 or median > budget + 10 or p95 > budget + 25:
                failures.append(row)
    return {'status': 'BLOCKED' if failures else 'verified', 'timings': summaries, 'failures': failures}


def summary(envelopes):
    result = []
    for budget in (100, 200):
        games = [e['record'] for e in envelopes if e['record']['budget_ms'] == budget]
        n = len(games)
        wins = sum(g['score_fp16'] == 1 for g in games)
        losses = sum(g['score_fp16'] == 0 for g in games)
        draws = n - wins - losses
        by_pair = {}
        for g in games:
            by_pair.setdefault(g['pair'], set()).add(g['color'])
        from collections import Counter
        result.append({'budget_ms': budget, 'games': n, 'complete_pairs': sum(len(v) == 2 for v in by_pair.values()),
                       'fp16_wins': wins, 's4_wins': losses, 'draws': draws,
                       'fp16_win_pct': 100*wins/n if n else None,
                       's4_win_pct': 100*losses/n if n else None, 'draw_pct': 100*draws/n if n else None,
                       'fp16_score': (wins + draws/2)/n if n else None,
                       'more_wins': 'tie' if wins == losses else ('fp16' if wins > losses else 's4'),
                       'absolute_win_margin': abs(wins-losses), 'pp_win_margin': 100*abs(wins-losses)/n if n else None,
                       'termination': dict(Counter(g['reason'] for g in games))})
    return result
