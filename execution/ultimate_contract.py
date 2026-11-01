"""Frozen Ultimate scientific contract and portable identity/validation helpers."""
from datetime import datetime, timezone
import hashlib
import math
import os
import json
import tempfile
import sys
import time
from pathlib import Path

import chess

from execution.support import fingerprint, identity, read_json

ROOT = Path(__file__).resolve().parents[1]
DEADLINE = datetime.fromisoformat('2026-11-01T11:45:00+05:30').timestamp()
A10 = ('fp16', 's1', 's2', 's3', 's4', 's4aa', 's0', 's1sp', 'swap_pol8', 'swap_val8')
B9 = tuple(m for m in A10 if m != 's0')
SEED = 20261030
CONTRACT = {
    'version': 1, 'seed': SEED, 'encoding': 'INPUT_CLASSICAL_112_PLANE',
    'history': 'FEN-only', 'halfmove': 0, 'syzygy': {'2': 1, '-2': -1, '0/1/-1': 0},
    'calibration': 512, 'development': 3360, 'selected': 's4',
    'recipe': {'percentile': 99.999, 'alpha': .7},
    'raw_models': A10, 'search_models': B9,
    'M': {'n': 6720, 'classes': 28, 'per_class': 240},
    'sensitivity_exclusions': [388, 611, 627, 965, 1100],
    'N': {'n': 24000, 'classes': 12, 'per_class': 2000, 'first_id': 11200,
          'max_attempts_per_class': 200000, 'evaluation_only': True},
    'challenge': 'FP16 raw best not preserving OR printed preserving mass < 0.8',
    'raw': {'nodes': 1, 'Threads': 1, 'MinibatchSize': 256, 'PolicyTemperature': 1.0},
    'nodes': {'budgets': [100, 400, 800, 1600], 'Threads': 1,
              'MinibatchSize': 256, 'SmartPruningFactor': 0},
    'low_nodes': [10, 25, 50, 100, 200], 's0_nodes': [100, 400],
    'timed': {'budgets_ms': [100, 200, 400, 800], 'MinibatchSize': 256,
              'MoveOverheadMs': 20, 'Threads': 'default', 'SmartPruningFactor': 'default'},
    'match': {'budgets_ms': [100, 200], 'pairs_per_budget': 250, 'distinct_openings': 250,
              'opening_plies': 6, 'MinibatchSize': 64, 'MoveOverheadMs': 20,
              'Ponder': False, 'Threads': 'default', 'SmartPruningFactor': 'default',
              'tb_men': 5, 'resign_cp': 1000, 'draw_cp': 10, 'draw_after': 80,
              'max_plies': 400, 'watchdog_seconds': 30, 'pair_batch': 5,
              'pilot_pairs_per_budget': 1, 'pilot_mean_median_extra_ms': 10,
              'pilot_p95_extra_ms': 25, 'time_forfeits': False},
    'statistics': {'primary': ['fp16', 's4'], 'secondary': 'other contrasts',
                   'mcnemar': 'exact nominal; Holm by study/cohort family',
                   'bootstrap_seed': SEED, 'bootstrap_replicates': 10000,
                   'bootstrap': 'material-stratified paired legal-symmetry groups',
                   'matches': 'descriptive counts, percentages, score, absolute/pp win margin'},
    'numeric_gate': {'wl': .005, 'd': .005, 'v': .005, 'm': .25, 'policy': .005,
                     'raw_outcome_changes': 0, 'performance_ratio': [.8, 1.2]},
    'background_flags_informational': True, 'single_GPU': True,
    'warmup': {'raw_nodes': 'imported convention, no extra excluded search',
               'timed': 'one excluded search on first root per engine/budget',
               'match': 'one excluded 2000ms search per resident engine'},
}


def atomic_json(path, value):
    """New mutable evidence uses LF without changing the frozen legacy helper."""
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    data=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode('utf-8')
    with tempfile.NamedTemporaryFile(prefix='lc0-ultimate-',suffix='.json',delete=False) as f:
        temp=Path(f.name)
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    try:
        if temp.stat().st_dev!=path.parent.stat().st_dev:
            raise RuntimeError('Atomic commit needs OS TEMP on destination volume')
        # Windows readers (including scanners/indexers) can briefly omit
        # FILE_SHARE_DELETE. Retry only the same atomic rename, never regenerate
        # data or truncate the last good checkpoint. Persistent errors propagate.
        for attempt in range(7):
            try:
                os.replace(temp,path)
                break
            except PermissionError as exc:
                if os.name!='nt' or getattr(exc,'winerror',None) not in (5,32,33) or attempt==6:
                    raise
                print(json.dumps({'atomic_replace_retry':attempt+1,'winerror':exc.winerror,
                                  'destination':str(path)}),file=sys.stderr,flush=True)
                time.sleep(.05*2**attempt)
    finally:
        if temp.exists():
            # Preserve the complete candidate on persistent failure as evidence.
            print('Uncommitted atomic candidate preserved: '+str(temp),file=sys.stderr,flush=True)


def portable_identity(path, root=ROOT):
    item = identity(path)
    try:
        item['path'] = Path(path).resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        pass
    return item


def freeze(path, value):
    # JSON canonicalization removes tuple/list differences on resumed contracts.
    import json
    value = json.loads(json.dumps(value, allow_nan=False))
    if Path(path).exists():
        if read_json(path) != value:
            raise ValueError('Immutable contract changed: ' + str(path))
    else:
        atomic_json(path, value)
    return value


def group_variants(board):
    if board.castling_rights:
        spatial = [board]
    elif board.pawns or board.ep_square is not None:
        spatial = [board, board.transform(chess.flip_horizontal)]
    else:
        transforms = [lambda x: x, chess.flip_horizontal, chess.flip_vertical,
                      lambda x: chess.flip_horizontal(chess.flip_vertical(x)),
                      chess.flip_diagonal, chess.flip_anti_diagonal,
                      lambda x: chess.flip_horizontal(chess.flip_diagonal(x)),
                      lambda x: chess.flip_vertical(chess.flip_diagonal(x))]
        spatial = [board.transform(t) for t in transforms]
    return spatial + [b.mirror() for b in spatial]


def group_id(board):
    key = min(' '.join(b.fen(en_passant='fen').split()[:4]) for b in group_variants(board))
    return hashlib.sha256(key.encode()).hexdigest()


def validate_root(row):
    b = chess.Board(row['fen'])
    if type(row['id']) is not int or row['id'] < 0 or not b.is_valid() or b.is_game_over():
        raise ValueError('Invalid root identity/board')
    if b.halfmove_clock != 0 or b.castling_rights or b.ep_square is not None:
        raise ValueError('Root history convention changed')
    if set(row['moves']) != {m.uci() for m in b.legal_moves}:
        raise ValueError('Root legal move labels incomplete')
    correct = {m for m, v in row['moves'].items() if v['res'] == row['root']}
    if row['root'] not in (0, 1) or correct != set(row['correct']) or not 0 < len(correct) < len(row['moves']):
        raise ValueError('Root decision-sensitive categories changed')
    if not set(row['dtz_opt']) <= correct or not row['dtz_opt']:
        raise ValueError('Invalid DTZ labels')
    return b


def finite_positive(x, label, allow_zero=False):
    if type(x) not in (int, float) or not math.isfinite(x) or (x < 0 if allow_zero else x <= 0):
        raise ValueError('Invalid ' + label + ': ' + repr(x))


def validate_observation(row, root, study, budget):
    key = 'movetime' if study == 'C' else 'budget'
    if row.get('id') != root['id'] or row.get(key) != budget:
        raise ValueError('Observation key mismatch')
    if row.get('best') not in root['moves']:
        raise ValueError('Missing/illegal selected move')
    finite_positive(row.get('nodes'), 'actual nodes')
    if study == 'A':
        p = row.get('p', {})
        if set(p) != set(root['moves']):
            raise ValueError('Printed policy legal-move coverage mismatch')
        for x in p.values():
            finite_positive(x, 'policy', True)
        finite_positive(sum(p.values()), 'policy mass')
        for k in ('wl', 'd', 'v', 'm'):
            if type(row.get(k)) not in (int, float) or not math.isfinite(row[k]):
                raise ValueError('Missing/nonfinite raw output: ' + k)
    else:
        finite_positive(row.get('wall'), 'wall seconds')
        finite_positive(row.get('time_ms'), 'engine ms')
    return row


def hard_ids(roots, raw):
    if set(raw) != {r['id'] for r in roots}:
        raise ValueError('Incomplete FP16 screening denominator')
    return [r['id'] for r in roots if raw[r['id']]['best'] not in r['correct'] or
            sum(p for m, p in raw[r['id']]['p'].items() if m in r['correct']) < .8]


def safe_output(path, root=ROOT):
    p = Path(path).resolve()
    reserved = [root / x for x in ('data', 'models', 'nets', 'archive', 'submissionready')]
    if p == root or any(p == r or r in p.parents for r in reserved):
        raise ValueError('Output overlaps immutable inputs/final release')
    return p


def authorized_deadline():
    # The submission remains usable after this project's reserved laptop window.
    # Its evaluator can supply a new window; current working/external replay
    # workers inherit the original explicit deadline from their controller.
    return float(os.environ.get('LC0_AUTHORIZED_UNTIL', DEADLINE if (ROOT/'.git').exists() else float('inf')))


def deadline_check(seconds_needed=0, now=None, deadline=None):
    now = datetime.now(timezone.utc).timestamp() if now is None else now
    if now + seconds_needed + 60 >= (authorized_deadline() if deadline is None else deadline):
        raise TimeoutError('Authorized working deadline reached; protect submission buffer')
