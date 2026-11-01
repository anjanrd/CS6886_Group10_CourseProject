"""Evaluation-only data extension using the imported construction and scorer."""
from collections import Counter
import json
from pathlib import Path
import random
import os
from concurrent.futures import ProcessPoolExecutor,as_completed

import chess

from tb import build_tbset as tb
from execution.support import fingerprint, read_json, strict_jsonl
from execution.ultimate_contract import (ROOT, CONTRACT, SEED, freeze, group_id, group_variants,
                                         portable_identity, validate_root, deadline_check, atomic_json)


def original_roots():
    rows = strict_jsonl(ROOT / 'data/tbset.jsonl')
    if len(rows) != 11200 or {r['id'] for r in rows} != set(range(11200)):
        raise ValueError('Original dataset identity/count mismatch')
    for r in rows:
        validate_root(r)
    return rows


def freeze_main(out):
    rows = original_roots()
    main = [r for r in rows if r['split'] == 'test']
    counts = Counter(r['cls'] for r in main)
    if len(main) != 6720 or len(counts) != 28 or set(counts.values()) != {240}:
        raise ValueError('M class coverage mismatch')
    order = [r['id'] for r in main]
    random.Random(SEED).shuffle(order)
    freeze(Path(out) / 'M.json', {'source': portable_identity(ROOT / 'data/tbset.jsonl'),
           'ids': order, 'class_counts': dict(counts), 'groups': {str(r['id']): group_id(chess.Board(r['fen'])) for r in main},
           'sensitivity_ids': sorted(r['id'] for r in main if r['id'] not in CONTRACT['sensitivity_exclusions'])})
    return rows, main


def verify_transform_labels(board, lab):
    original_moves = {m.uci(): m for m in board.legal_moves}
    # Categories must be invariant. Full label/DTZ verification on each variant
    # additionally catches accidental use of pawn-illegal D8 transformations.
    for variant in group_variants(board):
        if not variant.is_valid():
            raise ValueError('Audited symmetry produced illegal board')
        other = tb.label(variant)
        if other is None or other['root'] != lab['root'] or len(other['correct']) != len(lab['correct']):
            raise ValueError('Syzygy category symmetry validation failed')
        if sorted(v['res'] for v in other['moves'].values()) != sorted(v['res'] for v in lab['moves'].values()):
            raise ValueError('Symmetry move-category multiset changed')
        if sorted(v.get('dtz') for v in other['moves'].values() if 'dtz' in v) != \
                sorted(v.get('dtz') for v in lab['moves'].values() if 'dtz' in v):
            raise ValueError('Symmetry DTZ labels changed')


def material_key(cls):
    return tuple(sorted(''.join(sorted(side)) for side in cls.split('v')))


def _class_worker(out,syzygy,cls):
    try:
        generate_fresh(out,syzygy,only_class=cls)
        return cls
    finally:
        if tb._TB is not None:
            tb._TB.close()


def _finish_pool(out,classes,old):
    seen={group_id(chess.Board(r['fen'])) for r in old}
    all_rows,summaries=[],[]
    for ci,cls in enumerate(classes):
        path=out/'N'/(cls+'.json')
        checkpoint=read_json(path)
        roots=checkpoint['roots']
        if len(roots)!=2000 or checkpoint['contract']!=fingerprint(CONTRACT):
            raise ValueError('Incomplete/incompatible N class')
        if [r['id'] for r in roots]!=list(range(11200+ci*2000,11200+(ci+1)*2000)):
            raise ValueError('N class ordered ID namespace differs')
        for row in roots:
            board=validate_root(row)
            pieces=tuple(sorted(''.join(sorted(p.symbol().upper() for p in board.piece_map().values() if p.color==color))
                                for color in (chess.WHITE,chess.BLACK)))
            if row['cls']!=cls or pieces!=material_key(cls):
                raise ValueError('N material class differs')
            group=group_id(board)
            if group in seen:
                raise ValueError('N global legal-symmetry overlap')
            seen.add(group)
        all_rows.extend(roots)
        summaries.append({'class':cls,'attempts':checkpoint['attempts'],'n':2000,'input':portable_identity(path)})
    order=[r['id'] for r in all_rows]
    random.Random(SEED).shuffle(order)
    freeze(out/'N.json',{'contract':fingerprint(CONTRACT),'ids':order,'classes':summaries,
            'old_source':portable_identity(ROOT/'data/tbset.jsonl'),
            'groups':{str(r['id']):group_id(chess.Board(r['fen'])) for r in all_rows},
            'generation':'imported legal-board/scorer; verified legal symmetries; evaluation only'})
    return all_rows


def generate_fresh(out, syzygy, only_class=None):
    out = Path(out)
    old = original_roots()
    seen = {group_id(chess.Board(r['fen'])) for r in old}
    classes = [c for c in tb.DEFAULT_CLASSES if len(c.replace('v', '')) == 5]
    if len(classes) != 12:
        raise ValueError('Five-piece class definition changed')
    if only_class is None:
        # Color/spatial legal symmetries preserve the unordered per-side
        # material. Unique keys prove classes cannot overlap one another;
        # per-class seeds/ID blocks are independent of scheduling.
        if len({material_key(c) for c in classes})!=len(classes):
            raise ValueError('Class independence not proved')
        pending=[c for c in classes if not (out/'N'/(c+'.json')).exists() or
                 len(read_json(out/'N'/(c+'.json'))['roots'])!=2000]
        with ProcessPoolExecutor(max_workers=min(4,os.cpu_count() or 1)) as pool:
            tasks={pool.submit(_class_worker,str(out),str(syzygy),c):c for c in pending}
            for future in as_completed(tasks):
                print('CLASS WORKER VERIFIED '+future.result(),flush=True)
        return _finish_pool(out,classes,old)
    if only_class not in classes:
        raise ValueError('Unknown class worker')
    tb._init(str(syzygy))
    all_rows, summaries = [], []
    for ci, cls in enumerate(classes):
        if cls!=only_class:
            continue
        path = out / 'N' / (cls + '.json')
        existing = read_json(path) if path.exists() else None
        was_complete = existing is not None and len(existing['roots'])==2000
        if existing:
            if existing['contract'] != fingerprint(CONTRACT) or existing['class'] != cls:
                raise ValueError('N generation checkpoint mismatch')
            accepted = existing['roots']
            for r in accepted:
                validate_root(r)
                g = group_id(chess.Board(r['fen']))
                if g in seen:
                    raise ValueError('N checkpoint symmetry overlap')
                seen.add(g)
            tries = existing['attempts']
            rng = random.Random()
            def tuples(x):
                return tuple(tuples(y) for y in x) if isinstance(x, list) else x
            rng.setstate(tuples(existing['rng_state']))
        else:
            accepted, tries = [], 0
            rng = random.Random(f'{SEED}-{cls}')
        if len(accepted) > 2000:
            raise ValueError('N class exceeds frozen count')
        while len(accepted) < 2000 and tries < 200000:
            deadline_check()
            tries += 1
            board = tb.random_position(cls, rng)
            if board is None:
                continue
            g = group_id(board)
            if g in seen:
                continue
            lab = tb.label(board)  # missing tables propagate; no permissive skip
            if lab is None:
                continue
            verify_transform_labels(board, lab)
            row = {'id': 11200 + ci * 2000 + len(accepted), 'fen': board.fen(), 'cls': cls,
                   'split': 'test', **lab}
            validate_root(row)
            accepted.append(row)
            seen.add(g)
            if len(accepted) % 128 == 0:
                atomic_json(path, {'contract': fingerprint(CONTRACT), 'class': cls,
                            'roots': accepted, 'attempts': tries, 'rng_state': rng.getstate()})
                print(f'N {cls}: {len(accepted)}/2000 ({tries} attempts)', flush=True)
        if not was_complete:
            atomic_json(path, {'contract': fingerprint(CONTRACT), 'class': cls,
                        'roots': accepted, 'attempts': tries, 'rng_state': rng.getstate()})
        if len(accepted) != 2000:
            raise RuntimeError('N class generation bound exhausted: ' + cls)
        all_rows.extend(accepted)
        summaries.append({'class': cls, 'attempts': tries, 'n': len(accepted), 'input': portable_identity(path)})
        print(f'N VERIFIED {cls}: 2000/{tries}', flush=True)
    return all_rows


def load_pool(out, name):
    out = Path(out)
    manifest = read_json(out / (name + '.json'))
    if name == 'M':
        source = ROOT / manifest['source']['path']
        if portable_identity(source) != manifest['source']:
            raise ValueError('M source changed')
        rows = strict_jsonl(source)
    else:
        rows = []
        for cls in manifest['classes']:
            source = ROOT / cls['input']['path']
            if portable_identity(source) != cls['input']:
                raise ValueError('N source changed')
            rows.extend(read_json(source)['roots'])
    by_id = {r['id']: r for r in rows}
    return [by_id[i] for i in manifest['ids']]


def validate_openings(sequences, expected=None):
    result, seen = [], set()
    for moves in sequences:
        if len(moves) != 6:
            raise ValueError('Book opening must have six plies')
        b = chess.Board()
        for move in moves:
            b.push_uci(move)
        key = ' '.join(b.fen(en_passant='legal').split()[:4])
        if key not in seen:
            result.append(list(moves))
            seen.add(key)
    if expected is not None and len(result) != expected:
        raise ValueError('Opening terminal-position uniqueness mismatch')
    return result
