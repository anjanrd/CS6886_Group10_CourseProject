"""Declared Run 4 checks. Run as an external driver against --root fresh source.

compare-study/compare-speed are CPU-only. demo-capture explicitly launches lc0 and
belongs after the exclusive measurements. No mode runs at import/default help.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import time


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def lines(path): return [json.loads(x) for x in Path(path).read_text(encoding='utf-8-sig').splitlines() if x.strip()]


def idle_diagnosis(results, samples):
    bad = [s for s in samples if not isinstance(s.get('power_w'), (int, float)) or
           not math.isfinite(s['power_w']) or not 0 <= s['power_w'] <= 180]
    inconsistent = [r for r in results if any(not isinstance(r.get(k), (int, float)) or
                    not math.isfinite(r[k]) or r[k] <= 0 for k in ('gpu_j', 'seconds', 'power_w_mean')) or
                    abs((r['gpu_j'] / r['seconds']) / r['power_w_mean'] - 1) > .10]
    valid = bool(results) and bool(samples) and not bad and not inconsistent
    return {'above_idle_crosscheck_validated': valid, 'implausible_samples': len(bad),
            'counter_sample_mismatch_windows': len(inconsistent),
            'interpretation': 'idle API cross-check consistent' if valid else
                'idle telemetry unresolved; above-idle energy remains provisional. Total under-load energy and timing are separate.'}


def idle_telemetry(output):
    from hw.energy import Meter
    from execution.support import atomic_json
    results = []; samples = []
    for i in range(3):
        with Meter() as meter: time.sleep(10)
        if not meter.gpu.ok or 'gpu_j' not in meter.result: raise RuntimeError('GPU energy telemetry unavailable')
        results.append(meter.result); samples.extend(meter.samples)
    atomic_json(output, {'status': 'observed', 'windows': results, 'samples': samples,
                         **idle_diagnosis(results, samples),
                         'scope': 'exclusive idle telemetry diagnostic; no GPU kernels launched; does not silently change the energy estimator'})


def check_studies(contract, studies, output):
    import chess
    from execution.support import identity, atomic_json
    results = []; failures = []
    ids = set(contract['ids']); roots = {r['id']: r for r in contract['roots']}
    for model, spec in contract['models'].items():
        if identity(spec['fresh_weights'])['sha256'] != spec['weights_sha256']:
            raise ValueError('Fresh model bytes differ: ' + model)
        folder = Path(studies) / model
        actual_a = lines(folder / 'study_a.jsonl'); actual_b = lines(folder / 'study_b.jsonl')
        if len(actual_a) != len(ids) or {r['id'] for r in actual_a} != ids or len(actual_b) != len(ids):
            raise ValueError('Reproduction study ID/count mismatch: ' + model)
        if {(r['id'], r['budget']) for r in actual_b} != {(i, 100) for i in ids}:
            raise ValueError('Reproduction search keys differ')
        meta = read(folder / 'meta.json')
        if any('fp32' in w.lower() for w in meta.get('warnings', [])):
            raise ValueError('Backend FP32 warning: ' + model)
        reference = {r['id']: r for r in spec['reference_A']}
        b_ref = {r['id']: r for r in spec['reference_B100']}
        max_numeric = {}; policy_max = 0; outcome_changes = []; search_outcome_changes = []
        for row in actual_a:
            ident = row['id']; ref = reference[ident]; root = roots[ident]
            board = chess.Board(root['fen'])
            if chess.Move.from_uci(row['best']) not in board.legal_moves:
                raise ValueError('Illegal fresh selected move')
            for field, tolerance in contract['tolerances']['value_absolute'].items():
                a, b = row.get(field), ref.get(field)
                if a is None and b is None: continue
                if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or not math.isfinite(a):
                    raise ValueError('Missing/nonfinite value output: ' + field)
                diff = abs(a - b); max_numeric[field] = max(max_numeric.get(field, 0), diff)
                if diff > tolerance: failures.append(model + ':' + str(ident) + ':' + field)
            p, q = row['p'], ref['p']; ps, qs = sum(p.values()), sum(q.values())
            if (not p or not q or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
                                     for v in list(p.values()) + list(q.values())) or ps <= 0 or qs <= 0):
                raise ValueError('Empty/nonfinite policy output')
            drift = max(abs(p.get(k, 0) / ps - q.get(k, 0) / qs) for k in set(p) | set(q))
            policy_max = max(policy_max, drift)
            if drift > contract['tolerances']['normalized_policy_max_absolute']:
                failures.append(model + ':' + str(ident) + ':policy')
            if (row['best'] in root['correct']) != (ref['best'] in root['correct']): outcome_changes.append(ident)
        for row in actual_b:
            if chess.Move.from_uci(row['best']) not in chess.Board(roots[row['id']]['fen']).legal_moves:
                raise ValueError('Illegal fresh search move')
            if row['nodes'] <= 0 or row['time_ms'] <= 0 or row['wall'] <= 0:
                raise ValueError('Missing actual search observations')
            if (row['best'] in roots[row['id']]['correct']) != (b_ref[row['id']]['best'] in roots[row['id']]['correct']):
                search_outcome_changes.append(row['id'])
        if outcome_changes: failures.append(model + ':raw_outcome_vector_changed')
        results.append({'model': model, 'positions': len(ids), 'max_numeric_absolute': max_numeric,
                        'normalized_policy_max_absolute': policy_max, 'raw_outcome_changes': outcome_changes,
                        'search_outcome_changes_descriptive': search_outcome_changes,
                        'search_exact_moves_not_required': True})
    report = {'status': 'BLOCKED' if failures else 'verified', 'models': results, 'failures': failures,
              'tolerances_frozen_before_execution': contract['tolerances'],
              'scope': 'same frozen models/inputs; numerical/function reproduction, not a new accuracy study'}
    atomic_json(output, report)
    if failures: raise RuntimeError('Fresh numerical agreement requires review: ' + repr(failures[:8]))
    return report


def check_speed(contract, benchmark_root, sparse_root, capture_root, output):
    from execution.support import atomic_json
    from execution.benchmark import validate_search, validate_backend_rows
    from hw.trt_layers import classify_text
    results = []; failures = []; capture_proofs = []
    for model in contract['models']:
        result = read(Path(benchmark_root) / model / 'result.json'); search = result['search']
        validate_search(search, 10)
        for batch in (32, 64, 128, 256):
            validate_backend_rows([r for r in result['network'] if r['batch'] == batch], batch)
        reference = contract['benchmark_reference'][model]
        latency = next(r['ms'] for r in result['network'] if r['batch'] == 64)
        ratios = {'nps': search['nps'] / reference['nps'], 'batch64_latency': latency / reference['batch64_latency_ms']}
        low, high = contract['tolerances']['performance_ratio_review_band']
        if any(not low <= v <= high for v in ratios.values()): failures.append(model + ':performance_review_band')
        results.append({'model': model, 'ratios_vs_recorded_median': ratios,
                        'nps': search['nps'], 'batch64_latency_ms': latency,
                        'scope': 'one fresh trial; descriptive comparison, not a new precision-speed ranking'})
        trace = Path(capture_root) / model / 'trace.nsys-rep'
        db = Path(output).parent / (model + '-fresh-capture.sqlite')
        if not db.exists():
            command = [contract['tools']['nsys'], 'export', '--type=sqlite', '--force-overwrite=false',
                       '--output=' + str(db), str(trace)]
            with db.with_suffix('.log').open('x', encoding='utf-8') as log:
                proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if proc.returncode: raise RuntimeError('Fresh trace CPU export failed')
        with sqlite3.connect('file:' + db.as_posix() + '?mode=ro', uri=True) as con:
            rows = con.execute('SELECT DISTINCT s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName').fetchall()
        expected = 'fp16' if model == 'fp16' else 'int8'
        proof = [name for (name,) in rows if expected in classify_text(name)]
        if not proof: failures.append(model + ':no_required_precision_kernel_evidence')
        capture_proofs.append({'model': model, 'expected_tag': expected, 'matching_kernel_names': len(proof),
                               'examples': proof[:3], 'scope': 'name-tagged execution evidence, not operation coverage'})
    info = read(Path(sparse_root) / 'build_b256_layers.json')
    sparse = [l for l in info['Layers'] if l.get('LayerType') == 'gemm' and
              re.search(r'sparse|sptensor|spmma|_sp_', l.get('TacticName', ''), re.I)]
    idle = read(contract['idle_validation_path'])
    report = {'status': 'BLOCKED' if failures else 'verified', 'models': results, 'kernel_execution': capture_proofs,
              'above_idle_energy_crosscheck_validated': idle['above_idle_crosscheck_validated'],
              'energy_limitation': idle['interpretation'],
              'sparse_selected_tactics': len(sparse),
              'sparse_outcome': 'selected sparse tactics observed' if sparse else 'documented runtime limitation: no sparse tactics selected',
              'failures': failures, 'scope': 'fresh-source/Python/artifact same-host check; no old virtual environment copied'}
    atomic_json(output, report)
    if failures: raise RuntimeError('Fresh systems reproduction needs review: ' + repr(failures))
    return report


def demo_capture(contract, output):
    import chess
    from eval.lc0uci import Lc0
    from execution.support import atomic_json, require_session
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    model = contract['canonical_s4_weights']; log = output / 's4-demo.lc0.log'
    engine = Lc0(contract['tools']['lc0'], model, backend='onnx-trt', backend_opts='optimize=8',
                 options={'MinibatchSize': 64, 'MoveOverheadMs': 20}, log=str(log), name='s4-demo')
    board = chess.Board(); frames = []
    try:
        engine.go(fen=board.fen(), movetime=1000)  # excluded demo warm-up
        engine.newgame()
        for ply in range(12):
            if board.is_game_over(): break
            engine._log.flush()
            log_offset = engine._log.tell()
            before = board.fen(); result = engine.go(fen=before, movetime=500)
            move = chess.Move.from_uci(result['bestmove'])
            if move not in board.legal_moves: raise ValueError('Illegal demo move')
            if engine._log is not None: engine._log.flush()
            with log.open(encoding='utf-8', errors='replace') as stream:
                stream.seek(log_offset); text = stream.read()
            printed = re.findall(r'\bnps\s+(\d+)', text)
            if not printed: raise ValueError('No actual printed demo NPS')
            frames.append({'ply': ply, 'fen_before': before, 'move': move.uci(), 'nodes': result['nodes'],
                           'reported_ms': result['time_ms'], 'wall': result['wall'], 'printed_nps': int(printed[-1])})
            board.push(move)
    finally: engine.quit()
    atomic_json(output / 'demo-capture.json', {'model': 's4', 'backend': 'onnx-trt optimize=8',
                'frames': frames, 'final_fen': board.fen(), 'scope': 'actual UCI gameplay captured after measurements; video rendering may finish in normal Run5'})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=('compare-study', 'compare-speed', 'demo-capture', 'idle-telemetry'))
    ap.add_argument('--root', type=Path, required=True); ap.add_argument('--contract', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True); ap.add_argument('--studies', type=Path)
    ap.add_argument('--benchmarks', type=Path); ap.add_argument('--captures', type=Path); ap.add_argument('--sparse', type=Path)
    ap.add_argument('--session', type=Path)
    args = ap.parse_args(); root = args.root.resolve(); sys.path.insert(0, str(root)); os.chdir(root)
    contract = read(args.contract)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == 'idle-telemetry':
        from execution.support import require_session
        if not args.session: raise ValueError('Exclusive session required for controlled idle measurement')
        require_session(args.session); idle_telemetry(args.output)
    elif args.mode == 'compare-study': check_studies(contract, args.studies, args.output)
    elif args.mode == 'compare-speed': check_speed(contract, args.benchmarks, args.sparse, args.captures, args.output)
    else:
        from execution.support import require_session
        if not args.session: raise ValueError('Explicit exclusive session required for demo')
        require_session(args.session); demo_capture(contract, args.output)


if __name__ == '__main__': main()
