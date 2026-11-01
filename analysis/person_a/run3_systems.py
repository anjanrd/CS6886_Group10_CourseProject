"""CPU-only Run 3 analysis of immutable Run 2 evidence. No engine execution."""
import argparse
import collections
import csv
import json
import math
from pathlib import Path
import re
import sqlite3
import statistics
import subprocess
import time

from execution.support import ROOT, identity, read_json, atomic_json
from hw.trt_layers import classify_text

REPORT = ROOT / 'reports/takeover/run3'
TABLE = ROOT / 'analysis/tables'
MODELS = ('fp16', 's1', 's2', 's3', 's4', 's4aa', 's0', 's1sp')
OLD = ROOT / 'reports/takeover/run2'


def csvread(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def csvwrite(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: raise ValueError('Refusing empty analysis table: ' + str(path))
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def catalog():
    original = read_json(OLD / 'state.json')
    continuation = read_json(OLD / 'codex-continuation-v2/state.json')
    rows = {n: r for n, r in original['jobs'].items() if r['status'] == 'verified'}
    assert not set(rows) & set(continuation['jobs'])
    rows.update(continuation['jobs'])
    if len(rows) != 45 or any(r['status'] != 'verified' for r in rows.values()):
        raise ValueError('Incomplete verified measurement collection')
    return rows


def kernel_exports():
    """Explicit new SQLite destinations: never write next to the original trace."""
    nsys = read_json(ROOT / 'reports/takeover/run1/run2_jobs.json')['tools']['nsys']
    output = REPORT / 'kernels'; output.mkdir(parents=True, exist_ok=True)
    all_rows = []; tag_rows = []; receipts = []
    for model in MODELS:
        trace = OLD / ('capture-' + model) / 'attempt-001/trace.nsys-rep'
        db = output / (model + '.sqlite'); receipt_path = output / (model + '-export.json')
        source = identity(trace)
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            if receipt['source'] != source or receipt['database'] != identity(db):
                raise ValueError('Existing CPU export changed: ' + model)
        else:
            if db.exists(): raise FileExistsError('Preserve/review unrecorded SQLite export: ' + str(db))
            command = [nsys, 'export', '--type=sqlite', '--force-overwrite=false',
                       '--output=' + str(db), str(trace)]
            with (output / (model + '-export.log')).open('x', encoding='utf-8') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                atomic_json(output / (model + '-BLOCKED.json'), {'command': command, 'exit_code': result.returncode})
                raise RuntimeError('Nsight CPU export failed; preserve log: ' + model)
            receipt = {'source': source, 'command': command, 'exit_code': 0, 'database': identity(db),
                       'scope': 'CPU export of saved trace; no new GPU workload'}
            atomic_json(receipt_path, receipt)
        receipts.append(identity(receipt_path))
        with sqlite3.connect('file:' + db.as_posix() + '?mode=ro', uri=True) as con:
            tables = {x[0] for x in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if 'CUPTI_ACTIVITY_KIND_KERNEL' not in tables or 'StringIds' not in tables:
                raise ValueError('Missing saved CUDA kernel evidence: ' + model)
            fields = {x[1] for x in con.execute('PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)')}
            key = 'demangledName' if 'demangledName' in fields else 'shortName'
            data = con.execute('SELECT s.value,SUM(k.end-k.start),COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL k '
                               'JOIN StringIds s ON s.id=k.' + key + ' GROUP BY s.value ORDER BY 2 DESC').fetchall()
        total = sum(r[1] for r in data)
        if total <= 0: raise ValueError('No timed kernels: ' + model)
        tags = collections.defaultdict(float)
        for name, ns, instances in data:
            tag = '+'.join(sorted(classify_text(name))) or 'other'
            gemm = bool(re.search(r'gemm|mma|conv|matmul|bmm|fmha', name, re.I))
            all_rows.append({'model': model, 'kernel': name, 'total_ns': ns, 'instances': instances,
                             'fraction_all_kernel_time': ns / total, 'name_tag': tag, 'gemm_like_name': gemm})
            tags[tag] += ns
        tag_rows += [{'model': model, 'name_tag': tag, 'kernel_ms': ns / 1e6,
                      'fraction_all_kernel_time': ns / total,
                      'scope': 'whole capture includes startup/warmup; name-tag time is not operation coverage'}
                     for tag, ns in sorted(tags.items())]
        print('CPU_KERNEL_EXPORT_VERIFIED ' + model, flush=True)
    csvwrite(TABLE / 'T2-lc0-kernels.csv', all_rows)
    csvwrite(TABLE / 'T2-lc0-kernel-tags.csv', tag_rows)
    atomic_json(REPORT / 'kernel-export-verification.json', {'exports': receipts, 'models': len(MODELS),
                'original_traces_unchanged': True, 'new_GPU_workloads': 0,
                'classification_limit': 'Name tags are execution evidence/time fractions, not per-ONNX-operation precision proof'})
    return tag_rows


def metadata_nodes(layer):
    return set(re.findall(r'\[ONNX Layer: ([^\]]+)\]', layer.get('Metadata', '')))


def precision(layer):
    if layer.get('LayerType') != 'gemm': return 'opaque'
    inputs = [x.get('Format/Datatype', '') for x in layer.get('Inputs', [])]
    tactic = layer.get('TacticName', '').lower()
    if inputs and all('int8' in x.lower() for x in inputs) and re.search('i8|imma|int8', tactic): return 'int8'
    if inputs and all('half' in x.lower() or 'fp16' in x.lower() for x in inputs) and re.search('f16|h16816|hmma', tactic): return 'fp16'
    return 'unknown'


def sparse_tactic(layer):
    return bool(re.search(r'sparse|sptensor|spmma|_sp_', layer.get('TacticName', ''), re.I))


def selected_region(op, candidates, layers):
    gemms = [i for i in candidates if layers[i].get('LayerType') == 'gemm']
    picks = set(gemms or candidates)
    return next(iter(picks)) if len(picks) == 1 else None


def measured_peaks():
    folder = OLD / 'measured-ceilings/attempt-001/work'
    result = read_json(folder / 'result.json'); out = {}
    for name, wanted in (('fp16', 'fp16'), ('int8', 'int8'), ('bandwidth', 'fp16')):
        layers = read_json(folder / (name + '_layers.json'))['Layers']
        gemms = [l for l in layers if l.get('LayerType') == 'gemm']
        if len(gemms) != 1 or precision(gemms[0]) != wanted:
            raise ValueError('Microbenchmark precision not proved: ' + name)
        times = {r['name']: r['averageMs'] for r in read_json(folder / (name + '_profile.json')) if 'name' in r}
        ms = times[gemms[0]['Name']]
        if name == 'bandwidth':
            # Half operands and weight matrix 16384^2; logical weight traffic only.
            rate = 16384 ** 2 * 2 / (ms * 1e-3) / 1e9
            unit = 'modeled_weight_GB/s'
        else:
            rate = result[name]['arithmetic_ops'] / (ms * 1e-3) / 1e12
            unit = 'TFLOP/s' if name == 'fp16' else 'integer_TOP/s'
        out[name] = {'gemm_profile_ms': ms, 'gemm_reference_rate': rate, 'unit': unit,
                     'tactic': gemms[0]['TacticName'], 'visible_precision': wanted,
                     'end_to_end_latency_ms': result[name]['latency_ms'],
                     'scope': 'measured reference, not a universal architectural hard limit; profiling/clock conditions differ'}
    atomic_json(REPORT / 'measured-reference-peaks.json', out)
    return out


def profiles():
    ops = {r['name']: r for r in csvread(TABLE / 'roofline-operation-inventory.csv')}
    work_all = sum(int(r['arithmetic_ops_per_position']) for r in ops.values())
    peaks = measured_peaks(); detail = []; mapping = []; roof = []; summary = []
    configurations = [('fp16', 64, 'build_b64'), ('fp16', 256, 'reload_b256'),
                      ('s4', 64, 'build_b64'), ('s4', 256, 'reload_b256')]
    configurations += [(n, 256, 'build_b256') for n in ('s0-dense', 's0-sparse', 's1sp-dense', 's1sp-sparse')]
    for model, batch, prefix in configurations:
        folder = OLD / ('profile-' + model) / 'attempt-001/work'
        layers = read_json(folder / (prefix + '_layers.json'))['Layers']
        prof = {r['name']: r['averageMs'] for r in read_json(folder / (prefix + '_profile.json')) if 'name' in r}
        if set(prof) != {l['Name'] for l in layers}: raise ValueError('Layer/profile names disagree')
        candidates = collections.defaultdict(set)
        for i, layer in enumerate(layers):
            for node in metadata_nodes(layer) & set(ops): candidates[node].add(i)
        chosen = {n: selected_region(n, indices, layers) for n, indices in candidates.items()}
        chosen = {n: i for n, i in chosen.items() if i is not None}
        by_region = collections.defaultdict(list)
        for name, i in chosen.items(): by_region[i].append(name)
        integer_ops = 0; integer_count = 0; sparse_count = 0; times = collections.defaultdict(float)
        for i, layer in enumerate(layers):
            ms = prof[layer['Name']]; pr = precision(layer); names = by_region.get(i, [])
            sp = sparse_tactic(layer)
            if layer.get('LayerType') == 'gemm': category = pr + '_gemm'
            elif names: category = 'opaque_fused_matrix'
            elif re.search(r'/ln|norm|softmax', layer.get('Metadata', ''), re.I): category = 'norm_softmax_fused'
            elif layer.get('LayerType') in ('Reformat', 'cast') or ('Int8' in str(layer.get('Inputs')) and 'Half' in str(layer.get('Outputs'))): category = 'conversion'
            else: category = 'other_fused'
            times[category] += ms
            detail.append({'model': model, 'batch': batch, 'engine_layer': layer['Name'], 'type': layer.get('LayerType'),
                           'category': category, 'precision_evidence': pr, 'sparse_tactic': sp, 'average_ms': ms,
                           'matrix_names': json.dumps(names), 'tactic': layer.get('TacticName', ''),
                           'input_formats': json.dumps([x.get('Format/Datatype') for x in layer.get('Inputs', [])]),
                           'output_formats': json.dumps([x.get('Format/Datatype') for x in layer.get('Outputs', [])])})
            if layer.get('LayerType') == 'gemm' and sp: sparse_count += 1
            for name in names:
                if pr == 'int8': integer_count += 1; integer_ops += int(ops[name]['arithmetic_ops_per_position'])
            # Only directly attributed SINGLE-matrix GEMM regions have a simple
            # logical A/B/Y estimate. Packed/fused attention stays explicit/unplotted.
            if model in ('fp16', 's4') and len(names) == 1 and pr in ('fp16', 'int8') and ms > 0:
                op = ops[names[0]]; a = math.prod(json.loads(op['a_shape_batch1'])) * batch
                b = math.prod(json.loads(op['b_shape_batch1'])) * (batch if op['kind'] != 'activation_times_weight' else 1)
                y = math.prod(json.loads(op['output_shape_batch1'])) * batch
                outs = layer.get('Outputs', [])
                if len(outs) != 1: continue
                fmt = outs[0].get('Format/Datatype', '').lower()
                ybytes = 1 if 'int8' in fmt else 2 if 'half' in fmt or 'fp16' in fmt else 4 if 'float' in fmt else None
                if ybytes is None: continue
                operand_bytes = 1 if pr == 'int8' else 2
                logical = (a + b) * operand_bytes + y * ybytes
                work = int(op['arithmetic_ops_per_position']) * batch
                intensity = work / logical; achieved = work / (ms * 1e-3) / 1e12
                ref = peaks[pr]['gemm_reference_rate']; bw = peaks['bandwidth']['gemm_reference_rate']
                bound = min(ref, intensity * bw / 1000)
                roof.append({'model': model, 'batch': batch, 'node': names[0], 'layer': layer['Name'],
                             'precision': pr, 'arithmetic_ops': work, 'modeled_bytes': logical,
                             'ops_per_modeled_byte': intensity, 'profile_average_ms': ms,
                             'achieved_tera_ops_s': achieved, 'reference_roof_tera_ops_s': bound,
                             'above_reference': achieved > bound, 'reference_bound': 'memory' if intensity * bw / 1000 < ref else 'compute',
                             'sparse_tactic': sp, 'scope': 'logical matrix I/O+weights; not measured DRAM traffic; fused/packed regions excluded'})
        for name, op in ops.items():
            i = chosen.get(name); layer = layers[i] if i is not None else {}
            mapping.append({'model': model, 'batch': batch, 'node': name, 'kind': op['kind'],
                            'arithmetic_ops_per_position': int(op['arithmetic_ops_per_position']),
                            'engine_layer': layer.get('Name', ''), 'precision_evidence': precision(layer) if layer else 'unmapped',
                            'sparse_tactic': sparse_tactic(layer), 'basis': 'exact ONNX metadata; unique GEMM preferred' if layer else 'not resolved; no precision inferred'})
        total = sum(prof.values())
        if abs(sum(times.values()) - total) > 1e-8: raise ValueError('Double-counted profile durations')
        log = folder / (prefix + ('_trtexec.log' if prefix.startswith('build') else '.log'))
        text = log.read_text(errors='replace')
        match = re.search(r'GPU Compute Time:.*?mean = ([\d.]+) ms', text)
        if not match: raise ValueError('Missing non-profile GPU timing: ' + str(log))
        row = {'model': model, 'batch': batch, 'nonprofile_gpu_ms': float(match.group(1)),
               'profile_sum_ms': total, 'engine_layers': len(layers), 'matrix_ops_mapped': len(chosen),
               'matrix_ops_unmapped': len(ops) - len(chosen), 'int8_matrix_ops_proven': integer_count,
               'int8_matrix_arithmetic_fraction_proven': integer_ops / work_all,
               'sparse_gemm_tactics': sparse_count, 'timing_scope': 'standalone trtexec; profile overhead separate from lc0'}
        for cat in ('int8_gemm', 'fp16_gemm', 'unknown_gemm', 'opaque_fused_matrix', 'norm_softmax_fused', 'conversion', 'other_fused'):
            row[cat + '_ms'] = times[cat]
        summary.append(row)
    csvwrite(TABLE / 'T2-standalone-profiles.csv', summary)
    csvwrite(TABLE / 'T2-engine-layers.csv', detail)
    csvwrite(TABLE / 'T2-operation-precision.csv', mapping)
    csvwrite(TABLE / 'F6-roofline-points.csv', roof)
    sparse = []
    for model in ('s0', 's1sp'):
        dense = next(r for r in summary if r['model'] == model + '-dense')
        enabled = next(r for r in summary if r['model'] == model + '-sparse')
        sparse.append({'model': model, 'batch': 256, 'dense_gpu_ms': dense['nonprofile_gpu_ms'],
                       'sparse_gpu_ms': enabled['nonprofile_gpu_ms'],
                       'standalone_speed_ratio': dense['nonprofile_gpu_ms'] / enabled['nonprofile_gpu_ms'],
                       'selected_sparse_gemm_tactics': enabled['sparse_gemm_tactics'],
                       'scope': 'one standalone build/configuration, not lc0 sparse acceleration or a repeated speed CI'})
    csvwrite(TABLE / 'T5-sparsity.csv', sparse)
    result = {'profiles': summary, 'sparsity': sparse, 'reference_peaks': peaks,
              'roofline_points': len(roof), 'unmapped_precision_is_not_guessed': True,
              'profile_times_reconcile_without_duplication': True,
              'limitations': ['Whole-engine profile sum includes profiler overhead; non-profile latency is separate',
                 'Logical matrix bytes are not measured DRAM transactions; excluded packed/opaque regions remain visible',
                 'Sparse selection is proved by exported selected tactics; legacy sparse_layers_reported regex zero is not a selection count',
                 'End-to-end speed comes from lc0 benchmarks, not these standalone engines']}
    atomic_json(REPORT / 'profile-review.json', result)
    return result


def descriptives(values):
    a = sorted(values); q = statistics.quantiles(a, n=4, method='inclusive')
    return {'median': statistics.median(a), 'min': a[0], 'max': a[-1], 'mean': statistics.mean(a),
            'std': statistics.stdev(a) if len(a) > 1 else 0, 'q1': q[0], 'q3': q[2]}


def benchmarks():
    jobs = catalog(); trials = []; batches = []; energy = []; telemetry = []; sessions = []; idle_anomalies = []
    for name, receipt in jobs.items():
        if not name.startswith('bench-'): continue
        d = read_json(Path(receipt['attempt']) / 'work/result.json'); q = d['search']; e = q['energy']
        session = 'codex-v2' if 'policy' in receipt else 'original-strict'
        row = {'model': d['name'], 'trial': name, 'session': session, 'nodes': q['nodes'],
               'engine_time_ms': q['time_ms'], 'engine_nps': q['nps'], 'wall_nps': q['wall_nps'],
               'gpu_mJ_node_above_idle': q['gpu_j_per_node'] * 1000,
               'gpu_mJ_node_total': q['gpu_j_per_node_total'] * 1000,
               'search_idle_W': d['idle_search']['gpu_w'], 'search_temp_mean_C': e['temp_c_mean'],
               'search_temp_max_C': e['temp_c_max'], 'search_SM_mean_MHz': e['sm_mhz_mean'],
               'counter_mean_W': e['gpu_j'] / e['seconds'], 'sampled_mean_W': e['power_w_mean'],
               'model_sha256': d['model']['sha256'], 'receipt': str(Path(receipt['attempt']) / 'receipt.json')}
        trials.append(row)
        for nn in d['network']:
            batches.append({'model': d['name'], 'trial': name, 'session': session,
                            'batch': nn['batch'], 'latency_ms': nn['ms'], 'positions_s': nn['nps']})
        energy_row = {k: row[k] for k in ('model', 'trial', 'session', 'nodes', 'gpu_mJ_node_above_idle',
                         'gpu_mJ_node_total', 'search_idle_W', 'counter_mean_W', 'sampled_mean_W')}
        energy_row['above_idle_validity'] = 'reported subtraction; idle telemetry not cross-validated'
        energy_row['total_validity'] = 'under-load NVML counter/sample consistency checked; not external-meter validation'
        energy.append(energy_row)
        activity = [json.loads(line) for line in (Path(receipt['attempt']) / 'activity.log').read_text().splitlines()]
        bad = [r['gpu'] for r in activity if r.get('gpu', {}).get('power_w', 0) > 180]
        idle_anomalies.append({'model': d['name'], 'trial': name, 'samples_over_180W': len(bad),
                               'of_those_zero_utilization': sum(x.get('util') == 0 for x in bad),
                               'maximum_reported_W': max((r.get('gpu', {}).get('power_w', 0) for r in activity), default=0),
                               'counter_derived_idle_W': d['idle_search']['gpu_w'],
                               'interpretation': 'implausible instantaneous readings on120W-cap GPU; idle counter baseline remains plausible but not cross-validated'})
        telemetry.append(abs(row['counter_mean_W'] / row['sampled_mean_W'] - 1))
    summary = []; batch_summary = []
    for model in MODELS:
        rows = [r for r in trials if r['model'] == model]
        if len(rows) != (5 if model in ('fp16', 's4') else 3): raise ValueError('Wrong repeat count')
        row = {'model': model, 'trials': len(rows)}
        for key in ('engine_nps', 'wall_nps', 'gpu_mJ_node_above_idle', 'gpu_mJ_node_total',
                    'search_temp_mean_C', 'search_SM_mean_MHz'):
            for stat, value in descriptives([r[key] for r in rows]).items(): row[key + '_' + stat] = value
        summary.append(row)
        for batch in (32, 64, 128, 256):
            values = [r['latency_ms'] for r in batches if r['model'] == model and r['batch'] == batch]
            batch_summary.append({'model': model, 'batch': batch, 'trials': len(values),
                                  **{'latency_ms_' + k: v for k, v in descriptives(values).items()}})
        for session in ('original-strict', 'codex-v2'):
            r = [x for x in rows if x['session'] == session]
            if r: sessions.append({'model': model, 'session': session, 'trials': len(r),
                       'median_nps': statistics.median(x['engine_nps'] for x in r),
                       'median_SM_MHz': statistics.median(x['search_SM_mean_MHz'] for x in r),
                       'median_temp_C': statistics.median(x['search_temp_mean_C'] for x in r)})
    csvwrite(TABLE / 'T2-speed-trials.csv', trials); csvwrite(TABLE / 'T2-speed.csv', summary)
    csvwrite(TABLE / 'T2-network-batches.csv', batches); csvwrite(TABLE / 'T2-network-batch-summary.csv', batch_summary)
    csvwrite(TABLE / 'T3-energy.csv', energy); csvwrite(TABLE / 'T2-session-review.csv', sessions)
    csvwrite(TABLE / 'T3-idle-telemetry-review.csv', idle_anomalies)
    quality = {r['model']: r for r in csvread(TABLE / 'frontier-quality-ready.csv')}
    frontier = []
    for r in summary:
        q = quality[r['model']]
        frontier.append({'model': r['model'], 'n': int(q['n']), 'correct': int(q['correct']),
                         'raw_preserving_move_accuracy': float(q['accuracy']), 'accuracy_ci_low': float(q['ci_low']),
                         'accuracy_ci_high': float(q['ci_high']), 'search_nps_median': r['engine_nps_median'],
                         'search_nps_min': r['engine_nps_min'], 'search_nps_max': r['engine_nps_max'],
                         'trials': r['trials'], 'scope': 'raw NN quality versus lc0 search throughput; not fixed-time strength'})
    csvwrite(TABLE / 'F11-frontier.csv', frontier)
    review = {'trials': len(trials), 'batch_rows': len(batches), 'models': summary,
              'max_counter_vs_sample_relative_difference': max(telemetry),
              'max_idle_W': max(r['search_idle_W'] for r in trials),
              'clock_lock': False, 'session_policy_change_recorded': True,
              'uncertainty': 'repeat min/max/IQR/std are descriptive, not independent-trial confidence intervals',
              'scope': 'GPU energy only, both total and above-idle retained; no equal-energy accuracy claim',
              'trials_with_implausible_instantaneous_power': sum(r['samples_over_180W'] > 0 for r in idle_anomalies),
              'implausible_instantaneous_samples': sum(r['samples_over_180W'] for r in idle_anomalies),
              'warning': 'Repeated impossible instantaneous readings occur in inactive phases. Under-load counter/sample means agree, but counter-derived idle power lacks a trustworthy instantaneous cross-check. Above-idle numbers are as-reported/provisional; total GPU energy is retained separately. Run4 will repeat a declared idle telemetry check.'}
    atomic_json(REPORT / 'benchmark-review.json', review)
    return review


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('stage', choices=('kernels', 'profiles', 'benchmarks'))
    args = ap.parse_args(); REPORT.mkdir(parents=True, exist_ok=True)
    result = {'kernels': kernel_exports, 'profiles': profiles, 'benchmarks': benchmarks}[args.stage]()
    print('VERIFIED_CPU_ANALYSIS ' + args.stage, flush=True)


if __name__ == '__main__': main()
