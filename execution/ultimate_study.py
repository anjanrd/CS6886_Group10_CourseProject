"""Checked, atomic root blocks around the unchanged imported UCI study semantics."""
import contextlib
import ctypes
from ctypes import wintypes
from functools import lru_cache
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import time
import platform
from types import SimpleNamespace

from eval.lc0uci import Lc0
from execution.resources import ActivityMonitor, snapshot
from execution.support import fingerprint, identity, read_json, scoped_environment, single_workload
from execution.ultimate_contract import (ROOT, CONTRACT, deadline_check, freeze,
                                         portable_identity, validate_observation, atomic_json)


def conditions(monitor):
    if monitor.errors:
        raise RuntimeError('Telemetry monitor failure: ' + repr(monitor.errors))
    if not monitor.samples or any(s.get('ac_line') != 1 for s in monitor.samples):
        raise RuntimeError('AC power not positively observed')
    if any(not all(k in s.get('gpu', {}) for k in ('temp_c', 'sm_mhz', 'mem_mhz', 'util', 'power_w'))
           for s in monitor.samples):
        raise RuntimeError('Required GPU telemetry missing')
    # Background CPU/app/GPU activity is informational under the accepted policy.


class CheckedLc0(Lc0):
    """Use stock command/parser, checking live failure/deadline while waiting."""
    def __init__(self, *args, guard=lambda: None, **kwargs):
        self.guard = guard
        try:
            super().__init__(*args, **kwargs)
        except BaseException:
            if hasattr(self, 'p'):
                self.p.kill()
                self.p.wait()
            if getattr(self, '_log', None):
                self._log.close()
            raise

    def readline(self, timeout):
        end = time.monotonic() + timeout
        while True:
            self.guard()
            remaining = end - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('lc0 response watchdog expired')
            try:
                line = self.q.get(timeout=min(1, remaining))
            except queue.Empty:
                continue
            if line is None:
                raise RuntimeError('lc0 output closed; see ' + str(self.log_path))
            return line

    def go(self, *args, **kwargs):
        self.guard()
        result = super().go(*args, **kwargs)
        if any('fp32' in w.lower() for w in self.warnings):
            raise RuntimeError('Backend warned about FP32 execution')
        return result

    def _parse(self, lines, wall, fen):
        result = super()._parse(lines, wall, fen)
        result['printed_nps'] = None
        for line in lines:
            tokens = line.split()
            if line.startswith('info ') and 'nps' in tokens:
                result['printed_nps'] = int(tokens[tokens.index('nps')+1])
        return result


def preflight(config):
    deadline_check()
    snap = snapshot()
    if snap['ac_line'] != 1:
        raise RuntimeError('AC power unavailable')
    jobs = [p for p in snap['processes'] if p['name'].lower() in
            ('lc0.exe', 'trtexec.exe', 'nsys.exe', 'nsys-launcher.exe')]
    if jobs:
        raise RuntimeError('Existing engine/profiler workload: ' + repr(jobs))
    if shutil.disk_usage(ROOT).free < 100 * 1024**3:
        raise RuntimeError('Less than 100GiB free storage reserve')
    query = ['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total,temperature.gpu,clocks.sm,clocks.mem,power.draw,power.limit,utilization.gpu',
             '--format=csv,noheader,nounits']
    gpu = subprocess.check_output(query, text=True).strip()
    power = subprocess.check_output(['powercfg', '/getactivescheme'], text=True).strip()
    if '381b4222-f694-41f0-9685-ff5bb260df2e' not in power:
        raise RuntimeError('Windows power mode differs from historical Balanced')
    fields=[x.strip() for x in gpu.split(',')]
    freeze(ROOT/'reports/ultimate/runtime-identity.json',{'GPU_name':fields[0],'GPU_uuid':fields[1],
           'driver':fields[2],'memory_MiB':fields[3],'power_scheme':power,'python':platform.python_version()})
    return {'observed_unix': time.time(), 'snapshot': snap, 'gpu': gpu,
            'power_scheme': power, 'free_bytes': shutil.disk_usage(ROOT).free,
            'cooling': 'user confirms clear; temperature observed, physical clearance not programmatically observable',
            'OEM_mode': 'user confirms same; not programmatically observable',
            'clock_lock': False, 'background_flags_informational': True}


@contextlib.contextmanager
def keep_awake(record):
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.SetThreadExecutionState.argtypes=[wintypes.DWORD]
    kernel.SetThreadExecutionState.restype=wintypes.DWORD
    if not kernel.SetThreadExecutionState(0x80000003):
        raise ctypes.WinError(ctypes.get_last_error())
    record['requested']=True
    try:
        yield
    finally:
        record['released']=bool(kernel.SetThreadExecutionState(0x80000000))
        if not record['released']:
            raise ctypes.WinError(ctypes.get_last_error())


def record_workload(folder):
    record={'started_unix':time.time(),'pid':os.getpid(),'folder':str(folder),
            'scope':'monitored scientific workload attempt; controlled idle excluded'}
    path=ROOT/'reports/ultimate/workloads.jsonl'
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8',newline='\n') as f:
        f.write(json.dumps(record)+'\n');f.flush();os.fsync(f.fileno())
    progress=ROOT/'reports/takeover/progress.json'
    if progress.exists():
        state=read_json(progress)
        state['ultimate']['new_GPU_workloads']=state['ultimate'].get('new_GPU_workloads',0)+1
        state['ultimate']['GPU_workload_count_scope']=record['scope']
        state['ultimate']['last_workload']=record
        atomic_json(progress,state)


@contextlib.contextmanager
def gpu_interval(folder, config, *, workload=True):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'receipt.json').exists():
        raise RuntimeError('Prior attempt needs exact receipt reuse or reviewed recovery')
    observed = preflight(config)
    atomic_json(folder / 'preflight.json', observed)
    sleep_record = {}
    before = time.time()
    try:
        with single_workload(ROOT / 'reports/takeover/run2/ACTIVE.json'), \
                single_workload(ROOT / 'reports/ultimate/ACTIVE.json'), keep_awake(sleep_record):
            with ActivityMonitor(folder / 'activity.log') as monitor:
                conditions(monitor)
                if workload:
                    record_workload(folder)
                yield lambda: (deadline_check(), conditions(monitor))
            report = monitor.report(os.getpid())
            conditions(monitor)
            atomic_json(folder / 'conditions.json', report)
        atomic_json(folder / 'receipt.json', {'status': 'verified', 'started_unix': before,
                    'elapsed_seconds': time.time() - before, 'awake': sleep_record,
                    'conditions': portable_identity(folder / 'conditions.json')})
    except BaseException as exc:
        if 'monitor' in locals():
            atomic_json(folder / 'conditions.json', monitor.report(os.getpid()))
        atomic_json(folder / 'receipt.json', {'status': 'BLOCKED', 'error': repr(exc),
                    'started_unix': before, 'elapsed_seconds': time.time() - before, 'awake': sleep_record})
        raise


def engine(config, model, study, log, guard):
    opts = {'VerboseMoveStats': True, 'MinibatchSize': 256}
    if study in ('A', 'B'):
        opts['Threads'] = 1
    if study == 'A':
        opts['PolicyTemperature'] = 1.0
    elif study == 'B':
        opts['SmartPruningFactor'] = 0
    else:
        opts['MoveOverheadMs'] = 20
    return CheckedLc0(config['lc0'], ROOT / 'models' / (model + '.pb.gz'), backend='onnx-trt',
                      backend_opts='optimize=6' if model == 'fp16' else 'optimize=8',
                      options=opts, log=str(log), name=model, guard=guard)


def observation(root, result, study, budget):
    row = {'id': root['id'], 'best': result['bestmove'], 'nodes': result['nodes']}
    if study == 'A':
        row.update(budget=0, p={m: round(v['p'], 6) for m, v in result['moves'].items()},
                   **{k: result.get(k) for k in ('wl', 'd', 'v', 'm')})
    else:
        row.update({('movetime' if study == 'C' else 'budget'): budget,
                    'time_ms': result['time_ms'], 'wall': round(result['wall'], 4),
                    'depth': result['depth']})
        if study == 'B':
            row['score_cp'] = result['score_cp']
    return validate_observation(row, root, study, budget)


def block_contract(config, model, study, budget, roots):
    sources = ['common.py', 'eval/lc0uci.py', 'eval/run_study.py', 'execution/ultimate_study.py',
               'execution/ultimate_contract.py', 'execution/support.py', 'execution/resources.py', 'hw/energy.py']
    native = sorted(Path(config['lc0']).parent.glob('*.dll'))
    return {'protocol': fingerprint(CONTRACT), 'model': model, 'study': study, 'budget': budget,
            'backend':'onnx-trt','backend_opts':'optimize=6' if model=='fp16' else 'optimize=8',
            'weights': input_identity(ROOT / 'models' / (model + '.pb.gz')),
            'roots': fingerprint(roots), 'ids': [r['id'] for r in roots],
            'sources': [input_identity(ROOT / p) for p in sources],
            'runtime': [input_identity(p) for p in [Path(config['lc0']), *native]],
            'runtime_observation':input_identity(ROOT/'reports/ultimate/runtime-identity.json'),
            'search_options': CONTRACT[{'A': 'raw', 'B': 'nodes', 'C': 'timed'}[study]]}


@lru_cache(maxsize=256)
def _input_identity(path, size, mtime_ns):
    # In-process only. Every worker hashes its actual inputs on first use;
    # a changed size/mtime triggers a new hash. No persistent hash shortcuts.
    return portable_identity(path)


def input_identity(path):
    path=Path(path).resolve()
    stat=path.stat()
    return dict(_input_identity(str(path),stat.st_size,stat.st_mtime_ns))


def read_block(folder, contract, roots):
    p = Path(folder) / 'output.json'
    if not p.exists():
        return None
    output = read_json(p)
    if output['contract'] != fingerprint(contract) or output.get('status') != 'verified':
        raise ValueError('Block resume identity/status mismatch')
    rows = output['rows']
    if len(rows) != len(roots) or [r['id'] for r in rows] != [r['id'] for r in roots]:
        raise ValueError('Missing/duplicate block output')
    for row, root in zip(rows, roots):
        validate_observation(row, root, contract['study'], contract['budget'])
    return rows


def run_block(folder, config, model, study, budget, roots):
    if not roots:
        return []
    folder = Path(folder)
    contract = block_contract(config, model, study, budget, roots)
    freeze(folder / 'contract.json', contract)
    old = read_block(folder, contract, roots)
    if old is not None:
        receipt = read_json(folder / 'receipt.json')
        if receipt['status'] != 'verified' or receipt['output'] != portable_identity(folder / 'output.json'):
            raise ValueError('Missing/corrupt block receipt')
        return old
    # scoped PATH applies to this worker only. No global environment mutation.
    if os.environ.get('LC0_ULTIMATE_CHILD') != '1':
        raise RuntimeError('Scientific blocks run only in scoped monitored worker')
    rows = []
    with gpu_interval(folder, config) as guard:
        with engine(config, model, study, folder / 'engine.lc0.log', guard) as e:
            if study == 'C':
                e.go(fen=roots[0]['fen'], movetime=budget)
            for i, root in enumerate(roots):
                guard()
                e.newgame()
                res = e.go(fen=root['fen'], **({'movetime': budget} if study == 'C' else
                                               {'nodes': 1 if study == 'A' else budget}))
                rows.append(observation(root, res, study, budget))
                if (i + 1) % 32 == 0 or i + 1 == len(roots):
                    print(f'{model} {study}{budget}: {i+1}/{len(roots)}', flush=True)
            if e.warnings:
                atomic_json(folder / 'warnings.json', e.warnings)
        atomic_json(folder / 'output.json', {'status': 'verified', 'contract': fingerprint(contract), 'rows': rows})
    receipt = read_json(folder / 'receipt.json')
    receipt['output'] = portable_identity(folder / 'output.json')
    atomic_json(folder / 'receipt.json', receipt)
    return rows
