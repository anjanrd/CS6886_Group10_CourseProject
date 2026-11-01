"""Sequential Ultimate controller. Reference reuse and fresh replay are distinct."""
import argparse
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from execution.support import fingerprint, identity, read_json, scoped_environment
from execution.ultimate_contract import (ROOT, A10, B9, CONTRACT, DEADLINE, deadline_check,
                                         freeze, portable_identity, safe_output, authorized_deadline, atomic_json)


def config(path):
    value = read_json(path)
    value['_path'] = str(Path(path).resolve())
    base = Path(path).resolve().parent
    for key in ('lc0', 'trtexec', 'nsys', 'syzygy', 'bt3'):
        p = Path(os.path.expandvars(value[key])).expanduser()
        value[key] = str((base / p).resolve()) if not p.is_absolute() else str(p)
    return value


def check(cfg):
    """Read-only input/prerequisite inspection; does not start an engine."""
    for key in ('lc0', 'trtexec', 'nsys', 'syzygy', 'bt3'):
        if not Path(cfg[key]).exists():
            raise FileNotFoundError(cfg[key])
    from importlib.metadata import version
    import onnxruntime as ort
    packages = {p: version(p) for p in ('numpy', 'onnx', 'onnxruntime', 'python-chess', 'nvidia-ml-py')}
    preparation_providers=ort.get_available_providers()
    if any(p in preparation_providers for p in ('CUDAExecutionProvider','TensorrtExecutionProvider')):
        raise ValueError('Python preparation must use the declared CPU ONNX Runtime installation')
    if identity(cfg['bt3'])['sha256'] != 'e3067757d1fc2dfc66947b21d15ace0cedf4c54254fc1de83d77c378a3e8b8e1':
        raise ValueError('BT3 checkpoint identity changed')
    models = [portable_identity(ROOT / 'models' / (m + '.pb.gz')) for m in A10]
    manifest=ROOT/'artifact-manifest.json'
    if manifest.exists():
        from execution.ultimate_package import verify_manifest
        verified=verify_manifest(ROOT)
        # JSON round-trips tuples as lists; compare canonical protocol bytes.
        if fingerprint(verified['protocol'])!=fingerprint(CONTRACT):
            raise ValueError('Package protocol differs from executable contract')
        # Native executables and full Syzygy inventory are declared inputs.
        native=read_json(ROOT/'data/ultimate/native-inputs.json')
        for key,item in zip(('lc0','trtexec','nsys'),native['files']):
            actual=identity(cfg[key])
            if any(actual[k]!=item[k] for k in ('bytes','sha256')):
                raise ValueError('Native executable identity differs: '+key)
        for item in native['lc0_dependencies']:
            actual=identity(Path(cfg['lc0']).parent/Path(item['path']).name)
            if any(actual[k]!=item[k] for k in ('bytes','sha256')):
                raise ValueError('Native lc0 DLL identity differs: '+Path(item['path']).name)
        tb=read_json(ROOT/'data/ultimate/tablebases.json')
        for item in tb['files']:
            actual=identity(Path(cfg['syzygy'])/Path(item['path']).name)
            if any(actual[k]!=item[k] for k in ('bytes','sha256')):
                raise ValueError('Tablebase identity differs: '+item['path'])
    return {'status': 'checked_inputs_only', 'packages': packages,'preparation_available_providers':preparation_providers, 'models': models,
            'bt3': portable_identity(cfg['bt3']), 'protocol': fingerprint(CONTRACT)}


def log(message):
    stamp = datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime('%Y-%m-%d %H:%M:%S IST')
    with (ROOT / 'LOG.md').open('a', encoding='utf-8') as f:
        f.write('\n\n## ' + stamp + ' - Ultimate execution\n\n' + message + '\n')


def working_branch():
    if (ROOT / '.git').exists():
        branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
        if branch != 'anjan':
            raise RuntimeError('Working execution must preserve branch anjan')


def command(argv, folder, cfg, *, timeout=None, guard=None,cwd=None):
    """Single scoped worker; live deadline/monitor checks are inside scientific work."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    record_path = folder / 'command.json'
    if record_path.exists():
        previous = read_json(record_path)
        if previous.get('exit_code') != 0:
            raise RuntimeError('Previous real command failure requires review: ' + str(folder))
        raise FileExistsError('Command output already exists; validate and reuse its receipt')
    argv = list(map(str, argv))
    cwd=ROOT if cwd is None else Path(cwd).resolve()
    env = scoped_environment(cfg)
    env['LC0_ULTIMATE_CHILD'] = '1'
    env['LC0_AUTHORIZED_UNTIL'] = str(authorized_deadline())
    before = time.time()
    record = {'argv': argv, 'cwd': str(cwd), 'started_unix': before}
    atomic_json(record_path, record)
    log('START structured command ' + json.dumps(argv) + '; output ' + str(folder))
    try:
        with (folder / 'command.log').open('x', encoding='utf-8') as f:
            proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)
            record['pid'] = proc.pid
            atomic_json(record_path, record)
            while proc.poll() is None:
                if guard:
                    guard()
                if time.time() >= authorized_deadline() - 60 or (timeout and time.time() - before > timeout):
                    # Only this owned process tree. Windows taskkill handles children.
                    subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'], capture_output=True)
                    proc.wait()
                    if time.time()>=authorized_deadline()-60:
                        raise TimeoutError('Authorized working deadline reached; owned command quiesced')
                    raise TimeoutError('Owned command watchdog expired')
                time.sleep(1)
        record.update(exit_code=proc.returncode, elapsed_seconds=time.time() - before)
        atomic_json(record_path, record)
        log('EXIT ' + str(proc.returncode) + ' ' + json.dumps(argv) + '; elapsed ' + str(record['elapsed_seconds']))
        if proc.returncode:
            raise RuntimeError('Pipeline command exited ' + str(proc.returncode) + '; see ' + str(folder / 'command.log'))
    except BaseException as exc:
        if 'proc' in locals() and proc.poll() is None:
            subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'], capture_output=True)
            proc.wait()
        record.update(error=repr(exc), elapsed_seconds=time.time() - before)
        atomic_json(record_path, record)
        raise
    return record


class Run:
    def __init__(self, cfg, output, fresh=False, recovery=None):
        self.cfg, self.out, self.fresh = cfg, safe_output(output), fresh
        self.recovery = recovery
        self.data = self.out/'inputs' if fresh else ROOT/'data/ultimate'
        self.state_path = self.out / 'state.json'
        self.state = read_json(self.state_path) if self.state_path.exists() else {
            'protocol': fingerprint(CONTRACT), 'fresh': fresh, 'phases': {}}
        if self.state['protocol'] != fingerprint(CONTRACT) or self.state['fresh'] != fresh:
            raise ValueError('Run identity changed; no reference/fresh mixing')
        if fresh:
            from execution.ultimate_replay import prepare_runtime
            self.cfg=prepare_runtime(cfg,self.out)

    def phase(self, phase):
        deadline_check()
        working_branch()
        if not self.fresh and (ROOT/'progress.md').exists():
            # Read the live fixed-window tracker at every phase handoff.
            (ROOT/'progress.md').read_text(encoding='utf-8')
        previous = self.state['phases'].get(phase)
        history = []
        if previous:
            if previous['status'] != 'verified':
                if previous['status'] not in ('BLOCKED','checkpointed','awaiting_agent_review') or \
                        (previous['status']!='awaiting_agent_review' and not self.recovery):
                    raise RuntimeError('Unverified phase requires diagnosed recovery: ' + phase)
                history=[*previous.get('attempt_history',[]),
                         {k:v for k,v in previous.items() if k!='attempt_history'}]
                log('Resume '+phase+': '+(self.recovery or 'Visual-review checkpoint rechecked')+
                    '; prior attempt retained in attempt_history')
            else:
                for item in previous['outputs']:
                    if portable_identity(ROOT / item['path']) != item:
                        raise ValueError('Completed phase output changed')
                print('REUSE VERIFIED', phase, flush=True)
                return
        index = 10 if phase=='U10A' else int(phase[1:])
        predecessor = 'U10' if phase=='U10A' else ('U10A' if phase=='U11' else 'U'+str(index-1))
        if index and self.state['phases'].get(predecessor, {}).get('status') != 'verified':
            raise RuntimeError('Previous phase not verified')
        log('START phase ' + phase + '; output ' + str(self.out) + '; fresh=' + str(self.fresh))
        self.state['phases'][phase]={'status':'running','started_unix':time.time(),'attempt_history':history,
                                    'recovery':self.recovery}
        atomic_json(self.state_path,self.state)
        try:
            from execution import ultimate_phases
            from execution.ultimate_study import keep_awake
            awake={}
            # CPU derivation/analysis can outlast an AC idle timer too. This
            # request is scoped to actual owned phase work, with no power-mode
            # change, and is released before verification/completion stamping.
            with keep_awake(awake):
                result = getattr(ultimate_phases, phase)(self)
            log('Controller phase '+phase+' scoped awake request '+json.dumps(awake))
            if result.get('awaiting_agent_review'):
                self.state['phases'][phase]={'status':'awaiting_agent_review','attempt_history':history,
                     'outputs':[portable_identity(p) for p in result['outputs']],
                     'review_targets':result['awaiting_agent_review']}
                atomic_json(self.state_path,self.state)
                log('Numerical/structural work prepared for '+phase+'; actual agent visual review pending; no completion stamp')
                print('AGENT REVIEW REQUIRED',phase,flush=True)
                return False
            # A fresh clock follows the verification, never the estimated cell.
            stamp = datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime('%Y-%m-%d %H:%M:%S IST')
            row = {'status': 'verified', 'completed_IST': stamp,
                   'outputs': [portable_identity(p) for p in result['outputs']], 'summary': result.get('summary'),
                   'attempt_history':history}
            self.state['phases'][phase] = row
            atomic_json(self.state_path, self.state)
            if not self.fresh:
                self.update_tracker(phase, stamp)
            log('VERIFIED phase ' + phase + '; ' + json.dumps(row))
            print('VERIFIED', phase, stamp, flush=True)
        except BaseException as exc:
            window_end=isinstance(exc,TimeoutError) and 'Authorized working deadline' in str(exc)
            self.state['phases'][phase] = {'status': 'checkpointed' if window_end else 'BLOCKED', 'error': repr(exc), 'observed_unix': time.time(),
                                         'attempt_history':history}
            atomic_json(self.state_path, self.state)
            log(('CHECKPOINTED authorized window end ' if window_end else 'BLOCKED phase ') + phase + ': ' + repr(exc))
            if not self.fresh:
                p = ROOT / 'reports/takeover/progress.json'
                state = read_json(p)
                state['ultimate'].update(status=('checkpointed_window_end_' if window_end else 'BLOCKED_') + phase, error=repr(exc))
                atomic_json(p, state)
            raise

    def update_tracker(self, phase, stamp):
        import re
        path = ROOT / 'progress.md'
        text = path.read_text(encoding='utf-8')
        pattern = r'(^\| \*\*' + phase + r':[^\n]*\|)  \|$'
        text, n = re.subn(pattern, lambda m: m.group(1) + ' ' + stamp + ' |', text, flags=re.M)
        if n != 1:
            raise ValueError('Tracker completion cell is not blank/unique')
        next_phase = ('U10A' if phase=='U10' else 'U11' if phase=='U10A' else
                      'U' + str(int(phase[1:])+1) if phase != 'U12' else 'complete')
        text = re.sub(r'- Current phase:.*', '- Current phase: **' + next_phase + '**.', text)
        text = re.sub(r'\*\*State:.*?\*\*','**State: '+phase+' verified; '+next_phase+' next.**',text,count=1)
        path.write_text(text, encoding='utf-8',newline='\n')
        state_path = ROOT / 'reports/takeover/progress.json'
        state = read_json(state_path)
        state['ultimate'].update(status='running_' + next_phase if next_phase != 'complete' else 'completed_verified',
                                 new_code_implemented=True)
        state['ultimate'].setdefault('completed_phases', {})[phase] = stamp
        if phase=='U10A':
            state['ultimate'].update(status='stopped_after_verified_U10A',awaiting_explicit_U11_U12_green_light=True)
            text=path.read_text(encoding='utf-8').replace('**State: U10A verified; U11 next.**',
                '**State: U10A backup verified and published; STOPPED. U11/U12 pending explicit green light.**')
            path.write_text(text,encoding='utf-8',newline='\n')
        atomic_json(state_path, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check', 'all', 'stage', 'reference', 'render', 'review', '_block', '_native', '_match','_gate'))
    parser.add_argument('--config', default=str(ROOT / 'config.json'))
    parser.add_argument('--output', default=str(ROOT / 'reports/ultimate'))
    parser.add_argument('--fresh', action='store_true')
    parser.add_argument('--phase', choices=tuple('U'+str(i) for i in range(13))+('U10A',))
    parser.add_argument('--task', help=argparse.SUPPRESS)
    parser.add_argument('--recover-reviewed',help='Diagnosed repair/recovery note; retains failed attempt history')
    parser.add_argument('--source',help='render: editable report Markdown path')
    parser.add_argument('--pdf',help='render: destination PDF path')
    parser.add_argument('--note',help='review: attest actual visual inspection of every listed target; describe findings')
    args = parser.parse_args()
    if args.mode=='render':
        if not args.source or not args.pdf:parser.error('render requires --source and --pdf')
        from analysis.ultimate_report import render_markdown
        render_markdown(args.source,args.pdf)
        return
    if args.mode=='review':
        if args.phase not in ('U8','U9','U12') or not args.note:
            parser.error('review requires --phase U8/U9 (internal release U12) and --note describing actual completed visual inspection')
        out=safe_output(args.output)
        state=read_json(out/'state.json')
        if state['phases'].get(args.phase,{}).get('status')!='awaiting_agent_review':
            raise ValueError('Only prepared, awaiting-review work can be visually closed')
        stamp=datetime.now(timezone(timedelta(hours=5,minutes=30))).strftime('%Y-%m-%d %H:%M:%S IST')
        if args.phase=='U8':
            proof=read_json(out/'analysis/numerical-verification.json')
            for item in [*proof['sources'],*proof['outputs']]:
                if portable_identity(ROOT/item['path'])!=item:raise ValueError('Prepared analysis changed')
            atomic_json(out/'analysis/visual-review.json',{'status':'verified','figures':proof['figures'],
                'reviewed_IST':stamp,'reviewer_note':args.note,'actual_inspection_attested':True})
        else:
            reportfolder=out/('report-release' if args.phase=='U12' else 'report')
            proof=read_json(reportfolder/'structural-qa.json')
            for item in [proof['pdf'],proof['markdown'],proof['renderer'],*proof['figures']]:
                if portable_identity(ROOT/item['path'])!=item:raise ValueError('Prepared report changed')
            atomic_json(reportfolder/'visual-review.json',{'status':'verified','pdf':proof['pdf'],
                'reviewed_IST':stamp,'pages_reviewed':proof['pages'],'reviewer_note':args.note,'actual_inspection_attested':True})
        print('Visual inspection recorded; resume all/stage for phase verification and fresh completion timestamp')
        return
    cfg = config(args.config)
    if args.mode == 'check':
        print(json.dumps(check(cfg), indent=2))
    elif args.mode=='reference':
        check(cfg)
        if not (ROOT/'artifact-manifest.json').exists() or args.fresh:
            parser.error('reference requires an extracted package and no --fresh')
        from execution.ultimate_finish import analyze_phase
        run=Run(cfg,args.output,False)
        analyze_phase(run)
        if read_json(run.out/'analysis/quality.json')!=read_json(ROOT/'results/reference/analysis/quality.json'):
            raise ValueError('Deterministic reference analysis differs from bundled per-root analysis')
        if read_json(run.out/'analysis/T4-matches.json')!=read_json(ROOT/'results/reference/analysis/T4-matches.json'):
            raise ValueError('Deterministic complete-game analysis differs')
        atomic_json(run.out/'reference-verification.json',{'status':'verified','quality':portable_identity(run.out/'analysis/quality.json'),
                    'matches':portable_identity(run.out/'analysis/T4-matches.json'),
                    'scope':'Exact deterministic reanalysis of supplied observations; zero fresh scientific workloads; not full reproduction'})
    elif args.mode == '_block':
        from execution.ultimate_study import run_block
        task = read_json(args.task)
        run_block(task['folder'], cfg, task['model'], task['study'], task['budget'], task['roots'])
    elif args.mode == '_native':
        from execution.ultimate_phases import native_worker
        native_worker(read_json(args.task), cfg)
    elif args.mode == '_match':
        from execution.ultimate_phases import match_worker
        match_worker(read_json(args.task), cfg)
    elif args.mode=='_gate':
        from execution.ultimate_phases import U2
        run=Run(cfg,args.output,fresh=True)
        result=U2(run)
        atomic_json(run.out/'gate-verification.json',{'status':'verified','source_root':str(ROOT),
                    'python_prefix':sys.prefix,'python_executable':sys.executable,
                    'outputs':[portable_identity(p) for p in result['outputs']]})
    else:
        if args.mode == 'stage' and args.phase is None:
            parser.error('--phase is required for stage')
        if (ROOT/'artifact-manifest.json').exists() and not args.fresh:
            parser.error('Public scientific all/stage requires --fresh; use reference for supplied observations')
        if args.fresh and args.phase and (args.phase=='U10A' or int(args.phase[1:])>9):
            parser.error('Fresh research execution ends at U9 and does not assemble/rerun packages')
        run = Run(cfg, args.output, args.fresh, args.recover_reviewed)
        # Latest authorized root execution stops at the bounded backup gate.
        # U11/U12 remain explicit stages after a later user green light.
        phases = [args.phase] if args.mode == 'stage' else ['U'+str(i) for i in range(10 if args.fresh else 11)]+([] if args.fresh else ['U10A'])
        for phase in phases:
            if run.phase(phase) is False:
                break
