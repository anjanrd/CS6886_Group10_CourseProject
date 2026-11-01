"""Plan or explicitly execute the checked, sequential exclusive Run 2 manifest.

Import and --plan never start a GPU job. The known plugin-load failure alone may
defer its profiling branch; every other new error stops execution for review.
"""
import argparse
import json
from pathlib import Path
import re
import sys
import time
import os
import subprocess

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from execution.support import (atomic_json, fingerprint, identity, preserve_command,
                               read_json, require_session, scoped_environment, single_workload)


def known_plugin_failure(text):
    return bool(re.search(r'Unable to open library:\s*[\"\']?nvinfer_plugin_10\.dll',text,re.I))


def validate_manifest(manifest):
    jobs=manifest['jobs'];seen=set()
    for job in jobs:
        if job['id'] in seen:raise ValueError('Duplicate job '+job['id'])
        if not set(job.get('depends_on',[]))<=seen:raise ValueError('Dependency must precede job')
        if job['kind'] not in ('probe','profile','peaks','capture','benchmark','pilot'):
            raise ValueError('Unknown job kind')
        if not job['command'] or not isinstance(job['command'],list):raise ValueError('Expected argv array')
        seen.add(job['id'])
    return jobs


def validate_outputs(job,attempt):
    for relative in job['outputs']:
        p=Path(attempt)/relative
        if not p.is_file() or p.stat().st_size==0:raise ValueError('Missing output '+str(p))
        if p.suffix=='.json':
            data=read_json(p)
            if not data:raise ValueError('Empty JSON evidence')
        if p.suffix=='.nsys-rep' and p.stat().st_size<1024:raise ValueError('Empty trace')
    # Include the raw logs, serialized engines and detailed layer artifacts too,
    # rather than trusting an unchanged summary while its underlying files differ.
    return [identity(p) for p in sorted(Path(attempt).rglob('*'))
            if p.is_file() and p.name!='receipt.json']


def reusable(receipt,job,manifest_hash,runtime_hash):
    if receipt.get('status')!='verified':return False
    if receipt.get('manifest')!=manifest_hash or receipt.get('job')!=fingerprint(job):
        raise ValueError('Refusing a changed manifest/job checkpoint')
    if receipt.get('runtime')!=runtime_hash:raise ValueError('Runtime/power identity changed since prior job')
    for item in receipt['outputs']:
        if identity(item['path'])!=item:raise ValueError('Previously verified output changed')
    return True


def render_command(job,attempt,session):
    return [str(x).replace('{attempt}',str(attempt)).replace('{session}',str(session)) for x in job['command']]


def execute(manifest,session_path):
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='anjan':
        raise RuntimeError('Exclusive execution is authorized on anjan')
    from execution.resources import ActivityMonitor
    session=require_session(session_path)
    if session.get('resource_preflight_passed') is not True or not session.get('runtime'):
        raise RuntimeError('Record actual quiet-session preflight and runtime/power identities first')
    runtime=fingerprint(session['runtime']);h=fingerprint(manifest)
    for item in manifest['sources']:
        if identity(item['path'])!=item:raise ValueError('Frozen execution input changed: '+item['path'])
    output=Path(manifest['output']);output.mkdir(parents=True,exist_ok=True)
    env=scoped_environment(manifest['tools'])
    state_path=output/'state.json'
    state=read_json(state_path) if state_path.exists() else {'manifest':h,'jobs':{}}
    if state['manifest']!=h:raise ValueError('Use a new output directory for a changed manifest')
    with single_workload(output/'ACTIVE.json'):
        for job in validate_manifest(manifest):
            require_session(session_path)
            prior=state['jobs'].get(job['id'],{})
            if reusable(prior,job,h,runtime):continue
            if prior.get('status') in ('BLOCKED','invalid'):
                raise RuntimeError('Prior failed attempt requires explicit review: '+job['id'])
            if prior.get('status')=='deferred_known_plugin':continue
            if job.get('profiling_branch') and state.get('profiling_deferred'):
                state['jobs'][job['id']]={'status':'deferred_known_plugin','reason':'requires blocked trtexec branch'}
                atomic_json(state_path,state);continue
            if any(state['jobs'].get(d,{}).get('status')!='verified' for d in job.get('depends_on',[])):
                raise RuntimeError('Unverified dependency for '+job['id'])
            folder=output/job['id'];folder.mkdir(exist_ok=True)
            attempts=list(folder.glob('attempt-*'))
            attempt=folder/f'attempt-{len(attempts)+1:03d}';attempt.mkdir()
            command=render_command(job,attempt,session_path)
            receipt={'status':'running','manifest':h,'job':fingerprint(job),'runtime':runtime,
                     'session':identity(session_path),'attempt':str(attempt),'command':command,
                     'started_unix':time.time()}
            state['jobs'][job['id']]=receipt;atomic_json(state_path,state)
            print('START '+job['id'],flush=True)
            try:
                with ActivityMonitor(attempt/'activity.log') as monitor:
                    p=preserve_command(command,attempt/'command.log',env=env)
                conditions=monitor.report(os.getpid())
            except BaseException as exc:
                receipt.update(status='BLOCKED',error='launch/monitor interruption: '+repr(exc),
                               elapsed_seconds=time.time()-receipt['started_unix'])
                atomic_json(attempt/'receipt.json',receipt);atomic_json(state_path,state)
                raise
            atomic_json(attempt/'conditions.json',conditions)
            receipt['conditions']=identity(attempt/'conditions.json')
            receipt['exit_code']=p.returncode;receipt['elapsed_seconds']=time.time()-receipt['started_unix']
            try:
                if conditions['flags'] or conditions['monitor_errors']:
                    raise RuntimeError('Environmental interference/monitoring failure; preserve and review attempt')
                if p.returncode!=0:raise RuntimeError('Child command exited '+str(p.returncode))
                receipt['outputs']=validate_outputs(job,attempt)
                receipt['status']='verified'
                receipt['verification_scope']='exit status, structured outputs, identities and resource flags; scientific interpretation/precision mapping requires Run 3'
            except Exception as exc:
                receipt['error']=str(exc);receipt['status']='BLOCKED'
                if job['kind']=='probe' and not conditions['flags'] and not conditions['monitor_errors']:
                    text='\n'.join(p.read_text(encoding='utf-8',errors='replace')[-30000:]
                                   for p in attempt.rglob('*.log'))
                    if known_plugin_failure(text):
                        receipt['status']='deferred_known_plugin';state['profiling_deferred']=True
                atomic_json(attempt/'receipt.json',receipt);atomic_json(state_path,state)
                if receipt['status']=='BLOCKED':raise
            atomic_json(attempt/'receipt.json',receipt);atomic_json(state_path,state)
            print(receipt['status'].upper()+' '+job['id'],flush=True)
        # Final original-input audit catches edits made during a long sequence.
        for item in manifest['sources']:
            if identity(item['path'])!=item:
                state.update(status='BLOCKED',error='Execution input changed during run: '+item['path'])
                atomic_json(state_path,state)
                raise ValueError(state['error'])
        state['status']='collected_with_profiling_deferred' if state.get('profiling_deferred') else 'collected_for_run3_review'
        atomic_json(state_path,state)
    return state


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',default='reports/takeover/run1/run2_jobs.json')
    mode=ap.add_mutually_exclusive_group()
    mode.add_argument('--execute',action='store_true');mode.add_argument('--plan',action='store_true')
    ap.add_argument('--session')
    a=ap.parse_args();manifest=read_json(a.manifest);jobs=validate_manifest(manifest)
    if not a.execute:
        print(json.dumps({'mode':'plan_only','GPU_workloads_started':0,'jobs':len(jobs),
                          'order':[j['id'] for j in jobs]},indent=2));return
    if not a.session:raise ValueError('Explicit exclusive-session file is required')
    execute(manifest,Path(a.session).resolve())


if __name__=='__main__':main()
