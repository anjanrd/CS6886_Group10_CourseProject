"""Curated candidate/release construction; never archives the checkout wholesale."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

from execution.support import read_json,fingerprint
from execution.ultimate_contract import ROOT,A10,CONTRACT,atomic_json,portable_identity,freeze,deadline_check

NAME='CS6886_Group10_CourseProject'


def checked_transfer(source,destination):
    while True:
        deadline_check()
        block=source.read(8*1024*1024)
        if not block:break
        destination.write(block)


def deadline_copy(source,destination):
    with Path(source).open('rb') as left,Path(destination).open('xb') as right:
        checked_transfer(left,right)
    shutil.copystat(source,destination)
    return str(destination)


def verify_crc(stream):
    for item in stream.infolist():
        with stream.open(item) as source:
            while True:
                deadline_check()
                if not source.read(8*1024*1024):break


def verify_manifest(root):
    root=Path(root).resolve()
    manifest=read_json(root/'artifact-manifest.json')
    seen=set()
    for item in manifest['files']:
        deadline_check()
        relative=Path(item['path'])
        target=(root/relative).resolve()
        if relative.is_absolute() or not target.is_relative_to(root) or relative.as_posix() in seen:
            raise ValueError('Unsafe or duplicate artifact manifest entry')
        actual=portable_identity(target,root)
        if actual!=item:
            raise ValueError('Package input/code/reference artifact changed: '+item['path'])
        seen.add(relative.as_posix())
    return manifest


def archive(folder,path):
    folder,path=Path(folder),Path(path)
    if path.exists():
        raise FileExistsError('Preserve existing package archive')
    if path.is_relative_to(folder):
        raise ValueError('Archive cannot contain itself')
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as stream:
        for p in sorted(folder.rglob('*')):
            if p.is_file():
                info=zipfile.ZipInfo.from_file(p,(Path(folder.name)/p.relative_to(folder)).as_posix())
                info.compress_type=zipfile.ZIP_DEFLATED
                with p.open('rb') as source,stream.open(info,'w',force_zip64=True) as destination:
                    checked_transfer(source,destination)
    with zipfile.ZipFile(path) as stream:
        verify_crc(stream)
        if {Path(n).parts[0] for n in stream.namelist()}!={folder.name}:
            raise ValueError('Archive content/CRC/top-level-folder check failed')
    return path


def public_readme(native):
    return f'''# BT3 INT8 in lc0 - CS6886 Group10

Ritwik Mishra (ME23B237) and Anjan R D (ME23B157), IIT Madras.

Read report.pdf (editable report.md) for the question, methods, measured outcomes and limitations. Reference observations are under results/reference. Fresh measurements are written separately; reference results never satisfy a fresh scientific job.

## Tested prerequisites

Windows11/64-bit, Python3.11.9, RTX4060 Laptop8GB,32GB host memory, NVIDIA driver592.82. Other operating systems/GPUs are untested. Use AC, the recorded Balanced power mode and clear cooling, one GPU workload at a time. Natural clocks are logged; no clock lock is claimed. Background activity remains informational, while actual AC, monitor, command, numerical and output failures stop the affected sequence for diagnosis.

Provide lc0 v0.33.0-rc0 with its bundled GPU ORT1.22.1, TensorRT10.12.0.36/trtexec and matching DLLs, Nsight Systems2025.1.3, and the Syzygy3-5-piece tables. Python ORT1.31.1 is CPU preparation, a separate installation. Exact native file/table identities are in data/ultimate/native-inputs.json and tablebases.json. Native installations are declared prerequisites, not a hidden source/model dependency. The tested same-host paths were:

'''+ '\n'.join('- '+k+': '+v for k,v in native.items())+r'''

Official native sources: https://github.com/LeelaChessZero/lc0/releases , https://developer.nvidia.com/tensorrt , https://developer.nvidia.com/nsight-systems . Preserve upstream licenses/notices. Required research checkpoint/models/calibration/data are supplied here. Serialized reference engines are hardware-specific evidence; fresh runs build their own outputs/caches where applicable. Driver-level caching may persist and is disclosed.

## Setup and checks

Create a new environment locally; never copy a venv from another path/machine:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements/ultimate-report.lock.txt
Copy-Item config.example.json config.json
# Edit the lc0/trtexec/nsys/syzygy paths for your declared native installations.
.\.venv\Scripts\python.exe -X utf8 -B reproduce.py check
```

The check command verifies supplied code/input/reference hashes and prerequisites; it does not run inference. The manifest excludes itself and generated fresh results. Run from this extracted folder. Git, branch names, Codex and a Codex runtime are not required.

## Complete fresh reproduction

```powershell
.\.venv\Scripts\python.exe -X utf8 -B reproduce.py all --fresh --output results/reproduced
```

This is a substantial exclusive run, not a smoke test. It reconstructs export/quantization/pruning/swaps and frozen development-selection evidence, regenerates/audits datasets, repeats the A10 numerical gate and systems checks, runs the complete raw/node/challenge/timed matrix and both500-game matches, repeats reported hardware/energy/batch/F12 work, and regenerates analysis/report. Keep frozen HM/HN IDs: fresh screening is an audit, not reselection. Follow state.json and per-job receipts under the output. Completed exact-compatible keys from this fresh invocation can resume; bundled reference results cannot. A diagnosed repair retains failed evidence and uses a new attempt path. Source/input/settings changes require affected checks again.

The original complete-scope planning range was45-70hours, with72+ reserved; it is not an overnight promise. Earlier full main-node measurements alone estimated13.87h and full main timed allowances5.60h, before other studies/derivation/engine builds/matches. Actual early stopping and hardware change durations. Allow at least100GiB free space and the tested32GB host/8GB GPU memory. Fresh execution byte-verifies and copies lc0 plus its DLLs into its output runtime directory, excluding the original trt_cache: this version caches beside its executable. New engines/timing caches are built there; TensorRT/Nsight/driver/table installations remain declared prerequisites, and driver caches may persist.

The public stage option supports an interrupted pipeline; it enforces dependencies. Some final visual QA steps require inspection of the newly rendered figures/PDF before verification. The research execution does not assemble or recursively rerun another submission package. Reference-analysis checks use supplied per-root observations, separately from fresh scientific work.

At U8, inspect every PNG listed in state.json for text legibility, labels, clipping, legends and correspondence to tables. At U9, render/inspect every PDF page using a normal PDF viewer (or a separately installed Poppler); verify layout and numbers. Only after actual inspection, record it and resume:

```powershell
.\.venv\Scripts\python.exe -X utf8 -B reproduce.py review --fresh --output results/reproduced --phase U8 --note "Describe the completed inspection and findings"
.\.venv\Scripts\python.exe -X utf8 -B reproduce.py all --fresh --output results/reproduced
# Repeat review with --phase U9 after inspecting every newly generated PDF page.
```

For an interrupted diagnosed failure, add --recover-reviewed with the diagnosis/repair note; prior failed evidence remains. Review is an inspection attestation, not an automatic image-quality test. Deterministic reference reanalysis is available with `reproduce.py reference --output results/reference-reanalysis`; it must match supplied root/game summaries exactly and is never counted as fresh experimental reproduction. To render the editable supplied Markdown independently: `reproduce.py render --source report.md --pdf report-edited.pdf`.

Fresh search/match/performance results can vary. Report counts, timing, termination and interpretation differences; no required INT8 winner, significance or Elo threshold exists. Numerical/backend/protocol faults differ from valid changed scientific outcomes. See verification.json for the actual completed evaluator coverage; subset imports or plotting reference results cannot mean a full pass.

## File purposes

reproduce.py plus necessary core modules implement execution and analysis. data/nets/models are frozen research inputs and canonical model artifacts. results/reference contains compact root/game and essential systems evidence. report.pdf/report.md and figures/tables communicate the result. requirements/config and native identities make setup explicit. artifact-manifest.json binds the tested code/inputs/reference files. No archive, Git metadata, old failures, internal roadmap, venv or credentials is included.

The study is bounded to one network/device and synthetic endgames plus the stated opening book. Printed-prior KL is approximate; V drift is not full WDL-vector error. Above-idle historical energy remains provisional; total under-load GPU energy is separate. Standalone sparse tactics do not establish lc0 sparse acceleration. Full details and actual win margins are in the report.
'''


def candidate_phase(run):
    from execution.ultimate_finish import copy_exact
    from execution.ultimate_phases import existing_rows
    from execution.ultimate_data import load_pool
    prepared=run.out/'candidate.json'
    if prepared.exists():
        saved=read_json(prepared)
        verify_manifest(saved['folder'])
        return {'outputs':[prepared,Path(saved['zip'])]}
    base=Path(os.environ['LOCALAPPDATA'])/'LC0-BT3-Evaluator'/('candidate-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    folder=base/NAME
    folder.mkdir(parents=True,exist_ok=False)
    purposes={}
    def include(source,relative,purpose):
        deadline_check()
        copy_exact(source,folder/relative)
        purposes[Path(relative).as_posix()]=purpose
    for name in ('common.py','ladder.py','reproduce.py'):
        include(ROOT/name,name,'execution')
    for directory in ('quant','eval','tb','hw'):
        for p in sorted((ROOT/directory).glob('*.py')):
            include(p,p.relative_to(ROOT),'execution')
    selected=['__init__.py','support.py','resources.py','benchmark.py','gpu_job.py','run_jobs.py','run4_checks.py']
    selected += [p.name for p in (ROOT/'execution').glob('ultimate*.py')]
    for name in selected:
        include(ROOT/'execution'/name,Path('execution')/name,'checked execution/replay dependency')
    for name in ('ultimate.py','ultimate_report.py','ultimate_systems.py'):
        include(ROOT/'analysis'/name,Path('analysis')/name,'analysis/report')
    include(ROOT/'analysis/person_a/run3_systems.py','analysis/person_a/run3_systems.py','reused profile/operation precision analysis')
    include(ROOT/'tests/test_ultimate.py','tests/test_ultimate.py','focused correctness checks')
    for name in ('windows-prep.lock.txt','ultimate-report.lock.txt'):
        include(ROOT/'requirements'/name,Path('requirements')/name,'exact Python dependency lock')
    for name in ('tbset.jsonl','calib.npz','dev.npz'):
        include(ROOT/'data'/name,Path('data')/name,'frozen research input')
    for p in sorted((ROOT/'data/ultimate').rglob('*')):
        if p.is_file() and p.name!='reuse.json':
            include(p,p.relative_to(ROOT),'frozen cohorts/book/provenance')
    include(ROOT/'nets/BT3.pb.gz','nets/BT3.pb.gz','source checkpoint')
    model_files={'bt3_fp32.onnx'}
    for name in A10:
        model_files.add(name+'.pb.gz')
        if name!='fp16':model_files.add(name+'.onnx')
        if (ROOT/'models'/(name+'.json')).exists():model_files.add(name+'.json')
    for name in sorted(model_files):
        include(ROOT/'models'/name,Path('models')/name,'canonical retained model/recipe')
    native={k:run.cfg[k] for k in ('lc0','trtexec','nsys','syzygy')}
    example={k:run.cfg[k] for k in ('lc0','trtexec','nsys','syzygy')}
    example.update(work='.',bt3='nets/BT3.pb.gz',best='s4',
                   recipe={'s3_method':'percentile','s3_percentile':99.999,'s4_alpha':.7},ultimate=CONTRACT)
    for key in ('lc0','trtexec','nsys','syzygy'):
        example[key]='C:/declared-native-installation/'+('syzygy' if key=='syzygy' else key+'.exe')
    atomic_json(folder/'config.example.json',example);purposes['config.example.json']='machine-path template; frozen scientific values'
    atomic_json(folder/'data/ultimate/native-inputs.json',{'tested_paths':native,
                'files':[portable_identity(p) for k in ('lc0','trtexec','nsys') for p in [run.cfg[k]]],
                'lc0_dependencies':[portable_identity(p) for p in sorted(Path(run.cfg['lc0']).parent.glob('*.dll'))],
                'runtime':read_json(ROOT/'reports/ultimate/runtime-identity.json')})
    purposes['data/ultimate/native-inputs.json']='declared native prerequisites/identities'
    numeric=read_json(run.out/'U2/numeric-contract.json')
    for spec in numeric['models'].values():spec.pop('fresh_weights',None)
    numeric.pop('tools',None);numeric.pop('idle_validation_path',None)
    atomic_json(folder/'data/ultimate/numeric-reference.json',numeric)
    purposes['data/ultimate/numeric-reference.json']='frozen32-root numerical reference/tolerances'
    original=read_json(ROOT/'reports/takeover/run1/run2_jobs.json')
    tasks=[]
    for job in original['jobs']:
        if job['kind']=='pilot':continue
        spec={k:v for k,v in job.items() if k in ('id','kind','outputs','depends_on')}
        argv=[]
        for value in job['command']:
            if value==original['jobs'][0]['command'][0]:value='{python}'
            for key in ('lc0','trtexec','nsys'):
                if value==original['tools'][key]:value='{'+key+'}'
            prefix=str(ROOT/'models')
            if value.startswith(prefix):value='{models}'+value[len(prefix):]
            argv.append(value)
        spec['command']=argv;tasks.append(spec)
    if len(tasks)!=44:raise ValueError('Incorrect systems reproduction task coverage')
    atomic_json(folder/'data/ultimate/systems-tasks.json',{'jobs':tasks,'source':portable_identity(ROOT/'reports/takeover/run1/run2_jobs.json'),
                'scope':'All retained native workloads, old game-clock pilot excluded; new matches use frozen fixed movetime'})
    purposes['data/ultimate/systems-tasks.json']='frozen full systems replay task templates'
    # Compact observation views; original source paths/hashes remain provenance,
    # while duplicate task-root lists/controller histories are omitted.
    provenance=read_json(run.data/'reuse.json')
    observations=read_json(run.out/'observations/manifest.json')
    if len(observations['files'])!=22:raise ValueError('Compact retained study/model coverage differs')
    for item in observations['files']:
        path=run.out/'observations'/item['path']
        if portable_identity(path,run.out/'observations')!=item:raise ValueError('Prepared compact observations changed')
        include(path,Path('results/reference/observations')/item['path'],'compact actual per-root observations; exact unique keys')
    include(run.out/'observations/manifest.json','results/reference/observations/manifest.json','compact observation protocol/model/data bindings')
    for name in ('select_s3_dev.txt','select_s4_dev.txt','quick_compare_dev.txt'):
        include(ROOT/'reports'/name,Path('results/reference/selection')/name,'original development-grid and post-build numerical evidence')
    atomic_json(folder/'results/reference/provenance.json',provenance)
    purposes['results/reference/provenance.json']='source/condition reuse lineage; historical paths are provenance only'
    for p in (run.out/'matches/production').glob('game-*.json'):
        include(p,Path('results/reference/matches')/p.name,'complete-game atomic observation')
    for name in ('contract.json','summary.json','games.pgn'):
        include(run.out/'matches/production'/name,Path('results/reference/matches')/name,'frozen match contract/aggregate')
    for name in ('contract.json','timing-gate.json','summary.json','games.pgn'):
        include(run.out/'matches/pilot'/name,Path('results/reference/pilot')/name,'excluded technical pilot/gate evidence')
    for p in (run.out/'matches/pilot').glob('game-*.json'):
        include(p,Path('results/reference/pilot')/p.name,'excluded technical pilot atomic observation')
    for p in (run.out/'analysis').rglob('*'):
        if p.is_file():include(p,Path('results/reference/analysis')/p.relative_to(run.out/'analysis'),'reference analysis/figures')
    for name in ('report.md','report.pdf'):
        include(run.out/'report'/name,name,'technical report')
    markdown=folder/'report.md'
    markdown.write_text(markdown.read_text(encoding='utf-8').replace('](../analysis/','](results/reference/analysis/'),encoding='utf-8',newline='\n')
    import re
    for relative in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',markdown.read_text(encoding='utf-8')):
        target=(folder/relative).resolve()
        if not target.is_relative_to(folder.resolve()) or not target.is_file():
            raise ValueError('Editable report has a hidden/missing figure dependency')
    # Shared historical hardware source CSVs are needed to regenerate the
    # deterministic reference analysis without importing the original checkout.
    for p in (run.out/'analysis/hardware').glob('*'):
        if p.is_file() and p.suffix=='.csv':include(p,Path('analysis/tables')/p.name,'bounded reference hardware table')
    include(ROOT/'reports/c3_hw/batch_sweep/sweep.json','reports/c3_hw/batch_sweep/sweep.json','retained four-cap search sweep observations')
    for name in ('numeric-verification.json','systems-verification.json'):
        include(run.out/'U2'/name,Path('results/reference/U2')/name,'working-environment numerical/systems supplemental validation')
    independent=read_json(run.out/'U2/independent-environment.json')
    for item in independent['outputs']:
        p=Path(item['path'])
        if p.name in ('numeric-verification.json','systems-verification.json'):
            include(p,Path('results/reference/U2-independent')/p.name,'independent-environment bounded validation')
    include(run.out/'U2/idle/work/idle.json','results/reference/U2/idle/work/idle.json','unresolved idle-telemetry diagnostic')
    from analysis.person_a.run3_systems import catalog
    definitions={j['id']:j for j in original['jobs']}
    for name,receipt in catalog().items():
        if definitions[name]['kind']=='pilot':continue
        attempt=Path(receipt['attempt'])
        for p in attempt.rglob('*'):
            if not p.is_file():continue
            relative=p.relative_to(attempt)
            kind=definitions[name]['kind']
            retain=p.name in ('receipt.json','conditions.json','activity.log','result.json')
            retain=retain or p.suffix=='.nsys-rep'
            if kind in ('profile','peaks','probe'):
                retain=retain or p.suffix in ('.json','.log','.engine')
            if kind=='benchmark':retain=retain or p.suffix=='.log'
            if retain:
                include(p,Path('results/reference/systems')/name/'attempt-001'/relative,'raw retained systems evidence; hardware-specific engines labelled')
    for name in ('COPYING','ONNX-LICENSE','ONNX-ThirdPartyNotices.txt','mimalloc-LICENSE'):
        include(Path(run.cfg['lc0']).parent/name,Path('licenses')/name,'upstream lc0/runtime notice; native installs remain prerequisites')
    for name in ('current-channel-maxima.npz','collection.json','closure.json'):
        include(ROOT/'analysis/person_a/a5'/name,Path('results/reference/F12')/name,'version-matched calibration evidence')
    (folder/'README.md').write_text(public_readme(native),encoding='utf-8',newline='\n')
    purposes['README.md']='single starting point/setup/full reproduction protocol'
    atomic_json(folder/'verification.json',{'status':'candidate_pending_full_evaluator_replay',
                'full_reproduction_verified':False,'reference_phases_completed':'U0-U9',
                'scope':'Candidate only; not the final release or an actual evaluator pass'})
    purposes['verification.json']='honest integrated QA/reproduction status'
    files=[portable_identity(p,folder) for p in sorted(folder.rglob('*')) if p.is_file()]
    if set(purposes)!={p['path'] for p in files}:
        raise ValueError('Every curated file needs an explicit purpose')
    atomic_json(folder/'artifact-manifest.json',{'files':files,'purposes':purposes,'protocol':CONTRACT,
                'manifest_excludes_itself':True,'generated_outputs_excluded':['results/reproduced','LOG.md']})
    verify_manifest(folder)
    zipped=archive(folder,base/(NAME+'.zip'))
    atomic_json(prepared,{'folder':str(folder),'zip':str(zipped),'manifest':portable_identity(folder/'artifact-manifest.json'),
                'archive':portable_identity(zipped),'files':len(files),'bytes':sum(p['bytes'] for p in files),
                'status':'curated_candidate_pending_full_evaluator_replay'})
    return {'outputs':[prepared,zipped]}


def extract_verified(zipped,destination):
    """Extract a new single-root candidate; reject traversal/links/duplicates."""
    destination=Path(destination).resolve()
    destination.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(zipped) as stream:
        seen=set()
        for item in stream.infolist():
            name=item.filename.replace('\\','/')
            parts=name.split('/')
            target=(destination/Path(name)).resolve()
            if name in seen or parts[0]!=NAME or any(p in ('..','') for p in parts) or ':' in name or not target.is_relative_to(destination):
                raise ValueError('Unsafe candidate ZIP member')
            if (item.external_attr>>16)&0o170000==0o120000:
                raise ValueError('Candidate symlink refused')
            seen.add(name)
        verify_crc(stream)
        for item in stream.infolist():
            target=destination/Path(item.filename.replace('\\','/'))
            target.parent.mkdir(parents=True,exist_ok=True)
            with stream.open(item) as source,target.open('xb') as output:
                checked_transfer(source,output)
    folder=destination/NAME
    verify_manifest(folder)
    return folder


def evaluator_phase(run):
    """Actual public-command walk from external extraction/new local venv."""
    from execution.ultimate import command
    from execution.support import single_workload
    candidate=read_json(run.out/'candidate.json')
    if portable_identity(candidate['zip'])!=candidate['archive']:
        raise ValueError('Candidate archive changed before evaluator walk')
    setup_path=run.out/'evaluator-setup.json'
    if setup_path.exists():
        setup=read_json(setup_path)
        folder=Path(setup['folder']);base=Path(setup['workspace']);py=Path(setup['python'])
        verify_manifest(folder)
    else:
        base=Path(os.environ['LOCALAPPDATA'])/'LC0-BT3-Evaluator'/('walk-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
        base.mkdir(parents=True,exist_ok=False)
        folder=extract_verified(candidate['zip'],base/'extracted')
        env=base/'venv'
        command([sys.executable,'-X','utf8','-B','-m','venv',env],base/'setup/venv',run.cfg)
        py=env/'Scripts/python.exe'
        command([py,'-X','utf8','-B','-m','pip','install','-r',folder/'requirements/ultimate-report.lock.txt'],base/'setup/pip',run.cfg,cwd=folder)
        command([py,'-X','utf8','-B','-m','pip','check'],base/'setup/pip-check',run.cfg,cwd=folder)
        cfg={k:v for k,v in run.cfg.items() if not k.startswith('_')}
        cfg.update(work='.',bt3='nets/BT3.pb.gz')
        atomic_json(folder/'config.json',cfg)
        command([py,'-X','utf8','-B',folder/'reproduce.py','check'],base/'setup/check',run.cfg,cwd=folder)
        probe_script='''import json,sys,platform
from pathlib import Path
from importlib.metadata import version
import common
from execution.support import ROOT
expected={}
for name in ('windows-prep.lock.txt','ultimate-report.lock.txt'):
 for line in (ROOT/'requirements'/name).read_text().splitlines():
  if '==' in line and not line.lstrip().startswith('#'):
   key,value=line.strip().split('==',1);expected[key]=value
actual={key:version(key) for key in expected}
assert actual==expected,(actual,expected)
assert Path(common.__file__).resolve()==ROOT/'common.py'
print(json.dumps({'prefix':sys.prefix,'executable':sys.executable,'source_root':str(ROOT),'common':common.__file__,'python':platform.python_version(),'packages':actual}))
'''
        command([py,'-X','utf8','-B','-c',probe_script],base/'setup/environment-probe',run.cfg,cwd=folder)
        environment=read_json(base/'setup/environment-probe/command.log')
        if Path(environment['prefix']).resolve()!=env.resolve() or Path(environment['source_root']).resolve()!=folder.resolve():
            raise ValueError('Evaluator interpreter/source isolation probe failed')
        setup={'workspace':str(base),'folder':str(folder),'python':str(py),
               'environment':environment,
               'candidate_archive':candidate['archive'],'candidate_manifest':candidate['manifest'],
               'venv_created_locally':True,'old_venv_copied':False,'native_installations_shared':True,
               'source_and_research_inputs':'Extracted solely from candidate ZIP; no checkout imports or symlinks',
               'fresh_output':str(base/'replay'),'reference_reanalysis':str(base/'reference-reanalysis')}
        atomic_json(setup_path,setup)
    reference=base/'reference-reanalysis'
    if not (reference/'reference-verification.json').exists():
        command([py,'-X','utf8','-B',folder/'reproduce.py','reference','--output',reference],base/'reference-launch',run.cfg,cwd=folder)
    output=base/'replay'
    state=read_json(output/'state.json') if (output/'state.json').exists() else {'phases':{}}
    complete=all(state['phases'].get('U'+str(i),{}).get('status')=='verified' for i in range(10))
    if not complete:
        number=1
        while (base/f'launch-{number:03d}').exists():number+=1
        argv=[py,'-X','utf8','-B',folder/'reproduce.py','all','--fresh','--output',output]
        if run.recovery:argv+=['--recover-reviewed',run.recovery]
        # Root and extracted workers use different lock roots. Hold the original
        # locks to exclude legacy/root jobs throughout this owned external walk.
        with single_workload(ROOT/'reports/takeover/run2/ACTIVE.json'),single_workload(ROOT/'reports/ultimate/ACTIVE.json'):
            command(argv,base/f'launch-{number:03d}',run.cfg,cwd=folder)
        state=read_json(output/'state.json')
    pending=[p for p in state['phases'].values() if p['status']=='awaiting_agent_review']
    if pending:
        return {'awaiting_agent_review':[t for p in pending for t in p['review_targets']],
                'outputs':[setup_path,output/'state.json']}
    if not all(state['phases'].get('U'+str(i),{}).get('status')=='verified' for i in range(10)):
        raise ValueError('Evaluator replay incomplete; no full-reproduction pass')
    verify_manifest(folder)
    comparison=compare_replay(run,output)
    verification=run.out/'evaluator-verification.json'
    atomic_json(verification,{'status':'verified','full_reproduction_verified':True,
        'setup':portable_identity(setup_path),'fresh_state':portable_identity(output/'state.json'),
        'reference_reanalysis':portable_identity(reference/'reference-verification.json'),
        'derivation':portable_identity(output/'derivation/verification.json'),
        'systems':portable_identity(output/'systems/verification.json'),
        'F12':portable_identity(output/'F12/verification.json'),
        'comparison':portable_identity(comparison),'fresh_report':portable_identity(output/'report/report.pdf'),
        'scope':'Actual U0-U9 full fresh replay, all retained experimental conditions/model derivation/hardware/F12 and both500-game matches, regenerated analysis/report reviewed. Same host/native installations; new locally created venv/source extraction.'})
    return {'outputs':[verification,setup_path,comparison]}


def compare_replay(run,output):
    """Describe changed outcomes; valid differences are not acceptance failures."""
    old=read_json(run.out/'analysis/quality.json');new=read_json(output/'analysis/quality.json')
    key=lambda r:(r['cohort'],r['study'],r['budget'],r['model'])
    a={key(r):r for r in old['rows']};b={key(r):r for r in new['rows']}
    if a.keys()!=b.keys():raise ValueError('Fresh comparison condition coverage differs')
    quality=[]
    for k in a:
        if a[k]['n']!=b[k]['n']:raise ValueError('Fresh comparison denominator differs')
        quality.append({'cohort':k[0],'study':k[1],'budget':k[2],'model':k[3],'n':a[k]['n'],
            'reference_correct':a[k]['correct'],'fresh_correct':b[k]['correct'],
            'correct_difference':b[k]['correct']-a[k]['correct'],
            'reference_nodes_mean':a[k]['mean_actual_nodes'],'fresh_nodes_mean':b[k]['mean_actual_nodes'],
            'reference_wall_ms_mean':a[k]['mean_wall_ms'],'fresh_wall_ms_mean':b[k]['mean_wall_ms']})
    from execution.ultimate_finish import csvrows
    oldspeed={r['model']:r for r in csvrows(run.out/'analysis/hardware/T2-speed.csv')}
    newspeed={r['model']:r for r in csvrows(output/'analysis/hardware/T2-speed.csv')}
    speed=[{'model':m,'reference_nps_median':float(oldspeed[m]['engine_nps_median']),
            'fresh_nps_median':float(newspeed[m]['engine_nps_median'])} for m in oldspeed]
    path=run.out/'evaluator-comparison.json'
    atomic_json(path,{'status':'verified_descriptive_comparison','quality':quality,'speed':speed,
        'reference_matches':read_json(run.out/'analysis/T4-matches.json'),
        'fresh_matches':read_json(output/'analysis/T4-matches.json'),
        'screening_audit':read_json(output/'screening-audit.json'),
        'scientific_acceptance':'No required winner, p-value, speed band for full fresh scientific outcomes, or identical game count by result. Coverage and execution correctness are required.'})
    return path


def release_phase(run):
    from execution.ultimate_finish import copy_exact
    from analysis.ultimate_report import report_phase
    verification=read_json(run.out/'evaluator-verification.json')
    if not verification.get('full_reproduction_verified'):raise ValueError('D8 requires full evaluator reproduction')
    for name in ('setup','fresh_state','reference_reanalysis','derivation','systems','F12','comparison','fresh_report'):
        item=verification[name]
        if portable_identity(ROOT/item['path'])!=item:raise ValueError('Evaluator evidence changed before D8')
    candidate=read_json(run.out/'candidate.json');source=Path(candidate['folder'])
    original_manifest=verify_manifest(source)
    for item in original_manifest['files']:
        if Path(item['path']).suffix=='.py':
            current=portable_identity(ROOT/item['path'])
            if any(current[k]!=item[k] for k in ('bytes','sha256')):
                raise ValueError('Working code differs from evaluator-tested candidate; validate affected changes before D8')
    # Same tested renderer/source, updated numerical reconciliation is metadata.
    release_report=run.out/'report-release'
    report=report_phase(run,folder=release_report)
    if report.get('awaiting_agent_review'):return report
    parent=ROOT/'submissionready'
    destination=parent/NAME;destination_zip=parent/(NAME+'.zip')
    if destination.exists() or destination_zip.exists():raise FileExistsError('Preserve pre-existing final release for review')
    staging=Path(os.environ['LOCALAPPDATA'])/'LC0-BT3-Evaluator'/('release-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    staging.mkdir(parents=True,exist_ok=False)
    final=staging/NAME;zipped=staging/(NAME+'.zip')
    shutil.copytree(source,final,copy_function=deadline_copy)
    for name in ('report.md','report.pdf'):
        shutil.copy2(release_report/name,final/name)
    markdown=final/'report.md'
    markdown.write_text(markdown.read_text(encoding='utf-8').replace('](../analysis/','](results/reference/analysis/'),encoding='utf-8',newline='\n')
    fresh_setup=read_json(run.out/'evaluator-setup.json');fresh=Path(fresh_setup['fresh_output'])
    for name in ('report.md','report.pdf'):
        copy_exact(fresh/'report'/name,final/'results/evaluator/report'/name)
    for p in (fresh/'analysis').rglob('*'):
        if p.is_file() and p.suffix in ('.png','.csv','.json'):
            copy_exact(p,final/'results/evaluator/analysis'/p.relative_to(fresh/'analysis'))
    for p in (fresh/'observations').rglob('*'):
        if p.is_file():copy_exact(p,final/'results/evaluator/observations'/p.relative_to(fresh/'observations'))
    copy_exact(fresh/'F12/channel-maxima.npz',final/'results/evaluator/F12/channel-maxima.npz')
    copy_exact(fresh/'F12/verification.json',final/'results/evaluator/F12/verification.json')
    copy_exact(fresh/'systems/verification.json',final/'results/evaluator/systems/verification.json')
    copy_exact(fresh/'runtime/verification.json',final/'results/evaluator/runtime-verification.json')
    copy_exact(run.out/'evaluator-setup.json',final/'results/evaluator/setup.json')
    for kind in ('pilot','production'):
        for p in (fresh/'matches'/kind).glob('*'):
            if p.is_file() and (p.name.startswith('game-') or p.name in ('contract.json','summary.json','games.pgn','timing-gate.json')):
                copy_exact(p,final/'results/evaluator/matches'/kind/p.name)
    specs={j['id']:j for j in read_json(source/'data/ultimate/systems-tasks.json')['jobs']}
    for name,receipt in read_json(fresh/'systems/verification.json')['jobs'].items():
        attempt=Path(receipt['attempt']);kind=specs[name]['kind']
        for p in attempt.rglob('*'):
            if not p.is_file():continue
            retain=p.name in ('receipt.json','conditions.json','activity.log','result.json') or p.suffix=='.nsys-rep'
            if kind in ('profile','peaks','probe'):retain=retain or p.suffix in ('.json','.log','.engine')
            if kind=='benchmark':retain=retain or p.suffix=='.log'
            if retain:
                deadline_check()
                copy_exact(p,final/'results/evaluator/systems'/name/'attempt-001'/p.relative_to(attempt))
    for name in ('numeric-verification.json','systems-verification.json','idle/work/idle.json'):
        copy_exact(fresh/'U2'/name,final/'results/evaluator/U2'/name)
    copy_exact(fresh/'derivation/verification.json',final/'results/evaluator/derivation-verification.json')
    for name in ('select_s3_dev.txt','select_s4_dev.txt','quick_compare_dev.txt'):
        copy_exact(fresh/'derivation/reports'/name,final/'results/evaluator/selection'/name)
    for name in ('evaluator-comparison.json','evaluator-verification.json'):
        copy_exact(run.out/name,final/'results/evaluator'/name)
    stamp=datetime.now().astimezone().isoformat()
    record={'status':'verified_submission','full_reproduction_verified':True,'core_verification_at':stamp,
            'tested_candidate_archive':candidate['archive'],'same_tested_code_and_research_inputs':True,
            'final_changes':'Only report/reconciliation and verification metadata; no post-test code/model/scorer changes',
            'scope':verification['scope'],'limitations':'Same hardware/native installations, natural clocks; idle subtraction provisional; fresh outcomes descriptive.'}
    atomic_json(final/'verification.json',record)
    # Bind every final file; compare execution/model/input hashes to the tested
    # candidate, permitting only declared report/QA metadata differences.
    allowed={'verification.json','report.md','report.pdf'}
    for item in original_manifest['files']:
        if item['path'] not in allowed and portable_identity(final/item['path'],final)!=item:
            raise ValueError('Untested post-evaluator code/input mutation')
    files=[portable_identity(p,final) for p in sorted(final.rglob('*')) if p.is_file() and p.name!='artifact-manifest.json']
    purposes=dict(original_manifest['purposes'])
    for item in files:purposes.setdefault(item['path'],'full evaluator result/reconciliation evidence')
    atomic_json(final/'artifact-manifest.json',{**original_manifest,'files':files,'purposes':purposes})
    verify_manifest(final);archive(final,zipped)
    auditbase=Path(os.environ['LOCALAPPDATA'])/'LC0-BT3-Evaluator'/('final-audit-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    extracted=extract_verified(zipped,auditbase)
    if (extracted/'artifact-manifest.json').read_bytes()!=(final/'artifact-manifest.json').read_bytes():
        raise ValueError('Final archive/extraction identity differs')
    # Named final destinations are populated only after staged D8/CRC/extraction.
    staged_zip=portable_identity(zipped)
    deadline_check((sum(p['bytes'] for p in files)+3*staged_zip['bytes'])/(100*1024**2)+30)
    staging_root=staging.resolve()
    target_root=parent.resolve()
    if not final.resolve().is_relative_to(staging_root) or not zipped.resolve().is_relative_to(staging_root) or \
            not destination.resolve().is_relative_to(target_root) or not destination_zip.resolve().is_relative_to(target_root) or \
            staging_root.drive.lower()!=target_root.drive.lower():
        raise ValueError('Release publication paths are not verified same-volume staging/final destinations')
    parent.mkdir(parents=True,exist_ok=True)
    if destination.exists() or destination_zip.exists():raise FileExistsError('Final destination appeared during staged verification')
    final.rename(destination)
    zipped.rename(destination_zip)
    verify_manifest(destination)
    published_zip=portable_identity(destination_zip)
    if (destination/'artifact-manifest.json').read_bytes()!=(extracted/'artifact-manifest.json').read_bytes() or \
            any(published_zip[k]!=staged_zip[k] for k in ('bytes','sha256')):
        raise ValueError('Published release differs from verified staged artifacts')
    final,zipped=destination,destination_zip
    audit=run.out/'D8-verification.json'
    atomic_json(audit,{**record,'D8_completed_at':datetime.now().astimezone().isoformat(),'folder':str(final),'zip':portable_identity(zipped),
        'manifest':portable_identity(final/'artifact-manifest.json'),'extraction':str(extracted),
        'all_final_files_verified':len(files),'ZIP_CRC_and_single_top_folder':True})
    return {'outputs':[audit,final/'artifact-manifest.json',zipped]}
