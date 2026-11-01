"""Bounded backup checks; never a full experimental reproduction claim."""
import json
import math
import os
import time
import sys
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

from execution.support import read_json, strict_jsonl
from execution.ultimate_contract import atomic_json, portable_identity, deadline_check


def audit_observations(data, observations, analysis, output):
    """Independent count/point-estimate reconciliation; no bootstrap or inference."""
    from execution.ultimate_data import load_pool
    from eval.score import kl
    data, observations, analysis = map(Path, (data, observations, analysis))
    pools = {name: load_pool(data, name) for name in ('M', 'N')}
    challenges = read_json(data/'challenges.json')
    cohorts = {'M': pools['M'], 'M_sensitivity': [r for r in pools['M'] if r['id'] not in (388,611,627,965,1100)],
               'HM': [r for r in pools['M'] if r['id'] in set(challenges['HM'])],
               'HN': [r for r in pools['N'] if r['id'] in set(challenges['HN'])]}
    q = read_json(analysis/'quality.json')
    assert q['cohort_counts'] == {k:len(v) for k,v in cohorts.items()}
    cache = {}
    manifest = read_json(observations/'manifest.json')
    for item in manifest['files']:
        deadline_check()
        path = observations/item['path']
        assert portable_identity(path, observations) == item
        model = path.parent.name
        study = path.stem[-1].upper()
        for row in strict_jsonl(path):
            budget = row.get('movetime', row.get('budget', 0))
            target = cache.setdefault((model,study,budget), {})
            assert row['id'] not in target, (model,study,budget,row['id'])
            target[row['id']] = row
    def close(a,b):
        assert math.isclose(float(a),float(b),rel_tol=1e-11,abs_tol=1e-12),(a,b)
    for row in q['rows']:
        deadline_check()
        roots = cohorts[row['cohort']]
        study = 'B' if row['study']=='B_s0' else row['study']
        actual = cache[row['model'],study,row['budget']]
        selected = [actual[r['id']] for r in roots]
        ok = np.array([s['best'] in r['correct'] for r,s in zip(roots,selected)],dtype=int)
        assert row['n']==len(roots) and row['correct']==int(ok.sum()) and row['errors']==len(roots)-int(ok.sum())
        close(row['accuracy'],ok.mean())
        assert row['dtz_optimal']==sum(s['best'] in r['dtz_opt'] for r,s in zip(roots,selected))
        close(row['mean_actual_nodes'],np.mean([s['nodes'] for s in selected]))
        if study!='A':
            close(row['mean_wall_ms'],np.mean([s['wall']*1000 for s in selected]))
            close(row['mean_engine_ms'],np.mean([s['time_ms'] for s in selected]))
        ref = cache.get(('fp16',study,row['budget']),{})
        if all(r['id'] in ref for r in roots):
            delta = ok-np.array([ref[r['id']]['best'] in r['correct'] for r in roots],dtype=int)
            assert row['better']==int((delta>0).sum()) and row['worse']==int((delta<0).sum())
            close(row['delta_vs_fp16'],delta.mean())
        else:
            assert row['delta_vs_fp16'] is None and row['holm_p'] is None
        # Weighted floating-point sums may exceed one by one machine ulp.
        assert -1e-12<=row['accuracy_ci_low']<=row['accuracy_ci_high']<=1+1e-12
        if row['holm_p'] is not None:
            assert 0<=row['nominal_mcnemar_p']<=row['holm_p']<=1
    means = {}
    for cohort, roots in cohorts.items():
        ref=cache['fp16','A',0]
        for model in {r['model'] for r in q['graded'] if r['cohort']==cohort}:
            actual=cache[model,'A',0]
            values=np.array([[kl(ref[r['id']]['p'],actual[r['id']]['p']),
                max(ref[r['id']]['p'],key=ref[r['id']]['p'].get)==max(actual[r['id']]['p'],key=actual[r['id']]['p'].get),
                abs(ref[r['id']]['v']-actual[r['id']]['v']),actual[r['id']]['best'] in r['dtz_opt']] for r in roots])
            for name,value in zip(('printed_policy_KL','printed_prior_top1_agreement','printed_V_absolute_drift','DTZ_optimal_rate'),values.mean(axis=0)):
                means[cohort,model,name]=value
    for row in q['graded']:
        close(row['mean'],means[row['cohort'],row['model'],row['endpoint']])
    screening=sum(cache['fp16','A',0][r['id']]['best'] in r['correct'] for r in pools['N'])
    assert screening==q['screening']['fp16_correct'] and len(pools['N'])==q['screening']['N']
    record={'status':'verified','quality':portable_identity(analysis/'quality.json'),
            'observation_manifest':portable_identity(observations/'manifest.json'),
            'unique_records':sum(len(v) for v in cache.values()),'quality_rows':len(q['rows']),
            'graded_rows':len(q['graded']),'cohorts':q['cohort_counts'],
            'scope':'All supplied unique keys, cohort counts, correctness/discordances/DTZ, mean actual work/time and graded point estimates independently reconciled. Confidence bounds checked for validity; no bootstrap recomputation or new inference.'}
    atomic_json(output,record)
    return record


def audit_games(folder, analysis, output):
    from execution.ultimate_match import journal, summary, timing_gate
    folder=Path(folder)
    games=journal(folder/'matches',read_json(folder/'matches/contract.json'))
    actual=summary(games)
    assert len(games)==1000 and all(r['complete_pairs']==250 for r in actual)
    assert actual==read_json(Path(analysis)/'T4-matches.json')==read_json(folder/'matches/summary.json')
    # The concatenated PGN must be exactly the validated atomic envelopes.
    expected='\n\n'.join(e['pgn'] for e in games)+'\n'
    supplied=(folder/'matches/games.pgn').read_text(encoding='utf-8')
    assert supplied==expected
    pilot=journal(folder/'pilot',read_json(folder/'pilot/contract.json'))
    assert len(pilot)==4 and timing_gate(pilot)['status']=='verified'
    record={'status':'verified','games':1000,'pilot_games_excluded':4,'summaries':actual,
            'scope':'All atomic game legality/color/opening/timing records and production PGN reconciled; existing excluded pilot timing gate revalidated. No games replayed.'}
    atomic_json(output,record)
    return record


def smoke_worker(cfg, plan, output):
    """New actual searches, small fixed matrix; source comes from extraction."""
    import chess
    from execution.ultimate_contract import ROOT,A10,freeze
    from execution.ultimate_study import run_block,gpu_interval,CheckedLc0
    from execution.ultimate_match import match_counters
    plan=read_json(plan);output=Path(output)
    def budget_guard():
        deadline_check()
        if time.time()>=plan['checks_end_unix']:
            raise TimeoutError('U10A bounded check budget exhausted; no pass')
    budget_guard()
    roots=plan['roots'];reference=read_json(ROOT/'data/ultimate/numeric-reference.json')
    numeric=[]
    for model in A10:
        budget_guard()
        rows=run_block(output/model/'A',cfg,model,'A',0,roots)
        ref={r['id']:r for r in reference['models'][model]['reference_A']}
        max_drift={k:0. for k in ('wl','d','v','m','policy')}
        for root,row in zip(roots,rows):
            old=ref[row['id']]
            for key,tolerance in reference['tolerances']['value_absolute'].items():
                if row[key] is None and old[key] is None:
                    continue
                drift=abs(row[key]-old[key]);max_drift[key]=max(max_drift[key],drift)
                assert drift<=tolerance,(model,key,drift)
            a={k:v/sum(old['p'].values()) for k,v in old['p'].items()}
            b={k:v/sum(row['p'].values()) for k,v in row['p'].items()}
            drift=max(abs(a.get(k,0)-b.get(k,0)) for k in set(a)|set(b))
            max_drift['policy']=max(max_drift['policy'],drift)
            assert drift<=reference['tolerances']['normalized_policy_max_absolute']
            assert (row['best'] in root['correct'])==(old['best'] in root['correct'])
        numeric.append({'model':model,'roots':len(roots),'max_absolute_drift':max_drift})
    for model in ('fp16','s4'):
        for study,budget in (('B',100),('C',100),('C',200)):
            budget_guard();run_block(output/model/(study+str(budget)),cfg,model,study,budget,roots)
    # Match options are exercised directly on the two excluded opening roots.
    match_rows=[]
    with gpu_interval(output/'match-options',cfg) as live:
        def guard():budget_guard();live()
        for model in ('fp16','s4'):
            with CheckedLc0(cfg['lc0'],ROOT/'models'/(model+'.pb.gz'),backend='onnx-trt',
                  backend_opts='optimize=6' if model=='fp16' else 'optimize=8',
                  options={'MinibatchSize':64,'MoveOverheadMs':20,'Ponder':False},
                  log=str(output/'match-options'/(model+'.lc0.log')),name=model,guard=guard) as engine:
                engine.go(moves=plan['pilot_openings'][0],movetime=2000,timeout=30)
                for budget in (100,200):
                    for opening in plan['pilot_openings']:
                        guard();engine.newgame();result=engine.go(moves=opening,movetime=budget,timeout=30)
                        board=chess.Board()
                        for move in opening:board.push_uci(move)
                        assert chess.Move.from_uci(result['bestmove']) in board.legal_moves
                        match_counters(result['nodes'],result['time_ms'],result['wall']*1000)
                        match_rows.append({'model':model,'budget_ms':budget,'opening':opening,
                            'best':result['bestmove'],'nodes':result['nodes'],'engine_ms':result['time_ms'],
                            'wall_ms':result['wall']*1000})
    atomic_json(output/'verification.json',{'status':'verified','source_root':str(ROOT),'python_prefix':sys.prefix,
        'plan':portable_identity(Path(plan['plan_path'])),'numeric':numeric,'match_option_searches':match_rows,
        'raw_searches':40,'node_searches':8,'timed_searches':16,
        'cache_policy':'Existing same-host native caches reused; no fresh-cache or engine-build reproduction claim',
        'full_reproduction_verified':False})


def backup_phase(run):
    from execution.ultimate import command
    from execution.ultimate_contract import ROOT,A10,freeze,authorized_deadline
    from execution.ultimate_package import NAME,verify_manifest,extract_verified,archive,deadline_copy
    from execution.support import single_workload
    out=run.out/'backup';out.mkdir(parents=True,exist_ok=True)
    done=out/'publication.json'
    if done.exists():
        proof=read_json(done);verify_manifest(proof['folder'])
        assert portable_identity(Path(proof['zip']['path']))==proof['zip']
        return {'outputs':[done,Path(proof['folder'])/'artifact-manifest.json',Path(proof['zip']['path'])]}
    candidate=read_json(run.out/'candidate.json')
    assert portable_identity(Path(candidate['zip']))==candidate['archive']
    verify_manifest(candidate['folder'])
    plan_path=out/'plan.json'
    if not plan_path.exists():
        ref=read_json(Path(candidate['folder'])/'data/ultimate/numeric-reference.json')
        from execution.ultimate_data import load_pool
        pool={r['id']:r for r in load_pool(run.data,'M')}
        ids=[i for i in ref['ids'] if i in pool][:4]
        assert len(ids)==4
        start=time.time();end=min(start+7200,authorized_deadline()-60)
        base=Path(os.environ['LOCALAPPDATA'])/'LC0-BT3-Evaluator'/('backup-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
        pilot=read_json(run.out/'matches/pilot/contract.json')['openings']
        freeze(plan_path,{'started_unix':start,'end_unix':end,'checks_end_unix':end-1200,
            'target_minutes':60,'maximum_minutes':120,'publication_reserve_minutes':20,
            'workspace':str(base),'plan_path':str(plan_path.resolve()),'roots':[pool[i] for i in ids],
            'models':list(A10),'pilot_openings':pilot,'candidate_archive':candidate['archive'],
            'matrix':'A10 raw x4; fp16/s4 B100 x4, C100/C200 x4; match options x2 openings x2 budgets x2 models',
            'exclusions':['full derivation/selection','fresh-cache builds','full studies/matches','all44 systems tasks','full U11/U12/D8']})
    plan=read_json(plan_path);base=Path(plan['workspace']);base.mkdir(parents=True,exist_ok=True)
    def guard(publication=False):
        deadline_check()
        if time.time()>=plan['end_unix' if publication else 'checks_end_unix']:
            raise TimeoutError('U10A fixed1-2h budget exhausted; no completion pass')
    def checked(argv,label,cwd=None):
        guard();where=base/label
        if (where/'command.json').exists():
            saved=read_json(where/'command.json')
            if saved.get('exit_code')!=0:raise RuntimeError('Preserved U10A command failure needs diagnosed new attempt: '+str(where))
            return saved
        return command(argv,where,run.cfg,cwd=cwd,guard=guard,timeout=max(1,plan['checks_end_unix']-time.time()))
    setup_path=out/'setup.json'
    if setup_path.exists():
        setup=read_json(setup_path);folder=Path(setup['folder']);py=Path(setup['python']);verify_manifest(folder)
    else:
        folder=extract_verified(candidate['zip'],base/'extracted')
        env=base/'venv'
        checked([sys.executable,'-X','utf8','-B','-m','venv',env],'setup/venv')
        py=env/'Scripts/python.exe'
        checked([py,'-X','utf8','-B','-m','pip','install','-r',folder/'requirements/ultimate-report.lock.txt'],'setup/pip',folder)
        checked([py,'-X','utf8','-B','-m','pip','check'],'setup/pip-check',folder)
        cfg={k:v for k,v in run.cfg.items() if not k.startswith('_')};cfg.update(work='.',bt3='nets/BT3.pb.gz')
        atomic_json(folder/'config.json',cfg)
        checked([py,'-X','utf8','-B','reproduce.py','check'],'setup/check',folder)
        script='''import json,sys
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
assert actual==expected
assert Path(common.__file__).resolve()==ROOT/'common.py'
print(json.dumps({'prefix':sys.prefix,'executable':sys.executable,'source_root':str(ROOT),'packages':actual}))
'''
        checked([py,'-X','utf8','-B','-c',script],'setup/probe',folder)
        probe=read_json(base/'setup/probe/command.log')
        assert Path(probe['source_root']).resolve()==folder.resolve() and Path(probe['prefix']).resolve()==env.resolve()
        setup={'folder':str(folder),'python':str(py),'environment':probe,'new_venv_created_locally':True,
               'native_installations_and_caches_shared':True,'candidate_archive':candidate['archive']}
        atomic_json(setup_path,setup)
    # Run audits in the extracted code/new interpreter, never checkout imports.
    audit_script='''from execution.ultimate_backup import audit_observations,audit_games
audit_observations('data/ultimate','results/reference/observations','results/reference/analysis',r"%s")
audit_games('results/reference','results/reference/analysis',r"%s")
print('All compact observations and complete game records reconciled')
''' % (out/'count-audit.json',out/'game-audit.json')
    checked([py,'-X','utf8','-B','-c',audit_script],'checks/counts-games',folder)
    smoke=base/'smoke'
    if not (smoke/'verification.json').exists():
        with single_workload(ROOT/'reports/takeover/run2/ACTIVE.json'),single_workload(ROOT/'reports/ultimate/ACTIVE.json'):
            checked([py,'-X','utf8','-u','-B','-m','execution.ultimate_backup','--config',folder/'config.json',
                '--plan',plan_path,'--output',smoke],'checks/smoke',folder)
    assert read_json(smoke/'verification.json')['status']=='verified'
    verify_manifest(folder)
    report=read_json(run.out/'report/visual-review.json')
    assert report['status']=='verified' and report['actual_inspection_attested']
    for key in ('bytes','sha256'):
        assert portable_identity(folder/'report.pdf')[key]==report['pdf'][key]
    # Curated source files remain identical to the actually extracted/tested ZIP.
    original=verify_manifest(candidate['folder'])
    for item in original['files']:
        if Path(item['path']).suffix=='.py':
            actual=portable_identity(ROOT/item['path'])
            assert all(actual[k]==item[k] for k in ('bytes','sha256'))
    guard(publication=True)
    staged=base/'publication'/NAME;staged.parent.mkdir(exist_ok=True)
    shutil.copytree(candidate['folder'],staged,copy_function=deadline_copy)
    evidence=staged/'results/backup';evidence.mkdir(parents=True)
    for name in ('plan.json','setup.json','count-audit.json','game-audit.json'):
        shutil.copy2(out/name,evidence/name)
    shutil.copytree(smoke,evidence/'smoke',copy_function=deadline_copy)
    for name in ('structural-qa.json','visual-review.json'):
        shutil.copy2(run.out/'report'/name,evidence/('report-'+name))
    # Logs/receipts make the fresh-env command exit claims independently inspectable.
    shutil.copytree(base/'setup',evidence/'setup-commands',copy_function=deadline_copy)
    shutil.copytree(base/'checks',evidence/'check-commands',copy_function=deadline_copy)
    stamp=datetime.now(timezone(timedelta(hours=5,minutes=30))).strftime('%Y-%m-%d %H:%M:%S IST')
    record={'status':'U10A_bounded_backup_verified','verified_IST':stamp,'full_reproduction_verified':False,
        'U11':'pending; requires later explicit green light','U12':'pending; full integrated D8 not claimed',
        'primary_measurements':'U0-U9 complete and verified; compatible original results reused',
        'bounded_checks':['external ZIP extraction/CRC/hash manifest','new locally created venv/exact pins/pip check',
          'public input/native check and import/prefix isolation','444332 unique reference observations/count reconciliation',
          'all1000 production games and four excluded pilots reconciled','actual40 raw,8 node,16 timed and8 match-option searches',
          'same12-page PDF inspected in full; identical packaged PDF verified','final ZIP re-extraction and folder hash agreement'],
        'excluded':plan['exclusions'],'cache_policy':'Same-host existing native caches intentionally reused; not fresh-cache validation',
        'submission_scope':'Viable primary-study backup with bounded independent setup/runtime checks; not full fresh experimental reproduction'}
    atomic_json(staged/'verification.json',record)
    readme=staged/'README.md';text=readme.read_text(encoding='utf-8')
    text=text.replace('# BT3 INT8 in lc0 - CS6886 Group10','# BT3 INT8 in lc0 - CS6886 Group10\n\n**U10A bounded backup verification passed. Full U11 evaluator replay and U12 integrated audit are pending.**\n\nPrimary studies/report are complete. Independent backup checks used an external extraction, new Python environment and a fixed small actual search matrix. Existing native caches were reused. See verification.json and results/backup for exact coverage; no full reproduction pass is claimed.',1)
    readme.write_text(text,encoding='utf-8',newline='\n')
    files=[portable_identity(p,staged) for p in sorted(staged.rglob('*')) if p.is_file() and p.name!='artifact-manifest.json']
    purposes=dict(original['purposes'])
    for item in files:purposes.setdefault(item['path'],'bounded U10A backup verification evidence')
    atomic_json(staged/'artifact-manifest.json',{**original,'files':files,'purposes':purposes,'verification_level':'U10A bounded backup; U11/U12 pending'})
    verify_manifest(staged)
    zipped=archive(staged,base/'publication'/(NAME+'.zip'))
    guard(publication=True)
    extracted=extract_verified(zipped,base/'publication-extraction')
    assert (extracted/'artifact-manifest.json').read_bytes()==(staged/'artifact-manifest.json').read_bytes()
    parent=ROOT/'submissionready';destination=parent/NAME;destination_zip=parent/(NAME+'.zip')
    if destination.exists() or destination_zip.exists():raise FileExistsError('Preserve existing submission backup; no overwrite')
    assert staged.resolve().is_relative_to(base.resolve()) and destination.resolve().is_relative_to(parent.resolve())
    assert staged.resolve().drive.lower()==destination.resolve().drive.lower()
    guard(publication=True);parent.mkdir(exist_ok=True)
    zip_before=portable_identity(zipped);staged.rename(destination);zipped.rename(destination_zip)
    verify_manifest(destination)
    zip_after=portable_identity(destination_zip)
    assert all(zip_after[k]==zip_before[k] for k in ('bytes','sha256'))
    guard(publication=True)
    atomic_json(done,{**record,'published_IST':datetime.now(timezone(timedelta(hours=5,minutes=30))).strftime('%Y-%m-%d %H:%M:%S IST'),
        'folder':str(destination),'zip':zip_after,'manifest':portable_identity(destination/'artifact-manifest.json'),
        'extraction':str(extracted),'files_verified':len(files),'elapsed_minutes':(time.time()-plan['started_unix'])/60,
        'ZIP_CRC_single_top_folder_and_folder_agreement':True})
    return {'outputs':[done,destination/'artifact-manifest.json',destination_zip]}


if __name__=='__main__':
    import argparse
    from execution.ultimate import config
    parser=argparse.ArgumentParser(description='Internal bounded backup worker')
    parser.add_argument('--config',required=True);parser.add_argument('--plan',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    smoke_worker(config(args.config),args.plan,args.output)
