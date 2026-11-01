"""Ultimate phase implementations, sharing core pipeline code and contracts."""
from collections import Counter
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import urllib.request

import chess
import chess.pgn

from execution.support import fingerprint, identity, read_json, strict_jsonl
from execution.ultimate_contract import (ROOT, A10, B9, CONTRACT, SEED, freeze, portable_identity,
                                         validate_observation, hard_ids, deadline_check, atomic_json)


def U0(run):
    from execution.ultimate import check, log
    run.out.mkdir(parents=True, exist_ok=True)
    checked = check(run.cfg)
    # Preserve all original frozen inputs. Never refresh an expected hash to pass.
    public=(ROOT/'artifact-manifest.json').exists()
    if public:
        from execution.ultimate_package import verify_manifest
        source_manifest=verify_manifest(ROOT)
        preservation={'package_files_verified':len(source_manifest['files'])}
    else:
        source_manifest = read_json(ROOT / 'reports/takeover/run2/codex-continuation-v2/manifest.json')
        for item in source_manifest['sources']:
            if identity(item['path']) != item:
                raise ValueError('Frozen original Run2 input changed: ' + item['path'])
        if len(source_manifest['sources']) != 72:
            raise ValueError('Frozen source count differs')
        preservation={'frozen_Run2_sources_verified':72}
    freeze(run.out / 'protocol.json', CONTRACT)
    command = [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests', '-v']
    test_receipt=run.out/'U0-tests-receipt.json'
    tests=run.out/'U0-tests.log'
    if test_receipt.exists():
        saved=read_json(test_receipt);tests=ROOT/saved['log']['path']
        if saved['exit_code']!=0 or portable_identity(tests)!=saved['log']:
            raise ValueError('Completed offline test receipt changed')
    else:
        number=1
        while tests.exists():
            tests=run.out/f'U0-tests-attempt-{number:03d}.log';number+=1
        with tests.open('x', encoding='utf-8') as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        log('CPU tests command ' + json.dumps(command) + '; exit ' + str(result.returncode) + '; '+str(tests))
        if result.returncode:
            raise RuntimeError('Offline implementation tests failed')
        atomic_json(test_receipt,{'exit_code':0,'log':portable_identity(tests),'command':command})
    sources = [ROOT / 'reproduce.py', *sorted((ROOT / 'execution').glob('ultimate*.py')),
               ROOT / 'tests/test_ultimate.py']
    outputs = run.out / 'U0-verification.json'
    freeze(outputs, {'status': 'verified', **preservation,
           'new_sources': [portable_identity(p) for p in sources], 'check': checked,
           'tests': portable_identity(tests),
           'scientific_extensions': 'README.md / immutable protocol.json' if public else 'docs/ultimate_run.md U0 implementation note',
           'new_GPU_workloads': 0})
    if run.fresh:
        from execution.ultimate_replay import derive_models
        derivation=derive_models(run)
    else:
        derivation=None
    return {'outputs': [outputs, run.out/'protocol.json', tests,test_receipt,*([derivation] if derivation else [])],
            'summary': 'CPU implementation, contracts and full model/development derivation verified' if run.fresh else 'Offline implementation, contracts, preservation verified; no inference'}


def book(out):
    from execution.ultimate_data import validate_openings
    out = Path(out)
    source = out / 'opening-book.pgn'
    url = 'https://storage.lczero.org/files/opening_books/book-ply6-unifen-Q-0.0-0.25.pgn'
    if not source.exists():
        out.mkdir(parents=True, exist_ok=True)
        request=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0','Accept':'*/*',
                    'Referer':'https://github.com/jhorthos/lczero-training/wiki/Opening-Books'})
        with urllib.request.urlopen(request, timeout=120) as response:
            data = response.read()
        # Write the research input only after a complete successful download.
        source.write_bytes(data)
    openings = []
    with source.open(encoding='utf-8-sig') as stream:
        while True:
            game = chess.pgn.read_game(stream)
            if game is None:
                break
            if game.errors:
                raise ValueError('Opening source has illegal moves')
            openings.append([m.uci() for m in game.mainline_moves()])
    if len(openings) != 27772:
        raise ValueError('Opening source version/count differs')
    unique = validate_openings(openings)
    selected = random.Random(SEED).sample(unique, 252)
    validate_openings(selected, 252)
    path = out / 'book.json'
    freeze(path, {'seed': SEED, 'production': selected[:250], 'pilot': selected[250:],
                 'download_url': url, 'source': portable_identity(source), 'source_count': len(openings),
                 'unique_terminal_positions': len(unique),
                 'attribution': 'James Horsfall Thomas / jhorthos; Caissabase-derived balanced six-ply book',
                 'source_page': 'https://github.com/jhorthos/lczero-training/wiki/Opening-Books',
                 'source_page_version': '2020-10-19 as inspected 2026-10-30',
                 'selection': 'seeded sample without outcomes; 250 production plus 2 excluded pilot openings'})
    return path


def reuse_ledger(out):
    """Source-level provenance; rows stay in their preserved original files."""
    from execution.ultimate_data import original_roots
    roots = {r['id']: r for r in original_roots()}
    artifacts = {e['path'].replace('\\', '/'): e for e in read_json(ROOT/'artifacts/manifest.json')['files']}
    for model in A10:
        p = ROOT / 'models' / (model + '.pb.gz')
        expected = artifacts.get('models/' + p.name)
        if expected is None or identity(p)['sha256'] != expected['sha256']:
            raise ValueError('Canonical supplied model no longer matches preservation inventory: ' + model)
    sources = []
    for model in A10:
        folders = [(ROOT/'results'/model, ('A', 'B'))]
        if model == 's0':
            folders = [(ROOT/'results'/model, ('A',)), (ROOT/'results/s0_search'/model, ('B',))]
        if model in A10[:6]:
            folders.append((ROOT/'results/hard_lowbudget'/model, ('B',)))
        if model in ('fp16', 's4'):
            folders.append((ROOT/'results/studyc'/model, ('C',)))
        for folder, studies in folders:
            for study in studies:
                path = folder / ('study_' + study.lower() + '.jsonl')
                if not path.exists():
                    continue
                histories = strict_jsonl(folder/'meta.jsonl')
                if not histories and (folder/'meta.json').exists():
                    histories = [read_json(folder/'meta.json')]
                if not histories:
                    raise ValueError('No historical session provenance: ' + str(path))
                for meta in histories:
                    if meta['name'] != model or meta['backend'] != 'onnx-trt' or \
                       meta['backend_opts'] != ('optimize=6' if model == 'fp16' else 'optimize=8') or \
                       meta['minibatch'] != 256 or meta['threads'] != (0 if study == 'C' else 1) or \
                       meta.get('move_overhead', 20) != 20 or meta.get('warnings'):
                        raise ValueError('Historical settings incompatible: ' + str(path))
                rows = strict_jsonl(path)
                keys = set()
                counts = Counter()
                for row in rows:
                    if row['id'] not in roots:
                        raise ValueError('Unknown historical root ID')
                    budget = row['movetime' if study == 'C' else 'budget']
                    key = (row['id'], budget)
                    if key in keys:
                        raise ValueError('Duplicate historical result key')
                    keys.add(key)
                    validate_observation(row, roots[row['id']], study, budget)
                    if roots[row['id']]['split'] == 'test':
                        counts[str(budget)] += 1
                sources.append({'model': model, 'study': study, 'input': portable_identity(path),
                                'metadata': [portable_identity(folder/n) for n in ('meta.json', 'meta.jsonl') if (folder/n).exists()],
                                'test_counts': dict(counts), 'all_rows': len(rows),
                                'settings': histories,
                                'model_sha256': identity(ROOT/'models'/(model+'.pb.gz'))['sha256'],
                                'root_source': portable_identity(ROOT/'data/tbset.jsonl')})
    path = Path(out)/'reuse.json'
    freeze(path, {'sources': sources, 'protocol': fingerprint(CONTRACT),
                 'priority': 'main results before hard100 repeats; distinct observations never pooled',
                 'lineage': 'unchanged imported study implementation and preservation/model inventories; historical runtime C1/Run2 evidence'})
    return path


def U1(run):
    from execution.ultimate_data import freeze_main, generate_fresh
    old, main = freeze_main(run.data)
    reuse = reuse_ledger(run.data) if not run.fresh else None
    fresh = generate_fresh(run.data, run.cfg['syzygy'])
    if run.fresh:
        from execution.ultimate_data import load_pool
        canonical = {r['id']:r for r in load_pool(ROOT/'data/ultimate','N')}
        if {r['id']:r for r in fresh} != canonical:
            raise ValueError('Fresh regenerated N differs from frozen supplied research input')
        freeze(run.data/'challenges.json',read_json(ROOT/'data/ultimate/challenges.json'))
        (run.data/'opening-book.pgn').write_bytes((ROOT/'data/ultimate/opening-book.pgn').read_bytes())
    opening_path = book(run.data)
    verification = run.out/'U1-verification.json'
    freeze(verification, {'status': 'verified', 'M': len(main), 'N': len(fresh),
           'M_sensitivity': 6715, 'opening_production': 250, 'opening_pilot': 2,
           'N_no_legal_symmetry_overlap': True, 'N_transformed_labels_verified': True,
           'M_source': portable_identity(ROOT/'data/tbset.jsonl')})
    return {'outputs': [run.data/'M.json', run.data/'N.json', opening_path, verification,
                        *([reuse,run.data/'reuse-timed-provenance.json',run.data/'tablebases.json',
                           run.data/'generation-provenance.json'] if reuse else [])],
            'summary': 'M/N/book/reuse frozen and verified'}


def numeric_contract(run):
    prepared=ROOT/'data/ultimate/numeric-reference.json'
    if prepared.exists():
        contract=read_json(prepared)
        for name,spec in contract['models'].items():
            spec['fresh_weights']=str(ROOT/'models'/(name+'.pb.gz'))
        contract['tools']={k:run.cfg[k] for k in ('lc0','trtexec','nsys')}
        contract['idle_validation_path']=str(run.out/'U2/idle/work/idle.json')
        return contract
    old = read_json(ROOT/'reports/takeover/run3/reproduction-contract.json')
    ids = old['ids']
    models = {}
    for name in A10:
        a = {r['id']: r for r in strict_jsonl(ROOT/'results'/name/'study_a.jsonl')}
        b_path = ROOT/'results'/name/'study_b.jsonl' if name != 's0' else ROOT/'results/s0_search/s0/study_b.jsonl'
        b = {r['id']: r for r in strict_jsonl(b_path) if r['budget'] == 100}
        models[name] = {'fresh_weights': str(ROOT/'models'/(name+'.pb.gz')),
                        'weights_sha256': identity(ROOT/'models'/(name+'.pb.gz'))['sha256'],
                        'reference_A': [a[i] for i in ids], 'reference_B100': [b[i] for i in ids]}
    return {'ids': ids, 'roots': old['roots'], 'models': models, 'tolerances': old['tolerances'],
            'tools': {k: run.cfg[k] for k in ('lc0', 'trtexec', 'nsys')},
            'benchmark_reference': old['benchmark_reference'],
            'idle_validation_path': str(run.out/'U2/idle/work/idle.json')}


def native_worker(task, cfg):
    from execution.ultimate_study import gpu_interval
    from execution.ultimate import command
    folder = Path(task['folder'])
    with gpu_interval(folder, cfg,workload=task['kind']!='idle') as guard:
        # New condition-backed legacy leaf session, within the user's fixed window.
        # Historical twelve-hour guards and their sources remain unchanged.
        observed = read_json(folder/'preflight.json')
        session = folder/'session.json'
        atomic_json(session, {'root': str(ROOT), 'created_unix': observed['observed_unix'],
                    'exclusive_authorized': True, 'work_saved': True, 'other_work_paused': True,
                    'ac_power_confirmed': observed['snapshot']['ac_line'] == 1,
                    'authorized_until_unix': (lambda t:t if __import__('math').isfinite(t) else None)(
                        __import__('execution.ultimate_contract', fromlist=['authorized_deadline']).authorized_deadline()),
                    'condition_record': portable_identity(folder/'preflight.json')})
        if task['kind'] == 'idle':
            from execution.run4_checks import idle_telemetry
            idle_telemetry(folder/'work/idle.json')
        else:
            argv = [str(x).replace('{session}', str(session)).replace('{work}', str(folder/'work'))
                    .replace('{folder}', str(folder)) for x in task['argv']]
            command(argv, folder/'child', cfg, guard=guard)
        guard()
        for relative in task['required']:
            p = folder/relative
            if not p.is_file() or p.stat().st_size == 0:
                raise ValueError('Native workload output missing: ' + str(p))


def native(run, name, kind, argv, required):
    from execution.ultimate import command
    folder = run.out/'U2'/name
    task = {'folder': str(folder), 'kind': kind, 'argv': argv, 'required': required,
            'protocol': fingerprint(CONTRACT)}
    freeze(folder/'task.json', task)
    if (folder/'receipt.json').exists():
        receipt = read_json(folder/'receipt.json')
        if receipt['status'] != 'verified':
            raise ValueError('Native workload previously failed')
        for p in required:
            if not (folder/p).is_file():
                raise ValueError('Completed native output missing')
        return folder
    command([sys.executable, '-B', ROOT/'reproduce.py', '_native', '--config', run.cfg['_path'],
             '--task', folder/'task.json'], folder/'launch', run.cfg)
    return folder


def U2(run):
    independent=run.out/'U2/independent-environment.json'
    if independent.exists():
        record=read_json(independent)
        if record['status']!='verified':
            raise ValueError('Independent environment gate incomplete')
        for item in record['outputs']:
            if portable_identity(item['path'])!=item:
                raise ValueError('Independent environment evidence changed')
        return {'outputs':[independent,*[Path(i['path']) for i in record['outputs']]],
                'summary':'Independent locally-created environment, inventoried current-source overlay and all A10/systems gates verified; earlier working-env observations preserved'}
    if (ROOT/'.git').exists():
        raise RuntimeError('Working-checkout U2 requires the independently created environment gate and its verified source/environment record')
    from execution.ultimate import command
    from execution.run4_checks import check_studies, check_speed
    contract = numeric_contract(run)
    freeze(run.out/'U2/numeric-contract.json', contract)
    native(run, 'idle', 'idle', [], ['work/idle.json'])
    roots = contract['roots']
    study_out = run.out/'U2/numeric'
    for name in A10:
        for study, budget in (('A',0), ('B',100)):
            folder = study_out/name/study
            task = {'folder': str(folder), 'model': name, 'study': study, 'budget': budget, 'roots': roots}
            freeze(folder/'task.json', task)
            command([sys.executable, '-B', ROOT/'reproduce.py', '_block', '--config', run.cfg['_path'],
                     '--task', folder/'task.json'], folder/'launch', run.cfg)
            output = read_json(folder/'output.json')
            (study_out/name/('study_'+study.lower()+'.jsonl')).write_text(
                ''.join(json.dumps(r)+'\n' for r in output['rows']), encoding='utf-8')
        atomic_json(study_out/name/'meta.json', {'warnings': []})
    check_studies(contract, study_out, run.out/'U2/numeric-verification.json')
    tools = run.cfg
    for name in ('fp16','s4','s1sp'):
        opts = 'optimize=6' if name == 'fp16' else 'optimize=8'
        model = str(ROOT/'models'/(name+'.pb.gz'))
        argv = [sys.executable,'-B','-m','execution.benchmark','--name',name,'--lc0',tools['lc0'],
                '--weights',model,'--backend-opts',opts,'--session','{session}','--out','{work}',
                '--positions','10','--movetime','10000','--threads','0','--minibatch','64',
                '--batches','200','--conditioning-seconds','30']
        native(run,'bench-'+name,'benchmark',argv,['work/result.json'])
        # Review the fresh trial before proceeding; no repeat to seek a pass.
        actual = read_json(run.out/'U2'/('bench-'+name)/'work/result.json')
        ref = contract['benchmark_reference'][name]
        ratios = [actual['search']['nps']/ref['nps'],
                  next(r['ms'] for r in actual['network'] if r['batch']==64)/ref['batch64_latency_ms']]
        if any(not .8 <= x <= 1.2 for x in ratios):
            raise RuntimeError('Fresh speed/latency outside frozen review band: '+name+' '+repr(ratios))
    for name in ('fp16','s4','s1sp'):
        argv = [tools['nsys'],'profile','--trace=cuda,nvtx','--sample=none','-o','{folder}/trace',
                sys.executable,'-B','-m','execution.gpu_job','capture','--lc0',tools['lc0'],
                '--weights',str(ROOT/'models'/(name+'.pb.gz')),'--backend-opts',
                'optimize=6' if name=='fp16' else 'optimize=8','--session','{session}','--out','{work}']
        native(run,'capture-'+name,'capture',argv,['work/result.json','trace.nsys-rep'])
    argv = [sys.executable,'-B','-m','execution.gpu_job','profile','--onnx',str(ROOT/'models/s1sp.onnx'),
            '--trtexec',tools['trtexec'],'--int8','--large-only','--sparsity','--session','{session}','--out','{work}']
    native(run,'sparse','profile',argv,['work/result.json','work/build_b256_layers.json'])
    # Reuse existing strict systems comparison with a small path adapter.
    import tempfile
    # Adapter folders contain only symlinks/junction-free copied compact JSON; traces stay original.
    # check_speed expects model subfolders. Build this layout directly in a new meaningful view.
    view = run.out/'U2/systems'
    for name in ('fp16','s4','s1sp'):
        p=view/'benchmarks'/name
        p.mkdir(parents=True,exist_ok=True)
        (p/'result.json').write_bytes((run.out/'U2'/('bench-'+name)/'work/result.json').read_bytes())
        # Trace path adapter via hardlink avoids copying a large measurement.
        cap=view/'captures'/name
        cap.mkdir(parents=True,exist_ok=True)
        dest=cap/'trace.nsys-rep'
        if not dest.exists():
            os.link(run.out/'U2'/('capture-'+name)/'trace.nsys-rep',dest)
    speed_contract = {**contract, 'models': {n: contract['models'][n] for n in ('fp16','s4','s1sp')}}
    check_speed(speed_contract,view/'benchmarks',run.out/'U2/sparse/work',view/'captures',run.out/'U2/systems-verification.json')
    return {'outputs': [run.out/'U2/numeric-verification.json', run.out/'U2/systems-verification.json',
                        run.out/'U2/idle/work/idle.json'], 'summary': 'A10 numerical and fresh bounded systems gates verified'}


def existing_rows(run, model, study, budget):
    result = {}
    public_reference=(ROOT/'artifact-manifest.json').exists() and not run.fresh
    if public_reference:
        path=ROOT/'results/reference/observations/studies'/model/('study_'+study.lower()+'.jsonl')
        if path.exists():
            for row in strict_jsonl(path):
                if row['movetime' if study=='C' else 'budget']==budget:
                    if row['id'] in result:raise ValueError('Duplicate compact reference key')
                    result[row['id']]=row
    elif not run.fresh:
        ledger = read_json(run.data/'reuse.json')
        for source in ledger['sources']:
            if source['model'] != model or source['study'] != study:
                continue
            from execution.ultimate_study import input_identity
            if input_identity(ROOT/'models'/(model+'.pb.gz'))['sha256']!=source['model_sha256']:
                raise ValueError('Canonical model changed since compatible-result freeze')
            path = ROOT/source['input']['path']
            if portable_identity(path) != source['input']:
                raise ValueError('Historical result changed')
            for row in strict_jsonl(path):
                if row['movetime' if study == 'C' else 'budget'] == budget:
                    result.setdefault(row['id'], row)
    # Completed blocks are exact contracts; their immutable output identities live in receipts.
    for path in sorted((run.out/'studies'/model/(study+str(budget))).glob('*/output.json')):
        from execution.ultimate_study import block_contract
        contract = read_json(path.parent/'contract.json')
        task = read_json(path.parent/'task.json')
        if contract != block_contract(run.cfg,model,study,budget,task['roots']):
            raise ValueError('Completed block model/input/runtime/source contract changed')
        if contract['model'] != model or contract['study'] != study or contract['budget'] != budget:
            raise ValueError('Scientific block stored under wrong key')
        receipt = read_json(path.parent/'receipt.json')
        if receipt['status'] != 'verified' or receipt['output'] != portable_identity(path):
            raise ValueError('Scientific block receipt corrupt')
        output = read_json(path)
        if output['contract'] != fingerprint(contract) or output['status'] != 'verified':
            raise ValueError('Scientific block identity/status corrupt')
        for row in output['rows']:
            if row['id'] in result:
                raise ValueError('Fresh block duplicated an existing compatible observation')
            result[row['id']] = row
    return result


def collect(run, roots, models, study, budgets):
    from execution.ultimate import command
    from execution.ultimate_study import block_contract, read_block
    for model in models:
        for budget in budgets:
            done = existing_rows(run, model, study, budget)
            for root in roots:
                if root['id'] in done:
                    validate_observation(done[root['id']], root, study, budget)
            todo = [r for r in roots if r['id'] not in done]
            for start in range(0, len(todo), 512):
                block = todo[start:start+512]
                if study=='C':
                    estimate=len(block)*budget/1000*1.25+30
                elif study=='B':
                    measured=[r['wall'] for r in done.values() if r.get('wall')]
                    rate=sum(measured)/len(measured) if measured else budget/2000
                    estimate=len(block)*rate*1.5+30
                else:estimate=45
                deadline_check(estimate)
                folder = run.out/'studies'/model/(study+str(budget))/fingerprint([r['id'] for r in block])[:16]
                task = {'folder': str(folder), 'model': model, 'study': study, 'budget': budget, 'roots': block}
                if (folder/'launch/command.json').exists():
                    if not run.recovery:
                        raise RuntimeError('Interrupted leaf launch requires diagnosed recovery')
                    original=folder
                    number=1
                    while folder.exists():
                        folder=original.with_name(original.name+f'-r{number:03d}')
                        number+=1
                    task['folder']=str(folder)
                freeze(folder/'task.json', task)
                command([sys.executable, '-B', ROOT/'reproduce.py', '_block', '--config', run.cfg['_path'],
                         '--task', folder/'task.json'], folder/'launch', run.cfg)
                contract = block_contract(run.cfg, model, study, budget, block)
                verified = read_block(folder, contract, block)
                if verified is None:
                    raise ValueError('Successful child has missing output')
                prior=sum(r['id'] in done for r in roots)
                print(f'COMPLETE {model} {study}{budget}: {prior+start+len(block)}/{len(roots)}', flush=True)
            completed = existing_rows(run, model, study, budget)
            if not {r['id'] for r in roots} <= set(completed):
                raise ValueError('Incomplete required keys')


def phase_coverage(run, phase, roots, models, study, budgets):
    counts = []
    for model in models:
        for budget in budgets:
            data = existing_rows(run, model, study, budget)
            for root in roots:
                validate_observation(data[root['id']], root, study, budget)
            counts.append({'model': model, 'study': study, 'budget': budget, 'n': len(roots)})
    path = run.out/(phase+'-coverage-'+study+'.json')
    freeze(path, {'status': 'verified', 'root_ids': fingerprint([r['id'] for r in roots]), 'coverage': counts})
    return path


def U3(run):
    from execution.ultimate_data import load_pool
    main, fresh = load_pool(run.data, 'M'), load_pool(run.data, 'N')
    # Freeze baseline challenge IDs before other new challenge outcomes are collected.
    collect(run, main, ('fp16',), 'A', [0])
    collect(run, fresh, ('fp16',), 'A', [0])
    challenges = {}
    raw = existing_rows(run, 'fp16', 'A', 0)
    for name, roots in (('HM', main), ('HN', fresh)):
        ids = hard_ids(roots, {r['id']: raw[r['id']] for r in roots})
        challenges[name] = ids
    if not run.fresh:
        old43 = {int(i) for i in (ROOT/'reports/hard/ids.txt').read_text().split()}
        if not old43 <= set(challenges['HM']):
            raise ValueError('Historical43 not retained in HM')
        freeze(run.data/'challenges.json', {'protocol': fingerprint(CONTRACT), **challenges,
               'FP16_screening': fingerprint({str(r['id']): raw[r['id']] for r in main+fresh})})
    else:
        frozen = read_json(ROOT/'data/ultimate/challenges.json')
        atomic_json(run.out/'screening-audit.json', {'frozen': {k: frozen[k] for k in ('HM','HN')}, 'fresh': challenges,
                    'scope': 'fresh screening descriptive; original challenges remain frozen'})
        challenges = frozen
    collect(run, main, A10[1:], 'A', [0])
    hard_n = [r for r in fresh if r['id'] in set(challenges['HN'])]
    collect(run, hard_n, A10[1:], 'A', [0])
    a = phase_coverage(run, 'U3-M', main, A10, 'A', [0])
    b = phase_coverage(run, 'U3-HN', hard_n, A10, 'A', [0])
    return {'outputs': [a, b, run.data/'challenges.json'], 'summary': {k: len(challenges[k]) for k in ('HM','HN')}}


def U4(run):
    from execution.ultimate_data import load_pool
    roots = load_pool(run.data, 'M')
    collect(run, roots, B9, 'B', [100,400,800,1600])
    collect(run, roots, ('s0',), 'B', [100,400])
    return {'outputs': [phase_coverage(run, 'U4-main', roots, B9, 'B', [100,400,800,1600]),
                        phase_coverage(run, 'U4-s0', roots, ('s0',), 'B', [100,400])]}


def U5(run):
    from execution.ultimate_data import load_pool
    challenges = read_json(run.data/'challenges.json')
    outputs = []
    for pool, hard in (('M', 'HM'), ('N', 'HN')):
        roots = [r for r in load_pool(run.data, pool) if r['id'] in set(challenges[hard])]
        collect(run, roots, B9, 'B', [10,25,50,100,200])
        outputs.append(phase_coverage(run, 'U5-'+hard, roots, B9, 'B', [10,25,50,100,200]))
        if hard == 'HN':
            collect(run, roots, ('s0',), 'B', [100,400])
            outputs.append(phase_coverage(run, 'U5-s0-HN', roots, ('s0',), 'B', [100,400]))
    return {'outputs': outputs}


def U6(run):
    from execution.ultimate_data import load_pool
    challenges = read_json(run.data/'challenges.json')
    outputs = []
    for pool in ('M', 'N'):
        roots = load_pool(run.data, pool)
        if pool == 'N':
            roots = [r for r in roots if r['id'] in set(challenges['HN'])]
        collect(run, roots, ('fp16','s4'), 'C', [100,200,400,800])
        outputs.append(phase_coverage(run, 'U6-'+pool, roots, ('fp16','s4'), 'C', [100,200,400,800]))
    return {'outputs': outputs}


def match_worker(task, cfg):
    from execution.ultimate_study import gpu_interval, CheckedLc0
    from execution.ultimate_match import play, journal, make_envelope
    folder = Path(task['folder'])
    output = Path(task['journal'])
    contract = read_json(task['contract'])
    done = {(e['record']['budget_ms'],e['record']['pair'],e['record']['color'])
            for e in journal(output, contract)}
    remaining = [tuple(k) for k in task['keys'] if tuple(k) not in done]
    if not remaining:
        return
    with gpu_interval(folder,cfg) as guard:
        engines = []
        try:
            for name in ('fp16','s4'):
                e = CheckedLc0(cfg['lc0'],ROOT/'models'/(name+'.pb.gz'),backend='onnx-trt',
                        backend_opts='optimize=6' if name=='fp16' else 'optimize=8',
                        options={'MinibatchSize':64,'MoveOverheadMs':20,'Ponder':False},
                        log=str(folder/(name+'.lc0.log')),name=name,guard=guard)
                engines.append(e)
                e.go(fen=chess.STARTING_FEN,movetime=2000,timeout=30)
            for budget,pair,color in remaining:
                deadline_check(2*400*budget/1000+30)
                white,black = engines if color==0 else engines[::-1]
                before = time.perf_counter()
                board,result,reason,stats = play(white,black,contract['openings'][pair],budget,cfg['syzygy'],guard)
                envelope = make_envelope(board,result,reason,stats,budget,pair,color,
                                          time.perf_counter()-before,contract)
                path = output/f'game-{budget}-{pair:03d}-{color}.json'
                if path.exists():
                    raise FileExistsError('Completed game immutable')
                atomic_json(path,envelope)
                print(f'GAME {budget} pair{pair} color{color}: {result} ({reason})',flush=True)
        finally:
            for e in engines:
                e.quit()


def U7(run):
    from execution.ultimate import command
    from execution.ultimate_match import assignments,journal,timing_gate,summary
    from execution.ultimate_study import block_contract
    book_data = read_json(run.data/'book.json')
    base = {'protocol':fingerprint(CONTRACT),'settings':CONTRACT['match'],
            'engine_contracts':[block_contract(run.cfg,n,'C',100,[]) for n in ('fp16','s4')],
            'rules_source':portable_identity(ROOT/'eval/match.py'),
            'extension_source':portable_identity(ROOT/'execution/ultimate_match.py'),
            'book':portable_identity(run.data/'book.json')}
    for kind,openings in (('pilot',book_data['pilot']),('production',book_data['production'])):
        out = run.out/'matches'/kind
        contract = freeze(out/'contract.json',{**base,'openings':openings,'kind':kind})
        if kind=='pilot':
            batches = [[(100,0,0),(100,0,1),(200,1,1),(200,1,0)]]
        else:
            keys = list(assignments())
            batches = [keys[i:i+20] for i in range(0,len(keys),20)]
        for number,keys in enumerate(batches):
            done = {(e['record']['budget_ms'],e['record']['pair'],e['record']['color']) for e in journal(out,contract)}
            if set(map(tuple,keys)) <= done:
                continue
            folder = out/f'batch-{number:03d}'
            if (folder/'launch/command.json').exists():
                if not run.recovery:
                    raise RuntimeError('Interrupted match batch needs diagnosed recovery')
                original=folder;attempt=1
                while folder.exists():
                    folder=original.with_name(original.name+f'-r{attempt:03d}');attempt+=1
            task = {'folder':str(folder),'journal':str(out),'contract':str(out/'contract.json'),'keys':keys}
            freeze(folder/'task.json',task)
            command([sys.executable,'-B',ROOT/'reproduce.py','_match','--config',run.cfg['_path'],
                     '--task',folder/'task.json'],folder/'launch',run.cfg)
        envelopes = journal(out,contract)
        if len(envelopes)!=(4 if kind=='pilot' else 1000):
            raise ValueError('Incomplete match game coverage')
        if kind=='pilot':
            gate = timing_gate(envelopes)
            atomic_json(out/'timing-gate.json',gate)
            if gate['status']!='verified':
                raise RuntimeError('Pilot timing exceeds frozen engineering limits; review before production')
        atomic_json(out/'summary.json',summary(envelopes))
        (out/'games.pgn').write_text('\n\n'.join(e['pgn'] for e in envelopes)+'\n',encoding='utf-8')
    if any(r['complete_pairs']!=250 for r in summary(envelopes)):
        raise ValueError('Production pair coverage incomplete')
    return {'outputs':[run.out/'matches/pilot/timing-gate.json',run.out/'matches/production/summary.json',
                        run.out/'matches/production/games.pgn']}


def U8(run):
    if run.fresh:
        from execution.ultimate_replay import full_systems,f12_diagnostic
        full_systems(run)
        f12_diagnostic(run)
    from execution.ultimate_finish import analyze_phase
    return analyze_phase(run)


def U9(run):
    from analysis.ultimate_report import report_phase
    return report_phase(run)


def U10(run):
    from execution.ultimate_package import candidate_phase
    return candidate_phase(run)


def U11(run):
    from execution.ultimate_package import evaluator_phase
    return evaluator_phase(run)


def U10A(run):
    from execution.ultimate_backup import backup_phase
    return backup_phase(run)


def U12(run):
    from execution.ultimate_package import release_phase
    return release_phase(run)
