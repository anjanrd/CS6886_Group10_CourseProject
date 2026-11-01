"""Fresh model derivation and complete systems replay; reference results never resume jobs."""
import gc
import json
from pathlib import Path
import sys

import numpy as np

from execution.support import read_json
from execution.ultimate_contract import ROOT,A10,CONTRACT,atomic_json,freeze,portable_identity,deadline_check


def prepare_runtime(cfg,output):
    """Pinned lc0 uses BinaryDirectory/trt_cache; isolate by copying native bytes."""
    import shutil
    folder=Path(output)/'runtime/lc0';record=Path(output)/'runtime/verification.json'
    original=Path(cfg['lc0']).resolve()
    sources=[original,*sorted(original.parent.glob('*.dll'))]
    expected=[portable_identity(p) for p in sources]
    if record.exists():
        proof=read_json(record)
        if proof['inputs']!=expected:raise ValueError('Declared native installation changed')
        for item in proof['outputs']:
            if portable_identity(ROOT/item['path'])!=item:raise ValueError('Isolated native runtime changed')
    else:
        folder.mkdir(parents=True,exist_ok=True)
        if (folder/'trt_cache').exists():raise ValueError('Unrecorded evaluator cache is not fresh')
        for p in sources:
            target=folder/p.name
            if target.exists():
                actual=portable_identity(target)
                if any(actual[k]!=portable_identity(p)[k] for k in ('bytes','sha256')):
                    raise ValueError('Interrupted native copy differs')
            else:shutil.copy2(p,target)
        outputs=[portable_identity(folder/p.name) for p in sources]
        if any(any(a[k]!=b[k] for k in ('bytes','sha256')) for a,b in zip(expected,outputs)):
            raise ValueError('Native runtime copy identity differs')
        atomic_json(record,{'status':'verified','inputs':expected,'outputs':outputs,
            'cache_started_empty':True,'no_previous_engine_or_timing_cache_copied':True,
            'source':'lc0 v0.33.0-rc0 network_onnx.cc uses BinaryDirectory()/trt_cache; byte-identical isolated executable/DLLs',
            'shared_native_prerequisites':['trtexec','Nsight','driver','Syzygy'],
            'driver_caches':'May persist; no claim of a cold driver cache'})
    adjusted={k:v for k,v in cfg.items() if not k.startswith('_')}
    adjusted['lc0']=str(folder/original.name)
    path=Path(output)/'runtime/config.json';freeze(path,adjusted)
    adjusted['_path']=str(path)
    return adjusted


def checked_fresh_command(argv,folder,cfg):
    from execution.ultimate import command
    folder=Path(folder)
    previous=folder/'command.json'
    if previous.exists():
        record=read_json(previous)
        if record.get('exit_code')!=0:
            raise RuntimeError('Fresh step needs diagnosed recovery: '+str(folder))
        return record
    return command(argv,folder,cfg)


def derive_models(run):
    """Recompute export, frozen dev grid and every retained model into new paths."""
    from execution.ultimate_finish import copy_exact
    import onnx
    from onnx import numpy_helper
    from common import ort_session,INPUT
    folder=run.out/'derivation'
    state=folder/'verification.json'
    if state.exists():
        verified=read_json(state)
        for item in verified['outputs']:
            if portable_identity(ROOT/item['path'])!=item:
                raise ValueError('Fresh derivation output changed')
        return state
    for name in ('tbset.jsonl','calib.npz','dev.npz'):
        copy_exact(ROOT/'data'/name,folder/'data'/name)
    cfg={k:v for k,v in run.cfg.items() if not k.startswith('_')}
    cfg.update(work=str(folder.resolve()),bt3=str(ROOT/'nets/BT3.pb.gz'),best='s4',
               recipe={'s3_method':'percentile','s3_percentile':99.999,'s4_alpha':.7})
    freeze(folder/'config.json',cfg)
    inputs=[portable_identity(ROOT/p) for p in ('nets/BT3.pb.gz','data/calib.npz','data/dev.npz','data/tbset.jsonl')]
    freeze(folder/'contract.json',{'protocol':CONTRACT,'inputs':inputs,
            'sources':[portable_identity(ROOT/p) for p in ('ladder.py','quant/quantize.py','quant/sparsegpt24.py',
                        'quant/prune24.py','quant/wrap.py','quant/stitch_swap.py','eval/quick_compare.py')],
            'procedure':'Original export, all frozen development-grid candidates, frozen recipe build. No reselection from new tests.'})
    for step,extra in (('export',[]),('select',['--alpha']),('build',[])):
        checked_fresh_command([sys.executable,'-X','utf8','-B',ROOT/'ladder.py',step,
                    '--config',folder/'config.json',*extra],folder/'commands'/step,run.cfg)
    planes=np.load(ROOT/'data/calib.npz')['planes'][:32]
    comparisons=[]
    for name in A10:
        deadline_check()
        stem='bt3_fp32' if name=='fp16' else name
        supplied=onnx.load(ROOT/'models'/(stem+'.onnx'))
        rebuilt=onnx.load(folder/'models'/(stem+'.onnx'))
        onnx.checker.check_model(rebuilt)
        def nodes(model):
            output=[]
            for node in model.graph.node:
                node.doc_string=''
                output.append(node.SerializeToString())
            return output
        if nodes(supplied)!=nodes(rebuilt):
            raise ValueError('Rebuilt computation graph differs: '+name)
        a={t.name:numpy_helper.to_array(t) for t in supplied.graph.initializer}
        b={t.name:numpy_helper.to_array(t) for t in rebuilt.graph.initializer}
        if set(a)!=set(b) or any(a[k].dtype!=b[k].dtype or not np.array_equal(a[k],b[k]) for k in a):
            raise ValueError('Rebuilt model tensors differ: '+name)
        left=ort_session(supplied,cpu=True,threads=4)
        right=ort_session(rebuilt,cpu=True,threads=4)
        x,y=left.run(None,{INPUT:planes}),right.run(None,{INPUT:planes})
        errors=[]
        for first,second in zip(x,y):
            if first.shape!=second.shape or not np.isfinite(first).all() or not np.isfinite(second).all() or \
                    not np.allclose(first,second,rtol=1e-5,atol=1e-6):
                raise ValueError('Rebuilt CPU outputs differ: '+name)
            errors.append(float(np.max(np.abs(first-second))))
        comparisons.append({'model':name,'same_graph_and_tensors':True,'positions':32,
                            'CPU_output_max_absolute_differences':errors,
                            'canonical_onnx':portable_identity(ROOT/'models'/(stem+'.onnx')),
                            'rebuilt_onnx':portable_identity(folder/'models'/(stem+'.onnx')),
                            'rebuilt_wrapper':portable_identity(folder/'models'/(name+'.pb.gz'))})
        del supplied,rebuilt,a,b,left,right,x,y
        gc.collect()
    for item in inputs:
        if portable_identity(ROOT/item['path'])!=item:
            raise ValueError('Research input changed during derivation')
    outputs=[p for p in (folder/'models').glob('*') if p.is_file()]
    outputs += [folder/'reports/select_s3_dev.txt',folder/'reports/select_s4_dev.txt',folder/'reports/quick_compare_dev.txt']
    atomic_json(state,{'status':'verified','inputs':inputs,'comparisons':comparisons,
                'outputs':[portable_identity(p) for p in outputs],
                'selection':'Fresh grid evidence retained; frozen percentile99.999/alpha0.7/main s4 unchanged',
                'inference_inputs':'Canonical supplied files, validated against rebuilt identical computation/tensors; gzip metadata need not match'})
    return state


def full_systems(run):
    """All reported repeated benchmarks, captures, profiles, ceilings and batch sweep."""
    from execution.ultimate import command
    from execution.run_jobs import validate_outputs
    from execution.ultimate_phases import native_worker
    manifest=run.out/'systems/verification.json'
    if manifest.exists():
        saved=read_json(manifest)
        for receipt in saved['jobs'].values():
            for item in [receipt['receipt'],*receipt['outputs']]:
                # Legacy validate_outputs uses absolute provenance paths.
                actual=portable_identity(ROOT/item['path'])
                if any(actual[k]!=item[k] for k in ('bytes','sha256')):
                    raise ValueError('Completed fresh systems evidence changed')
        item=saved['batch_sweep']
        if portable_identity(ROOT/item['path'])!=item:raise ValueError('Fresh batch evidence changed')
        return manifest
    tasks=read_json(ROOT/'data/ultimate/systems-tasks.json')['jobs']
    if len(tasks)!=44:
        raise ValueError('Full systems replay requires44 declared native jobs')
    receipts={}
    for spec in tasks:
        folder=run.out/'systems'/spec['id']/'attempt-001'
        argv=[]
        for value in spec['command']:
            for token,replacement in {'{python}':sys.executable,'{lc0}':run.cfg['lc0'],
                    '{trtexec}':run.cfg['trtexec'],'{nsys}':run.cfg['nsys'],
                    '{models}':str(ROOT/'models'),'{attempt}':str(folder)}.items():
                value=value.replace(token,replacement)
            argv.append(value)
        task={'folder':str(folder),'kind':spec['kind'],'argv':argv,'required':spec['outputs'],
              'protocol':CONTRACT}
        freeze(folder/'task.json',task)
        receipt_path=folder/'receipt.json'
        if receipt_path.exists():
            receipt=read_json(receipt_path)
            if receipt.get('status')!='verified':
                raise RuntimeError('Failed systems attempt requires diagnosed recovery')
        else:
            command([sys.executable,'-B',ROOT/'reproduce.py','_native','--config',run.cfg['_path'],
                    '--task',folder/'task.json'],folder/'launch',run.cfg)
            receipt=read_json(receipt_path)
        outputs=validate_outputs(spec,folder)
        receipts[spec['id']]={'attempt':str(folder),'status':'verified','outputs':outputs,
                              'receipt':portable_identity(receipt_path),'session':'fresh-evaluator'}
    # The retained four-cap batch sweep is a separate workload from NN batching.
    folder=run.out/'systems/batch-sweep/attempt-001'
    argv=[sys.executable,'-B',ROOT/'hw/batch_sweep.py','--lc0',run.cfg['lc0'],
          '--sizes','32,64,128,256','--model','fp16='+str(ROOT/'models/fp16.pb.gz')+':optimize=6',
          '--model','s4='+str(ROOT/'models/s4.pb.gz')+':optimize=8','--out','{work}']
    task={'folder':str(folder),'kind':'batch','argv':argv,'required':['work/sweep.json','work/sweep.csv'],
          'protocol':CONTRACT}
    freeze(folder/'task.json',task)
    if not (folder/'receipt.json').exists():
        command([sys.executable,'-B',ROOT/'reproduce.py','_native','--config',run.cfg['_path'],
                '--task',folder/'task.json'],folder/'launch',run.cfg)
    if read_json(folder/'receipt.json')['status']!='verified':
        raise ValueError('Batch sweep did not pass')
    rows=read_json(folder/'work/sweep.json')['rows']
    if {(r['model'],r['minibatch']) for r in rows}!={(m,b) for m in ('fp16','s4') for b in (32,64,128,256)} or \
            any(not isinstance(r['nps'],(int,float)) or r['nps']<=0 for r in rows):
        raise ValueError('Fresh batch sweep missing actual work')
    atomic_json(manifest,{'status':'verified','jobs':receipts,
                'batch_sweep':portable_identity(folder/'work/sweep.json'),
                'scope':'Actual complete fresh native workloads; no reference-result resumption'})
    return manifest


def f12_diagnostic(run):
    """Repeat the matched final-model512-input activation diagnostic on CPU."""
    state=run.out/'F12/verification.json'
    if state.exists():
        cached=read_json(state)
        if cached['output']!=portable_identity(ROOT/cached['output']['path']):
            raise ValueError('Fresh F12 diagnostic changed')
        return state
    import onnx
    from onnx import numpy_helper
    from common import INPUT,ort_session
    from quant.quantize import with_outputs,load_planes,batches
    tensor='/encoder14/ln2'
    model=onnx.load(ROOT/'models/s4.onnx')
    inverse=numpy_helper.to_array(next(t for t in model.graph.initializer if t.name==tensor+'/smooth_inv'))
    session=ort_session(with_outputs(model,[tensor,tensor+'/smoothed']),cpu=True,threads=4)
    maxima=[None,None];max_error=0.
    for planes in batches(load_planes(ROOT/'data/calib.npz')):
        deadline_check()
        before,after=session.run([tensor,tensor+'/smoothed'],{INPUT:planes})
        if not np.isfinite(before).all() or not np.isfinite(after).all() or \
                not np.allclose(before*inverse,after,rtol=1e-5,atol=1e-12):
            raise ValueError('Matched SmoothQuant scaling check failed')
        max_error=max(max_error,float(np.max(np.abs(before*inverse-after))))
        for i,value in enumerate((before,after)):
            row=np.abs(value).max(axis=tuple(range(value.ndim-1)))
            maxima[i]=row if maxima[i] is None else np.maximum(maxima[i],row)
    folder=run.out/'F12';folder.mkdir(parents=True,exist_ok=True)
    np.savez(folder/'channel-maxima.npz',s4_raw=maxima[0],s4_smoothed=maxima[1],smooth_inverse=inverse)
    atomic_json(folder/'verification.json',{'status':'verified','positions':512,'channels':768,
                'max_scaling_absolute_error':max_error,'model':portable_identity(ROOT/'models/s4.onnx'),
                'calibration':portable_identity(ROOT/'data/calib.npz'),'output':portable_identity(folder/'channel-maxima.npz'),
                'providers':session.get_providers(),'scope':'CPU calibration diagnostic; not GPU precision or independent quality'})
    return folder/'verification.json'
