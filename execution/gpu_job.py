"""Explicit exclusive-session GPU probes/captures; import is CPU-only."""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'eval'))
from execution.support import atomic_json, checked_result, identity, positive, preserve_command, require_session


def parse_latency(log):
    text=Path(log).read_text(encoding='utf-8',errors='replace')
    pattern=r'GPU Compute Time: min = [\d.eE+-]+ ms, max = [\d.eE+-]+ ms, mean = ([\d.eE+-]+) ms'
    m=re.search(pattern,text)
    if not m:raise ValueError('No successful trtexec GPU-compute latency in '+str(log))
    return positive(float(m.group(1)),'trtexec latency ms')


def run_gemm(a,tag,shape,int8):
    from hw.roofline import gemm_onnx
    folder=Path(a.out);folder.mkdir(parents=True,exist_ok=True)
    model=folder/(tag+'.onnx');log=folder/(tag+'.log')
    gemm_onnx(model,*shape,int8)
    cmd=[a.trtexec,f'--onnx={model}','--fp16','--useCudaGraph','--noDataTransfers',
         '--builderOptimizationLevel=5','--profilingVerbosity=detailed',
         f'--exportLayerInfo={folder/(tag+"_layers.json")}',
         f'--exportProfile={folder/(tag+"_profile.json")}','--separateProfileRun']
    if int8:cmd.append('--int8')
    p=preserve_command(cmd,log)
    checked_result(p,'trtexec '+tag)
    ms=parse_latency(log)
    layers=json.loads((folder/(tag+'_layers.json')).read_text())
    if not layers.get('Layers'):raise ValueError('Empty layer evidence for '+tag)
    return {'shape':list(shape),'latency_ms':ms,'arithmetic_ops':2*shape[0]*shape[1]*shape[2],
            'model':identity(model),'layer_info':identity(folder/(tag+'_layers.json')),
            'precision_requires_review':'Inspect real tactic/operand formats before assigning a compute roof'}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['probe','peaks','capture','profile'])
    ap.add_argument('--session',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--trtexec');ap.add_argument('--lc0');ap.add_argument('--weights')
    ap.add_argument('--backend-opts');ap.add_argument('--onnx')
    ap.add_argument('--int8',action='store_true');ap.add_argument('--sparsity',action='store_true')
    ap.add_argument('--large-only',action='store_true');a=ap.parse_args()
    require_session(a.session)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    if a.action=='probe':
        result=run_gemm(a,'probe',(16,16,16),False)
        atomic_json(out/'result.json',result)
    elif a.action=='peaks':
        fp=run_gemm(a,'fp16',(8192,4096,4096),False)
        q=run_gemm(a,'int8',(8192,4096,4096),True)
        bw=run_gemm(a,'bandwidth',(8,16384,16384),False)
        atomic_json(out/'result.json',{'fp16':fp,'int8':q,'bandwidth':bw,
                  'fp16_candidate_tflops':fp['arithmetic_ops']/(fp['latency_ms']*1e9),
                  'int8_candidate_tops':q['arithmetic_ops']/(q['latency_ms']*1e9),
                  'bandwidth_modelled_weight_gbs':16384*16384*2/(bw['latency_ms']*1e6),
                  'scope':'Measured microbenchmark ceilings; review precision. Bandwidth is effective modeled traffic, not measured DRAM bytes.'})
    elif a.action=='profile':
        from types import SimpleNamespace
        from hw.trt_layers import cmd_build
        batch=256 if a.large_only else 64
        prefix=out/f'build_b{batch}'
        cmd_build(SimpleNamespace(out=str(prefix),trtexec=a.trtexec,onnx=a.onnx,
                      opt_level=5,min_batch=4,opt_batch=256,max_batch=1024,batch=batch,
                      int8=a.int8,bf16=False,sparsity=a.sparsity,layer_norm_fp32=True))
        profiles={str(batch):{'layers':identity(str(prefix)+'_layers.json'),
                              'profile':identity(str(prefix)+'_profile.json')}}
        if not a.large_only:
            larger=out/'reload_b256'
            cmd=[a.trtexec,f'--loadEngine={prefix}.engine','--shapes=/input/planes:256x112x8x8',
                 '--useCudaGraph','--noDataTransfers','--profilingVerbosity=detailed',
                 f'--exportLayerInfo={larger}_layers.json',f'--exportProfile={larger}_profile.json',
                 f'--exportTimes={larger}_times.json','--separateProfileRun']
            p=preserve_command(cmd,out/'reload_b256.log');checked_result(p,'same-engine large-batch profile')
            parse_latency(out/'reload_b256.log')
            profiles['256']={'layers':identity(str(larger)+'_layers.json'),
                              'profile':identity(str(larger)+'_profile.json')}
        for record in profiles.values():
            layers=json.loads(Path(record['layers']['path']).read_text())
            profile=json.loads(Path(record['profile']['path']).read_text())
            if not layers.get('Layers') or not any(r.get('averageMs',r.get('timeMs',0))>0
                 for r in profile if isinstance(r,dict)):
                raise ValueError('Missing nonempty layer/timing evidence')
        atomic_json(out/'result.json',{'model':identity(a.onnx),'engine':identity(str(prefix)+'.engine'),
                    'int8_permission':a.int8,'sparse_permission':a.sparsity,'profiles':profiles,
                    'same_engine_both_batches':not a.large_only,'layer_norm_constraint':'prefer FP32 by name glob',
                    'scope':'Standalone TensorRT profiles; inspect actual formats/tactics. Not asserted identical to lc0/ORT partitioning.'})
    else:
        from eval.lc0uci import Lc0
        from hw.bench import SEARCH_FENS
        records=[]
        e=Lc0(a.lc0,a.weights,backend='onnx-trt',backend_opts=a.backend_opts,
              options={'MoveOverheadMs':20,'Ponder':False},log=str(out/'capture.lc0.log'))
        try:
            e.go(fen=SEARCH_FENS[0],movetime=2000)
            for fen in SEARCH_FENS[:3]:
                e.newgame();r=e.go(fen=fen,nodes=20000)
                positive(r.get('nodes'),'capture actual nodes')
                records.append({'fen':fen,**{k:r.get(k) for k in ('nodes','time_ms','wall','bestmove')}})
        finally:e.quit()
        atomic_json(out/'result.json',{'model':identity(a.weights),'backend_opts':a.backend_opts,
                   'requested_nodes':20000,'positions':records,'scope':'Instrumented precision evidence; excluded from headline timing'})


if __name__=='__main__':main()
