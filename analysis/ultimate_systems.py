"""Analyze only the fresh evaluator's actual systems and calibration outputs."""
from pathlib import Path
import collections
import json
import re
import sqlite3
import subprocess

import numpy as np

from execution.support import read_json
from execution.ultimate_contract import ROOT,atomic_json,portable_identity
from analysis.ultimate import write_csv


def analyze_systems(run,quality):
    from analysis.person_a import run3_systems as helper
    from execution.ultimate_finish import csvrows
    import onnx
    out=run.out/'analysis/hardware';out.mkdir(parents=True,exist_ok=True)
    proof=read_json(run.out/'systems/verification.json')
    receipts=proof['jobs']
    # Geometry is a model input, never a reused performance observation.
    inventory=csvrows(ROOT/'analysis/tables/roofline-operation-inventory.csv')
    graph=onnx.load(ROOT/'models/bt3_fp32.onnx')
    nodes={n.name:n for n in graph.graph.node}
    initializers={t.name:t for t in graph.graph.initializer}
    if len(inventory)!=195 or {r['name'] for r in inventory}!={n.name for n in graph.graph.node if n.op_type=='MatMul'}:
        raise ValueError('Operation inventory does not match the active graph')
    for r in inventory:
        n=nodes[r['name']]
        shape=json.loads(r['output_shape_batch1']);k=int(r['K'])
        if int(np.prod(shape))*k!=int(r['macs_per_position']) or int(r['arithmetic_ops_per_position'])!=2*int(r['macs_per_position']):
            raise ValueError('Operation geometry arithmetic differs')
        if r['unique_weight_name']:
            weight=initializers[r['unique_weight_name']]
            if weight.name not in n.input or int(np.prod(weight.dims))!=int(r['weight_parameters']):
                raise ValueError('Operation weight geometry differs')
    write_csv(out/'roofline-operation-inventory.csv',inventory)
    raw=[r for r in quality['rows'] if r['cohort']=='M' and r['study']=='A']
    ready=[]
    for r in raw:
        lo,hi=r['accuracy_ci_low'],r['accuracy_ci_high']
        ready.append({'model':r['model'],'n':r['n'],'correct':r['correct'],'accuracy':r['accuracy'],'ci_low':lo,'ci_high':hi})
    write_csv(out/'frontier-quality-ready.csv',ready)
    # Existing pure analysis is reused with every input/output root explicit.
    saved={k:getattr(helper,k) for k in ('OLD','TABLE','REPORT','catalog')}
    helper.OLD=run.out/'systems';helper.TABLE=out;helper.REPORT=run.out/'systems-analysis'
    helper.REPORT.mkdir(parents=True,exist_ok=True)
    helper.catalog=lambda:receipts
    try:
        profile=helper.profiles()
        benchmark=helper.benchmarks()
    finally:
        for k,v in saved.items():setattr(helper,k,v)
    from execution.ultimate_finish import copy_exact
    benchmark['session_policy_change_recorded']=False
    benchmark['warning']='Fresh evaluator trials; idle subtraction remains provisional unless independently validated. Inspect the actual counter/sample and idle-anomaly fields; no external-meter validation.'
    atomic_json(run.out/'systems-analysis/benchmark-review.json',benchmark)
    for name in ('measured-reference-peaks.json','profile-review.json','benchmark-review.json'):
        copy_exact(run.out/'systems-analysis'/name,out/name)
    # The legacy helper's two historical session names are inapplicable here.
    for p in out.glob('*.csv'):
        text=p.read_text(encoding='utf-8')
        text=text.replace('original-strict','fresh-evaluator').replace('codex-v2','fresh-evaluator')
        p.write_text(text,encoding='utf-8',newline='')
    kernels(run,out)
    figures(run,out)
    sources=[portable_identity(run.out/'systems/verification.json'),portable_identity(run.out/'F12/verification.json')]
    atomic_json(out/'fresh-analysis-verification.json',{'status':'verified','sources':sources,
        'profile':profile,'benchmark':benchmark,
        'scope':'Actual fresh 44 native jobs, four-cap batch sweep,512-input F12; static checked model geometry reused. No historical timing/energy observations copied.',
        'energy':'Above-idle subtraction remains provisional unless the independent idle check validates it; fresh total-load consistency reported separately.'})
    return out


def kernels(run,out):
    from hw.trt_layers import classify_text
    tags=[];allrows=[];proof=[]
    folder=run.out/'systems-analysis/kernels';folder.mkdir(parents=True,exist_ok=True)
    for model in ('fp16','s1','s2','s3','s4','s4aa','s0','s1sp'):
        trace=run.out/'systems'/('capture-'+model)/'attempt-001/trace.nsys-rep'
        db=folder/(model+'.sqlite');receipt=folder/(model+'.json')
        if not receipt.exists():
            if db.exists():raise FileExistsError('Unrecorded fresh kernel export')
            argv=[run.cfg['nsys'],'export','--type=sqlite','--force-overwrite=false','--output='+str(db),str(trace)]
            with (folder/(model+'.log')).open('x',encoding='utf-8') as stream:
                p=subprocess.run(argv,stdout=stream,stderr=subprocess.STDOUT)
            if p.returncode:raise RuntimeError('Fresh trace export failed: '+model)
            atomic_json(receipt,{'source':portable_identity(trace),'database':portable_identity(db),'command':argv,'exit_code':0})
        verified=read_json(receipt)
        if verified['source']!=portable_identity(trace) or verified['database']!=portable_identity(db):
            raise ValueError('Fresh kernel export changed')
        with sqlite3.connect('file:'+db.as_posix()+'?mode=ro',uri=True) as con:
            fields={r[1] for r in con.execute('PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)')}
            key='demangledName' if 'demangledName' in fields else 'shortName'
            rows=con.execute('SELECT s.value,SUM(k.end-k.start),COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.'+key+' GROUP BY s.value ORDER BY 2 DESC').fetchall()
        total=sum(r[1] for r in rows)
        if total<=0:raise ValueError('Fresh trace has no timed kernels')
        sums=collections.defaultdict(float)
        for name,ns,count in rows:
            tag='+'.join(sorted(classify_text(name))) or 'other';sums[tag]+=ns
            allrows.append({'model':model,'kernel':name,'total_ns':ns,'instances':count,'fraction_all_kernel_time':ns/total,'name_tag':tag,'gemm_like_name':bool(re.search('gemm|mma|conv|matmul|bmm|fmha',name,re.I))})
        tags.extend({'model':model,'name_tag':tag,'kernel_ms':ns/1e6,'fraction_all_kernel_time':ns/total,'scope':'whole capture incl startup/warmup; name-tag time is not operation coverage'} for tag,ns in sums.items())
        proof.append(portable_identity(receipt))
    write_csv(out/'T2-lc0-kernels.csv',allrows);write_csv(out/'T2-lc0-kernel-tags.csv',tags)
    atomic_json(folder/'verification.json',{'status':'verified','exports':proof})


def figures(run,out):
    from execution.ultimate_finish import csvrows
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def save(fig,name):
        fig.tight_layout(rect=(0,.18,1,1) if fig.legends else (0,0,1,1))
        for ext in ('png','pdf','svg'):fig.savefig(out/(name+'.'+ext),dpi=180)
        plt.close(fig)
    rows=csvrows(out/'T2-standalone-profiles.csv')
    rows=[r for r in rows if r['model'] in ('fp16','s4')]
    fig,ax=plt.subplots(figsize=(7.2,4.2));bottom=np.zeros(len(rows))
    for key,label in (('fp16_gemm_ms','Visible FP16 GEMM'),('int8_gemm_ms','Visible INT8 GEMM'),('opaque_fused_matrix_ms','Opaque matrix'),('norm_softmax_fused_ms','Norm/softmax/fusion'),('conversion_ms','Conversion'),('other_fused_ms','Other fusion')):
        vals=np.array([float(r[key]) for r in rows]);ax.bar(range(len(rows)),vals,bottom=bottom,label=label);bottom+=vals
    ax.set_xticks(range(len(rows)),[r['model']+' B'+r['batch'] for r in rows]);ax.set_ylabel('Summed layer profile time (ms)')
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.01),fontsize=8.5,ncol=3)
    save(fig,'F6-profile-breakdown')
    points=csvrows(out/'F6-roofline-points.csv');peaks=read_json(out/'measured-reference-peaks.json')
    fig,axes=plt.subplots(1,2,figsize=(8,3.8));x=np.logspace(-1,5,250)
    for ax,pr in zip(axes,('fp16','int8')):
        ax.plot(x,np.minimum(peaks[pr]['gemm_reference_rate'],x*peaks['bandwidth']['gemm_reference_rate']/1000),'--',label='Measured reference')
        for model,mark in (('fp16','o'),('s4','s')):
            rr=[r for r in points if r['precision']==pr and r['model']==model]
            ax.scatter([float(r['ops_per_modeled_byte']) for r in rr],[float(r['achieved_tera_ops_s']) for r in rr],s=10,alpha=.5,marker=mark,label=model)
        ax.set(xscale='log',yscale='log',title=pr.upper(),xlabel='Ops / modeled byte',ylabel='Tera ops/s');ax.legend(fontsize=9)
    save(fig,'F6-roofline')
    speed=csvrows(out/'T2-speed.csv');fig,axes=plt.subplots(1,2,figsize=(8,3.8))
    for ax,key,title in zip(axes,('gpu_mJ_node_total','gpu_mJ_node_above_idle'),('Total under load','Above idle: provisional')):
        y=np.array([float(r[key+'_median']) for r in speed]);lo=y-[float(r[key+'_min']) for r in speed];hi=[float(r[key+'_max']) for r in speed]-y
        ax.bar(range(len(y)),y,yerr=[lo,hi],capsize=2);ax.set_xticks(range(len(y)),[r['model'] for r in speed],rotation=60);ax.set(ylabel='GPU mJ / visit',title=title)
    save(fig,'F7-energy')
    source=(run.out/'F12/channel-maxima.npz') if run.fresh else ((ROOT/'results/reference/F12/current-channel-maxima.npz') if (ROOT/'artifact-manifest.json').exists() else ROOT/'analysis/person_a/a5/current-channel-maxima.npz')
    maxima=np.load(source);fig,axes=plt.subplots(1,2,figsize=(8,3))
    for ax,key,title in zip(axes,('s4_raw','s4_smoothed'),('Before scaling','After scaling')):
        ax.plot(maxima[key],linewidth=.7);ax.set(xlabel='Channel',ylabel='Max |activation|',title=title)
    save(fig,'F12-outliers')
