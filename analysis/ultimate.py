"""Strict paired extended-cohort analysis; no scientific jobs at import."""
from collections import defaultdict
from functools import lru_cache
import csv
import json
import math
from pathlib import Path

import chess
import numpy as np

from eval.score import kl, mcnemar_p, ne_curve
from execution.support import read_json
from execution.ultimate_contract import A10, B9, ROOT, SEED, CONTRACT, group_id, hard_ids, atomic_json,deadline_check,portable_identity,freeze
from execution.ultimate_data import load_pool
from execution.ultimate_phases import existing_rows


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError('Empty required table: '+str(path))
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@lru_cache(maxsize=50000)
def bootstrap_group(fen):
    return group_id(chess.Board(fen))


def bootstrap(roots, values, replicates=10000):
    """Stratified cluster pairs; bounded128-replicate draws, including group sizes."""
    values = np.asarray(values,dtype=float)
    if values.ndim==1:
        values=values[:,None]
    if len(roots)!=len(values) or not np.isfinite(values).all():
        raise ValueError('Invalid paired bootstrap inputs')
    if not roots:
        return [[None,None] for _ in range(values.shape[1])]
    # Every possible resample has this identical value; retain the exact
    # degenerate interval, which does not establish statistical equivalence.
    if np.all(values==values[0]):
        return np.column_stack((values[0],values[0])).tolist()
    strata = defaultdict(lambda:defaultdict(list))
    for i,r in enumerate(roots):
        strata[r['cls']][bootstrap_group(r['fen'])].append(i)
    rng=np.random.default_rng(SEED)
    sums=np.zeros((replicates,values.shape[1]))
    for cls,groups in sorted(strata.items()):
        deadline_check()
        indices=list(groups.values())
        totals=np.array([values[idx].sum(axis=0) for idx in indices])
        sizes=np.array([len(idx) for idx in indices])
        class_weight=sizes.sum()/len(roots)
        for start in range(0,replicates,128):
            deadline_check()
            n=min(128,replicates-start)
            draws=rng.integers(0,len(indices),size=(n,len(indices)))
            means=totals[draws].sum(axis=1)/sizes[draws].sum(axis=1)[:,None]
            sums[start:start+n]+=class_weight*means
    return np.quantile(sums,[.025,.975],axis=0).T.tolist()


def holm(rows):
    families=defaultdict(list)
    for r in rows:
        study='B' if r['study']=='B_s0' else r['study']
        r['holm_family']=r['cohort']+':'+study
        if r['model']!='fp16':
            families[(r['cohort'],study)].append(r)
        else:
            r['holm_p']=1.
    for family in families.values():
        ordered=sorted(family,key=lambda r:r['nominal_mcnemar_p'])
        previous=0.
        for i,r in enumerate(ordered):
            previous=max(previous,min(1.,r['nominal_mcnemar_p']*(len(ordered)-i)))
            r['holm_p']=previous


def node_interval(accuracy,points):
    xs,ys=ne_curve(points)
    eps=1e-12
    if accuracy<ys[0]-eps:
        return [None,None,'below reference range']
    if accuracy>ys[-1]+eps:
        return [None,None,'above reference range']
    equal=[i for i,y in enumerate(ys) if abs(y-accuracy)<eps]
    if equal:
        return [math.exp(xs[min(equal)]),None if max(equal)==len(xs)-1 else math.exp(xs[max(equal)]),
                'flat/endpoint interval; no precise node-equivalent' if len(equal)>1 or max(equal)==len(xs)-1 else 'observed point']
    i=next(i for i,y in enumerate(ys) if y>accuracy)
    value=math.exp(xs[i-1]+(accuracy-ys[i-1])/(ys[i]-ys[i-1])*(xs[i]-xs[i-1]))
    return [value,value,'interpolated on observed log-node envelope']


def analyze(run):
    cache={}
    def data(model,study,budget):
        key=(model,study,budget)
        if key not in cache:
            cache[key]=existing_rows(run,*key)
        return cache[key]
    out=run.out/'analysis'
    pools={p:load_pool(run.data,p) for p in ('M','N')}
    challenges=read_json(run.data/'challenges.json')
    screening_raw=data('fp16','A',0)
    screen_ids=hard_ids(pools['N'],{r['id']:screening_raw[r['id']] for r in pools['N']})
    if not run.fresh and screen_ids!=challenges['HN']:
        raise ValueError('Frozen HN no longer matches its original screening observations')
    screen_correct=sum(screening_raw[r['id']]['best'] in r['correct'] for r in pools['N'])
    screening={'N':len(pools['N']),'fp16_correct':screen_correct,'fp16_errors':len(pools['N'])-screen_correct,
               'HN_frozen':len(challenges['HN']),'HN_fresh_rule':len(screen_ids),
               'scope':'FP16-only full N screening; other models evaluated on frozen HN, not all N'}
    write_csv(out/'T1-screening.csv',[screening])
    cohorts={'M':pools['M'],
             'M_sensitivity':[r for r in pools['M'] if r['id'] not in (388,611,627,965,1100)],
             'HM':[r for r in pools['M'] if r['id'] in set(challenges['HM'])],
             'HN':[r for r in pools['N'] if r['id'] in set(challenges['HN'])]}
    rows,graded,adjacent,node_equiv=[],[],[],[]
    for cohort,roots in cohorts.items():
        if not roots:
            # An empty predeclared challenge is a valid finding, never a zero score.
            continue
        conditions=[('A',0,A10)]
        conditions += [('B',b, B9 if b in (800,1600) else A10) for b in (100,400,800,1600)] if cohort in ('M','M_sensitivity') else []
        if cohort in ('HM','HN'):
            conditions += [('B',b,B9) for b in (10,25,50,100,200)]
            if cohort=='HN':
                conditions += [('B_s0',b,('s0',)) for b in (100,400)]
        conditions += [('C',b,('fp16','s4')) for b in (100,200,400,800)]
        for study,budget,models in conditions:
            actual_study='B' if study=='B_s0' else study
            ref=data('fp16',actual_study,budget)
            # HN s0@400 has no B9@400 condition: compare to no invented reference.
            has_ref=all(r['id'] in ref for r in roots)
            for model in models:
                observed=data(model,actual_study,budget)
                selected=[observed[r['id']] for r in roots]
                ok=np.array([s['best'] in r['correct'] for s,r in zip(selected,roots)],dtype=int)
                rok=np.array([ref[r['id']]['best'] in r['correct'] for r in roots],dtype=int) if has_ref else None
                delta=ok-rok if has_ref else None
                intervals=bootstrap(roots,np.column_stack((ok,delta)) if has_ref else ok)
                absolute_low,absolute_high=intervals[0]
                lo,hi=intervals[1] if has_ref else (None,None)
                better=int((delta>0).sum()) if has_ref else None
                worse=int((delta<0).sum()) if has_ref else None
                nominal=mcnemar_p(worse,better) if has_ref else None
                row={'cohort':cohort,'study':study,'budget':budget,'model':model,'n':len(roots),
                     'correct':int(ok.sum()),'errors':int(len(roots)-ok.sum()),'accuracy':float(ok.mean()),
                     'accuracy_ci_low':absolute_low,'accuracy_ci_high':absolute_high,
                     'dtz_optimal':sum(s['best'] in r['dtz_opt'] for s,r in zip(selected,roots)),
                     'delta_vs_fp16':float(delta.mean()) if has_ref else None,'delta_ci_low':lo,'delta_ci_high':hi,
                     'better':better,'worse':worse,
                     'nominal_mcnemar_p':nominal,'nominal_p_float_underflow':nominal==0.0 if has_ref else False,
                     'mean_actual_nodes':float(np.mean([s['nodes'] for s in selected])),
                     'median_actual_nodes':float(np.median([s['nodes'] for s in selected])),
                     'p95_actual_nodes':float(np.percentile([s['nodes'] for s in selected],95)),
                     'min_actual_nodes':min(s['nodes'] for s in selected),'max_actual_nodes':max(s['nodes'] for s in selected),
                     'mean_wall_ms':float(np.mean([s['wall']*1000 for s in selected])) if actual_study!='A' else None,
                     'mean_engine_ms':float(np.mean([s['time_ms'] for s in selected])) if actual_study!='A' else None,
                     'median_wall_ms':float(np.median([s['wall']*1000 for s in selected])) if actual_study!='A' else None,
                     'p95_wall_ms':float(np.percentile([s['wall']*1000 for s in selected],95)) if actual_study!='A' else None,
                     'early_stop_fraction':float(np.mean([s['wall']*1000<budget-20 for s in selected])) if actual_study=='C' else None,
                     'inference':'primary fp16/s4; secondary others; baseline-conditioned challenges' if has_ref else 'supplement only; no matched FP16400 HN condition'}
                rows.append(row)
                if study=='A':
                    refrows=[ref[r['id']] for r in roots]
                    vals=np.array([[kl(a['p'],b['p']),
                       max(a['p'],key=a['p'].get)==max(b['p'],key=b['p'].get),abs(a['v']-b['v']),
                       b['best'] in r['dtz_opt']] for a,b,r in zip(refrows,selected,roots)],dtype=float)
                    intervals=bootstrap(roots,vals)
                    for j,name in enumerate(('printed_policy_KL','printed_prior_top1_agreement','printed_V_absolute_drift','DTZ_optimal_rate')):
                        graded.append({'cohort':cohort,'model':model,'endpoint':name,'n':len(roots),
                                       'mean':float(vals[:,j].mean()),'ci_low':intervals[j][0],'ci_high':intervals[j][1]})
        if cohort in ('M','M_sensitivity','HN'):
            for left,right in zip(A10[:6],A10[1:6]):
                a,b=data(left,'A',0),data(right,'A',0)
                ref=data('fp16','A',0)
                vals=np.array([[float(b[r['id']]['best'] in r['correct'])-float(a[r['id']]['best'] in r['correct']),
                      kl(ref[r['id']]['p'],b[r['id']]['p'])-kl(ref[r['id']]['p'],a[r['id']]['p']),
                      abs(b[r['id']]['v']-ref[r['id']]['v'])-abs(a[r['id']]['v']-ref[r['id']]['v']),
                      float(max(b[r['id']]['p'],key=b[r['id']]['p'].get)==max(ref[r['id']]['p'],key=ref[r['id']]['p'].get))-float(max(a[r['id']]['p'],key=a[r['id']]['p'].get)==max(ref[r['id']]['p'],key=ref[r['id']]['p'].get)),
                      float(b[r['id']]['best'] in r['dtz_opt'])-float(a[r['id']]['best'] in r['dtz_opt'])] for r in roots])
                intervals=bootstrap(roots,vals)
                for j,name in enumerate(('accuracy_change','KL_change','V_drift_change','top1_agreement_change','DTZ_optimal_change')):
                    adjacent.append({'cohort':cohort,'from':left,'to':right,'endpoint':name,'n':len(roots),
                                     'mean':float(vals[:,j].mean()),'ci_low':intervals[j][0],'ci_high':intervals[j][1]})
        print('ANALYZED',cohort,len(roots),flush=True)
    # Only comparisons with an actually matched reference enter inferential families.
    holm([r for r in rows if r['nominal_mcnemar_p'] is not None])
    for r in rows:
        r.setdefault('holm_p',None)
        r.setdefault('holm_family',r['cohort']+':'+('B' if r['study']=='B_s0' else r['study']))
    mainrows=[r for r in rows if r['cohort']=='M']
    refpts=[(r['mean_actual_nodes'],r['accuracy']) for r in mainrows if r['model']=='fp16' and r['study'] in ('A','B')]
    for r in mainrows:
        if r['study']=='B':
            lo,hi,scope=node_interval(r['accuracy'],refpts)
            node_equiv.append({'model':r['model'],'budget':r['budget'],'n':r['n'],
                               'reference_nodes_low':lo,'reference_nodes_high':hi,
                               'ratio_low':lo/r['mean_actual_nodes'] if lo is not None else None,
                               'ratio_high':hi/r['mean_actual_nodes'] if hi is not None else None,'scope':scope})
    write_csv(out/'T1-quality.csv',rows)
    write_csv(out/'T1-graded.csv',graded)
    write_csv(out/'T1-adjacent-ladder.csv',adjacent)
    write_csv(out/'F3-node-equivalents.csv',node_equiv)
    atomic_json(out/'quality.json',{'cohort_counts':{k:len(v) for k,v in cohorts.items()},'screening':screening,'rows':rows,
                'graded':graded,'adjacent':adjacent,'node_equivalents':node_equiv,
                'bootstrap':'10000 material-stratified paired legal-symmetry-group replicates; seed20261030',
                'bootstrap_implementation':'Joint absolute/paired intervals; memoized exact legal-group keys; all-constant arrays return the mathematically identical10000-replicate degenerate interval directly',
                'limitations':['Synthetic decision-sensitive endgames, not general chess',
                  'HM/HN selected using frozen FP16; N screening denominator24000 differs from paired HN',
                  'Printed KL is approximate; V drift is not full WDL-vector error',
                  'Degenerate bootstrap does not establish equivalence',
                  'Original exact McNemar integer-tail calculation converts to float; underflow zeros are flagged, not a claim that probability is exactly zero',
                  'Budgets and shared roots are not independent observations']})
    plots(out,rows,graded)
    export_observations(run,cache,{r['id'] for pool in pools.values() for r in pool})
    return out


def export_observations(run,cache,allowed):
    grouped=defaultdict(list)
    for (model,study,budget),values in cache.items():
        grouped[model,study].extend(r for i,r in values.items() if i in allowed)
    folder=run.out/'observations';outputs=[]
    for (model,study),rows in sorted(grouped.items()):
        deadline_check()
        budget_key='movetime' if study=='C' else 'budget'
        rows.sort(key=lambda r:(r['id'],r[budget_key]))
        keys=[(r['id'],r[budget_key]) for r in rows]
        if len(set(keys))!=len(keys):raise ValueError('Duplicate compact observation key')
        path=folder/'studies'/model/('study_'+study.lower()+'.jsonl')
        path.parent.mkdir(parents=True,exist_ok=True)
        text=''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows)
        if path.exists():
            if path.read_text(encoding='utf-8')!=text:raise ValueError('Preserve changed/partial compact observation file')
        else:path.write_text(text,encoding='utf-8',newline='\n')
        outputs.append(portable_identity(path,folder))
    freeze(folder/'manifest.json',{'status':'verified','protocol':CONTRACT,'fresh':run.fresh,
        'models':[portable_identity(ROOT/'models'/(m+'.pb.gz')) for m in A10],
        'datasets':[portable_identity(run.data/(name+'.json')) for name in ('M','N','challenges')],
        'files':outputs,'file_paths_relative_to':'manifest_directory',
        'scope':'Compact actual unique per-root/budget observations used by this analysis; full FP16 N screening retained. No inferred or missing-result zeros.'})


def plots(out,rows,graded):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out)
    plt.rcParams.update({'font.size':11.5,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        fig.tight_layout(rect=(0,.16,1,1) if fig.legends else (0,0,1,1))
        for ext in ('png','pdf','svg'):
            fig.savefig(out/(name+'.'+ext),dpi=180)
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    for ax,cohort in zip(axes,('M','HN')):
        rr=[r for r in rows if r['cohort']==cohort and r['study']=='A']
        ax.bar([r['model'] for r in rr],[100*r['accuracy'] for r in rr],color=['#5d778f' if r['model']!='s4' else '#cb793e' for r in rr])
        ax.set(title=f'{cohort}: raw preserving accuracy',ylabel='Correct moves (%)',ylim=(0,101))
        ax.tick_params(axis='x',rotation=60)
    save(fig,'F1-raw')
    for name,study,cohorts,models in [('F2-nodes','B',('M','HM','HN'),B9),
                 ('F4-swaps','B',('M','HM','HN'),('fp16','s4','swap_pol8','swap_val8')),
                 ('F5-time','C',('M','HM','HN'),('fp16','s4'))]:
        fig,axes=plt.subplots(1,3,figsize=(9.2,4.2))
        for ax,cohort in zip(axes,cohorts):
            for model in models:
                rr=sorted([r for r in rows if r['cohort']==cohort and r['study']==study and r['model']==model],key=lambda r:r['budget'])
                if rr:
                    ax.plot([r['budget'] for r in rr],[100*r['accuracy'] for r in rr],'o-',label=model,markersize=3)
            ax.set(title=cohort,xlabel='Requested nodes' if study=='B' else 'Requested ms',ylabel='Correct moves (%)')
            ax.set_xscale('log',base=2)
            budgets=sorted({r['budget'] for r in rows if r['cohort']==cohort and r['study']==study and r['model'] in models})
            ax.set_xticks(budgets,[str(b) for b in budgets])
            ax.minorticks_off()
        handles,labels=axes[-1].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.01),ncol=min(5,len(models)),fontsize=9)
        save(fig,name)
    fig,axes=plt.subplots(1,2,figsize=(9,3.6))
    for ax,endpoint in zip(axes,('printed_policy_KL','printed_V_absolute_drift')):
        rr=[r for r in graded if r['cohort']=='M' and r['endpoint']==endpoint]
        ax.bar([r['model'] for r in rr],[r['mean'] for r in rr],color='#5d778f')
        ax.set(title=endpoint.replace('_',' ')+' (M)',ylabel='Mean; log above 0.001')
        ax.set_yscale('symlog',linthresh=.001)
        ax.tick_params(axis='x',rotation=60)
    save(fig,'F1-graded')
