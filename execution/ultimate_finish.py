"""Extended analysis, report and release operations through the public controller."""
import csv
import json
from pathlib import Path
import shutil

from execution.ultimate_contract import ROOT,atomic_json,portable_identity
from execution.support import read_json


def copy_exact(source,destination):
    source,destination=Path(source),Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        from execution.support import identity
        a,b=identity(destination),identity(source)
        if any(a[k]!=b[k] for k in ('bytes','sha256')):
            raise ValueError('Refusing different pre-existing curated file: '+str(destination))
    else:
        shutil.copy2(source,destination)


def csvrows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def analyze_phase(run):
    from analysis.ultimate import analyze,write_csv
    from execution.ultimate_match import journal,summary
    out=run.out/'analysis'
    prepared=out/'numerical-verification.json'
    # A visual-review resume need not recompute10000-replicate analysis.
    if prepared.exists():
        cached=read_json(prepared)
        for item in [*cached['sources'],*cached['outputs']]:
            if portable_identity(ROOT/item['path'])!=item:
                raise ValueError('Prepared analysis output changed before visual closure')
        review=out/'visual-review.json'
        figures=cached['figures']
        if not review.exists() or read_json(review).get('figures')!=figures or read_json(review).get('status')!='verified':
            return {'awaiting_agent_review':[str(ROOT/p['path']) for p in figures], 'outputs':[prepared]}
        return {'outputs':[prepared,review,out/'quality.json',out/'T4-matches.json']}
    out=analyze(run)
    matchroot=ROOT/'results/reference/matches' if (ROOT/'artifact-manifest.json').exists() and not run.fresh else run.out/'matches/production'
    contract=read_json(matchroot/'contract.json')
    games=journal(matchroot,contract)
    match=summary(games)
    if len(games)!=1000 or any(r['complete_pairs']!=250 for r in match):
        raise ValueError('T4 requires1000 games and250 pairs per condition')
    atomic_json(out/'T4-matches.json',match)
    for r in match:
        r['termination']=json.dumps(r['termination'],sort_keys=True)
    write_csv(out/'T4-matches.csv',match)
    timing=[]
    for envelope in games:
        game=envelope['record']
        for move in game['moves']:
            timing.append({'budget_ms':game['budget_ms'],'pair':game['pair'],'color':game['color'],
                           'model':game[move['side']],**move})
    write_csv(out/'T4-move-timing.csv',timing)
    # Primary hardware observations remain the bounded historical repeats;
    # U2 is a separately labelled validation, not a silent replacement trial.
    hardware=out/'hardware'
    sources=[]
    if run.fresh:
        from analysis.ultimate_systems import analyze_systems
        analyze_systems(run,read_json(out/'quality.json'))
        sources=[portable_identity(hardware/'fresh-analysis-verification.json')]
    else:
        for p in sorted((ROOT/'analysis/tables').glob('*.csv')):
            if p.name.startswith(('T2-','T3-','T5-','F6-','roofline-','smoothquant-')):
                copy_exact(p,hardware/p.name)
                sources.append(portable_identity(p))
        for name in ('measured-reference-peaks.json','profile-review.json','benchmark-review.json'):
            source=(ROOT/'results/reference/analysis/hardware'/name) if (ROOT/'artifact-manifest.json').exists() else ROOT/'reports/takeover/run3'/name
            copy_exact(source,hardware/name);sources.append(portable_identity(source))
        from analysis.ultimate_systems import figures
        figures(run,hardware)
        source=(ROOT/'results/reference/F12/current-channel-maxima.npz') if (ROOT/'artifact-manifest.json').exists() else ROOT/'analysis/person_a/a5/current-channel-maxima.npz'
        sources.append(portable_identity(source))
    refresh_frontier_and_batch(out,run)
    validation=ROOT/'results/reference/U2' if (ROOT/'artifact-manifest.json').exists() and not run.fresh else run.out/'U2'
    atomic_json(out/'hardware-provenance.json',{'sources':sources,
                'U2_numeric':portable_identity(validation/'numeric-verification.json'),
                'U2_systems':portable_identity(validation/'systems-verification.json'),
                'U2_idle':portable_identity(validation/'idle/work/idle.json'),
                'scope':'Actual complete fresh hardware replay' if run.fresh else 'Original repeated hardware observations retained; fresh gate separately labelled; no retroactive idle validation'})
    outputs=sorted(p for p in out.rglob('*') if p.is_file())
    outputs+=sorted(p for p in (run.out/'observations').rglob('*') if p.is_file())
    atomic_json(out/'numerical-verification.json',{'status':'numerically_verified_pending_visual_review',
                'sources':[portable_identity(ROOT/p) for p in ('analysis/ultimate.py','analysis/ultimate_systems.py','execution/ultimate_finish.py')],
                'quality':portable_identity(out/'quality.json'),'matches':portable_identity(out/'T4-matches.json'),
                'figures':[portable_identity(p) for p in outputs if p.suffix=='.png'],
                'outputs':[portable_identity(p) for p in outputs],
                'visual_review_required':True})
    # Explicit visual review is performed by the orchestrator before phase closure.
    review=out/'visual-review.json'
    if not review.exists() or read_json(review).get('status')!='verified':
        return {'awaiting_agent_review':[str(p) for p in outputs if p.suffix=='.png'],
                'outputs':[out/'quality.json',out/'T4-matches.json',out/'numerical-verification.json']}
    return {'outputs':[out/'quality.json',out/'T4-matches.json',out/'visual-review.json',out/'numerical-verification.json']}


def refresh_frontier_and_batch(out,run=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from analysis.ultimate import write_csv
    q=read_json(out/'quality.json')
    speed=csvrows(out/'hardware/T2-speed.csv')
    raw={r['model']:r for r in q['rows'] if r['cohort']=='M' and r['study']=='A'}
    frontier=[]
    fig,ax=plt.subplots(figsize=(6.5,4))
    for r in speed:
        model=r['model'];quality=raw[model]
        x=float(r['engine_nps_median']);y=100*quality['accuracy']
        ax.plot(x,y,'o',color='#cb793e' if model=='s4' else '#4f718b')
        ax.annotate(model,(x,y),xytext=(4,4),textcoords='offset points',fontsize=8)
        frontier.append({'model':model,'n':quality['n'],'correct':quality['correct'],
                         'raw_accuracy':quality['accuracy'],'historical_search_nps_median':x,
                         'historical_search_nps_min':float(r['engine_nps_min']),
                         'historical_search_nps_max':float(r['engine_nps_max']),
                         'scope':'Common M raw quality; historical bounded search repeats. No swap speed claims.'})
    ax.set(xlabel='Historical lc0 search visits/s (median)',ylabel='M raw preserving accuracy (%)')
    fig.tight_layout()
    for ext in ('png','pdf','svg'):
        fig.savefig(out/('F11-frontier.'+ext),dpi=180)
    plt.close(fig)
    write_csv(out/'F11-frontier.csv',frontier)
    # F3 shows interval status rather than a fabricated precise saturated inverse.
    values=csvrows(out/'F3-node-equivalents.csv')
    fig,ax=plt.subplots(figsize=(8,3.8))
    budgets=(100,400,800,1600)
    import numpy as np
    models=('fp16','s1','s2','s3','s4','s4aa','s1sp','swap_pol8','swap_val8')
    colors=plt.get_cmap('tab10')
    for index,model in enumerate(models):
        rr=[r for r in values if r['model']==model]
        first=True
        for r in rr:
            if r['ratio_low']:
                x=int(r['budget'])*np.exp((index-4)*.025);lo=float(r['ratio_low'])
                hi=float(r['ratio_high']) if r['ratio_high'] else None
                ax.plot(x,lo,'^' if hi is None else 'o',markersize=4,color=colors(index),label=model if first else None)
                if hi is not None:ax.vlines(x,lo,hi,color=colors(index),linewidth=1)
                first=False
    ax.set(xlabel='Requested nodes',ylabel='FP16 node-equivalent ratio\n(lower endpoint)',xscale='log')
    ax.set_xticks(budgets,[str(b) for b in budgets]);ax.minorticks_off()
    ax.set_title('Triangles: unbounded upper endpoint; absent points: outside reference range\nSlight x-offset separates models',fontsize=9)
    ax.legend(fontsize=7,ncol=3)
    fig.tight_layout()
    for ext in ('png','pdf','svg'):
        fig.savefig(out/('F3-node-equivalents.'+ext),dpi=180)
    plt.close(fig)
    batch_source=run.out/'systems/batch-sweep/attempt-001/work/sweep.json' if run and run.fresh else ROOT/'reports/c3_hw/batch_sweep/sweep.json'
    if not batch_source.exists():
        candidates=list((ROOT/'reports/c3_hw').glob('batch_sweep*/sweep.json'))
        if len(candidates)!=1:
            raise ValueError('Four-point historical batch evidence not uniquely located')
        batch_source=candidates[0]
    batches=read_json(batch_source)['rows']
    if {(r['model'],r['minibatch']) for r in batches}!={(m,b) for m in ('fp16','s4') for b in (32,64,128,256)}:
        raise ValueError('Batch evidence coverage differs')
    write_csv(out/'hardware/F9-batch.csv',batches)
    fig,ax=plt.subplots(figsize=(6,3.5))
    for model in ('fp16','s4'):
        rr=sorted([r for r in batches if r['model']==model],key=lambda r:r['minibatch'])
        ax.plot([r['minibatch'] for r in rr],[r['nps'] for r in rr],'o-',label=model)
    ax.set(xlabel='MinibatchSize cap',ylabel='Observed search visits/s',xscale='log')
    ax.set_xticks([32,64,128,256],['32','64','128','256']);ax.minorticks_off()
    ax.legend()
    fig.tight_layout()
    for ext in ('png','pdf','svg'):
        fig.savefig(out/('F9-batch.'+ext),dpi=180)
    plt.close(fig)
