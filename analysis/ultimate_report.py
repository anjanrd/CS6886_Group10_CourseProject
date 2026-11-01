"""Portable Markdown-to-PDF report, generated from verified numerical tables."""
from pathlib import Path
import re
from xml.sax.saxutils import escape

from execution.support import read_json
from execution.ultimate_contract import ROOT,atomic_json,portable_identity


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'---|'*len(headers),
                       *['| '+' | '.join(str(x) for x in row)+' |' for row in rows]])


def build_markdown(run):
    from execution.ultimate_finish import csvrows
    out=run.out/'analysis'
    q=read_json(out/'quality.json')
    raw=[r for r in q['rows'] if r['cohort']=='M' and r['study']=='A']
    main={r['model']:r for r in raw}
    graded={(r['model'],r['endpoint']):r for r in q['graded'] if r['cohort']=='M'}
    speed={r['model']:r for r in csvrows(out/'hardware/T2-speed.csv')}
    latency={(r['model'],int(r['batch'])):r for r in csvrows(out/'hardware/T2-network-batch-summary.csv')}
    nps_gain=100*(float(speed['s4']['engine_nps_median'])/float(speed['fp16']['engine_nps_median'])-1)
    latency_drop=100*(1-float(latency['s4',64]['latency_ms_median'])/float(latency['fp16',64]['latency_ms_median']))
    matches=read_json(out/'T4-matches.json')
    counts=q['cohort_counts']
    screening=q['screening']
    group_source=run.out/'derivation/models/s4.json' if run.fresh else ROOT/'models/s4.json'
    smooth_groups=read_json(group_source)['smoothed_groups']
    improved=sum(v['outlier_after']<v['outlier_before'] for v in smooth_groups.values())
    def fmt(x):
        return 'undefined' if x is None else f'{x:.4f}'
    raw_table=table(['Model','Correct/n','Accuracy %','Approx. KL','Top1 %','Mean abs dV'],[
        [r['model'],f"{r['correct']}/{r['n']}",f"{100*r['accuracy']:.4f}",
         f"{graded[r['model'],'printed_policy_KL']['mean']:.5f}",
         f"{100*graded[r['model'],'printed_prior_top1_agreement']['mean']:.2f}",
         f"{graded[r['model'],'printed_V_absolute_drift']['mean']:.5f}"] for r in raw])
    timed=[r for r in q['rows'] if r['study']=='C' and r['cohort'] in ('M','HN')]
    timed_table=table(['Cohort','ms','Model','Correct/n','Visits (mean)','Wall ms (mean)'],[
        [r['cohort'],r['budget'],r['model'],f"{r['correct']}/{r['n']}",
         f"{r['mean_actual_nodes']:.1f}",f"{r['mean_wall_ms']:.2f}"] for r in timed])
    timed_pairs=table(['Cohort','ms','s4 delta pp','95% paired interval pp','-worse/+better','Holm p'],[
        [r['cohort'],r['budget'],f"{100*r['delta_vs_fp16']:.4f}",
         f"[{100*r['delta_ci_low']:.4f}, {100*r['delta_ci_high']:.4f}]",
         f"-{r['worse']}/+{r['better']}",f"{r['holm_p']:.4f}"] for r in timed if r['model']=='s4'])
    match_table=table(['ms','Games','FP16 wins','s4 wins','Draws','FP16 score'],[
        [r['budget_ms'],r['games'],r['fp16_wins'],r['s4_wins'],r['draws'],f"{r['fp16_score']:.4f}"] for r in matches])
    match_text=[]
    for r in matches:
        match_text.append(f"At {r['budget_ms']} ms, "+
            ('the models tie in win count.' if r['more_wins']=='tie' else
             f"{r['more_wins']} has more wins: an absolute margin of {r['absolute_win_margin']} games ({r['pp_win_margin']:.2f} percentage points).")+
            f" FP16/s4 win percentages are {r['fp16_win_pct']:.2f}%/{r['s4_win_pct']:.2f}%; draws are {r['draw_pct']:.2f}%.")
    node_table=table(['Cohort','Nodes','Model','Correct/n','Delta pp','95% paired interval pp','-worse/+better'],[
        [r['cohort'],r['budget'],r['model'],f"{r['correct']}/{r['n']}",
         f"{100*r['delta_vs_fp16']:.4f}",f"[{100*r['delta_ci_low']:.4f}, {100*r['delta_ci_high']:.4f}]",
         f"-{r['worse']}/+{r['better']}"] for r in q['rows']
        if r['study']=='B' and r['model'] in ('fp16','s4') and r['cohort'] in ('M','HM','HN')])
    systems_table=table(['Model','Trials','Search visits/s','Batch64 NN ms','GPU total mJ/visit'],[
        [m,r['trials'],f"{float(r['engine_nps_median']):.0f}",f"{float(latency[m,64]['latency_ms_median']):.2f}",
         f"{float(r['gpu_mJ_node_total_median']):.4f}"] for m,r in speed.items()])
    winners='; '.join(f"{r['budget_ms']}ms: "+('equal win counts' if r['more_wins']=='tie' else r['more_wins']+' has more wins') for r in matches)
    hardware_word='Fresh evaluator' if run.fresh else 'Historical comparable'
    profiles=csvrows(out/'hardware/T2-standalone-profiles.csv')
    mapped=next(r for r in profiles if r['model']=='s4' and int(r['batch'])==64)
    geometry=csvrows(out/'hardware/roofline-operation-inventory.csv')
    total_ops=sum(int(r['arithmetic_ops_per_position']) for r in geometry)
    peaks=read_json(out/'hardware/measured-reference-peaks.json')
    roof=csvrows(out/'hardware/F6-roofline-points.csv')
    above=sum(r['above_reference']=='True' for r in roof)
    sparse={r['model']:r for r in csvrows(out/'hardware/T5-sparsity.csv')}
    energy=read_json(out/'hardware/benchmark-review.json')
    reproduction=''
    reproduction_status='Full independent evaluator reproduction (U11) and the subsequent integrated release audit (U12) are pending. The supplied primary experiments and report are complete; any bounded backup checks are recorded separately in verification.json. Such checks do not establish full experimental reproducibility.'
    comparison=run.out/'evaluator-comparison.json'
    if comparison.exists():
        reproduction_status='The full evaluator reproduction is documented below and in verification.json.'
        comp=read_json(comparison)
        changed=sum(r['correct_difference']!=0 for r in comp['quality'])
        replaymatches=' '.join(f"At{r['budget_ms']}ms: FP16 {r['fp16_wins']} wins, s4 {r['s4_wins']} wins, {r['draws']} draws; "+('equal win counts.' if r['more_wins']=='tie' else f"{r['more_wins']} leads by {r['absolute_win_margin']} wins ({r['pp_win_margin']:.2f}pp).") for r in comp['fresh_matches'])
        reproduction=f"\n\nThe full evaluator walk completed from an external ZIP extraction with a newly created local Python environment and explicitly shared native installations on this same laptop. It rebuilt every retained model and development-grid evidence, regenerated datasets, reran the complete retained raw/node/time/challenge matrix, all44 systems jobs, batch sweep, matched512-input activation diagnostic, and both500-game settings, and regenerated/reviewed its own figures and report. {changed}/{len(comp['quality'])} matched conditions changed their correct count. {replaymatches} Full per-condition changes, timings and audit evidence are supplied under results/evaluator. These are additional descriptive observations, not a requirement to reproduce a particular winner."
    def result(cohort,study,budget,model):
        return next(r for r in q['rows'] if (r['cohort'],r['study'],r['budget'],r['model'])==(cohort,study,budget,model))
    hfp,hs4=(result('HN','A',0,m) for m in ('fp16','s4'))
    raw_findings=(f"On M, s4 and FP16 preserve {main['s4']['correct']}/{main['s4']['n']} and {main['fp16']['correct']}/{main['fp16']['n']} outcomes, with {main['s4']['better']} s4 improvements and {main['s4']['worse']} regressions. "
        f"Approximate mean policy KL is {graded['s3','printed_policy_KL']['mean']:.5f} for s3 and {graded['s4','printed_policy_KL']['mean']:.5f} for s4; mean printed V drift is {graded['s3','printed_V_absolute_drift']['mean']:.5f} and {graded['s4','printed_V_absolute_drift']['mean']:.5f}. "
        f"On HN, FP16 preserves {hfp['correct']}/{hfp['n']} and s4 {hs4['correct']}/{hs4['n']}, with {hs4['better']} improvements and {hs4['worse']} regressions. Conditioning HN on FP16 is essential to interpreting that comparison.")
    node_findings='Main-pair M counts (FP16/s4) are '+', '.join(f"{b} nodes: {result('M','B',b,'fp16')['correct']}/{result('M','B',b,'s4')['correct']}" for b in (100,400,800,1600))+'. '
    node_findings+='HN endpoint counts (FP16/s4) are '+', '.join(f"{b} nodes: {result('HN','B',b,'fp16')['correct']}/{result('HN','B',b,'s4')['correct']}" for b in (10,200))+'. Search repairs many raw errors, but the paired tables retain every compression regression.'
    timed_findings='Timed preserving-count changes (s4 minus FP16) on M are '+', '.join(f"{b} ms: {result('M','C',b,'s4')['correct']-result('M','C',b,'fp16')['correct']:+d}" for b in (100,200,400,800))+'; on HN they are '+', '.join(f"{b} ms: {result('HN','C',b,'s4')['correct']-result('HN','C',b,'fp16')['correct']:+d}" for b in (100,200,400,800))+'. '
    timed_findings+='At 800 ms, mean visits (FP16/s4) are '+', '.join(f"{c}: {result(c,'C',800,'fp16')['mean_actual_nodes']:.1f}/{result(c,'C',800,'s4')['mean_actual_nodes']:.1f}" for c in ('M','HN'))+'. Workload and accuracy changes must be assessed separately.'
    match_interpretation=('The 100 ms setting shows a practical win-count advantage for s4 in this opening set; the 200 ms setting is much closer and has 85% draws. This budget dependence limits a general strength claim and does not isolate throughput from numerical drift as the cause.' if not run.fresh else 'These fresh counts are additional observations. Compare each budget and its draw rate separately; no required winner or hypothesis threshold applies.')
    text=f'''# Does INT8 make Leela stronger per second?

BT3 post-training quantization in lc0: neural outputs, search repair and measured systems tradeoffs

Ritwik Mishra (ME23B237) and Anjan R D (ME23B157)  
CS6886 - Systems for Deep Learning, IIT Madras

## Abstract

We evaluate a frozen BT3 chess transformer in lc0 through TensorRT, comparing a wrapped FP16 baseline with a cumulative signed W8A8 post-training ladder, two pruning variants and policy/value swaps. The selected SmoothQuant model s4 achieves {main['s4']['correct']}/{main['s4']['n']} raw outcome-preserving moves, versus {main['fp16']['correct']}/{main['fp16']['n']} for FP16 on the common expanded test cohort. {hardware_word} repeated measurements show {nps_gain:.2f}% higher median search throughput and {latency_drop:.2f}% lower batch64 network latency for s4. Fixed-node, baseline-conditioned challenge, fixed-time and full-game results are reported separately. The two match settings contain500 games each ({winners}). These observations test separate links from compression to inference speed, visits per second and decisions; a faster network alone cannot establish improved playing strength. Adverse sparse results and limits of energy/precision evidence remain part of the answer.

## 1. Motivation and research questions

Leela combines a neural evaluator with Monte Carlo tree search (MCTS). Quantization can change both the policy that guides exploration and the values backed up through the tree. Faster evaluation might buy more search within a fixed time, but search might correct, hide or amplify the resulting errors. We therefore ask three separate questions: RQ1, how does compression change neural outputs and selected moves? RQ2, what does a fixed amount of search repair, and do policy/value swaps distinguish sensitivity? RQ3, do measured speed gains translate to more visits, better timed decisions or more game wins?

Haque, Wei and Mueller [1] motivate separating raw Lc0 decisions from search with exact endgame labels. Their older network and corpus do not predict BT3 INT8 outcomes. SmoothQuant [2] and SparseGPT [3] motivate rescaling and compensated pruning; their language-model results do not guarantee chess improvements. GPTQ [4] is related background, not the quantizer used here. Sparse execution [5], roofline analysis [6] and NNUE [7] provide systems context with different computation and evaluation boundaries.

## 2. BT3, runtime and compression

The supplied BT3-768x15x24h-swa-2790000 checkpoint uses INPUT_CLASSICAL_112_PLANE, with input shape[batch,112,8,8]. Outputs are a1858-entry policy, a three-component win/draw/loss (WDL) head, and moves-left. Policy supplies legal-move priors; WDL supplies a neural position evaluation; moves-left is auxiliary. The final move comes from search statistics. A selected move's tablebase accuracy is not its neural WDL probability.

The checkpoint is exported with lc0 leela2onnx. A metadata-preserving wrapper embeds the ONNX graph in an lc0 network. FP16 means the exported graph through onnx-trt optimize=6, permitting FP16. Quantized variants use optimize=8, permitting INT8 plus floating exceptions. Python ONNX Runtime1.31.1 performs CPU preparation; lc0 v0.33.0-rc0 bundles GPU ORT1.22.1 and TensorRT10.12. The benchmark device is an RTX4060 Laptop8GB on Windows with32GB host memory. Flags express permissions; actual kernels/layer inspection establish bounded execution evidence.

'''+table(['Model','Added method'],[
 ['fp16','Wrapped exported floating graph; runtime FP16 permission'],
 ['s1','Signed symmetric per-tensor W8A8; minmax calibration'],
 ['s2','Per-channel weight scales'],['s3','Percentile99.999 calibration'],
 ['s4','SmoothQuant alpha0.7; frozen main INT8'],['s4aa','s4 plus dynamic attention products'],
 ['s0','Magnitude2:4 pruning then best-recipe quantization'],
 ['s1sp','Sequential SparseGPT2:4 then best-recipe quantization'],
 ['swap_pol8 / swap_val8','Policy/value from different complete bodies; quality only']])+f'''

Calibration stays512 inputs:256 random-game/history and256 tablebase inputs. Development selection uses the original3360 roots and freezes s4 before expanded tests. The ladder uses the adopted custom ORT-based calibration implementation, not ModelOpt. No new calibration, training or test-driven retuning occurs. SmoothQuant implements XW=(X/s)(sW) using explicit compensating Mul branches because a shared LayerNorm output also feeds residual consumers. SparseGPT uses sequential calibration compensation before quantization. Sparse masks alone do not prove sparse GPU execution.

![F12. Selected-model rescaling: a matched512-input channel example with compensated-scaling checks.](../analysis/hardware/F12-outliers.png)

The matched final-model diagnostic checks768 channels at one layer using all512 calibration inputs and agrees with compensated scaling. It is a calibration illustration, not GPU precision or test accuracy. The build records {improved}/{len(smooth_groups)} groups with a lower peak/median activation ratio after rescaling, retaining non-improving groups. These build summaries must not be confused with the plotted single-layer diagnostic.

## 3. Data, scorer and frozen protocol

'''+f'''The original11200 positions are synthetic legal, nonterminal, decision-sensitive3-5-piece boards, then labelled by local Syzygy tables; they were not downloaded from tablebase files or sampled from human games. The original calibration/development/test pools contain1120/3360/6720 roots. M includes all6720 test roots,240 per28 classes. The prior2437-root cohort was a time-budgeted class-round-robin FEN-prefix sample, not an unbiased general-chess sample. A6715-root sensitivity view excludes IDs388,611,627,965,1100 without reinference or deletion from M.

N contains24000 fresh five-piece roots,2000 per12 classes, seed20261030, IDs11200-35199. Exact/legal-symmetry overlap is rejected against all11200 old roots and within N. The legal relation uses pawn-preserving file reflection, pawnless D8 and color/rank mirroring, with transformed tablebase-category checks. The accepted IDs and seeded order are frozen before inference. N is evaluation-only. HM/HN apply the frozen FP16 rule: a wrong raw selected move OR printed preserving policy mass below0.8. Their observed sizes are {counts['HM']}/{counts['HN']}; no size or significance target changes them. N's24000 screening denominator is distinct from the HN paired denominator. Challenges are baseline-conditioned.

FP16 selects a preserving move on {screening['fp16_correct']}/{screening['N']} full-N screening roots ({screening['fp16_errors']} errors). The fresh application of the challenge rule selects {screening['HN_fresh_rule']} roots; the evaluated frozen HN retains {screening['HN_frozen']}. Other models were not evaluated on all N.

The scorer maps Syzygy+2 to win,-2 to loss, and0/+1/-1 to draw. A correct selected move preserves the root category; DTZ optimality is secondary. Root halfmove clocks remain zero with the adopted FEN-only history. Losing and all-moves-correct roots were excluded by construction. Tablebases supply labels and match adjudication only; evaluated engines never receive SyzygyPath.

'''+table(['Study','Models / cohorts','Frozen settings'],[
 ['Raw','A10 on M/HN; FP16 screens N','go nodes1; PolicyTemperature1'],
 ['Main nodes','B9 on M; s0 supplement','100/400/800/1600; s0 only100/400'],
 ['Challenge','B9 on HM/HN','10/25/50/100/200; s0 HN100/400'],
 ['Timed','fp16/s4 on M/HN','100/200/400/800ms; minibatch256, overhead20ms'],
 ['Games','fp16/s4,250 paired openings','500 games each at100/200ms; minibatch64, overhead20ms']])+'''

Fixed-node search uses Threads1, MinibatchSize256 and SmartPruningFactor0. Timed search retains default threads/pruning. Actual nodes can overshoot and timed searches can stop early. Reset/cache conventions are retained; each bounded block records its fresh engine and timed excluded warm-up. Compatible old observations retain source/model/settings provenance, while missing conditions are measured. Distinct compatible keys are stored once and are not independent repeats in multiple views.

Matches use250 distinct six-ply balanced openings sampled with seed20261030 from the attributed jhorthos/Caissabase-derived book [8], with colors swapped and the same openings at both budgets. Both resident engines are warmed; one search runs at a time, Ponder=false. Two disjoint excluded pilot openings check shared timing limits. The imported legal-game/tablebase/resignation/draw rules and400-ply cap apply. Ordinary soft overruns are retained as timings, not clock forfeits. Complete-game atomic records support exact-contract resumption.

We report paired correct/n, discordances and accuracy differences. Primary contrasts are FP16/s4; others are secondary. McNemar p-values are exact nominal values, with Holm adjustment within study/cohort contrast families when interpreted inferentially. Confidence intervals use10000 fixed-seed material-stratified paired resamples over legal symmetry groups. Shared roots/budgets are not independent observations; a degenerate interval does not prove equivalence. Printed-prior KL is approximate, and printed V deviation is not full WDL-vector error. All conditions are retained irrespective of outcome.

## 4. RQ1: neural outputs and raw move quality

'''+raw_table+'''

![F1. Common-M and baseline-conditioned HN raw preserving accuracy.](../analysis/F1-raw.png)

![Graded output differences on M. The vertical scale is linear below 0.001 and logarithmic above it; the reference has zero drift.](../analysis/F1-graded.png)

Equal correct counts do not imply equal moves or probabilities. '''+raw_findings+'''

The machine-readable graded table contains paired uncertainty for KL, printed-prior top1 agreement, V drift and DTZ optimality, and the adjacent-ladder table retains changes between cumulative steps. The sparse and pruned models' quality results remain part of the evaluation irrespective of outcome.

## 5. RQ2: search repair, challenges and swaps

'''+node_table+'''

![F2. Fixed-node comparisons on M/HM/HN; actual visits are retained in tables.](../analysis/F2-nodes.png)

The main and low-node comparisons use the same roots at each compared condition. The historical43-root hard view remains recoverable. Challenge conditioning and a small number of discordances limit generalization even when an observed difference is nonzero. Search need not improve monotonically at every finite budget. No p-value, minimum HN count or INT8 gain was required to complete collection.

'''+node_findings+'''

![F4. Policy/value swaps use two complete bodies and are quality-only probes.](../analysis/F4-swaps.png)

Swaps route policy and value/moves-left between s4 and the floating-side graph. They are not timed frontier points, and backend numerical changes can prevent a perfectly isolated intervention on one scalar. Saturated or tied swap outcomes cannot establish equal policy/value importance. Magnitude retains only its predeclared100/400-node supplement; no missing high-budget curve is invented.

![F3. Node-equivalent interval endpoints; flat reference curves remain undefined or unbounded.](../analysis/F3-node-equivalents.png)

The FP16 reference uses its observed monotone envelope in log actual nodes. A flat accuracy curve admits an interval of inverse node counts; the figure/table preserve that uncertainty instead of fabricating a precise efficiency ratio.

## 6. RQ3: fixed time and complete games

'''+timed_table+'\n\n'+timed_pairs+'''

![F5. Timed preserving accuracy at all four frozen requested budgets.](../analysis/F5-time.png)

Requested time, actual wall/engine time and visits are distinct. Early stopping and skewed visit distributions can make a fixed-time visit ratio differ from a throughput benchmark's ratio. A one-root accuracy difference or more visits alone cannot demonstrate stronger full-game play. HM timed views are derived from M without counting a second observation.

'''+timed_findings+'''

'''+match_table+'\n\n'+'\n\n'.join(match_text)+'''

'''+match_interpretation+'''

Matches are descriptive: counts/percentages, score, explicit win margins, move timing and all termination reasons. There is no required Elo/LOS, significance test, hypothesis acceptance threshold, result-based stopping or resizing. The complete-game and per-move tables preserve what actually ran. Production and an independent evaluator replay are separate observations; different winners or timing need not be a software failure.

## 7. Systems explanation and energy limits

'''+systems_table+f'''

{hardware_word} s4 median throughput is {nps_gain:.2f}% above FP16 and batch64 network latency is {latency_drop:.2f}% lower. FP16/s4 have five trials each and other single-body variants three each. These are observed repeats with session/thermal context, not automatically independent confidence-interval replicates. AC/Balanced mode and natural clocks are recorded; clock locking was denied. Background activity flags are informational, not assumed equal across trials. U2 numerical/load and bounded systems checks are labelled separately.

![F11. Common-M raw quality versus bounded {hardware_word.lower()} search throughput; swaps excluded.](../analysis/F11-frontier.png)

Actual standalone s4 inspector evidence proves {mapped['int8_matrix_ops_proven']} INT8 matrix operations, representing {100*float(mapped['int8_matrix_arithmetic_fraction_proven']):.4f}% of matrix arithmetic among {len(geometry)} matrix operations ({total_ops} arithmetic operations,{total_ops//2} MACs per position). This is not parameter fraction, all-operation precision or runtime share. {mapped['matrix_ops_unmapped']} matrix operations remain unmapped by exact metadata; their precision is unresolved. lc0 traces demonstrate named precision kernels, not whole-operation coverage; standalone TensorRT mappings do not automatically describe lc0's internal ORT engine.

![F6. Whole standalone regions are counted once; fusion and conversion remain visible.](../analysis/hardware/F6-profile-breakdown.png)

![F6. Partial empirical-reference roofline uses logical matrix bytes, not measured DRAM traffic.](../analysis/hardware/F6-roofline.png)

The partial roofline maps {len(roof)} simple GEMM points. Empirical reference rates are {peaks['fp16']['gemm_reference_rate']:.2f} FP16 TFLOP/s, {peaks['int8']['gemm_reference_rate']:.2f} INT8 TOP/s and modelled {peaks['bandwidth']['gemm_reference_rate']:.2f}GB/s; {above} above-reference observations remain plotted. Shape, fusion, accumulation and clock differences prevent treating those lines as universal hardware ceilings or proof of stationary dataflow.

Standalone selected sparse GEMM tactics number {sparse['s0']['selected_sparse_gemm_tactics']} for magnitude and {sparse['s1sp']['selected_sparse_gemm_tactics']} for SparseGPT; the preserved old regex count0 was an incomplete historical summary. Single-configuration dense/sparse ratios are {float(sparse['s0']['standalone_speed_ratio']):.3f}x/{float(sparse['s1sp']['standalone_speed_ratio']):.3f}x at batch256. lc0 does not enable that standalone sparsity flag, so these results do not establish lc0 sparse acceleration. Pruning can reduce network latency while worsening search throughput and move quality.

![F7. Median GPU energy per visit; error bars span observed trial minima and maxima. Total energy and provisional above-idle subtraction are separated.](../analysis/hardware/F7-energy.png)

These logs contain {energy['implausible_instantaneous_samples']} implausible inactive-phase power samples across {energy['trials_with_implausible_instantaneous_power']}/{energy['trials']} trials on a120W-cap laptop GPU. The maximum relative difference between under-load counter/sample means is {100*energy['max_counter_vs_sample_relative_difference']:.4f}%; this is an internal cross-check, not external-meter validation. Total GPU mJ/visit is retained separately. Historical idle baselines lack a trustworthy instantaneous cross-check, so above-idle subtraction remains provisional; a successful fresh idle diagnosis cannot retroactively validate it. No CPU/whole-machine energy or equal-energy accuracy claim is made.

![F9. Four measured search batch caps; no batch-accuracy claim or global optimum.](../analysis/F9-batch.png)

The four-point32/64/128/256 sweep distinguishes search throughput from full-batch network latency. An observed best cap does not establish global optimality for all positions or short move budgets. Swaps have two bodies and are excluded from speed/energy coordinates.

## 8. Limits, reproducibility and conclusion

This is one network on one laptop, with synthetic decision-sensitive endgames, a ceiling-prone outcome metric, limited printed-output precision and baseline-conditioned challenges. The sensitivity view discloses cross-pool symmetry without redefining M. Development selection is frozen; new results cannot choose a new recipe. Numerical drift, preservation of tablebase categories, actual visits, throughput and playing strength remain separate empirical endpoints. Negative, saturated or inconclusive results are valid outcomes.

The package contains the canonical models, checkpoint/calibration/development inputs, scorer, frozen roots/challenges/openings, per-root/game observations, necessary systems evidence and a public reproduction entry. Model derivation and the frozen development-selection evidence can be replayed separately from canonical-input checksum checks. Fresh searches, matches and performance may vary; deterministic analysis must reproduce exact counts/rounding from identical inputs. Serialized TensorRT caches are hardware-specific and fresh caches are rebuilt where required. The final verification record states the actual evaluator coverage; a subset smoke test cannot count as full reproduction.

The selected INT8 model is faster in the measured runtime and improves some practical game counts, especially at 100 ms. The expanded endgame tests show near-ceiling main accuracy and mixed challenge/timed effects. Thus the evidence supports the compression-to-speed link more strongly than a general speed-to-decision-quality claim. Sparse/pruned variants show that acceleration permissions and zero weights alone do not guarantee better search or decisions.

{reproduction_status}{reproduction}

## References

[1] Haque, R., Wei, T.-H., Mueller, M. On the Road to Perfection? Evaluating Leela Chess Zero Against Endgame Tablebases. ACG2021 proceedings, LNCS13262,142-152,2022. https://doi.org/10.1007/978-3-031-11488-5_13

[2] Xiao, G. et al. SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. ICML,2023. https://proceedings.mlr.press/v202/xiao23c.html

[3] Frantar, E., Alistarh, D. SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot. ICML,2023. https://proceedings.mlr.press/v202/frantar23a.html

[4] Frantar, E. et al. GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. ICLR,2023. https://arxiv.org/abs/2210.17323

[5] Mishra, A. et al. Accelerating Sparse Deep Neural Networks.2021. https://arxiv.org/abs/2104.08378

[6] Williams, S., Waterman, A., Patterson, D. Roofline: An Insightful Visual Performance Model for Multicore Architectures. Communications of the ACM,2009. https://doi.org/10.1145/1498765.1498785

[7] Stockfish developers. NNUE documentation. https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html

[8] Thomas, J. H. Opening books derived from Caissabase; balanced six-ply Q0-0.25 tier. https://github.com/jhorthos/lczero-training/wiki/Opening-Books

Software: Leela Chess Zero, TensorRT10.12, ONNX Runtime (distinct CPU/GPU versions), python-chess/Syzygy; exact identities, licenses and sources are supplied in the package.
'''
    # Readable typesetting without changing model names, versions or paths.
    words='a|an|A|contains|contain|stays|original|prior|all|matched|checks|uses|use|seed|IDs|below|above|only|batch|minibatch|overhead|with|Windows|per|on|at|and|the|historical|predeclared|across|from|LNCS|ICML|ICLR|Network|nodes|PolicyTemperature|MinibatchSize|SmartPruningFactor|percentile|alpha|Magnitude|SparseGPT|point|count'
    text=re.sub(r'\b('+words+r')(?=\d)',r'\1 ',text)
    text=re.sub(r'(?<=\d)(?=ms\b|GB\b|W\b)', ' ',text)
    text=text.replace('shape[','shape [').replace('a 1858-entry','an 1858-entry').replace('batch64','batch 64')
    return text


def render_markdown(source,destination):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,KeepTogether,CondPageBreak
    from reportlab.lib.utils import ImageReader
    source,destination=Path(source),Path(destination)
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyReport',fontName='Times-Roman',fontSize=10.5,leading=13.5,spaceAfter=7))
    styles.add(ParagraphStyle(name='CaptionReport',fontName='Helvetica',fontSize=8.2,leading=10.5,spaceAfter=10))
    styles.add(ParagraphStyle(name='TableReport',fontName='Helvetica',fontSize=8,leading=9.8))
    styles['Title'].fontName='Helvetica-Bold';styles['Title'].fontSize=23;styles['Title'].leading=27
    styles['Heading2'].fontName='Helvetica-Bold';styles['Heading2'].fontSize=13;styles['Heading2'].leading=16
    doc=SimpleDocTemplate(str(destination),pagesize=A4,rightMargin=43,leftMargin=43,topMargin=45,bottomMargin=43,
                          title='Does INT8 make Leela stronger per second?',author='Ritwik Mishra and Anjan R D')
    story=[]
    lines=source.read_text(encoding='utf-8').splitlines()
    def inline(text):
        text=escape(text)
        text=re.sub(r'`([^`]+)`',r'<font name="Courier">\1</font>',text)
        text=re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',text)
        return text
    i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line:
            i+=1;continue
        if line.startswith('# '):
            story.append(Paragraph(inline(line[2:]),styles['Title']));i+=1
        elif line.startswith('## '):
            story.extend([CondPageBreak(100),Paragraph(inline(line[3:]),styles['Heading2'])]);i+=1
        elif line.startswith('!['):
            match=re.fullmatch(r'!\[(.*)\]\((.*)\)',line)
            if not match:
                raise ValueError('Malformed report image reference')
            path=(source.parent/match[2]).resolve()
            width,height=ImageReader(str(path)).getSize()
            scale=min(doc.width/width,235/height)
            story.append(KeepTogether([Image(str(path),width=width*scale,height=height*scale),
                                       Paragraph(inline(match[1]),styles['CaptionReport'])]));i+=1
        elif line.startswith('|'):
            grid=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                cells=[c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch('[-:]+',c) for c in cells):
                    grid.append([Paragraph(inline(c),styles['TableReport']) for c in cells])
                i+=1
            if any(len(r)!=len(grid[0]) for r in grid):
                raise ValueError('Report table column mismatch')
            t=Table(grid,colWidths=[doc.width/len(grid[0])]*len(grid[0]),repeatRows=1,hAlign='LEFT')
            t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8eef3')),
                                  ('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#61778a')),
                                  ('BOTTOMPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),
                                  ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f6f8fa')])]))
            story.extend([t,Spacer(1,9)])
        else:
            paragraph=[]
            while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','![','|')):
                paragraph.append(lines[i].strip());i+=1
            story.append(Paragraph(inline(' '.join(paragraph)),styles['BodyReport']))
    def footer(canvas,doc):
        canvas.saveState();canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#536574'))
        canvas.drawString(43,25,'CS6886 | BT3 INT8 in lc0')
        canvas.drawRightString(A4[0]-43,25,str(doc.page));canvas.restoreState()
    doc.build(story,onFirstPage=footer,onLaterPages=footer)


def report_phase(run,folder=None):
    from pypdf import PdfReader
    folder=Path(folder) if folder else run.out/'report';folder.mkdir(parents=True,exist_ok=True)
    md=folder/'report.md';pdf=folder/'report.pdf'
    text=build_markdown(run)
    figures=[portable_identity((folder/p).resolve()) for p in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',text)]
    qa=folder/'structural-qa.json'
    cached=read_json(qa) if qa.exists() else None
    unchanged=md.exists() and md.read_text(encoding='utf-8')==text and pdf.exists() and cached and cached['pdf']==portable_identity(pdf) and cached.get('renderer')==portable_identity(__file__) and cached.get('figures')==figures
    if not unchanged:
        md.write_text(text,encoding='utf-8',newline='\n')
        render_markdown(md,pdf)
    reader=PdfReader(pdf)
    text='\n'.join(page.extract_text() or '' for page in reader.pages)
    for required in ('References','Tablebases','100','200','24000','6720','provisional','unresolved'):
        if required not in text:
            raise ValueError('Report missing required content: '+required)
    atomic_json(folder/'structural-qa.json',{'pages':len(reader.pages),'pdf':portable_identity(pdf),'renderer':portable_identity(__file__),
                'figures':figures,
                'markdown':portable_identity(md),'visual_review_required':True})
    review=folder/'visual-review.json'
    if not review.exists() or read_json(review).get('status')!='verified' or read_json(review).get('pdf')!=portable_identity(pdf):
        return {'awaiting_agent_review':[str(pdf)],'outputs':[md,pdf,folder/'structural-qa.json']}
    return {'outputs':[md,pdf,folder/'structural-qa.json',review]}
