# Does INT8 make Leela stronger per second?

**CS6886 Systems for Deep Learning, IIT Madras - Group 10**  
Ritwik Mishra (ME23B237) and Anjan R D (ME23B157)

Repository: [anjanrd/CS6886_Group10_CourseProject](https://github.com/anjanrd/CS6886_Group10_CourseProject)

This project measures how INT8 post-training quantization and 2:4 pruning affect Leela Chess Zero's BT3 transformer: neural outputs, endgame decisions, search, latency, GPU energy and complete games at equal time per move. This directory contains the source, canonical models, frozen evaluation inputs, measured observations, analysis and reports.

## 1. Start here

| Evaluation goal | Instructions | New GPU measurements? |
|---|---|---|
| Read the report and inspect supplied results | Reports below and Section 5 | No |
| Check installation and artifact identities | Section 4: `reproduce.py check` | No inference |
| Recompute statistics and figures from recorded observations | Section 5: `reproduce.py reference` | No |
| Rebuild models and collect new experimental results | Section 6: `reproduce.py all --fresh` | Yes |

- [Report (5 pages)](CS6886_Group10_Report.pdf): course report with the current repository URL.
- [Extended Report](extended_report.pdf): expanded methods, figures, results and limitations;

The primary experiments are complete. Independent checks covered an extracted copy, a new Python environment, reference-data reconciliation and a bounded set of actual engine searches using existing native caches. **These checks are not a completed cold-cache reproduction of the entire experiment.** A subsequent independent replay regenerated models and datasets, but stopped at a numerical tolerance check for newly built floating-point engines. Section 7 describes this limitation. The original [verification record](verification.json) and [bounded-check evidence](results/backup) are retained; their historical identifiers describe recorded checks, not additional evaluator tasks.

### Selected results

The selected INT8 model is `s4` (SmoothQuant, alpha 0.7); the baseline is `fp16`.

| Measurement | FP16 | Selected INT8 |
|---|---:|---:|
| Correct raw decisions, main set | 6,706 / 6,720 | 6,706 / 6,720 |
| Correct main-set decisions at 400 requested nodes | 6,720 / 6,720 | 6,720 / 6,720 |
| Median search visits/s, repeated benchmark | 4,842 | 6,818 |
| Median network latency, batch 64 | 18.08 ms | 13.04 ms |
| Median total GPU energy per search visit | 17.3372 mJ | 12.7155 mJ |
| Wins / draws / losses, 100 ms per move | 50 / 357 / 93 | 93 / 357 / 50 |
| Wins / draws / losses, 200 ms per move | 36 / 425 / 39 | 39 / 425 / 36 |

Each match setting contains 500 games: 250 openings played in both colours. INT8 has 43 more wins at 100 ms and three more at 200 ms. These are descriptive results for the stated protocol. Neural error, decision accuracy, throughput and game outcomes are distinct measurements.

## 2. Obtain the complete files

### GitHub checkout

Install Git and [Git LFS](https://git-lfs.com/) before cloning. ONNX models, network files and serialized engines use LFS; pointer files alone cannot run the experiments.

From a parent directory with sufficient space:

```powershell
git lfs install
git clone https://github.com/anjanrd/CS6886_Group10_CourseProject.git CS6886-source
git -C .\CS6886-source lfs pull
git -C .\CS6886-source lfs fsck
```

Run evaluator commands from a separate working copy without Git metadata. The supplied controller retains development-checkout guards; an extracted working copy uses its evaluator path. This also separates generated configuration, environments and measurements from the source clone.

```powershell
# Use a new destination; do not merge with an earlier evaluation run.
New-Item -ItemType Directory -Path .\CS6886-evaluation -ErrorAction Stop
Get-ChildItem -LiteralPath .\CS6886-source -Force |
    Where-Object { $_.Name -ne '.git' } |
    Copy-Item -Destination .\CS6886-evaluation -Recurse
Set-Location .\CS6886-evaluation
```

All subsequent commands run from this evaluation directory. Do not assume a GitHub-generated source ZIP includes the real LFS payloads.

### Supplied submission archive

Extract the complete archive and enter its `CS6886_Group10_CourseProject` directory. Git and LFS are unnecessary when the archive already contains the actual model files. Create a new Python environment locally; do not copy a virtual environment between machines or paths.

## 3. Prerequisites

| Component | Tested configuration |
|---|---|
| Operating system | 64-bit Windows 11 |
| Python | 3.11.9 |
| GPU / host memory | NVIDIA RTX 4060 Laptop 8 GB / 32 GB RAM |
| NVIDIA driver | 592.82 |
| Engine | lc0 v0.33.0-rc0, bundled GPU ONNX Runtime 1.22.1 |
| TensorRT | 10.12.0.36, including `trtexec` and matching libraries |
| Profiler | Nsight Systems 2025.1.3 |
| Ground truth | Required Syzygy 3-5-piece WDL/DTZ files |

Python uses **CPU ONNX Runtime 1.31.1** for model preparation, separate from lc0's GPU runtime. Do not replace it with `onnxruntime-gpu`. Other platforms and GPUs have not been validated. Strict checks compare native executables, lc0 DLLs and tablebases with [native-inputs.json](data/ultimate/native-inputs.json) and [tablebases.json](data/ultimate/tablebases.json); a newer native version is not automatically equivalent.

Install native tools separately from the official [lc0 releases](https://github.com/LeelaChessZero/lc0/releases), [TensorRT](https://developer.nvidia.com/tensorrt) and [Nsight Systems](https://developer.nvidia.com/nsight-systems) distributions. Obtain the Syzygy files listed in the supplied inventory. Keep lc0 and its matching DLLs together and preserve upstream licenses/notices. The research checkpoint, prepared models, calibration/development/test inputs and opening book are included. Serialized reference engines are hardware-specific evidence, not portable substitutes for new builds.

For measurements, use AC power, clear cooling and a consistent power mode. The recorded laptop used Balanced mode and natural clocks; no clock lock is claimed. Run one GPU workload at a time. Actual power, telemetry, process, numerical and output failures stop affected work; background-activity indicators are informational.

Reserve **at least 100 GiB free space** for a full replay, beyond any source clone/LFS storage. Complete-scope planning was **45-70 hours, with 72+ hours reserved**. This is a planning range, not a measured fresh-reproduction completion time or a guarantee for other hardware. Main fixed-node replay alone was later estimated at about 16 hours on the tested laptop.

## 4. Install dependencies and check inputs

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements/ultimate-report.lock.txt
.\.venv\Scripts\python.exe -m pip check
Copy-Item config.example.json config.json
```

Edit these entries in `config.json` for your declared native installations:

| Key | Expected value |
|---|---|
| `lc0` | Full path to `lc0.exe`, alongside matching DLLs |
| `trtexec` | Full path to TensorRT's `trtexec.exe` |
| `nsys` | Full path to the supported `nsys.exe` |
| `syzygy` | Directory containing the required tablebase files |

Use forward slashes in JSON paths (for example, `C:/tools/lc0/lc0.exe`) or escape backslashes. Keep the supplied checkpoint path, recipe and experimental settings unchanged. Save machine-specific changes in `config.json`, not the example file.

```powershell
.\.venv\Scripts\python.exe -X utf8 -B reproduce.py check
```

This verifies dependencies, model/checkpoint identities, the artifact manifest and declared native inputs. It does **not** run inference or certify full reproduction. On a mismatch, inspect the LFS download, file modifications or native versions; do not change expected scientific hashes to bypass it.

## 5. Inspect and recompute recorded results

### Direct inspection

These files can be read without a GPU or Python environment:

| Location | Contents |
|---|---|
| [Reference analysis](results/reference/analysis) | Tables, paired statistics, confidence intervals and figures |
| [quality.json](results/reference/analysis/quality.json) | All 240 quality conditions, graded endpoints and node-equivalent intervals |
| [Hardware analysis](results/reference/analysis/hardware) | Latency, execution coverage, roofline and qualified energy results |
| [Complete games](results/reference/matches) | Production games, PGN, settings and summaries |
| [Per-position observations](results/reference/observations) | Compact measurements and provenance bindings |
| [Development selection](results/reference/selection) | Frozen selection evidence |
| [Activation diagnostic](results/reference/F12) | Matched SmoothQuant rescaling evidence |

The reference collection has 444,332 unique position/condition observations and all 1,000 production games. Hard-subset and sensitivity views reuse corresponding observations; they are not independent extra trials.

### Deterministic reference reanalysis

After completing setup and the input check:

```powershell
.\.venv\Scripts\python.exe -X utf8 -B reproduce.py reference --output results/reference-reanalysis
```

This recomputes statistics and figures, including the 10,000-replicate paired bootstrap, from supplied observations and checks quality/game summaries against their reference counterparts. A successful comparison writes `results/reference-reanalysis/reference-verification.json`. No new scientific engine searches run, although this command still checks the configured native prerequisites. Allow time for bootstrap analysis.

This verifies analysis of **recorded measurements**, not newly collected engine outcomes.

To render the editable detailed report separately:

```powershell
.\.venv\Scripts\python.exe -X utf8 -B reproduce.py render --source report.md --pdf report-rendered.pdf
```

This renders the detailed Markdown report, not the separately supplied course-report PDF. Inspect any newly rendered PDF before use.

## 6. Run a complete fresh experiment

Read Section 7's known limitation first. From the configured evaluation directory:

```powershell
.\.venv\Scripts\python.exe -X utf8 -B reproduce.py all --fresh --output results/reproduced
```

Use a new output directory for a new invocation. The workflow is designed to rebuild exports, quantization/pruning/swaps and development comparisons; regenerate/audit datasets and openings; verify numerical/runtime behaviour; repeat the complete experimental matrix; and regenerate analysis, figures and the detailed report. Frozen hard-subset identities remain fixed rather than being reselected using new outcomes.

The engine executable and matching DLLs are copied into the output runtime directory without the original engine cache. This lc0 version caches beside its executable. New engine/timing caches stay separate from reference engines; driver-level caching may persist and is not represented as cleared.

### Experimental coverage

| Study | Conditions |
|---|---|
| Raw outputs and moves | Ten variants on Main (6,720) and Hard-Fresh (632); FP16 additionally screens Fresh (24,000) |
| Main fixed-node search | Nine variants at 100/400/800/1,600 requested nodes; magnitude supplement at 100/400 |
| Hard-subset search | Nine variants on Hard-Main (99) and Hard-Fresh (632) at 10/25/50/100/200 nodes; magnitude on Hard-Fresh at 100/400 |
| Fixed-time decisions | FP16 and selected INT8 at 100/200/400/800 ms on Main and Hard-Fresh; Hard-Main derived from Main |
| Full games | FP16 versus selected INT8: 500 games at 100 ms and 500 at 200 ms, plus excluded technical pilots |
| Systems / mechanisms | Eight single-body variants; precision/profiling/sparsity, qualified GPU energy, separate batch sweep and matched activation-rescaling diagnostic |

Variants are FP16 (`fp16`), per-tensor (`s1`), per-channel (`s2`), percentile-clipped (`s3`), SmoothQuant (`s4`), attention-extended (`s4aa`), magnitude 2:4 (`s0`), SparseGPT 2:4 (`s1sp`), and head swaps (`swap_pol8`, `swap_val8`). Both pruning methods use the selected quantization recipe. Swaps contain two complete bodies and are quality probes, excluded from timing/energy comparisons.

### Visual review and completion

The controller pauses for visual inspection. Review every listed figure for labels, clipping, legends, scales and correspondence to tables. Then record the actual inspection using the interface's figure-review identifier:

```powershell
.\.venv\Scripts\python.exe -X utf8 -B reproduce.py review --fresh --output results/reproduced --phase U8 --note "Describe the actual inspection and findings"
.\.venv\Scripts\python.exe -X utf8 -B reproduce.py all --fresh --output results/reproduced
```

When the generated report is ready, inspect every PDF page in a viewer or render pages with Poppler. Record the report review and resume:

```powershell
.\.venv\Scripts\python.exe -X utf8 -B reproduce.py review --fresh --output results/reproduced --phase U9 --note "Describe the completed page-by-page inspection and findings"
.\.venv\Scripts\python.exe -X utf8 -B reproduce.py all --fresh --output results/reproduced
```

The literal review identifiers are required command-line values. Notes attest to actual inspection; they are not automated image-quality checks. Fresh measurements remain separate from `results/reference`. State and per-job receipts are under `results/reproduced`; generated tables/figures and reports are in its `analysis/` and `report/` directories.

### Interruptions and failures

Preserve output directories, state, caches, logs and receipts. Completed compatible work is reused where supported; changing code, inputs or settings invalidates affected contracts. Do not delete failed evidence or rebuild engines repeatedly until favourable numbers appear.

After diagnosing and resolving a specific failure, record the recovery through:

```powershell
.\.venv\Scripts\python.exe -X utf8 -B reproduce.py all --fresh --output results/reproduced --recover-reviewed "Describe the diagnosed cause, repair and verification"
```

This records a reviewed recovery; it does not repair faults or waive tolerances. The numerical-check collection loop has a known resume limitation: it can refuse pre-existing command folders even when collection receipts are complete. Preserve those receipts and diagnose the resume path rather than deleting or overwriting them.

## 7. Validation status and interpretation limits

The independent setup and bounded engine checks passed. A later cold-cache replay regenerated all ten models, development comparisons and evaluation inputs; sampled CPU outputs agreed exactly. It completed the small raw/node collection matrix, but **new floating-containing TensorRT engines exceeded frozen raw-output tolerances**. Therefore the full fresh scientific replay is not certified.

Affected models were FP16 and the two head swaps. For example, a moves-left output changed from 40.7 to 41.2 (bound 0.25), and a normalized policy difference was approximately 0.00540054 (bound 0.005). Repeating the new cache reproduced the new values; a diagnostic with the original cache reproduced reference values with unchanged source, weights and native executable identities. This supports engine/cache-dependent numerical variation; no particular internal tactic was established as its cause. Unchanged sampled move outcomes do not satisfy a failed numerical check. Reusing old caches or widening tolerances would not establish the specified cold-cache verification.

Other limits to preserve when interpreting or extending the results:

- Main and Fresh are synthetic decision-sensitive endgames with at most five pieces. Hard subsets are selected only from frozen FP16 raw outputs. Other variants were evaluated on Hard-Fresh, not all 24,000 Fresh positions.
- Syzygy supplies scoring/adjudication only, never the engine's search backend. Categories map +2 to win, -2 to loss, and 0/+1/-1 to draw.
- The recipe is fixed: 512 calibration inputs, 3,360 separate development positions, percentile 99.999 and SmoothQuant alpha 0.7. Do not tune on test or match outcomes.
- Printed-prior KL is approximate; scalar value drift is not full WDL-vector distance. Nominal and Holm-adjusted tests have different roles.
- Saturated node-equivalent intervals do not support precise efficiency ratios. Accuracy need not increase monotonically at every search budget.
- Standalone INT8 matrix-arithmetic coverage is not whole-network precision coverage. Standalone sparse tactics do not establish sparse acceleration within lc0.
- Historical above-idle GPU energy remains provisional because idle telemetry was inconsistent. Total GPU energy under load is separate; neither measures whole-machine energy.
- Fresh timing, search paths and game outcomes may vary. Record valid differences; the pipeline does not require INT8 to win, meet an Elo target or cross a significance threshold.

## 8. Directory guide

| Path | Purpose |
|---|---|
| `reproduce.py` | Public checking, reanalysis, fresh-experiment, rendering and review interface |
| `quant/`, `eval/`, `tb/`, `hw/` | Quantization, evaluation, scoring and systems measurements |
| `execution/`, `analysis/` | Checked orchestration, analysis and report generation |
| `nets/`, `models/` | Source checkpoint and canonical models |
| `data/` | Calibration/development/test inputs, book, settings and provenance |
| `results/reference/` | Recorded observations, games, analysis and systems evidence |
| `results/backup/`, `verification.json` | Independent bounded-check evidence and scope |
| `requirements/`, `config.example.json` | Exact dependencies and configuration template |
| `artifact-manifest.json` | Sizes/checksums for supplied research artifacts and documentation |

The course-report URL revision and evaluator README are documentation changes, not new experimental results or a full-reproduction pass. The retained verification record describes its original checked snapshot. Keep generated results separate from supplied reference evidence and retain upstream licenses/notices.
