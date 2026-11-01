# BT3 INT8 in lc0 - CS6886 Group10

**U10A bounded backup verification passed. Full U11 evaluator replay and U12 integrated audit are pending.**

Primary studies/report are complete. Independent backup checks used an external extraction, new Python environment and a fixed small actual search matrix. Existing native caches were reused. See verification.json and results/backup for exact coverage; no full reproduction pass is claimed.

Ritwik Mishra (ME23B237) and Anjan R D (ME23B157), IIT Madras.

Read report.pdf (editable report.md) for the question, methods, measured outcomes and limitations. Reference observations are under results/reference. Fresh measurements are written separately; reference results never satisfy a fresh scientific job.

## Tested prerequisites

Windows11/64-bit, Python3.11.9, RTX4060 Laptop8GB,32GB host memory, NVIDIA driver592.82. Other operating systems/GPUs are untested. Use AC, the recorded Balanced power mode and clear cooling, one GPU workload at a time. Natural clocks are logged; no clock lock is claimed. Background activity remains informational, while actual AC, monitor, command, numerical and output failures stop the affected sequence for diagnosis.

Provide lc0 v0.33.0-rc0 with its bundled GPU ORT1.22.1, TensorRT10.12.0.36/trtexec and matching DLLs, Nsight Systems2025.1.3, and the Syzygy3-5-piece tables. Python ORT1.31.1 is CPU preparation, a separate installation. Exact native file/table identities are in data/ultimate/native-inputs.json and tablebases.json. Native installations are declared prerequisites, not a hidden source/model dependency. The tested same-host paths were:

- lc0: C:\Users\HP\Desktop\LC0\.runtime\tools\lc0\lc0.exe
- trtexec: C:\Users\HP\Desktop\LC0\.runtime\tools\TensorRT-10.12.0.36\bin\trtexec.exe
- nsys: C:\Users\HP\Desktop\LC0\.runtime\tools\nsys\nsight_systems-windows-x86_64-2025.1.3.140-archive\nsight-systems\2025.1.3\target-windows-x64\nsys.exe
- syzygy: C:\Users\HP\Desktop\LC0\.runtime\syzygy

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
