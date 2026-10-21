# PHYCOM R1 experiment package — V4.0

For manuscript PHYCOM-D-26-02646, submitted 16 September 2026. This is executable
code for new revision experiments, not completed publication results. The
experiment plan responds to the three supplied reviewer reports dated September
and October 2026. Read REVIEWER_MAP.md and PROTOCOL.md before interpreting results.

## Start on Kaggle

1. Import **PHYCOM_V4_KAGGLE_T4x2.ipynb** into a new Kaggle notebook.
2. Select **GPU T4 x2** in the notebook accelerator settings.
3. Leave `SMOKE_TEST = False` for the research run. Choose `SESSION_HOURS` shorter
   than the session time and GPU quota shown in your own Kaggle account. The
   default is 10 work hours; leave at least an hour for reporting and archives.
4. Run all cells. The notebook already embeds every V4 Python source file; no
   external code download, Mamba wheel installation or input dataset is needed
   for the first session. It uses the GPU PyTorch provided by Kaggle.
5. Keep the output notebook version and download the **RESUME.zip** after each
   session. For the next session, attach that archive as a **private Kaggle
   dataset**, leave `RESUME_INPUT = "auto"`, and run the same notebook again.
   If multiple resume files are attached, give the exact path to the intended
   archive. An extracted previous output directory also works.
6. At completion download **PHYCOM_V4_PUBLICATION_REVIEW.zip** and
   **PHYCOM_V4_PUBLICATION_RESUME.zip**. The small REVIEW archive contains the
   tables, figures, plan, audit and source. RESUME also contains selected weights,
   optimizer checkpoints and independent-window error arrays.

The scheduler uses one process per GPU and dynamically assigns independent
training groups. It keeps small training windows on each GPU and avoids repeated
host transfers. It does not use DataParallel/DDP or change the matched batch size
to manufacture a speed increase. More GPU utilization is not a scientific
outcome. Speed depends on the selected models and Kaggle hardware allocation;
no fixed completion time or twofold speedup is promised.

The full plan has **430 fits**: ten independent training/data seeds for the
central iid comparisons, five for the supplementary initialization and
physical/practical studies. It may require several sessions. Checkpoints commit
after every epoch. If interrupted within an epoch, that epoch restarts with the
same shuffle. Do not stop the notebook during final archive creation. A hard
session kill cannot guarantee a final ZIP; retain a saved output version or the
complete output directory where available.

## Smoke test

Set `SMOKE_TEST = True` for a short pipeline check with 108 tiny two-epoch fits.
Its separate output directory/archive names contain `SMOKE`. These outputs must
never be used as scientific evidence. Change back to False to start the formal
study; smoke checkpoints cannot be imported into the publication run.

Local CPU validation:

```bash
python -m unittest discover -s tests -v
python run_revision.py --smoke --devices cpu --output ./SMOKE_RUN --resume-input '' --hours 0.25
```

Command-line publication run after extracting the source ZIP:

```bash
python run_revision.py --devices 0,1 --output /kaggle/working/PHYCOM_V4_RUN --hours 10
```

Run `python run_revision.py --plan-only` to inspect the full plan without starting
training. Run with `--preflight-only` to check CUDA forward/backward compatibility.

## What to return for the manuscript revision

Send **REVIEW.zip**, preferably with **RESUME.zip** retained as the reproducibility
record. Check FINAL_AUDIT.json and SESSION_STATUS.json first. `complete: true`
means the scheduled tasks finished, not that neural estimators beat LMMSE or the
journal will accept the paper. A run that hits the epoch cap with recent
validation gains needs scientific review before claiming convergence. Do not
pick only favorable seeds or cases.

The outputs needed for a response letter are:

| File in Paper_Data/ | Use |
|---|---|
| summary_mean_std.csv | Full SNR curves and variability across fitted runs |
| paired_comparisons.csv | Gated/ungated, raw/CNN, width, initialization, LS and LMMSE comparisons |
| convergence.csv | Per-fit epoch history summary, validation choices, elapsed training cost |
| representation_diagnostics.csv | Effective ranks, numerical ranks, domain nullities and conditioning |
| singular_value_spectra.csv | Every saved weight/feature singular value |
| reporting_errors_paired.csv | Paired error-versus-accurate-report and clipped-versus-unclipped effects |
| evaluation_coverage.csv | True/reported SNR ranges and extrapolation fractions |
| latency.csv | Isolated same-device inference measurements |
| covariance_fitting_cost.csv / linear_filter_setup_cost.csv | Costs excluded from cached-filter inference |
| RESULTS_SUMMARY.md | Automatically generated results and limits |

Raw `jobs/.../evaluation/.../*.npz` files retain per-independent-window squared
errors, channel powers, correction terms, and diagnostic BER/rate blocks.
Selected `best.pt`, resumable `last.pt`, and complete `training.json` histories
are in each model folder. These are included in RESUME.zip.

## Preserve the submitted study

V4 recreates the original LSTM architecture and simulator before adding controls.
The tests compare original and new initialization/predictions exactly for the
overlapping configurations. Fresh seeds/data intentionally make this a new
confirmation study. The original V3.2 files are unchanged, and the source ZIP
includes `reference_v32/` as a reference copy.

The new suite focuses on the reviewers' central CNN/raw-LSTM issues. It does
not retrain the earlier Mamba/GRU, long-sequence, or other V3.2 experiments.
Retain and label those earlier results separately. This package does not revise
the manuscript wording or add Reviewer 3's suggested references. Those tasks
must follow the new evidence and verification of the cited papers.

## Reproducibility and failure handling

The run ID covers source, complete scientific configuration, and numerical
runtime versions. A changed source/runtime cannot silently resume an old run.
Keep the same Kaggle environment; if Kaggle updates a numerical package, use
the saved environment or start a separate run. Do not combine different runs.
`runtime` in RUN_MANIFEST.json records the relevant versions and GPU model.
FP32, fixed component seeds, deterministic algorithms and a cuBLAS workspace
configuration are used. This is not a guarantee of bitwise reproducibility on
different devices or PyTorch releases.

CUDA forward/backward preflight runs on every selected GPU. The local delivery
was checked on CPU; the preflight is not a claim that T4 training was already
run here. If an error occurs, keep the printed log and failure archive, and
return them for diagnosis rather than removing the checks.

The complete selection is frozen before any confirmation evaluation. Changing
the protocol after inspecting these results makes later work exploratory unless
a new untouched confirmation set is used. No test result is used to select a
checkpoint, seed, residual scaling coefficient or covariance shrinkage value.
