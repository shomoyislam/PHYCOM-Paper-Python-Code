# V4 revision experiments: SNR-conditioned MIMO channel estimation

This folder contains the completed V4 experiment outputs and source code for the major revision of manuscript **PHYCOM-D-26-02646**:

**Feature-Wise SNR Conditioning for Jakes-Fading MIMO Channel Estimation: Controlled Benefits and Architectural Limits**

Authors: Md Shaikhul Islam Shomoy and Ziqiang Hao  
Corresponding author: shomoyislampabna@gmail.com  
Repository: https://github.com/shomoyislam/PHYCOM-Paper-Python-Code

## Relationship to the submitted manuscript

The original V3.2 files elsewhere in this repository document the study submitted on **September 16, 2026**. This `revision_v4/` folder contains additional experiments conducted for the major revision. Preserve both versions and identify which version supports each reported result.

V4 focuses on CNN–LSTM and raw-LSTM controls. Earlier Mamba, GRU, and other V3.2 experiments were not retrained in this suite. The results here do not by themselves constitute a revised manuscript or a completed response to reviewers.

## Contents

| Path | Description |
|---|---|
| `source/` | Python implementation, tests, protocol, and reviewer-to-experiment map |
| `Paper_Data/` | Result tables, statistical comparisons, convergence summaries, diagnostics, and timing measurements |
| `Figures/` | Generated PDF figures |
| `config.json` | Scientific configuration and training budget |
| `EXPERIMENT_PLAN.json` | Planned experiment groups and models |
| `RUN_MANIFEST.json` | Run identity, source hash, and numerical environment |
| `FROZEN_SELECTION.json` | Validation-selected model records frozen before confirmation evaluation |
| `FINAL_AUDIT.json` | Completion and consistency checks |
| `SESSION_STATUS.json` | Final session status |
| `SESSIONS.json` | Session history |
| `README.md` | This guide |

Start with [the generated results summary](Paper_Data/RESULTS_SUMMARY.md), [the protocol](source/PROTOCOL.md), and [the reviewer map](source/REVIEWER_MAP.md).

## Compressed singular-value table

The large `singular_value_spectra.csv` table is distributed as `Paper_Data/singular_value_spectra.zip` for browser uploading.

**Extract `Paper_Data/singular_value_spectra.zip` into `Paper_Data/` before using the singular-value spectra data.** The resulting path should be:

```text
revision_v4/Paper_Data/singular_value_spectra.csv
```

If the CSV is already present, no extraction is needed. The ZIP is a storage convenience; the CSV contents must remain unchanged.

## Completed run

The saved audit records a publication-mode run completed on **October 20, 2026 (UTC)**:

| Item | Recorded result |
|---|---:|
| Completed / planned fits | 430 / 430 |
| Central independent training/data seeds | 10 |
| Supplementary seeds | 5 |
| Metric rows | 43,000 |
| Paired-comparison rows | 11,586 |
| Primary statistical tests | 12 |
| Confirmation evaluation | Complete |
| Timing evaluation | Complete |
| Final session error | None |

Run ID:

```text
2c9a7bc7073a1db1c0c5dec4e8285ce09847aa04a4858aa1da24d7fca458bd89
```

The run used two independent Tesla T4 workers for experiment groups. Recorded environment: Python 3.13.15, PyTorch 2.11.0+cu128, CUDA 12.8, cuDNN 91900, NumPy 2.1.3, SciPy 1.16.3, pandas 2.3.3, and Matplotlib 3.10.0. Consult `RUN_MANIFEST.json` for the authoritative record.

Completion means that the scheduled tasks finished. It does not establish convergence of every model or universal superiority of neural estimators.

## Experimental scope

The suite includes:

- 2×2, 3×3, and 4×4 channel configurations.
- Matched gated and ungated CNN–LSTM and raw-LSTM comparisons.
- Separate projection-width, recurrent-width, and output-head controls.
- Conditioning initialization controls and LS-residual outputs.
- LS, matched-statistics LMMSE, and training-data empirical LMMSE baselines.
- Synthetic spatial correlation and unknown/held-out channel-statistics cases.
- SNR reporting errors, clipping, extrapolation, and noise-calibration tests.
- Representation spectra, paired statistical comparisons, and isolated inference timing.

Training uses a maximum of 240 epochs with validation-based checkpoint selection and early stopping. The full settings are in `config.json`.

## Statistical interpretation and limitations

- The 12 primary comparisons use paired two-sided sign tests with Holm correction. These concern directional consistency across training runs; they are not tests of the population mean.
- Seed-bootstrap intervals are descriptive. Five-seed supplementary studies remain exploratory.
- Confirmation results were evaluated after model selection was frozen. Subsequent tuning must not reuse these observations as an untouched confirmation set.
- Results support conditional conditioning benefits. Larger-array variability, classical-baseline performance, and adverse or null results must remain visible.
- Synthetic Jakes and spatial channel simulations do not establish performance on measured channels or massive-MIMO deployments.
- Calibration-based SNR estimation uses 64 additional noise-only complex samples per window. The reported rate proxy does not include all pilot, calibration, or training overhead.
- Timing results require their batch size, hardware, and measurement scope to be stated. Covariance estimation and filter setup costs are recorded separately.

### Convergence qualification

Of 430 fits, **26 reached the 240-epoch limit**. One exceeded the configured 1% recent-validation-gain threshold:

- Group: `iid_4x4_seed31079`
- Model: `raw_p0_h32_head32_none`
- Best validation MSE: 0.0436955444 at epoch 238
- Improvement relative to the best through epoch 220: approximately 1.58%

Report these results under the specified training budget. Full convergence and an inherent architectural limitation are not established by this audit. Any longer-budget follow-up should be documented separately, use matched controls, and preserve the original V4 record.

## Reproducing the experiments

The exported tables and figures can be inspected without rerunning training. A new full experiment is computationally substantial.

### Source-based execution

From the repository root:

```bash
cd revision_v4/source
python -m unittest discover -s tests -v
python run_revision.py --plan-only
python run_revision.py --preflight-only --devices 0,1
python run_revision.py --devices 0,1 --output /kaggle/working/PHYCOM_V4_RUN --hours 10
```

These commands require a suitable Python/PyTorch environment. `source/requirements.txt` lists broad dependencies; `RUN_MANIFEST.json` records the versions used for the completed run. Use the installed GPU PyTorch build on Kaggle rather than indiscriminately upgrading it.

The `--hours 10` option is a session work budget, not a guarantee that the entire study finishes in ten hours. Multiple sessions may be needed. See [source/README.md](source/README.md) for resume and smoke-test instructions.

### Kaggle notebook

If `PHYCOM_V4_KAGGLE_T4x2.ipynb` has also been added to this folder, import it into Kaggle, select **GPU T4 ×2**, and use `SMOKE_TEST = False` for a publication run. Smoke-test outputs are pipeline checks and must not be reported as research results.

The notebook embeds its source; modifying `source/` alone does not update the embedded notebook code. Keep the notebook and source versions consistent.

## Checkpoints and raw outputs

This review package contains source code and processed results. The large `PHYCOM_V4_PUBLICATION_RESUME.zip` is retained separately as a backup and is not included in this folder. It contains model checkpoints, optimizer states, detailed training histories, and raw evaluation arrays.

The `raw_hashes_verified` audit field records a check performed during the original run; this smaller review package does not contain those raw arrays for independent rechecking. Contact the corresponding author about access to additional artifacts.

## Version and citation information

When using these materials, identify **V4**, the run ID above, and the repository commit used. Retain the V3.2 submission record separately. Consult the repository's license file, if present, for reuse terms; this README does not grant an additional license.

