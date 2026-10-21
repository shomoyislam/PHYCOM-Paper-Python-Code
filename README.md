# Feature-Wise SNR Conditioning for Jakes-Fading MIMO Channel Estimation

Code, experimental results, and reproducibility records for:

**Feature-Wise SNR Conditioning for Jakes-Fading MIMO Channel Estimation: Controlled Benefits and Architectural Limits**

**Journal:** Physical Communication  
**Original submission:** September 16, 2026

## Start here

**The completed major-revision experiments are in [revision_v4/](revision_v4/).**

- [V4 README and reproduction instructions](revision_v4/README.md)
- [V4 results summary](revision_v4/Paper_Data/RESULTS_SUMMARY.md)
- [V4 result tables](revision_v4/Paper_Data/)
- [V4 figures](revision_v4/Figures/)
- [V4 source code](revision_v4/source/)
- [V4 experimental protocol](revision_v4/source/PROTOCOL.md)
- [Reviewer-to-experiment map](revision_v4/source/REVIEWER_MAP.md)
- [V4 completion audit](revision_v4/FINAL_AUDIT.json)

## Versions and repository layout

| Version | Purpose | Location |
|---|---|---|
| V3.2 | Original submitted study and its reproducibility record | Root-level `Paper_Data/`, `Figures/`, `source/`, JSON records, and `PHYCOM_V3_2_CODE/` |
| V4 | Additional experiments conducted for the major revision | `revision_v4/` |

The original V3.2 files are retained. V4 is a separately identified study with fresh training/data seeds and a frozen confirmation protocol. Do not combine tables from different versions without explicitly identifying their origin.

V4 concentrates on CNN–LSTM and raw-LSTM controls. Earlier Mamba, GRU, sequence-window, and other V3.2 experiments were not all retrained in V4.

## V4 experiment status

The recorded publication run completed on October 20, 2026 (UTC):

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

The final session records no error. The authoritative configuration, environment, and run identity are available in [config.json](revision_v4/config.json), [RUN_MANIFEST.json](revision_v4/RUN_MANIFEST.json), and [SESSION_STATUS.json](revision_v4/SESSION_STATUS.json).

Run ID:

```text
2c9a7bc7073a1db1c0c5dec4e8285ce09847aa04a4858aa1da24d7fca458bd89
```

Completion refers to the scheduled computations. It does not establish convergence of every fit or universal superiority over classical estimators.

## What V4 evaluates

- Matched gated/ungated CNN–LSTM and raw-LSTM comparisons for 2×2, 3×3, and 4×4 channels.
- Separate projection-width, recurrent-width, and output-head controls.
- Conditioning initialization controls and LS-residual estimators.
- LS, matched-statistics LMMSE, and empirical LMMSE baselines.
- Synthetic spatial correlation and unknown or held-out channel statistics.
- SNR reporting errors, clipping, extrapolation, and calibration overhead.
- Representation spectra, paired statistical comparisons, convergence summaries, and isolated timing.

See the [protocol](revision_v4/source/PROTOCOL.md) for definitions and the [results summary](revision_v4/Paper_Data/RESULTS_SUMMARY.md) for measured outcomes. Favorable, adverse, and null results are retained.

## Reading the data

V4 tables are in [revision_v4/Paper_Data/](revision_v4/Paper_Data/). The main files are:

| File | Use |
|---|---|
| `summary_mean_std.csv` | Mean NMSE and variation across seeds |
| `metrics_per_seed.csv` | Individual seed/case/method results |
| `paired_comparisons.csv` | Matched effects and statistical comparisons |
| `convergence.csv` | Training budget, validation selection, and stopping summaries |
| `representation_diagnostics.csv` | Rank and conditioning diagnostics |
| `reporting_errors_paired.csv` | Paired effects of SNR reporting errors |
| `evaluation_coverage.csv` | SNR coverage and extrapolation information |
| `latency.csv` | Isolated inference timing |
| `covariance_fitting_cost.csv` | Empirical covariance estimation cost |
| `linear_filter_setup_cost.csv` | Linear-filter setup cost |

### Compressed singular-value spectra

The full spectra table is stored in [singular_value_spectra.zip](revision_v4/Paper_Data/singular_value_spectra.zip).

**Extract that ZIP into `revision_v4/Paper_Data/` before using the spectra data.** The resulting path should be:

```text
revision_v4/Paper_Data/singular_value_spectra.csv
```

Compression changes only storage; the underlying CSV is preserved.

## Reproducing V4

Read [revision_v4/README.md](revision_v4/README.md) and [source/PROTOCOL.md](revision_v4/source/PROTOCOL.md) before running experiments. Existing result tables and figures can be inspected without retraining.

With the required Python/PyTorch environment, run from the repository root:

```bash
cd revision_v4/source
python -m unittest discover -s tests -v
python run_revision.py --plan-only
python run_revision.py --preflight-only --devices 0,1
python run_revision.py --devices 0,1 --output /kaggle/working/PHYCOM_V4_RUN --hours 10
```

The completed study used two independent Tesla T4 workers. Its recorded environment includes Python 3.13.15, PyTorch 2.11.0+cu128, CUDA 12.8, NumPy 2.1.3, SciPy 1.16.3, pandas 2.3.3, and Matplotlib 3.10.0. Use the [run manifest](revision_v4/RUN_MANIFEST.json) for the complete record.

The ten-hour argument is a session work budget, not a guaranteed completion time. Read the source README for checkpoint/resume instructions. Smoke-test outputs are diagnostic and must not be used as research evidence.

## Original V3.2 study

The original study remains available through:

- [V3.2 Kaggle notebook](PHYCOM_V3_2_KAGGLE_FINAL.ipynb)
- [V3.2 source and run instructions](PHYCOM_V3_2_CODE/README.md)
- [V3.2 mathematical conventions](PHYCOM_V3_2_CODE/METHODS.md)
- [V3.2 validation report](PHYCOM_V3_2_CODE/TEST_REPORT.md)
- [Original result tables](Paper_Data/)
- [Original figures](Figures/)
- [Original audit](FINAL_AUDIT.json)
- [Original environment record](requirements_frozen.txt)

The V3.2 notebook targets its own recorded environment and includes official Mamba dependencies. Its resume archives and settings are different from V4. Use the instructions for the intended version.

## Interpretation and limitations

- Primary V4 inference uses paired two-sided sign tests with Holm correction across 12 comparisons. Supplementary five-seed studies are exploratory.
- Model selection was frozen before confirmation evaluation. Later tuning requires a separately documented analysis; previously inspected confirmation data cannot be described as untouched.
- Twenty-six V4 fits reached the 240-epoch cap. One 4×4 ungated raw-LSTM fit (seed 31079) showed approximately 1.58% recent validation improvement. These are results under a specified training budget; full convergence is not established.
- Synthetic channel simulations do not establish measured-channel or massive-MIMO performance.
- Calibration-based SNR estimation uses additional observations. The rate proxy omits some acquisition and training overhead.
- Inference timing must be interpreted alongside batch size, hardware, covariance estimation, and filter setup costs.
- Any longer-budget follow-up should preserve the original V4 results and be identified separately.

## Raw outputs and checkpoints

The large `PHYCOM_V4_PUBLICATION_RESUME.zip` is retained separately as a backup. It contains training histories, model/optimizer checkpoints, and raw evaluation arrays. Those files are not part of this smaller GitHub review package.

The audit's raw-hash verification records checks performed during the original run. The review package does not include the raw arrays needed to independently repeat those hash checks.

## Referencing this repository

When referring to these materials, identify the manuscript title, experiment version (V3.2 or V4), and the exact repository commit or release used. Do not describe this revision package as an accepted publication or invent a DOI.

Repository: https://github.com/shomoyislam/PHYCOM-Paper-Python-Code

Consult the repository's license file, if present, for reuse terms. This README does not grant an additional license.

