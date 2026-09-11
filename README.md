# PHYCOM V3.2 — controlled confirmation study

A self-contained Kaggle experiment for feature-wise nominal-SNR conditioning in time-varying MIMO channel estimation. This revision builds on the user's audited V3.1 run and produces new evidence for writing the manuscript. It does not promise a particular ranking or journal acceptance.

## Kaggle: one Run All

1. Import `PHYCOM_V3_2_KAGGLE_FINAL.ipynb` into a fresh Kaggle notebook.
2. Select **GPU T4 ×2** and enable **Internet**. Keep the default notebook settings for the full experiment.
3. Optional: attach your own `PHYCOM_V3_1_PUBLICATION_RESULTS (1).zip` as a Kaggle input. The notebook detects the audited run and, when its numerical runtime matches, reuses its 93 trained states. With no compatible archive, it trains from scratch.
4. Choose **Run All**. The project code is embedded in the notebook; no separate code upload is needed.
5. Download `PHYCOM_V3_2_PUBLICATION_RESULTS.zip`. Start with `Paper_Data/RESULTS_SUMMARY.md`, `convergence.csv`, and `FINAL_AUDIT.json`.

The installer targets your successful Python 3.12 / PyTorch 2.10.0+cu128 / CUDA 12.8 T4 environment. It retains official Mamba-1 from `mamba-ssm==2.3.2.post1` and `causal-conv1d==1.6.2.post1`, with checked official binary wheels. It protects the existing core numerical stack and performs CUDA forward/backward and actual estimator optimizer-resume checks before training.

The complete default experiment contains **129 fits**, up to **240 total epochs** each, with validation patience 10. It can take several hours. The run pauses at a safe boundary after a default 10.5-hour invocation budget, leaving time to create a partial archive; completion within one Kaggle session is not guaranteed. Set the budget lower if the current session is already old. Checkpoints are committed each epoch. An abrupt Kaggle shutdown cannot be recovered unless the outputs were retained or downloaded.

## What changed

- All original 13 main models are retained for both 2×2 and 4×4 arrays and seeds 42, 123, 2024: **78 fits**.
- The CNN–LSTM head is widened from 32 to 64 units for 4×4 only, with otherwise identical gated and ungated designs: **6 additional fits**. ReLU, temporal width, data, optimizer and shared component initialization remain matched.
- All five sequence windows, T = 3, 5, 10, 15, 20, now use three seeds for LSTM-gate, official Mamba-gate and LS-residual-gate: **45 fits**. Each uses 40,000/8,000 training/validation samples and the same stopping policy. Sequence accuracy is evaluated at 15 dB.
- The best validation checkpoint within the original 60-epoch budget is saved separately from the extended best checkpoint. Both are evaluated on the same fresh test set for main experiments.
- Gated and ungated LSTM, Mamba and LS-residual estimators receive the same temporal, spatial and SNR-side-information stress tests. The wide-head pair is included too.
- An additional Jakes baseline uses the same erroneous reported SNR in side-information tests. True-SNR classical references are explicitly identified in raw rows.
- Training for all models finishes and checkpoint hashes are frozen **before** confirmation data is generated. No model is selected using test NMSE, BER, rate or stress results.
- Classical estimates are simulated once per array/window. Their Monte Carlo uncertainty is not represented as zero training-seed variability.
- Separate paired Monte Carlo and training-seed intervals, convergence diagnostics, observed head-activity diagnostics and raw-statistic audits are exported.

## Resume and provenance

To continue an interrupted **V3.2** session, attach its `_PARTIAL.zip` to a new Kaggle notebook and set `RESUME_ARCHIVE` to that input path. The same notebook restores the archive and checks code, configuration and numerical-runtime identity. An existing matching output directory resumes automatically. Do not put a V3.1 ZIP in `RESUME_ARCHIVE`; the `PARENT_V31` setting handles that separate migration.

V3.1 is opened read-only. Compatible old model, optimizer, scheduler, training history and random-number states are imported under an explicitly new run identity. Parent run and checkpoint hashes are retained. Completed early stopping remains respected. Original test results are not mixed into confirmation summaries. New wide-head models and extra sequence seeds start from scratch. An incompatible optional parent runtime triggers a fresh run, recorded in the manifest; an incompatible V3.2 resume is rejected.

If a central fit remains at the epoch cap with improving validation loss, the report flags it. The notebook does not silently reset patience, search new hyperparameters, select a lucky seed or hide a stronger classical baseline.

## GitHub / command line

The notebook extracts the same readable project contained in the companion ZIP. Upload the source files, tests, documentation and notebook to your repository. Large results and checkpoints can be distributed separately; `.gitignore` excludes generated output, CUDA wheel caches and credentials.

```bash
# Full experiment (two GPUs used when present)
python run_phycom.py --output /kaggle/working/PHYCOM_V3_2_PUBLICATION

# Optional audited parent archive
python run_phycom.py --parent '/path/to/PHYCOM_V3_1_PUBLICATION_RESULTS (1).zip' --output new_run

# Restore a matching V3.2 run
python run_phycom.py --resume '/path/to/PHYCOM_V3_2_PUBLICATION_PARTIAL.zip' --output restored_run

# Small CPU diagnostic; NOT manuscript results
python run_phycom.py --smoke --no-mamba --parent none --session-hours 0 --output smoke_output

# Mathematical, installer and checkpoint tests
python -m unittest discover -s tests -v
```

Normal full mode requires official CUDA Mamba. There is no alternative model silently substituted if Mamba installation fails. `--skip-setup` skips installation but keeps full-mode CUDA checks. Dependencies are installed into the running Python executable used by the notebook.

## Output map

| File or folder | Use |
|---|---|
| `Paper_Data/RESULTS_SUMMARY.md` | Computed results and manuscript interpretation |
| `Paper_Data/summary_mean_std.csv` | All main, stress and sequence summaries |
| `Paper_Data/paired_comparisons.csv` | Gating, head width and training-extension contrasts |
| `Paper_Data/convergence.csv` | Selected epochs, stopping reason and recent validation gains |
| `Paper_Data/head_diagnostics.csv` | Observed inactive head units and output-span diagnostics |
| `Paper_Data/complexity.csv`, `latency_vs_T.csv` | Isolated timing and explicitly approximate arithmetic counts |
| `Paper_Data/nmse_*.tex` | Compact computed LaTeX tables at 0, 15, 30 dB |
| `Figures/` | PDF and PNG plots, plus caption guidance |
| `jobs/` | Best, 60-epoch-reference and resumable checkpoints; raw block statistics |
| `frozen_selection.json` | Validation-selected checkpoint hashes recorded before testing |
| `FINAL_AUDIT.json` | Completeness and recomputation checks; not scientific acceptance |
| `source/`, `requirements_frozen.txt`, `run_manifest.json` | Exact code, installed environment and run identity |

See `METHODS.md` for mathematical conventions and `TEST_REPORT.md` for the scope of local verification.

## Primary implementation references

- [Official Mamba 2.3.2.post1 release](https://github.com/state-spaces/mamba/releases/tag/v2.3.2.post1)
- [Official causal-conv1d 1.6.2.post1 release](https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.6.2.post1)
- [PyTorch 2.10 optimizer-state loading](https://docs.pytorch.org/docs/2.10/generated/torch.optim.Optimizer.load_state_dict.html)
- [PyTorch 2.10 checkpoint serialization](https://docs.pytorch.org/docs/2.10/notes/serialization.html)
