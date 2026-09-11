# Development verification — 2026-09-09

This report distinguishes local checks from the full GPU experiment that must run on Kaggle.

## Passed locally

- **27 tests** across the mathematical, installer, checkpoint and validation suites. The original 16 scientific/setup checks and 11 V3.2 validation checks passed.
- A complete **CPU smoke study with 28 non-Mamba fits**, including original and wide heads, separate budget/extended evaluation, all smoke mismatch conditions and two sequence windows.
- The smoke study exported **574 metric rows**, **778 paired-comparison rows**, compact LaTeX tables, convergence/head diagnostics, and **54 PDF figures**. Every saved scalar NMSE, BER and rate matched recomputation from its raw arrays. All nine final audit checks passed. Smoke numerical values are not publication results.
- Interrupted CPU training resumed the identical selected-weight trajectory. A crash after the atomic epoch write but before its selected-checkpoint exports was repaired on restart.
- Direct continuation from a legacy-format optimizer checkpoint matched an uninterrupted CPU trajectory exactly. The reference-budget checkpoint remained unchanged.
- Import of the user's **actual 93 V3.1 trained states** from the uploaded result ZIP was exercised. All 93 budget references retained identical selected tensors, epochs and validation values. This verified migration and provenance, not GPU training execution.
- A deliberately interrupted evaluation archive restored through the command-line runner, evaluated the missing case, and reproduced **all 574 metric rows exactly**, without retraining.
- A forced short invocation budget returned the designated pause status and produced a resumable partial ZIP.
- Tests verified the head-width comparison's shared initialization, fresh test namespace, unchanged observations under SNR-side-information errors, ungated invariance to that scalar, and equality of reported-SNR and true-SNR LMMSE at zero reporting error.
- A three-seed synthetic contrast verified that training-seed uncertainty remains distinct from Monte Carlo uncertainty conditional on fitted models.
- The installer, official wheel manifest and channel simulation files were checked byte-for-byte against the audited V3.1 source. They are unchanged.
- Representative head-control and SNR-bias plot layouts were visually inspected for readable axes and legends.
- The delivered notebook passed nbformat validation and compilation of every code cell. Its embedded source matched the companion GitHub package. Every notebook code cell was executed in Python against a completed CPU smoke fixture, exercising extraction, subprocess launch, completed-run reuse and result links. This did not use a Kaggle browser or a locally hosted Jupyter kernel.

## Runs on the user's GPUs before full training

The notebook verifies both official CUDA extensions, convolution against a PyTorch reference, Mamba forward/backward passes at T = 3,5,20,256 and fused/unfused consistency. It additionally runs actual CNN–LSTM, wide-head CNN–LSTM and official Mamba estimator training/Adam-resume checks on each detected GPU, including optimizer-state device placement.

No CUDA device is available in the development environment. The full 129-fit experiment, its runtime, convergence and accuracy have **not** been measured here. The maximum training budget and extra controls improve the evidence available for writing; they do not guarantee a better NMSE, a preferred model ranking or PHYCOM acceptance.
