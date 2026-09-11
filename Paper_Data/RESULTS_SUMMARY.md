# Results guide for the manuscript

Mode: **publication**. Completed 129 fits. The maximum is 240 epochs; 8 fits reached that limit and 8 require a convergence review. A patience stop does not prove a global optimum. A capped fit can only support a stated training-budget result.

All designs and validation-selected checkpoints were frozen before fresh test generation. The 60-epoch snapshot is the best validation checkpoint within that budget; it is not necessarily epoch 60 itself. Extended and budget60 predictions use the same new test channels.

## Main NMSE at 30 dB

| Method | Array | Mean NMSE | Uncertainty |
|---|---|---:|---|
| Jakes LMMSE (fixed covariance) | 2×2 | 0.00096151 | 95% MC half-width 9.23e-06 |
| Jakes LMMSE (matched covariance) | 2×2 | 0.00096151 | 95% MC half-width 9.23e-06 |
| AR(1) Kalman | 2×2 | 0.000998789 | 95% MC half-width 9.59e-06 |
| Instantaneous LMMSE | 2×2 | 0.00100061 | 95% MC half-width 9.6e-06 |
| LS | 2×2 | 0.00100147 | 95% MC half-width 9.62e-06 |
| CNN + gate | 2×2 | 0.00191566 | Seed std 0.000533 |
| CNN–GRU + gate | 2×2 | 0.00285096 | Seed std 4e-05 |
| CNN–GRU | 2×2 | 0.0040575 | Seed std 0.000218 |
| LS + gated LSTM correction | 2×2 | 0.00133473 | Seed std 0.000106 |
| LS + LSTM correction | 2×2 | 0.00252203 | Seed std 0.000137 |
| CNN–LSTM + SNR concat | 2×2 | 0.00397899 | Seed std 0.000132 |
| CNN–LSTM + FiLM | 2×2 | 0.00646468 | Seed std 0.000521 |
| CNN–LSTM + gate | 2×2 | 0.00283394 | Seed std 0.000132 |
| CNN–LSTM | 2×2 | 0.00395207 | Seed std 0.00024 |
| CNN–LSTM + residual gate | 2×2 | 0.00261166 | Seed std 9.96e-05 |
| CNN–Mamba + gate | 2×2 | 0.00259011 | Seed std 0.000289 |
| CNN–Mamba | 2×2 | 0.00302355 | Seed std 6.89e-05 |
| Raw LSTM + gate | 2×2 | 0.00256042 | Seed std 7.74e-05 |
| Jakes LMMSE (fixed covariance) | 4×4 | 0.000957918 | 95% MC half-width 4.64e-06 |
| Jakes LMMSE (matched covariance) | 4×4 | 0.000957918 | 95% MC half-width 4.64e-06 |
| AR(1) Kalman | 4×4 | 0.000995464 | 95% MC half-width 4.81e-06 |
| Instantaneous LMMSE | 4×4 | 0.00099716 | 95% MC half-width 4.82e-06 |
| LS | 4×4 | 0.000998057 | 95% MC half-width 4.83e-06 |
| CNN + gate | 4×4 | 0.0131909 | Seed std 0.0185 |
| CNN–GRU + gate | 4×4 | 0.0263273 | Seed std 0.0148 |
| CNN–GRU | 4×4 | 0.0370008 | Seed std 0.000648 |
| LS + gated LSTM correction | 4×4 | 0.000999835 | Seed std 1.04e-06 |
| LS + LSTM correction | 4×4 | 0.00147317 | Seed std 0.000135 |
| CNN–LSTM + SNR concat | 4×4 | 0.0788878 | Seed std 0.0181 |
| CNN–LSTM + FiLM | 4×4 | 0.0812895 | Seed std 0.0178 |
| CNN–LSTM + gate | 4×4 | 0.0784264 | Seed std 0.0359 |
| CNN–LSTM | 4×4 | 0.0371179 | Seed std 0.0013 |
| CNN–LSTM + residual gate | 4×4 | 0.037062 | Seed std 0.000332 |
| CNN–LSTM + gate (head 64) | 4×4 | 0.0277243 | Seed std 0.0171 |
| CNN–LSTM (head 64) | 4×4 | 0.0398202 | Seed std 0.000424 |
| CNN–Mamba + gate | 4×4 | 0.0384256 | Seed std 0.000292 |
| CNN–Mamba | 4×4 | 0.0682331 | Seed std 0.0324 |
| Raw LSTM + gate | 4×4 | 0.00708141 | Seed std 0.000165 |

## How to write the conclusions

- Use `paired_comparisons.csv` for gating, head width, and training-extension differences. Negative NMSE/BER differences favor the candidate. Each pair uses identical test channels and data-link random draws.
- Report all three seeds, including reversals. Monte Carlo intervals condition on fitted models; the separate seed-t interval conditions on the finite test set and is imprecise with three seeds. These are pointwise intervals without multiplicity correction.
- Keep the fixed Jakes LMMSE baseline in every main accuracy comparison. Under the matched scenario its covariance is already matched; it is not disadvantaged by an unknown test covariance.
- Compare wide gated against wide ungated, and each wide model against its original-width counterpart. Changing width tests one architectural factor; it does not by itself identify an attenuation mechanism.
- Head diagnostics show units inactive on one observed test set and their output span. They do not establish that a unit was always inactive during training.
- Robustness claims require gated/ungated pairs at the same mismatch. SNR-error plots include a classical estimator using the same reported SNR.
- Sequence experiments use three seeds per window and the same train/validation counts. They evaluate accuracy at 15 dB. Timing is measured after concurrent training has stopped, at a fixed batch size.
- Describe the payload BER as a frozen-channel diagnostic. The effective ZF Gaussian-input rate excludes pilot overhead and is not measured QPSK throughput.
- Read `convergence.csv` before using the word converged. Keep the 240-epoch cap visible if any central fit still improves. Do not tune again on this confirmation set to obtain a preferred ranking.

## Files to use

`summary_mean_std.csv`, `paired_comparisons.csv`, `convergence.csv`, `head_diagnostics.csv`, `complexity.csv`, `latency_vs_T.csv`, the compact `.tex` tables, and `Figures/CAPTIONS.md`. All scalar metrics are traceable to saved raw per-channel statistics. `FINAL_AUDIT.json` checks completeness and arithmetic only.
