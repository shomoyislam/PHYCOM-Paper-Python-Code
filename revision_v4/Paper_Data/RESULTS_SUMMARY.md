# PHYCOM V4 revision results

Mode: **publication**. Complete: **True**.
Fits: 430/430; central independent seeds: 10; supplemental seeds: 5.
Epoch-cap fits: 26; capped fits with >1% recent validation gain: 1.

All confirmation results are generated after all model selections are frozen. The experiment is pre-specified in config.json; it is not an externally registered protocol. Do not tune on these confirmation results and then call the same set fresh.

Primary inference: exact two-sided sign tests on paired training-run differences with Holm across 12 predeclared tests (three arrays, CNN/raw LSTM, 15/30 dB). This tests directional consistency/median effects, not the population mean. Bootstrap intervals across seeds are descriptive; paired-t intervals are sensitivity checks. Conditional channel/BER intervals hold the fitted ensemble fixed and do not include training uncertainty. Five-seed supplementary studies remain exploratory; more fits do not replace replication.

## Main 30 dB comparisons

### 2x2

| Method | Mean NMSE | Seed SD |
|---|---:|---:|
| Empirical_LMMSE_K100000 | 0.0009648561 | 1.3e-08 |
| Empirical_LMMSE_K2048 | 0.0009649977 | 6.4e-08 |
| Empirical_LMMSE_K256 | 0.0009662031 | 1.39e-07 |
| Empirical_LMMSE_shrink001_K100000 | 0.0009866705 | 8.94e-09 |
| Empirical_LMMSE_shrink001_K2048 | 0.0009866832 | 9e-08 |
| Empirical_LMMSE_shrink001_K256 | 0.0009867196 | 1.45e-07 |
| Jakes_LMMSE_fixed_iid | 0.0009648534 | 0 |
| LS | 0.001004489 | 0 |
| Oracle_component_reportedSNR | 0.0009648534 | 0 |
| Oracle_component_trueSNR | 0.0009648534 | 0 |
| Oracle_training_distribution_LMMSE | 0.0009648534 | 0 |
| cnn_p32_h32_head16_concat | 0.003990478 | 0.000317 |
| cnn_p32_h32_head16_concat_identity | 0.003950189 | 0.000318 |
| cnn_p32_h32_head16_film | 0.006608824 | 0.000658 |
| cnn_p32_h32_head16_film_identity | 0.006228638 | 0.000354 |
| cnn_p32_h32_head16_gate | 0.002980085 | 0.00017 |
| cnn_p32_h32_head16_gate_identity | 0.003087716 | 0.000101 |
| cnn_p32_h32_head16_gate_lsres | 0.001368693 | 0.000116 |
| cnn_p32_h32_head16_gate_lsres__val_scaled | 0.001356175 | 0.000112 |
| cnn_p32_h32_head16_none | 0.004039059 | 0.000327 |
| cnn_p32_h32_head16_none_lsres | 0.002646541 | 0.000137 |
| cnn_p32_h32_head16_none_lsres__val_scaled | 0.002584289 | 0.000146 |
| cnn_p32_h32_head16_resgate | 0.002634036 | 0.000276 |
| cnn_p32_h32_head16_resgate_identity | 0.002854947 | 0.000172 |
| raw_p0_h32_head16_gate | 0.002488298 | 0.000211 |
| raw_p0_h32_head16_none | 0.003032454 | 0.000193 |

### 3x3

| Method | Mean NMSE | Seed SD |
|---|---:|---:|
| Empirical_LMMSE_K100000 | 0.0009564776 | 9.82e-09 |
| Empirical_LMMSE_K2048 | 0.0009568244 | 9.29e-08 |
| Empirical_LMMSE_K256 | 0.0009606739 | 6.37e-07 |
| Empirical_LMMSE_shrink001_K100000 | 0.0009772833 | 8.54e-09 |
| Empirical_LMMSE_shrink001_K2048 | 0.0009773081 | 4.52e-08 |
| Empirical_LMMSE_shrink001_K256 | 0.0009776057 | 1.53e-07 |
| Jakes_LMMSE_fixed_iid | 0.0009564698 | 0 |
| LS | 0.0009946036 | 0 |
| Oracle_component_reportedSNR | 0.0009564698 | 0 |
| Oracle_component_trueSNR | 0.0009564698 | 0 |
| Oracle_training_distribution_LMMSE | 0.0009564698 | 0 |
| cnn_p32_h32_head18_gate | 0.01053495 | 0.0175 |
| cnn_p32_h32_head18_none | 0.0108795 | 0.0175 |
| raw_p0_h32_head18_gate | 0.02080459 | 0.0269 |
| raw_p0_h32_head18_none | 0.02069668 | 0.0369 |

### 4x4

| Method | Mean NMSE | Seed SD |
|---|---:|---:|
| Empirical_LMMSE_K100000 | 0.0009578193 | 8.75e-09 |
| Empirical_LMMSE_K2048 | 0.000958433 | 4.55e-08 |
| Empirical_LMMSE_K256 | 0.0009683149 | 6.04e-07 |
| Empirical_LMMSE_shrink001_K100000 | 0.0009791963 | 8.48e-09 |
| Empirical_LMMSE_shrink001_K2048 | 0.0009792502 | 4.78e-08 |
| Empirical_LMMSE_shrink001_K256 | 0.0009800464 | 8.64e-08 |
| Jakes_LMMSE_fixed_iid | 0.0009578034 | 0 |
| LS | 0.0009968015 | 0 |
| Oracle_component_reportedSNR | 0.0009578034 | 0 |
| Oracle_component_trueSNR | 0.0009578034 | 0 |
| Oracle_training_distribution_LMMSE | 0.0009578034 | 0 |
| cnn_p128_h128_head32_gate | 0.0192428 | 0.0219 |
| cnn_p128_h128_head32_none | 0.02590058 | 0.0294 |
| cnn_p128_h32_head32_gate | 0.01304369 | 0.0124 |
| cnn_p128_h32_head32_none | 0.01340036 | 0.00907 |
| cnn_p32_h128_head32_gate | 0.06208513 | 0.0424 |
| cnn_p32_h128_head32_none | 0.09381277 | 0.0575 |
| cnn_p32_h32_head32_concat | 0.06355236 | 0.0266 |
| cnn_p32_h32_head32_concat_identity | 0.06950838 | 0.0218 |
| cnn_p32_h32_head32_film | 0.09601164 | 0.0404 |
| cnn_p32_h32_head32_film_identity | 0.04624618 | 0.0134 |
| cnn_p32_h32_head32_gate | 0.05302461 | 0.0297 |
| cnn_p32_h32_head32_gate_identity | 0.03763083 | 0.00116 |
| cnn_p32_h32_head32_gate_lsres | 0.001001994 | 5.14e-06 |
| cnn_p32_h32_head32_gate_lsres__val_scaled | 0.001001788 | 4.92e-06 |
| cnn_p32_h32_head32_none | 0.06093042 | 0.0203 |
| cnn_p32_h32_head32_none_lsres | 0.001525642 | 0.000153 |
| cnn_p32_h32_head32_none_lsres__val_scaled | 0.00150716 | 0.000145 |
| cnn_p32_h32_head32_resgate | 0.03145754 | 0.0132 |
| cnn_p32_h32_head32_resgate_identity | 0.043395 | 0.0137 |
| cnn_p32_h32_head64_gate | 0.03124569 | 0.012 |
| cnn_p32_h32_head64_none | 0.03501802 | 0.0103 |
| cnn_p32_h64_head32_gate | 0.03216166 | 0.023 |
| cnn_p32_h64_head32_none | 0.0258922 | 0.0209 |
| cnn_p64_h32_head32_gate | 0.009822635 | 0.00961 |
| cnn_p64_h32_head32_none | 0.01229229 | 0.01 |
| cnn_p64_h64_head32_gate | 0.01383585 | 0.015 |
| cnn_p64_h64_head32_none | 0.01045204 | 0.0132 |
| raw_p0_h32_head32_gate | 0.009824727 | 0.00941 |
| raw_p0_h32_head32_none | 0.01249787 | 0.00949 |
| raw_p0_h44_head32_gate | 0.007226011 | 0.00976 |
| raw_p0_h44_head32_none | 0.006876698 | 0.00994 |

## Limits

- Wider projection, recurrent width, and output head are distinct interventions. Spectra diagnose association, not causation.
- Exact-Gaussian test cases are separate from finite-16-sinusoid Jakes cases. LMMSE is only guaranteed MMSE for the Gaussian known-statistics case.
- Spatial controls assume half-wavelength ULAs and a uniform angular power spectrum; they are synthetic, not measured channels.
- Empirical LMMSE uses subsets of the same clean labeled training windows available to the neural estimator. It sees no confirmation targets.
- The unknown-statistics suite withholds per-window covariance parameters from neural and empirical methods. Component oracles have extra knowledge and are labeled.
- Calibration-based SNR uses 64 extra noise-only complex samples per window; no free-SNR-acquisition or end-to-end throughput claim is supported.
- QPSK blocks hold each channel fixed. Rate is a ZF Gaussian-input proxy without pilot/training/calibration overhead.
- V3.2 Mamba/GRU/sequence results are preserved in the old study, not silently retrained or replaced by this focused LSTM revision.
- Report runtime from latency.csv only when measured with the training workers stopped. Include covariance setup and fitting costs separately.
- No result establishes publication acceptance. Preserve adverse and null results.
