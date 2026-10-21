# Reviewer-to-experiment map

Source: the supplied four-page PHYCOM-D-26-02646-reviews.pdf. The table maps
reviewer concerns to implemented evidence; it does not claim they have already
been resolved by favorable results.

| Concern | Implemented control | Evidence to examine | Limit |
|---|---|---|---|
| R1: all neural estimators lose to matched LMMSE; practical benefit unproven | Same-information empirical LMMSE from 256, 2,048 and 100,000 labeled training windows; raw and fixed 1% shrinkage covariance variants; unknown per-window covariance mixture and held-out parameters; isolated timing including optimized iid temporal filter | Classical comparisons, main/unknown_stats summaries, inference/training/covariance costs | An advantage may remain absent. Oracle knowledge is explicitly identified; never imply LMMSE is generally nonlinear-MMSE optimal for finite sinusoid channels. |
| R1/R2/R3: 4x4 degradation and output-head-only explanation | Separate 32/64/128 projection and 32/64/128 recurrent widths, joint 64/64 and 128/128, and head64-only controls; each gated and ungated across ten seeds | Width/head paired effects, all-seed plot, spectra | Controlled changes improve attribution but do not prove a unique cause. |
| R1/R2: CNN has no demonstrated role on spatially iid channels | Raw LSTM with and without gating; nearest-parameter-count raw-LSTM controls; physically specified ULA covariance | raw_vs_cnn and parameter_matched_encoder rows | Uniform angular-spectrum ULA model is synthetic. Parameter counts are near matched, not exactly identical. |
| R1: residual outputs fail to improve LS | Initial output exactly LS; per-window correction energy and cross term; paired LS differences; one global coefficient in [0,1] fitted only on validation data | correction_energy, correction_cross, LS paired intervals, validation_residual_scale | Scaling is an additional estimator and offers no test-risk guarantee; report unscaled and scaled results. |
| R1/R2/R3: only three seeds and fragile inference | Ten new independent data/optimization seeds in central comparisons, five for supplements; fixed separate confirmation namespace; paired sign tests and Holm over twelve primary hypotheses | Frozen selections, full paired table, every seed retained | Ten runs are an improvement, not a guarantee of adequate power. Supplementary five-seed and architecture-selection results remain exploratory. |
| R1: intermediate antenna configuration | 3x3 CNN/raw LSTM gated/ungated, all ten central seeds | n=3 rows and curves | Does not support massive-MIMO claims. |
| R2: conditioning rules have different initialization | Legacy gate/FiLM/concatenation/residual gate and new identity-initialized counterparts; shared compatible backbone/head parameters | initialization rows and original parity tests | Functionally equal starts do not make parameterizations or optimization trajectories identical. |
| R2: numerical rank alone misses conditioning | Singular values, float32-tolerance rank, domain nullity, variance-entropy effective rank, stable rank, nonzero-spectrum condition number for projection/head weights and intermediate features | diagnostics CSVs, saved spectra and spectrum figure | Observational diagnostics are not standalone causality proofs. |
| R2: reporting error mixed with extrapolation | In-range/out-of-range masks before clipping, true-SNR report reference on identical windows, paired clipped/unclipped tests, explicit calibration with 64 noise-only samples | reporting_errors_paired.csv and evaluation_coverage.csv | Extra calibration observations have a cost; throughput proxy omits that overhead. |
| R2: too much emphasis on 129 fits | New report distinguishes independent seeds from total fit count; old results preserved | Protocol and results notes | Abstract/introduction still need revision after results exist. |
| R3: limited array/SNR realism; useful references | Intermediate array, physical correlation, mixture/held-out statistics, explicit SNR-estimation overhead and conservative scope | Robustness and practical suites | Literature updates and broader channel/large-array claims require separate work; the code does not fabricate those experiments or citations. |

After running, write a point-by-point response with measured effects and exact
table/figure references. If the neural estimator still lacks a practical
advantage, state that finding and adjust the contribution; code changes alone
cannot guarantee acceptance.
