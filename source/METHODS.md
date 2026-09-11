# Protocol notes for writing the methods section

These notes describe the implementation. Rewrite them in the manuscript's notation and cite the original scientific references after checking their bibliographic details.

## Observation and channel model

The code simulates independent square MIMO channel windows H_t with N_r = N_t in {2,4}, T pilot-block observations, and unit expected complex power per channel coefficient. The finite sum-of-sinusoids generator uses M = 16 sinusoids per quadrature, independent phases and randomized angle offsets. It approximates circular Gaussian Rayleigh fading and follows J0 temporal second moments; it is not an assertion that the finite-M distribution is exactly Gaussian. No channel is normalized by its realized sample norm.

Training uses lag-one correlation alpha = 0.8. The normalized Doppler f_D T_s is obtained by inverting J0(2 pi f_D T_s) on its first positive lobe. Spatially independent channels are used in training. Spatial stress applies exponential transmit/receive covariance with entry rho^|i-j|. Temporal stress changes the Jakes lag-one correlation, with the remaining generator assumptions held fixed.

The pilot matrix X is a unitary N_t-point DFT. Pilot observations are H_t X + N_t and despreading produces Y_t = (H_t X + N_t) X^H. Complex noise variance is 10^(-SNR/10) per entry. SNR is nominal expected-power pilot SNR and is constant inside a window. Neural inputs contain real and imaginary components of Y_1,...,Y_T. Targets are real and imaginary components of H_T. Training SNR is sampled uniformly in dB between 0 and 30; the supplied scalar is normalized as (SNR - 15)/15.

## Estimators

CNN encoders have 2→16→8 channels, 3×3 kernels, BatchNorm and ReLU. CNN–LSTM/GRU/Mamba models project each frame's features to dimension 32. LSTM and GRU hidden width is 32. Official Mamba-1 uses d_model = 32, d_state = 16, d_conv = 4 and expand = 2. The raw LSTM receives vectorized despread observations directly. The CNN-only estimator uses the last observation.

The multiplicative feature gate is sigmoid(W_s s + b_s). Matched CNN–LSTM alternatives include no conditioning, scalar concatenation, unconstrained affine FiLM, and a multiplier 1 + sigmoid(W_s s + b_s). FiLM's scale is not initialized to an identity multiplier. Original output heads are Linear→ReLU→Linear with hidden width max(16, 2 N_t N_r): 16 for 2×2 and 32 for 4×4. The new 4×4 head-width control uses 64 hidden units for both gated and ungated CNN–LSTM, leaving the temporal width and all other rules unchanged.

LS-residual estimators add the learned correction to Y_T. Their final layer starts at zero, so their initial prediction is LS. This initialization does not guarantee that a trained correction improves on LS. Initialization seeds are assigned by component so shared shapes have identical initial weights within each matched comparison. Models still have different parameter counts when conditioning or width adds parameters; those counts are reported.

Classical estimators are LS, instantaneous iid LMMSE, a fixed-alpha AR(1) Kalman filter, a fixed-covariance Jakes temporal LMMSE and an oracle separable temporal/spatial Jakes LMMSE. The fixed Jakes covariance uses training alpha = 0.8. Fixed and oracle Jakes estimators coincide in the matched main scenario, up to floating-point arithmetic. The oracle is allowed the stress covariance parameters but never estimates them from the evaluated targets. These are linear baselines; the code does not assert a universal nonlinear MMSE lower bound for finite-M fading.

## Training and selection

Main experiments use 100,000/20,000 training/validation windows for each of three seeds, arrays 2×2 and 4×4, and T = 5. Training and validation generation seeds are disjoint and are unchanged from V3.1. All models within a job use the same arrays and epoch-specific sample order. Adam uses initial learning rate 0.001, real-component MSE loss and batch size 256. ReduceLROnPlateau halves the learning rate after five non-improving epochs under its default relative-threshold rule. Early stopping uses ten epochs without any strict validation-MSE improvement. Gradients are not clipped. The maximum is 240 total epochs. Every model is selected by minimum validation MSE.

Main fits retain the best validation checkpoint within the first 60 epochs as a budget reference, including an earlier patience stop if one occurs. The extended checkpoint is the best within the full permitted trajectory. Continued V3.1 fits retain optimizer, scheduler, history, random-number state and any accumulated early-stopping patience. New models use the same initialization/data rules from epoch one. No patience or optimizer reset is performed to promote a desired result. Official CUDA kernels may remain nondeterministic despite controlled seeds and deterministic settings; cross-platform bitwise equality is not claimed.

Sequence experiments use T in {3,5,10,15,20}, 40,000/8,000 training/validation windows for every window length, three seeds, and LSTM-gate, Mamba-gate and LS-residual-gate. Their sample and epoch budgets are equal across T; compute cost need not be equal. Sequence NMSE is tested at 15 dB. The sequence T=5 study has a different training sample budget from the main T=5 study and must be labeled separately.

All training finishes and every selected checkpoint is hashed before confirmation evaluation starts. New test seeds use the namespace `phycom-v3.2-confirmation-20260909`, which differs from the previously inspected V3.1 test namespace. Test targets are not available to model selection. All prespecified designs are reported; ranking on test results is descriptive, not model selection for deployment.

## Evaluation and uncertainty

Main NMSE uses 20,000 independent channel windows per condition at SNR 0,5,...,30 dB. It is the ratio of summed squared channel errors to summed true channel energy, not the unweighted average of per-window normalized errors. Monte Carlo uncertainty for this ratio uses a delta method over independent windows. Training seeds share test channels and noise to allow paired comparisons.

Diagnostic QPSK BER uses 2,048 independent channel blocks and 1,000 bits per block. The channel is held fixed inside each payload block. The data observation is sqrt(rho/N_t) H x + n with unit-energy symbols and unit complex noise variance. A zero-forcing receiver uses the estimated channel; a 1e-8 numerical diagonal stabilization is present in its normal-equation solve. Paired estimators use the same bits and noise. BER percentile intervals resample independent channel blocks, not individual bits, with 2,000 bootstrap resamples.

The effective ZF Gaussian-input sum-rate proxy uses 4,096 independent blocks, residual interstream interference and receiver noise enhancement. It does not subtract pilot overhead or model coded throughput, and must not be called achieved QPSK spectral efficiency. A perfect-CSI diagnostic is retained in main raw metrics.

Neural summary spread is sample standard deviation across three training seeds. Classical methods are evaluated once per common array/window test set, with independent-channel Monte Carlo intervals instead of an artificial zero seed variance. Paired comparisons average differences across the fitted models before estimating conditional test-block intervals. A separate Student-t interval describes training-seed variability conditional on the finite test set; it is imprecise with three seeds. Neither interval alone combines both uncertainty sources. Intervals are pointwise and do not include a multiple-comparison adjustment.

Temporal stress uses alpha in {0.3,0.5,0.65,0.8,0.9,0.95}; spatial stress uses rho in {0,0.3,0.5,0.7,0.9}, each with 10,000 windows at 15 dB. SNR side-information bias is {-3,-1,0,1,3} dB and error standard deviation is {0,1,2,3} dB, evaluated at true SNR 15 and 30 dB. These side-information tests change only the reported scalar, never the already generated observation noise or channel. Reported SNR is not clipped. Gated/ungated pairs share exactly the same cases. The reported-SNR Jakes comparator uses each window's reported noise variance; true-SNR classical references are marked separately.

## Diagnostics and timing

A convergence table reports stopping reason, selected epoch, improvement over the 60-epoch reference and recent best-validation gains. Capped fits with a late best epoch or at least 1% recent improvement are marked for review. These diagnostics do not prove stationarity or optimality.

Head-activity diagnostics observe which ReLU head units activate on the frozen main 30-dB test windows and compute the span of active output columns. The reported affine-span lower bound concerns this observed output support; for LS-residual models it is applied to the correction target. It is descriptive and cannot establish global inactivity or a causal mechanism.

Latency is measured after all competing training/evaluation workers have finished. CUDA synchronization surrounds each timed call, with ten warm-ups and 100 trials. Model complexity tables use batch size one. Sequence timing uses a fixed batch size of 32 at T = 5,8,16,32,64,128,256; its fresh weights and timing-only Transformer are explicitly labeled. Arithmetic counts are approximate real operations with MAC = 2, excluding several nonlinearities, memory operations and implementation overhead. Empirical speed claims must follow measured timing, not asymptotic complexity alone.
