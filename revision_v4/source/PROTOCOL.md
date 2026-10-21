# Fixed protocol for the V4 revision experiments

This is a test-independent, versioned design, not a registered study. Source and
configuration are hashed before training. All 430 model selections and
validation-only residual coefficients are frozen before confirmation evaluation.
The original V3.2 results motivated the design; none of those old test samples
are used for V4 confirmation.

## Model counts and replication

| Family | Independent seeds | Fits |
|---|---:|---:|
| iid 2x2, including supplementary initialization controls | 10 core / 5 initialization | 95 |
| iid 3x3 | 10 | 40 |
| iid 4x4, including width/head/parameter controls and initialization | 10 core / 5 initialization | 255 |
| Spatial ULA 4x4 | 5 | 20 |
| Unknown per-window statistics mixture 4x4 | 5 | 20 |
| Total | Seeds are shared across matched models | 430 |

Seeds: 31001, 31013, 31019, 31033, 31039, 31051, 31063, 31069, 31079, 31081.
The first five are used for supplementary experiments. Each seed determines a
fresh training set, validation set, initialization and shuffling. Thus variation
across seeds includes both training-data and optimization variability. Matched
models within a group use the same arrays, permutations and compatible
component-initialization seeds. Width changes cannot preserve all tensor shapes.

CNN: 2→16→8 padded 3x3 convolutions, BatchNorm and ReLU, then flatten and linear
projection. Base projection/LSTM widths are 32/32. The 4x4 controls add 64/32,
128/32, 32/64, 32/128, 64/64 and 128/128. The output hidden layer is
max(16,2N²), with a separate 64-unit head-only control. Raw LSTM consumes the
real/imaginary antenna values directly. Near-parameter-matched raw hidden
widths minimize the absolute parameter-count difference over integers 8…256;
counts are saved for every fit. Physical/practical suites compare CNN128/128
with this near-matched raw LSTM, each gated and ungated.

With normalized reported SNR s=(SNR_dB−15)/15, the original gate is sigmoid(Ws+b).
The identity version is 2 sigmoid(Ws+b), with W=b=0. The original residual gate
is 1+sigmoid(Ws+b); its identity control is 1+tanh(Ws+b), initialized at zero.
These are different function classes, explicitly named. Identity FiLM uses
(1+Ws+b)z+(W's+b') with all modulation parameters zero. Identity concatenation
sets only the extra SNR column of the head to zero. Base weights are shared.
The identity controls are evaluated at 2x2 and 4x4 with five seeds each.

## Data and optimization

- Training/validation: 100,000/20,000 independent windows per group; nominal
  pilot SNR uniform over [0,30] dB. T=5, M=16 finite sinusoids, alpha=0.8 for the
  main iid setting. DFT pilots are unitary, and observations are despread LS
  windows; complex per-entry noise power is 10^(−SNR/10).
- No overlapping windows are used between train, validation and confirmation.
  No normalization is fitted on confirmation targets. Sample namespaces are
  saved in config.json. Confirmation channels are shared across training seeds,
  so they are not additional independent Monte Carlo replicates per seed.
- Adam lr=0.001, batch256, ReduceLROnPlateau factor0.5/patience5, validation
  early stopping patience10, maximum240 epochs. Validation aggregate MSE alone
  chooses checkpoints. FP32 is retained; no mixed-precision change to the
  scientific comparison. No gradient clipping is introduced.
- The maximum budget is identical for matched models. Report actual epochs and
  validation histories. Epoch-cap models with >1% best-validation improvement
  over the previous20 epochs are flagged. This flag diagnoses unresolved
  convergence, not an automatic favorable-result stopping rule.
- Each LS-residual network starts exactly at LS because the final affine layer
  is zero. For the additional scaled estimator, beta=clip(−Σe·c/Σ||c||²,0,1) is
  selected on validation real/imaginary components, where e=LS−H and c=f(Y,s).
  One global beta per model is then fixed for all confirmation SNRs.

## Physical and practical comparisons

The ULA spatial covariance has half-wavelength spacing and a uniform angular
power spectrum on mean±spread degrees. The fixed spatial study uses mean20°,
spread15°. It applies R_rx^(1/2) H R_tx^(1/2) with the same covariance at both
ends. Under row-major vectorization the spatial covariance is R_rx⊗R_tx^T.
Numerical angular integration uses513 evenly spaced angles. Unit diagonal
preserves nominal per-entry channel power.

Unknown statistics: each window is sampled uniformly from the nine combinations
alpha∈{0.3,0.6,0.9}, spread∈{10°,30°,60°}, with mean20°. Neural and empirical
linear estimators do not receive the component label. They receive the same
reported SNR and have access to the same supervised training windows. Additional
held-out tests use alpha0.75/spread20° at15 and30dB.

Empirical LMMSE estimates the centered complex full-window covariance and mean
using nested prefixes of 256,2,048,100,000 clean training windows. Both the raw
sample covariance and a fixed 1% identity-shrinkage version are reported; the
unregularized full-training reference is used in the primary classical
comparison columns. No shrinkage level is selected on test targets. Labeled
training channels are required by both neural and empirical methods; this does
not model completely unsupervised deployment.

Other baselines: LS, an assumed iid alpha0.8 filter, an oracle training-distribution
second-moment filter, a per-component true-SNR oracle, and a per-component
reported-SNR oracle. The last two have extra covariance knowledge unavailable to
the learned practical models. The true-SNR version also has extra noise knowledge
in reporting-error tests. Fixed-SNR inverse filters are cached; varying reported
SNR uses a single covariance eigendecomposition and per-window diagonal updates.

Finite-16-sinusoid Jakes has the J0 second moments but is not exactly Gaussian.
The separate exact-Gaussian control draws jointly complex-Gaussian windows with
the same covariance. Only there is known-statistics LMMSE guaranteed to equal
nonlinear MMSE. Neural models are not claimed optimal in either case.

## Confirmation outcomes

Main curves:20,000 independent windows at0,5,10,15,20,25,30dB. Stress cases:5,000
windows. Primary outcome is ratio-of-sums NMSE, Σ||prediction−H||²/Σ||H||².
All architectures and methods retain individual-window error/power arrays.

Twelve primary gating comparisons are fixed: CNN32 and raw-LSTM32, each at2x2,
3x3 and4x4, at15 and30dB. For each comparison an exact two-sided sign test on
ten paired training-run differences tests directional/median consistency.
Holm corrects these12 p-values together once all are available. Bootstrap
intervals over paired seeds describe mean effects; paired-t intervals are
sensitivity checks. Neither creates extra independent training runs. Other
SNR/architecture/initialization/practical comparisons are exploratory and are
not labeled multiplicity-adjusted significance claims.

Conditional channel uncertainty uses paired independent windows with the fitted
ensemble held fixed. It is reported separately from training-seed uncertainty.
Do not pool ten evaluations on the same channels as200,000 independent channels.

Residual-risk identity is saved and verified per window:
Δ||error||² = ||correction||² + 2 Re[(LS−H)^H correction]. Unscaled and
validation-scaled corrections remain distinct output methods.

Reporting-error cases at true0/15/30dB include ±1/±3dB bias and Gaussian error
SD1/3dB, with paired unclipped and clipped reports. A separate accurate-report
stress reference uses identical H,Y. The original report determines in-range
versus extrapolation masks even when clipping is applied. Results are available
for each subset and for the overall paired effect. A calibration illustration
uses64 independent noise-only complex samples to estimate the noise variance,
then reports −10log10(variance). Those extra observations are explicitly counted.

Additional controls include alpha0.3/0.6/0.95 shifts, physical spatial shift and
the exact-Gaussian test. At main SNRs, diagnostic QPSK BER uses1,024 independent
fixed-channel blocks with512 bits per stream; the ZF Gaussian-input rate proxy
uses2,048 blocks. Neither is a time-evolving payload nor a net-throughput result.

## Cost and representation diagnostics

Projection, head and feature singular spectra are computed from stored FP32
values using float64 SVD. Numerical-rank tolerance is max(shape)×eps_float32×s_max.
Domain nullity is saved separately so a full-row-rank32×128 matrix cannot be
misread as information preserving. Effective rank is exp(entropy(s²/Σs²)); stable
rank is Σs²/s_max². Nonzero-spectrum condition number uses the retained spectrum;
the full-spectrum condition number is also saved in diagnostics.json.
Feature spectra use the first2,048 main30dB windows. Weight spectra refer to the
selected checkpoint; gate-value diagnostics additionally probe0/15/30dB. These
fixed diagnostic windows do not select a model or architecture.

Runtime is measured only after GPU workers stop, on the first selected GPU, with
all methods starting from resident despread observations at the same batch sizes
1/32/256 and T=5.100 synchronized wall-time trials follow20 warmups. CPU/GPU,
precision, mean/median/p95 latency and incremental allocated GPU bytes are
recorded. LS includes an output copy. Cached full covariance filters and an
optimized iid temporal-only LMMSE are both timed. Covariance fitting, CPU filter
construction and neural training costs are recorded separately. Neither data
transfer nor pilot/SNR acquisition is included in inference latency. Approximate
arithmetic counts cover convolution/projection, recurrent matrix products and
output-head matrix products. They exclude SNR-modulation operations, activation,
BatchNorm, indexing and memory movement; they are not complete hardware FLOP counts.

The two-GPU scheduler accelerates independent experiments. It is not a claim
that deploying one estimator needs or benefits from two GPUs.
