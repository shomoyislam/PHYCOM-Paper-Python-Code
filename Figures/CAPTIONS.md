# Figure interpretation

Neural error bars show one sample standard deviation across training seeds. Classical NMSE bars show 95% Monte Carlo half-widths across independent channel sequences. These describe different sources of uncertainty. For paired differences use Paper_Data/paired_comparisons.csv.

Main and robustness figures use extended checkpoints selected only on validation MSE. Sequence experiments use identical train/validation counts and three training seeds per window in publication mode; accuracy is measured at 15 dB. Budget60 comparisons are in the exported tables and CSV files.

Reported-SNR robustness applies the same erroneous side information to neural gates and the reported-SNR Jakes estimator. The fixed-covariance Jakes estimator in raw CSV side-information rows uses true nominal SNR and is a reference only.

QPSK payload blocks hold the channel fixed. The Gaussian-input effective ZF rate is a diagnostic proxy without pilot overhead.
