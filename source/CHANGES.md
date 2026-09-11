# V3.2 changes from the audited V3.1 source

- The verified official wheel installer and its wheel checksum manifest are retained unchanged.
- Maximum epochs: 60 → 240; optimizer, scheduler, data generation, model selection and patience remain unchanged.
- Explicit, checked migration of the audited V3.1 checkpoints into a new run identity; no parent file changes.
- The committed resume checkpoint now includes selected states, making selected-checkpoint exports recoverable after interruption. Native optimizer loading preserves correct device placement of Adam counters and moments.
- Two matched 4×4 CNN–LSTM head-width controls, 32 → 64 hidden units, trained from scratch.
- Three seeds per sequence window, with equal 40k/8k sample budgets and unchanged T grid.
- Separate train/freeze/evaluate phases; fresh confirmation seed namespace.
- Gated/ungated mismatch controls, raw statistics for all conditions, and a reported-SNR classical comparator.
- Classical rows evaluated once per array/window; distinct Monte Carlo and training-seed uncertainty.
- Per-case evaluation recovery, a graceful session pause, diagnostic archives, human-readable plots, compact LaTeX tables and a computed manuscript guide.

The old 60-epoch numbers remain an archived study. V3.2 budget60 uses those selected weights when imported but evaluates them on a new test set, so its numbers need not equal the previous tables. No model replacement at inference and no fabricated results are used.
