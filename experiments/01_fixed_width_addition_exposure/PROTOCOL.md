# Fixed-width addition protocol — frozen before training

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/01_fixed_width_addition_exposure/PROTOCOL.md>

**TLDR:** A paired experiment changes carry-chain coverage while holding the supervised task, training compute, and evaluation questions fixed. The claim is restricted to the fraction of a prespecified accuracy gap recovered in this setup.

Frozen before any model training or observation of model performance on 10-16-2026. Changes after launch must be recorded separately, without editing this protocol. These are agent-proposed operational thresholds, not established community standards.

## Scientific question (locked)

Does replacing mostly no-carry training examples with balanced carry-chain coverage recover more than half of a substantial baseline no-carry versus length-three-carry accuracy gap, while retaining no-carry performance?

Let E be mean no-carry accuracy under scarce-long-carry training; H the same models' length-three accuracy; B the balanced models' length-three accuracy; and C their no-carry accuracy. Define baseline gap G = E - H, gain D = B - H, and recovered fraction R = D/G. R is not a global fraction of difficulty explained or a variance-explained statistic. It can exceed one and is not clipped.

Support requires all of: G >= 0.20; E >= 0.90; B >= 0.90; C - E >= -0.05; the paired-bootstrap 95% lower endpoint for R exceeds 0.50; and shuffled-label mean test accuracy is below 0.05. Nonpositive baseline-gap bootstrap denominators make R uninterpretable. Failure of any criterion means the strong claim is not established by this run; do not change the thresholds to obtain a positive result.

## Locked parameters

- Domain: two zero-padded three-digit nonnegative integers, sorted so a <= b; four-digit sum output. All 500,500 unordered pairs are enumerated.
- Stratum: maximum consecutive carry-out chain under ordinary decimal addition (0, 1, 2, 3). The actual reference algorithm executes three columns for every example.
- Split seed 20261016. Test: 1,000 distinct pairs per stratum (4,000 total). Validation: 250 per stratum (1,000 total). Test and validation are disjoint and excluded from all training, including swapped operands.
- Train: 16,000 distinct pairs per condition and seed, sampled without replacement. Scarce-long-carry counts [15520,160,160,160]; balanced [4000,4000,4000,4000]; rich-long-carry [160,160,160,15520]. Shuffled-label control uses balanced inputs with the 16,000 full answer vectors randomly permuted.
- Seeds [101,202,303,404,505,606,707,808], paired across all four conditions; 32 complete training cells. Within a seed, initialize the same parameters and use the same sequence of sampled minibatch positions. Training examples differ as prescribed.
- Model: digit one-hot inputs (60 numbers), fully connected layers 60→128→128→40 with rectified-linear activations; four independent categorical output heads, ten possibilities each. No pretrained weights. Parameter count is verified at runtime.
- AdamW optimizer, learning rate 0.001, weight decay 0.0001; batch 256, exactly 3,000 updates (768,000 example presentations per model), mean cross-entropy over output digits. Two processor threads; deterministic operations; no graphics processor.
- Final checkpoint only. Validation results are descriptive and cannot choose checkpoints or hyperparameters. No score-dependent early stopping, extra seeds, retries, or selective condition replacement.
- Test metrics: exact full-sum accuracy and sum of output-digit negative log base-two probabilities, in bits per answer. Every test item is identical across all 32 cells.
- Eight independent training/data-sampling seeds, one shared test set. Paired percentile bootstrap: 20,000 draws, seed 16102026. Resample whole seeds jointly across conditions and strata. Report sample standard deviations across seeds. The intervals are conditional on the fixed test set, architecture, and optimizer, not population-wide uncertainty.
- Primary supporting p-value: exhaustive one-sided sign-flip test over eight paired values D_s - 0.5G_s (256 assignments), null nonpositive mean at its boundary under symmetry of paired differences. The prospective verdict uses the bootstrap criterion, not this auxiliary p-value. Other descriptive estimates use p-val=n/a. Rich-long-carry and predictive-code-length outcomes are secondary; no multiple-testing claims.

## Scale verification

Accuracy is on [0,1] and is multiplied by 100 only for percentage display. Code length uses log base two, summed over four output digits, not normalized a second time. Percentages of the recovered gap must not be called percentages of all task difficulty.

## Data schema and integrity

Commit heldout_items.csv, train_<seed>_<condition>.csv, predictions_<seed>_<condition>.csv, cells.json, metrics.csv, runtime.json, learning traces, summary.json, and the generated plot. Predictions include item identifier, operands, stratum, correct sum, predicted sum, correctness, and predictive bits. Analyze independently from per-item predictions; reject missing or duplicate cells and any train/test overlap. Model checkpoint files may stay in an external artifact store with hashes and retrieval instructions.

The independent column-addition oracle must agree with Python integer addition on all 500,500 pairs. Verify carry boundary examples, disjointness, training counts, identical evaluation rows, and exact-match recomputation. No language-model judge or curated gold-reference benchmark is modified.

## Preconditions and bounds

The remote Python environment must import the pinned numerical packages. No model-provider credentials or external inference services are needed. The protocol and source must be published before training. A named remote supervisor writes durable status and final exit codes, survives a disconnected laptop, and terminates the child process group at 5,400 seconds. Node reboot is not automatically recovered; partial cells and outputs must be preserved. No automatic experimental reruns are authorized by this protocol.

## Expected duration and stopping

Estimated 5–45 minutes, initially unmeasured, with a hard 90-minute wall cap for training. Abort on overlap, oracle mismatch, nonfinite loss, source/config mismatch, insufficient disk, or timeout; retain completed and pending cells. A complete run requires 32/32 cells and independent artifact checks. A poor accuracy result is not an execution failure and is not grounds for retries.

## Interpretation

The data intervention also changes digit combinations, output frequencies, and learning curriculum. They are part of the changed training distribution; this design does not identify carry exposure as the unique mediator. Fixed task definition is an operational construct control, not proof of latent strategy equivalence. The extreme exposure distributions are deliberate interventions, not estimated real-world frequencies. A broader claim needs additional task families, architectures, milder exposure ranges, and stronger feature-matched controls.
