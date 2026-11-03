Run on 10-20-2026 on the shared 4-core CPU box, branch `results/cpu-expts-11-15`, protocol frozen at commit b2f1db7b before any Experiment 21 training. All 256 of 256 runs completed (16 targets × 2 shares × 8 seeds, budget 2^18 examples) in about 21,800 s of grid wall time, inside the frozen 43,200 s cap. `verify.py` result: see Verification below.

## Takeaway

**Experiment 20's alignment floor was a budget artifact where the target has enough data: the stalled runs learn late. Pre-registered overall verdict: LATE_LEARNING (predicted, about 60% confidence).**

- **Late learning at α = 0.15, n = 4 and 8.** Within the same runs, ε falls from 0.842 to 0.093 bits (n = 4, drop 0.748, CI 0.682 to 0.814; 29 of 32 runs cross 0.5 bits) and from 0.942 to 0.272 bits (n = 8, drop 0.671, CI 0.598 to 0.732; 22 of 32 cross). At 2^16 none of these 64 runs had crossed.
- **At 2^16 the network had barely started on any language, distractors included.** The mean distractor loss was 0.991 to 0.997 bits at 2^16 and is 0.894 to 0.919 at 2^18. The Experiment 20 "floor" was the network as a whole not having learned yet, not a limit specific to the target.
- **Mixing still costs more than its share of the data.** Interference I > 0 in every cell: at 2^18 the mixed runs are worse than Experiment 20's pure (α = 1) runs at the same number of target examples (decisive cells: cell means 0.62 to 5.25 in log2 units, every interval above 0). Alignment shows up as a delay, and a large one, not as a floor.
- **The rest is undecided or flat.** n = 16 at α = 0.15 and n = 4, 8, 16 at α = 0.05 start to move by 2^18 (drops 0.03 to 0.17 bits) but meet neither the LATE_LEARNING nor the DESCENDING bar: UNRESOLVED. n = 32 is FLOOR_HOLDS at both shares, but there even α = 1 does not learn the language, so it says nothing about alignment (PROTOCOL Section 4.3).
- **The replicate check holds.** ε(2^16) on the re-drawn string order matches Experiment 20 within 0.008 bits in every cell.

Predictions (PROTOCOL Section 5): overall LATE_LEARNING right; (4, 0.15) and (8, 0.15) right; n = 32 FLOOR_HOLDS right at both shares; interference > 0 in every decisive cell right; replicate within ±0.05 right. Wrong: (16, 0.15) and (4, 0.05) were predicted DESCENDING or LATE_LEARNING, and (8, 0.05) and (16, 0.05) DESCENDING or FLOOR_HOLDS; all four came out UNRESOLVED, their drop intervals straddling the 0.05-bit bar. Cell predictions right: 4 of 8.

**What this changes for Experiment 20.** Its G6 (an alignment floor a1(1 − α)) should be read as "within 2^16 examples" only. At low shares the LSTM's cost of misalignment is a delayed onset whose length grows with n, and for n = 16 at α = 0.15 and n = 8 and 16 at α = 0.05 even 2^18 is too short to tell.

**Next step.** Per Brando (10-20-2026), no further LSTM follow-ups: the same floor-versus-delay question, if pursued, goes to the transformer learner, and any rerun goes to Sherlock CPU nodes.

## Deviations and notes

1. **Concurrent load.** From 04:17 to about 06:20 the grid shared the 4 cores with Experiment 17b (AT-453). Runs are seeded and single-threaded, so results are unaffected (`verify.py` replays checkpoints and retrains the first cell); only wall time changed.
2. **An interrupted chunk.** The third chunk was killed at 06:35, mid-cell, when the worker running it stopped. `run.py --resume` redid the cells in flight (done markers are written last), and it is missing from `run_state.json`'s chunk list (about 1,211 s, 24 cells). Chunks ran at most about 1 h, within the frozen 2 h limit.
3. **Cell cost.** PROTOCOL Section 3.2 item 4 said cell cost does not depend on n. Median cell wall time was 180 to 490 s for most cells, but 815 s for n = 32 at α = 0.05, the last cells run. The cause was not diagnosed, and cell time is not a result.
4. **report.py** was written at 06:15, while the grid was running and before `analysis.json` existed. It only formats `analysis.json` and `metrics.csv` and decides nothing. These notes (takeaway, deviations) were written after the analysis.

## Verification

`verify.py`: **PASS** in 1,966 s. Training sources match the manifest; tasks (Hopcroft = Moore, 32 of 32) and all data regenerate identically and are byte-identical to Experiment 20's; 256 of 256 mixes regenerate, none reuses the pool, and all 256 task-slot sequences through 2^16 equal Experiment 20's; 2,560 checkpoints replay with max |Δ| = 0 in excess bits, accuracy and per-string predictions; cell 11101_dfa_n04_t1_0.15 retrained from scratch matches exactly; metrics.csv equals the cell records; analysis.json recomputes identically and an independent recomputation of the cell means, drops, crossing counts and verdicts agrees; all 256 cells' distractor losses replay identically (the protocol asked for 8). Receipt: `expt_v1/results/verification.json`.
