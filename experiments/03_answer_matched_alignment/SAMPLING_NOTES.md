# Sampling scope and warmup accounting

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/03_answer_matched_alignment/SAMPLING_NOTES.md>

**TLDR:** Exact answer histograms and intervention-pool sizes match across all conditions. The warmup-plus-intervention union is equal at the two primary endpoints, but varies across intermediate mixtures.

Recorded during training before inspecting accuracy or loss outcomes. This clarifies the frozen protocol without changing the sampling code or analysis.

Each mixture uses 4,680 distinct intervention pairs (40 per answer); warmup uses a common 4,680-pair balanced pool. Prefix nesting makes the union of warmup and intervention pools 6,552 pairs at both q=.1 and q=.9, 5,616 at q=.3 and q=.7, and 4,680 at q=.5. Within each mixture the union still has the same count for each of the 117 answers. The two primary endpoint interventions therefore match both intervention and total pool sizes. Intermediate-mixture predictive analyses do not control this union size independently of the mixture; starting difficulty is measured before the intervention.

The intervention ratio is 9:1 at the endpoints. Including 500 balanced warmup and 3,000 intervention updates, the expected cumulative two-carry sampling fractions are approximately 15.7% and 84.3%; the quoted 9:1 describes the intervention, not the entire training history. Actual batch draws are reproducible from the recorded generators.

Input-digit marginals, operand distributions and internal learned strategies are not claimed matched. The primary causal contrast concerns these complete training-distribution interventions; it does not isolate carry familiarity from every correlated input feature.
