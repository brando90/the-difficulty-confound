# Results: fixed-width addition exposure

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/archive/01_fixed_width_addition_exposure/results.md>

**Status:** COMPLETE — all 32 runs verified; strict prospective criterion not established.
**Last updated:** 10-16-2026 11:44 PDT

Balanced exposure raised length-three-carry accuracy from 9.60% (768/8,000 predictions) to 94.89% (7,591/8,000) on the same 1,000 questions across eight seeds. It recovered 94.34% [92.74, 95.86] of the baseline gap, but no-carry accuracy fell 6.08 points against a permitted five-point decline.

[Full report and uncertainty](RESULTS_addition_exposure.md) · [All stratum results](expt_v1/results/table.md) · [Verification receipt](expt_v1/results/verification.json)

| Phase | Status | Evidence |
|---|---|---|
| Protocol frozen before training | DONE | Source revision db2dc6d |
| Deterministic checks | DONE | 500,500 oracle pairs agree; zero split overlap |
| Training | DONE | 32/32 cells, 159.02 seconds |
| Independent analysis | DONE | Local and remote summaries identical |
| Scientific criterion | NOT ESTABLISHED | Five criteria passed; control-performance retention failed |

**TLDR-end:** [the-difficulty-confound: experiments/archive/01_fixed_width_addition_exposure/] The fixed-task experiment shows a large data-exposure effect and reversed difficulty ordering; the stronger criterion failed its retention requirement.
