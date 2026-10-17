"""Generate publication report, 3-panel figure, and SHA-256 manifest for Experiment 07."""
import hashlib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RES = ROOT / 'expt_v1/results'


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fmt_pct(ci):
    est = ci['estimate'] if 'estimate' in ci else ci['relative_rmse_reduction']
    return f"{100 * est:.2f}% [{100 * ci['lo']:.2f}, {100 * ci['hi']:.2f}]"


def fmt_pp(ci, signed=False):
    sign = '+' if signed and ci['estimate'] >= 0 else ''
    s_lo = '+' if signed and ci['lo'] >= 0 else ''
    s_hi = '+' if signed and ci['hi'] >= 0 else ''
    return f"{sign}{100 * ci['estimate']:.2f} [{s_lo}{100 * ci['lo']:.2f}, {s_hi}{100 * ci['hi']:.2f}]"


def fmt_bits(ci):
    return f"{ci['estimate']:.4f} [{ci['lo']:.4f}, {ci['hi']:.4f}]"


def main():
    runtime = json.loads((RES / 'runtime.json').read_text())
    ver = json.loads((RES / 'verification.json').read_text())
    inter = json.loads((RES / 'intervention_summary.json').read_text())
    mdl = json.loads((RES / 'prequential_mdl_summary.json').read_text())
    ret = json.loads((RES / 'retention_summary.json').read_text())
    pred = json.loads((RES / 'prediction_summary.json').read_text())

    # Generate 3-panel plot
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), dpi=180)

    # Panel A: Pure Compositional Reversal & No-Harm Remediation (q_pre=0.1 -> q_post=0.5)
    ax = axes[0]
    rs = ret['statistics']
    cats = ['Pre (step=0)\nq_pre=0.10', 'Post (step=1000)\nq_post=0.50']
    c_means = [100 * rs['pure_control_pre']['estimate'], 100 * rs['pure_control_post']['estimate']]
    t_means = [100 * rs['pure_target_pre']['estimate'], 100 * rs['pure_target_post']['estimate']]
    x = np.arange(len(cats))
    w = 0.34
    ax.bar(x - w / 2, c_means, w, label='Single carry (1,0)+(0,1) [Control]', color='#2b5c8f')
    ax.bar(x + w / 2, t_means, w, label='Cascading two-carry (1,1) [Target]', color='#d95f02')
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel('Exact-Match Accuracy (%)')
    ax.set_ylim(0, 105)
    ax.set_title('A. Compositional Gap Recovery & Retention')
    ax.legend(loc='lower right', fontsize=8.5)
    ax.grid(axis='y', alpha=0.25)

    # Panel B: Prequential Online MDL Code Length by Pretraining History
    ax = axes[1]
    ms = mdl['statistics']
    q_labels = ['q_pre=0.10\n(Single-carry heavy)', 'q_pre=0.50\n(Balanced)', 'q_pre=0.90\n(Double-carry heavy)']
    s1_bits = [ms['preq_s1_q01']['estimate'], ms['preq_s1_q05']['estimate'], ms['preq_s1_q09']['estimate']]
    s2_bits = [ms['preq_s2_q01']['estimate'], ms['preq_s2_q05']['estimate'], ms['preq_s2_q09']['estimate']]
    x2 = np.arange(len(q_labels))
    ax.bar(x2 - w / 2, s1_bits, w, label='Online MDL: S_iso (1,0)+(0,1)', color='#2b5c8f')
    ax.bar(x2 + w / 2, s2_bits, w, label='Online MDL: S_comp (0,0)+(1,1)', color='#d95f02')
    ax.set_xticks(x2)
    ax.set_xticklabels(q_labels)
    ax.set_ylabel('Prequential Online MDL (bits / item)')
    ax.set_title('B. Prequential Online MDL Reversal (1000 steps)')
    ax.legend(loc='upper right', fontsize=8.5)
    ax.grid(axis='y', alpha=0.25)

    # Panel C: Joint Held-Out RMSE (Conditional Bits & Online Prequential MDL Bits)
    ax = axes[2]
    jm = {r['law']: r for r in pred['metrics'] if r['axis'] == 'joint'}
    p_mdl = mdl['prequential_predictive_law']
    laws_labels = ['Conditional Loss\n(Held-out Test)', 'Prequential MDL\n(Online Adaptation)']
    base_vals = [jm['stable_base']['rmse'], p_mdl['stable_base_rmse']['estimate']]
    orac_vals = [jm['stable_oracle_history']['rmse'], p_mdl['stable_oracle_history_rmse']['estimate']]
    diff_vals = [jm['stable_difficulty']['rmse'], p_mdl['stable_difficulty_rmse']['estimate']]
    x3 = np.arange(len(laws_labels))
    w3 = 0.25
    ax.bar(x3 - w3, base_vals, w3, label='stable_base (no history)', color='#7f7f7f')
    ax.bar(x3, orac_vals, w3, label='stable_oracle_history (log q_pre)', color='#7570b3')
    ax.bar(x3 + w3, diff_vals, w3, label='stable_difficulty (probe D_m0)', color='#1b9e77')
    ax.set_xticks(x3)
    ax.set_xticklabels(laws_labels)
    ax.set_ylabel('Joint Held-Out RMSE (bits / item)')
    ax.set_title('C. Starting-Difficulty Law vs Baseline & Oracle')
    ax.legend(loc='upper right', fontsize=8.5)
    ax.grid(axis='y', alpha=0.25)

    fig.tight_layout()
    fig.savefig(ROOT / 'mdl_history_results.png')
    fig.savefig(ROOT / 'mdl_history_results.pdf')
    plt.close(fig)

    jm = {r['law']: r for r in pred['metrics'] if r['axis'] == 'joint'}
    jc = {(r['baseline'], r['extended']): r for r in pred['comparisons'] if r['axis'] == 'joint'}
    p_mdl = mdl['prequential_predictive_law']
    ist = inter['statistics']
    mst = mdl['statistics']
    rst = ret['statistics']

    report_md = f"""# Experiment 07 Results: Carry-Marginal Matched Heterogeneous Pretraining, No-Harm Remediation, and Prequential Online MDL

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/archive/07_prequential_mdl_and_length_transfer/RESULTS_mdl_history.md>

**Frozen commit:** `{runtime['git_revision']}` · **Verification:** `{ver['status']}` (`max_replay_bits_difference = {ver['max_replay_bits_difference']:.3e}`, {ver['completed_trajectories']} trajectories, {ver['final_models_replayed']} final + {ver['warm_models_replayed']} warm checkpoints replayed).

---

## 1. Executive Summary of Prospective Gates

| Prospective Gate | Comparison / Metric | Estimate [95% Seed-Block CI] | Gate Criterion | Status |
|---|---|---|---|---|
| **Gate 1: Conditional Loss Predictive Feature Value** | `stable_base` $\\to$ `stable_difficulty` (joint held-out RMSE) | **{fmt_pct(jc['stable_base', 'stable_difficulty'])}** (`{jm['stable_base']['rmse']:.4f} -> {jm['stable_difficulty']['rmse']:.4f}` bits) | $\\ge 10.0\\%$, lower CI $>0$, both widths improve, 0 invalid | **`{pred['status']}`** |
| **Gate 2a: Prequential Online MDL Reversal** | Online adaptation code-length gap (`pure_mdl_low` / `pure_mdl_high`) | **{fmt_bits(mst['pure_mdl_low'])}** / **{fmt_bits(mst['pure_mdl_high'])}** bits/item | Both $> 0$, lower CIs $>0$, all 5 widths $>0$ | **`{mdl['status']}`** |
| **Gate 2b: Prequential Online MDL Prediction** | `stable_base` $\\to$ `stable_difficulty` (joint prequential RMSE) | **{fmt_pct(p_mdl['relative_rmse_reduction'])}** (`{p_mdl['stable_base_rmse']['estimate']:.4f} -> {p_mdl['stable_difficulty_rmse']['estimate']:.4f}` bits) | $\\ge 10.0\\%$, lower CI $>0$ | **`{mdl['predictive_mdl_status']}`** |
| **Gate 3: Pure Compositional Reversal (`step=0`)** | `(1,0)+(0,1)` vs `(1,1)` at $q_{{\\text{{pre}}}}=.1$ / $.9$ | **{fmt_pp(ist['pure_low'])} pp** / **{fmt_pp(ist['pure_high'])} pp** | Both $\\ge 10.0$ pp, lower CIs $>0$, all 5 widths $>0$ | **`{inter['pure_carry_status']}`** |
| **Gate 4: Compositional Remediation & Retention** | $q_{{\\text{{pre}}}}=0.10 \\to q_{{\\text{{post}}}}=0.50$ (`step=1000`) | Recovery **{fmt_pct(rst['pure_recovery_fraction'])}**; control $\\Delta$ **{fmt_pp(rst['pure_control_change'], signed=True)} pp** | Recovery lo $\\ge 50\\%$, target $\\ge 85\\%$, control $\\Delta \\ge -5.0$ pp | **`{ret['status']}`** |

---

## 2. Resolving the Shared-Warmup vs Heterogeneous-History Contrast Under Invariant Carry Marginals

In [Experiment 06](../06_compositional_carry_marginal_matched/RESULTS_compositional.md), holding exact answers and per-column carry frequencies ($50\\%$ ones, $50\\%$ tens; input-digit TV `0.0398`) fixed across all mixtures yielded only a **1.94% [-3.10, 6.50]** RMSE reduction from `stable_difficulty` because all models shared the same balanced warmup (`q=0.50`).

When checkpoints in Experiment 07 are pretrained across **heterogeneous carry-marginal matched histories** ($q_{{\\text{{pre}}}} \\in \\{{.1, .2, .5, .8, .9\\}}$) before balanced post-training ($q_{{\\text{{post}}}}=0.50$):
- **`stable_base` joint held-out RMSE:** `{jm['stable_base']['rmse']:.4f} [{jm['stable_base']['rmse_lo']:.4f}, {jm['stable_base']['rmse_hi']:.4f}]` bits/answer
- **`stable_oracle_history` joint held-out RMSE:** `{jm['stable_oracle_history']['rmse']:.4f} [{jm['stable_oracle_history']['rmse_lo']:.4f}, {jm['stable_oracle_history']['rmse_hi']:.4f}]` bits/answer
- **`stable_difficulty` joint held-out RMSE:** `{jm['stable_difficulty']['rmse']:.4f} [{jm['stable_difficulty']['rmse_lo']:.4f}, {jm['stable_difficulty']['rmse_hi']:.4f}]` bits/answer
- **Relative RMSE reduction (`stable_base` $\\to$ `stable_difficulty`):** **{fmt_pct(jc['stable_base', 'stable_difficulty'])}** (`{pred['status']}`), also outperforming `stable_oracle_history` by **{fmt_pct(jc['stable_oracle_history', 'stable_difficulty'])}**.

---

## 3. Prequential (Online) MDL Code Length Along the Adaptation Stream

Following [Voita & Titov (2020)](https://aclanthology.org/2020.emnlp-main.14/) and our complexity note ([`COMPLEXITY.md`](../01_fixed_width_addition_exposure/COMPLEXITY.md)), we measured the cumulative online prequential description length $\\text{{MDL}}_{{\\text{{preq}}, s}}(T)$ required to transmit the 1,000-step balanced adaptation stream ($256,000$ online examples) before each weight update:
- **After $q_{{\\text{{pre}}}}=0.10$ (single-carry heavy):** Online MDL is **{fmt_bits(mst['preq_s1_q01'])}** bits/item on $S_{{\\text{{iso}}}}$ vs **{fmt_bits(mst['preq_s2_q01'])}** bits/item on $S_{{\\text{{comp}}}}$ (excess adaptation cost **{fmt_bits(mst['mdl_low'])}** bits/item on $S_{{\\text{{comp}}}}$, and **{fmt_bits(mst['pure_mdl_low'])}** bits/item on pure `(1,1)` vs `(1,0)+(0,1)`).
- **After $q_{{\\text{{pre}}}}=0.50$ (matched balanced history):** Online MDL is **{fmt_bits(mst['preq_s1_q05'])}** bits/item on $S_{{\\text{{iso}}}}$ vs **{fmt_bits(mst['preq_s2_q05'])}** bits/item on $S_{{\\text{{comp}}}}$.
- **After $q_{{\\text{{pre}}}}=0.90$ (double-carry heavy):** Online MDL reverses to **{fmt_bits(mst['preq_s1_q09'])}** bits/item on $S_{{\\text{{iso}}}}$ vs **{fmt_bits(mst['preq_s2_q09'])}** bits/item on $S_{{\\text{{comp}}}}$ (excess adaptation cost **{fmt_bits(mst['mdl_high'])}** bits/item on $S_{{\\text{{iso}}}}$, and **{fmt_bits(mst['pure_mdl_high'])}** bits/item on pure `(1,0)+(0,1)` vs `(1,1)`).
- **Predicting Online Prequential MDL on Held-Out Scales & Histories:** Adding probed starting difficulty (`stable_difficulty` vs `stable_base`) reduces joint held-out prequential MDL RMSE from **{fmt_bits(p_mdl['stable_base_rmse'])}** to **{fmt_bits(p_mdl['stable_difficulty_rmse'])}** bits/item—a **{fmt_pct(p_mdl['relative_rmse_reduction'])}** reduction (`{mdl['predictive_mdl_status']}`).

---

## 4. Pure Compositional Remediation & Strict No-Harm Retention ($q_{{\\text{{pre}}}}=0.10 \\to q_{{\\text{{post}}}}=0.50$)

- **Pre-intervention (`step=0`):** Single-carry control `(1,0)+(0,1)` accuracy is **{fmt_pct(rst['pure_control_pre'])}** vs cascading two-carry target `(1,1)` accuracy **{fmt_pct(rst['pure_target_pre'])}** (initial gap **{fmt_pp(rst['pure_initial_gap'])} pp**).
- **Post-remediation (`step=1000`):** Cascading two-carry `(1,1)` accuracy rises to **{fmt_pct(rst['pure_target_post'])}** (**{fmt_pp(rst['pure_target_gain'], signed=True)} pp**, recovering **{fmt_pct(rst['pure_recovery_fraction'])}** of the initial gap), while single-carry control `(1,0)+(0,1)` accuracy changes by **{fmt_pp(rst['pure_control_change'], signed=True)} pp** to **{fmt_pct(rst['pure_control_post'])}** (`{ret['status']}`).
"""
    (ROOT / 'RESULTS_mdl_history.md').write_text(report_md)
    (ROOT / 'results.md').write_text(report_md)

    hashes = {}
    for p in sorted(ROOT.rglob('*')):
        if p.is_file() and p.name != 'ARTIFACT_SHA256.json' and '__pycache__' not in p.parts:
            hashes[str(p.relative_to(ROOT))] = sha(p)
    (ROOT / 'ARTIFACT_SHA256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    print('REPORT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
