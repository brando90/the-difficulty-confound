"""Generate publication report, 3-panel figure, and SHA-256 manifest for Experiment 08."""
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
    rlu = json.loads((RES / 'rl_utility_summary.json').read_text())
    ret = json.loads((RES / 'retention_summary.json').read_text())
    pred = json.loads((RES / 'prediction_summary.json').read_text())

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), dpi=180)

    # Panel A: Pure 0/1 Exact-Sum RLVR Gap Recovery & No-Harm Retention (q_pre=0.1 -> q_env=0.5)
    ax = axes[0]
    rs = ret['statistics']
    cats = ['Pre-RLVR (step=0)\nq_pre=0.10', 'Post-RLVR (step=1000)\nG=16, 0/1 Exact Sum']
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
    ax.set_title('A. Verifiable-Reward RL (RLVR) Remediation')
    ax.legend(loc='lower right', fontsize=8.5)
    ax.grid(axis='y', alpha=0.25)

    # Panel B: RL Improvement Utility U = Acc_1000 - Acc_0 by Pretraining History
    ax = axes[1]
    us = rlu['statistics']
    q_labels = ['q_pre=0.10\n(Single-carry heavy)', 'q_pre=0.50\n(Balanced)', 'q_pre=0.90\n(Double-carry heavy)']
    u1_vals = [100 * us['u_s1_q01']['estimate'], 100 * us['u_s1_q05']['estimate'], 100 * us['u_s1_q09']['estimate']]
    u2_vals = [100 * us['u_s2_q01']['estimate'], 100 * us['u_s2_q05']['estimate'], 100 * us['u_s2_q09']['estimate']]
    x2 = np.arange(len(q_labels))
    ax.bar(x2 - w / 2, u1_vals, w, label='RL Utility U: S_iso (1,0)+(0,1)', color='#2b5c8f')
    ax.bar(x2 + w / 2, u2_vals, w, label='RL Utility U: S_comp (0,0)+(1,1)', color='#d95f02')
    ax.set_xticks(x2)
    ax.set_xticklabels(q_labels)
    ax.set_ylabel('RL Improvement Utility U = Acc_T - Acc_0 (pp)')
    ax.set_title('B. RL Improvement Utility Reversal (1000 steps)')
    ax.legend(loc='upper right', fontsize=8.5)
    ax.grid(axis='y', alpha=0.25)

    # Panel C: Held-Out RL Utility Scaling Law RMSE (utility_base vs oracle vs signal+difficulty)
    ax = axes[2]
    um = rlu['utility_law_metrics']['joint']
    laws_labels = ['utility_base\n(Scale+Budget)', 'utility_oracle\n(+log q_pre)', 'utility_signal\n(+rho, H)', 'utility_signal_diff\n(+rho, H, D_m0)']
    u_rmses = [
        100 * um['utility_base']['estimate'],
        100 * um['utility_oracle_history']['estimate'],
        100 * um['utility_signal_coverage']['estimate'],
        100 * um['utility_signal_difficulty']['estimate'],
    ]
    colors = ['#7f7f7f', '#7570b3', '#e7298a', '#1b9e77']
    x3 = np.arange(len(laws_labels))
    ax.bar(x3, u_rmses, 0.55, color=colors)
    ax.set_xticks(x3)
    ax.set_xticklabels(laws_labels, fontsize=8.5)
    ax.set_ylabel('Joint Held-Out Utility RMSE (pp)')
    ax.set_title('C. RL Utility Scaling Law (Held-Out Scales & Histories)')
    ax.grid(axis='y', alpha=0.25)

    fig.tight_layout()
    fig.savefig(ROOT / 'rlvr_utility_results.png')
    fig.savefig(ROOT / 'rlvr_utility_results.pdf')
    plt.close(fig)

    jm = {r['law']: r for r in pred['metrics'] if r['axis'] == 'joint'}
    jc = {(r['baseline'], r['extended']): r for r in pred['comparisons'] if r['axis'] == 'joint'}
    um = rlu['utility_law_metrics']['joint']
    uc = rlu['utility_law_comparisons']['joint']
    ist = inter['statistics']
    ust = rlu['statistics']
    rst = ret['statistics']

    report_md = f"""# Experiment 08 Results: Reinforcement Learning with Verifiable Exact-Sum Rewards (RLVR) and Utility Scaling Laws

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/archive/08_rlvr_verifiable_reward_utility_laws/RESULTS_rlvr_utility.md>

**Frozen commit:** `{runtime['git_revision']}` · **Verification:** `{ver['status']}` (`max_replay_bits_difference = {ver['max_replay_bits_difference']:.3e}`, {ver['completed_trajectories']} RLVR trajectories, {ver['final_models_replayed']} final + {ver['warm_models_replayed']} warm checkpoints replayed).

---

## 1. Executive Summary of Prospective Gates

| Prospective Gate | Comparison / Metric | Estimate [95% Seed-Block CI] | Gate Criterion | Status |
|---|---|---|---|---|
| **Gate 1: Post-RLVR Loss Predictive Feature Value** | `stable_base` $\\to$ `stable_difficulty` (joint held-out RMSE) | **{fmt_pct(jc['stable_base', 'stable_difficulty'])}** (`{jm['stable_base']['rmse']:.4f} -> {jm['stable_difficulty']['rmse']:.4f}` bits) | $\\ge 10.0\\%$, lower CI $>0$, both widths improve, 0 invalid | **`{pred['status']}`** |
| **Gate 2: RL Improvement Utility Scaling Law** | `utility_base` $\\to$ `utility_signal_difficulty` (joint $U$ RMSE) | **{fmt_pct(uc['utility_base_to_utility_signal_difficulty'])}** (`{100*um['utility_base']['estimate']:.2f} -> {100*um['utility_signal_difficulty']['estimate']:.2f}` pp) | $\\ge 10.0\\%$, lower CI $>0$, both widths improve | **`{rlu['utility_law_status']}`** |
| **Gate 3: RL Utility Reversal Across Histories** | $U = \\text{{Acc}}_{{1000}} - \\text{{Acc}}_0$ contrast (`u_low` / `u_high`) | **{fmt_pp(ust['u_low'])} pp** / **{fmt_pp(ust['u_high'])} pp** | Both $\\ge 10.0$ pp, lower CIs $>0$, all 5 widths $>0$ | **`{rlu['utility_reversal_status']}`** |
| **Gate 4: Pure 0/1 RLVR Remediation & Retention** | $q_{{\\text{{pre}}}}=0.10 \\to \\text{{RLVR }} (q_{{\\text{{env}}}}=0.50, G=16)$ | Recovery **{fmt_pct(rst['pure_recovery_fraction'])}**; control $\\Delta$ **{fmt_pp(rst['pure_control_change'], signed=True)} pp** | Recovery lo $\\ge 50\\%$, control $\\Delta \\ge -5.0$ pp | **`{ret['status']}`** |

---

## 2. Bridging to `rl-4-code-verification`: Predicting Held-Out RL Improvement Utility $U = \\text{{Acc}}_T - \\text{{Acc}}_0$

In [`experiments/ideas/predictability_bridge/README.md`](../../ideas/predictability_bridge/README.md), we hypothesized that training history governs starting-policy success probabilities $p_i(m_0)$, which determine usable RL signal density $\\rho_i(G) = 1 - p_i^G - (1-p_i)^G$, coverage $H_i(G) = 1 - (1-p_i)^G$, and starting difficulty $D_{{m_0}}$, thereby predicting downstream RL improvement utility $U = \\text{{Acc}}_T - \\text{{Acc}}_0$.

On 192 joint held-out targets (unseen widths `96, 384` $\\times$ unseen pretraining mixtures `0.20, 0.80` $\\times$ checkpoints `100, 300, 1000` $\\times$ strata `1, 2`):
- **`utility_base` (Scale + Budget $s(C)$ + Stratum only):** RMSE **{fmt_pp(um['utility_base'])} percentage points**
- **`utility_oracle_history` (+ Oracle $\\log q_{{\\text{{pre}}}}$):** RMSE **{fmt_pp(um['utility_oracle_history'])} percentage points** (relative reduction **{fmt_pct(uc['utility_base_to_utility_oracle_history'])}**)
- **`utility_signal_coverage` (+ Probed $\\log \\rho_{{m_0}}$ and headroom $1 - p_{{m_0}}$):** RMSE **{fmt_pp(um['utility_signal_coverage'])} percentage points** (relative reduction **{fmt_pct(uc['utility_base_to_utility_signal_coverage'])}**)
- **`utility_signal_difficulty` (+ Probed $\\log \\rho_{{m_0}}$, headroom $1 - p_{{m_0}}$, and $\\log D_{{m_0}}$):** RMSE **{fmt_pp(um['utility_signal_difficulty'])} percentage points** — a **{fmt_pct(uc['utility_base_to_utility_signal_difficulty'])} reduction** vs `utility_base` (`{rlu['utility_law_status']}`), and **{fmt_pct(uc['utility_oracle_history_to_utility_signal_difficulty'])}** lower RMSE than `utility_oracle_history`!

---

## 3. Pure 0/1 Verifiable-Reward RL Remediation & Natural No-Harm Retention

During RLVR post-training, the model receives **only** a binary $0/1$ indicator of whether its 4 sampled digits equal the exact integer sum $a+b$, using leave-one-out group-relative advantage ($G=16$). Because already-mastered items have $\\bar{{R}}_b = 1 \\implies \\hat{{A}}_{{b,g}} = 0$, group-relative RL naturally avoids gradient interference on mastered items while focusing updates on unmastered items with $\\rho_i(G) > 0$:
- **Pre-RLVR (`step=0`, $q_{{\\text{{pre}}}}=0.10$):** Single-carry control `(1,0)+(0,1)` accuracy is **{fmt_pct(rst['pure_control_pre'])}** vs cascading two-carry target `(1,1)` accuracy **{fmt_pct(rst['pure_target_pre'])}** (initial gap **{fmt_pp(rst['pure_initial_gap'])} pp**).
- **Post-RLVR (`step=1000`, $q_{{\\text{{env}}}}=0.50$):** Cascading two-carry `(1,1)` accuracy rises to **{fmt_pct(rst['pure_target_post'])}** (**{fmt_pp(rst['pure_target_gain'], signed=True)} pp**, recovering **{fmt_pct(rst['pure_recovery_fraction'])}** of the initial gap), while single-carry control `(1,0)+(0,1)` accuracy changes by **{fmt_pp(rst['pure_control_change'], signed=True)} pp** to **{fmt_pct(rst['pure_control_post'])}** (`{ret['status']}`).
- **RL Utility Reversal:** At $q_{{\\text{{pre}}}}=0.10$, RLVR improves $S_{{\\text{{comp}}}}$ by **{fmt_pp(ust['u_s2_q01'], signed=True)} pp** vs **{fmt_pp(ust['u_s1_q01'], signed=True)} pp** on $S_{{\\text{{iso}}}}$ (`u_low = ` **{fmt_pp(ust['u_low'])} pp**). At $q_{{\\text{{pre}}}}=0.90$, the RL utility reverses: $S_{{\\text{{iso}}}}$ improves by **{fmt_pp(ust['u_s1_q09'], signed=True)} pp** vs **{fmt_pp(ust['u_s2_q09'], signed=True)} pp** on $S_{{\\text{{comp}}}}$ (`u_high = ` **{fmt_pp(ust['u_high'])} pp**, `{rlu['utility_reversal_status']}`).
"""
    (ROOT / 'RESULTS_rlvr_utility.md').write_text(report_md)
    (ROOT / 'results.md').write_text(report_md)

    hashes = {}
    for p in sorted(ROOT.rglob('*')):
        if p.is_file() and p.name != 'ARTIFACT_SHA256.json' and '__pycache__' not in p.parts:
            hashes[str(p.relative_to(ROOT))] = sha(p)
    (ROOT / 'ARTIFACT_SHA256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    print('REPORT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
