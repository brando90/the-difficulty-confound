# Does Task2Vec relate to Kolmogorov complexity? (research note)

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/ideas/complexity_vs_alignment/TASK2VEC_KOLMOGOROV.md>

Written 10-18-2026. The question is Brando's: "task2vec claims to have a relationship to kolmogorov complexity, is that true?" Every claim below was checked against the paper's PDF (arXiv version, read with `pdftotext`) and carries one of three labels:

- **[proven]**: a theorem or exact identity, with its assumptions stated;
- **[approximate]**: holds under a named approximation or heuristic;
- **[unsupported]**: claimed or implied, but not shown.

Notation: $K(x)$ is prefix Kolmogorov complexity, the length of a shortest program that prints $x$ on a fixed optimal universal machine. FIM means Fisher information matrix, $F = \mathbb E_{x\sim\hat p,\,y\sim p_w(y|x)}[\nabla_w \log p_w(y|x)\,\nabla_w \log p_w(y|x)^\top]$. KL means Kullback–Leibler divergence.

## 1. What Task2Vec itself claims

Task2Vec (Achille, Lam, Tewari, Ravichandran, Maji, Fowlkes, Soatto, Perona; arXiv 1902.03545, v1 02-10-2019; ICCV 2019) embeds a task as the diagonal of the FIM of a fixed "probe network". Only the probe's final classifier is retrained, and Fisher values are averaged within each filter (Sect. 2.1).

**The only Kolmogorov sentence.** Sect. 2: "The FIM is also related to the (Kolmogorov) complexity of a task, a property that can be used to define a computable metric of the learning distance between tasks [3]." Reference [3] is the 2018 NeurIPS workshop paper by Achille, Mbeng, Paolini and Soatto, "The dynamic distance between learning tasks: From Kolmogorov complexity to transfer learning via quantum physics and the information bottleneck of the weights of deep networks", which Task2Vec lists with arXiv ID 1810.02440. Task2Vec contains no definition, equation or theorem involving $K$. **[unsupported]** inside Task2Vec: the link is asserted by citation only.

**Norm and "difficulty".** Sect. 1: "The norm of the embedding correlates with the complexity of the task". The argument is Sect. 2.2 ("Encoding task difficulty") together with Appendix A, for a two-layer sigmoid network: $F = \mathbb E[(y-p)^2\, S\otimes xx^\top]$ with $y \sim p_w(y|x)$, so $\mathbb E[(y-p)^2] = p(1-p)$. In the paper's words: "if the fit model is very confident in its predictions, $E[(y-p)^2]$ goes to zero. Hence, the norm of the task embedding $\|F\|_\star$ scales with the difficulty of the task for a given feature extractor $\phi$."

- **[proven]** for that two-layer model: the norm weights each input by the probe's predictive uncertainty $p(1-p)$.
- **[approximate/empirical]** for real networks: Fig. 2 (right) and Sect. 5.1 show a scatter plot of the L1 norm against test error. No correlation coefficient is reported.
- "Difficulty" here means residual uncertainty *for this probe's features*. That is not $K$. A random-label task drives $p$ toward $1/2$ and maximizes the weight, even though it has no structure to learn.

**Asymmetric distance.** Sect. 3.1 defines $d_{\text{asym}}(t_a\to t_b) = d_{\text{sym}}(t_a,t_b) - \alpha\, d_{\text{sym}}(t_a,t_0)$, where $t_0$ is the trivial embedding and $\alpha = 0.15$ was tuned for a ResNet-34 probe. The justification: "complexity can be measured as the distance from the trivial embedding." This is a definition plus a tuned heuristic **[approximate]**. The same section warns that "the norm of the embedding is affected by complexity of the task and the number of samples used to compute the embedding."

**A real structural link.** Task2Vec's "robust Fisher" estimator (Sect. 2.1; Appendix B) chooses a Gaussian weight perturbation $\mathcal N(\hat w,\Lambda^{-1})$ by minimizing $\mathbb E_{w}[H_{p_w,\hat p}(y|x)] + \beta\,\mathrm{KL}(\mathcal N(\hat w,\Lambda^{-1})\,\|\,\mathcal N(0,\lambda^2 I))$. This is exactly the Lagrangian $C_\beta(D;P,Q)$ of Achille et al. 2021 (Definition 5.1, eq. 5) with Gaussian $P$ and $Q$ **[proven: identical objective]**. The embedding, however, reports the optimal diagonal precision, not the KL value. Two implementation details also matter:

- The classifier is trained for only 2 epochs before $\Lambda$ is fitted (Appendix C.2), so $\hat w$ is not a converged minimum.
- $\lambda^2$ is learned per layer (Appendix B). A prior fitted to the data is not a fixed code.

## 2. The information-complexity line: what is proven and what is assumed

**Version history of [3].** arXiv 1810.02440 v1 (10-04-2018) is titled "The Dynamics of Differential Learning I: Information-Dynamics and Task Reachability". Its Sect. 2.1 defines Kolmogorov's structure function $S_D(t) = \min_{L_D(M) \lt t} K(M)$ (eq. 1). Sect. 2.2 then *replaces* $K(M)$: "Under this encoding, the cost of encoding the model $M = q(w) \sim \mathcal N(\mu,\Sigma)$ once the prior is fixed is given by the KL-divergence." A second-order expansion gives a Hessian log-determinant (eq. 3), which equals the FIM at a local minimum. So that "Kolmogorov" structure function is a Gaussian-code structure function **[approximate: quadratic loss, Gaussian code, local minimum]**. v2 (05-29-2019), retitled "Dynamics and Reachability of Learning Tasks", drops the section; "Kolmogorov" survives only in the discussion and references.

**Achille, Paolini, Mbeng, Soatto, "The Information Complexity of Learning Tasks, their Structure and their Distance"** (arXiv 1904.03292; *Information and Inference* 10(1):51–72, 2021) is where the formal link lives.

| Result | Statement (paraphrased; equation numbers are the paper's) | Assumptions | Label |
|---|---|---|---|
| Def. 3.1 | $C(D) = \min_{p(y\mid x)} L_D(p) + K(p)$, with true Kolmogorov complexity over computable, factorized $p$ | none; uncomputable | definition |
| Prop. 3.2 | $K(y\mid x) = C_K(D)$ up to $O(1)$; $C(D) \le C_{\det}(D)$ | additive constants | **[proven]** |
| Prop. 3.3 | $N H_p(y\mid x) \le \mathbb E[C(D)] \le N H_p(y\mid x) + K(p(y\mid x))$; equality w.h.p. for large $N$ | i.i.d. samples, computable $p(y\mid x)$ | **[proven]** |
| Ex. 3.4, Prop. 3.6 | random labels: $C(D)\approx N\log\lvert Y\rvert$, a "complex memorization task, but a trivial learning task" | typical random-label dataset | **[proven]** |
| Eq. 3, Sect. 3.1 | structure function $S_D(t)=\min_{K(p)\le t} L_D(p)$; "cannot be computed in general" | cites Vereshchagin–Vitányi Sect. VII | **[proven]** (by citation) |
| Def. 4.1, Lemma 4.2, Thm. 4.4, Cor. 4.5 | asymmetric task distance via $K(p_2\mid p_1)$; positivity, triangle inequality, $d_\beta(D_1\sqcup D_2\to D_1)=O(1)$ | up to $O(1)$; i.i.d. from computable $p_i$ | **[proven]** |
| Def. 5.1 | $C_\beta(D;P,Q) = \mathbb E_{Q}[L_D(p_w)] + \beta\,\mathrm{KL}(Q(w\mid D)\,\Vert\,P(w))$. Text: "Kolmogorov complexity has been replaced by the Kullbach-Liebler (KL) divergence" | arbitrary $P$, $Q$ | definition |
| Prop. 5.2 | universal prior $P(w)\propto e^{-K(w)}$ and Dirac $Q$ give $\mathrm{KL} = K(w^{\star}) + \log Z$ | one-line proof; uncomputable; this is $K$ *of the weights*, not of the task | **[proven]** |
| Prop. 5.3 | optimal "adapted" prior $P=\mathbb E_D[Q(w\mid D)]$ gives $\mathbb E_D\,\mathrm{KL} = I(w;D)$ (Shannon) | needs the distribution over datasets $\pi(D)$, which the authors call unrealistic | **[proven]** |
| Thm. 5.4 | Gaussian $P=\mathcal N(0,\lambda^2 I)$, $Q=\mathcal N(w^{\star},\Sigma)$, $\lambda\to\infty$: $\mathrm{KL} = \tfrac12\log\lvert F\rvert + \tfrac12 k\log\lambda^2 + O(1)$ | $w^{\star}$ a local minimum; "quadratic approximation holds in a sufficiently large neighborhood"; $H=F$ at the critical point | **[approximate]** (second order) |
| Thm. 5.5 | Probably Approximately Correct (PAC)-Bayes: test error $\le$ an affine function of $C_\beta(D;P,Q)$ (McAllester) | bounded loss, $\beta \gt 1/2$, i.i.d. data | **[proven]** |
| Sect. 7, Fig. 4 | "the FIM trace correlates with the error obtained on the task, and so the FIM is indeed a sensible measure of the complexity of a task" | plot reproduced from Task2Vec | empirical; the "complexity" reading is **[unsupported]** |
| Sect. 7 | "A robust version of the asymmetric distance between tasks described here has been used in [1]" (that is, Task2Vec) | Task2Vec's distance is a cosine heuristic, not Def. 6.1's KL difference | **[unsupported]** as an identity |

Four points decide Brando's question.

1. **The Kolmogorov, Shannon and Fisher cases are siblings.** They are three choices of $(P,Q)$ in one Lagrangian. No theorem bounds the Fisher quantity by $K$, approximates $K$ with it, or shows the two rank tasks the same way. The authors themselves (end of Sect. 4): "quantities defined in terms of Kolmogorov complexity are difficult to handle in practice, and may behave well only in an asymptotic regime."
2. **The Fisher term that appears is $\tfrac12\log\lvert F\rvert$ of the full FIM.** Task2Vec's "norm" is the L1 norm of a filter-averaged diagonal: a different statistic. "Where is the Information in a Deep Neural Network?" (Achille, Paolini, Soatto, arXiv 1905.12213, Remark 2.7) adds that the $\log\lambda$ term diverges as the prior becomes improper.
3. **The assumptions fail by construction for Task2Vec.** The probe's feature extractor was trained on ImageNet, not the new task, so it is generally not at a critical point and Theorem 5.4's $H=F$ step does not apply. 1905.12213 patches this with Lemma 2.4 ($H\approx F$ when "almost all training samples are predicted correctly") and Remark 2.6 (Fisher as a positive semi-definite curvature stand-in) **[approximate]**.
4. **The authors' own framing is "information in the weights".** 1905.12213 (Sect. 1) says Kolmogorov-based measures "cannot be measured in practice for modern deep neural networks".

**What a Fisher-based quantity does bound.** At $\beta=1$, $C_1(D;P,Q)$ is the length of a variational ("bits-back") code for the labels (Blier & Ollivier, arXiv 1802.07044, Sect. 3.3, Def. 2). Any computable code length upper-bounds $K$ up to a constant: $K(x) \le \log 1/f(x) + K(f) + O(1)$ for computable $f$ (Grünwald & Vitányi, arXiv cs/0410002, Thms. 2.8–2.9). So:

- $K(y\mid x,\text{probe}) \lesssim C_1 + \text{const}$ is **[proven]** for the exact variational code;
- it is only **[approximate]** once the Gaussian/Fisher formula of Thm. 5.4 replaces the exact code.

The bound is loose. In Blier & Ollivier Table 1 the variational code takes 22.2 kbits on MNIST against 4.10 kbits for the prequential code, and 89.0 against 45.3 kbits on CIFAR-10.

This also corrects Brando's own Appendix C.3 of "Beyond Scale" (arXiv 2306.13840). That appendix says Achille's work "suggests that FIM can serve as an upper bound for Kolmogorov Complexity in the context of learning tasks." That holds only for the variational code length just described. Neither the FIM, nor its diagonal, nor its norm, nor a cosine distance between diagonals is an upper bound on $K$ **[unsupported]**. The appendix itself concedes: "explicit relationships quantifying how distances based on Fisher information matrices correlate with true Kolmogorov complexity are not provided."

**Training state.** In "Critical Learning Periods in Deep Networks" (Achille, Rovere, Soatto, arXiv 1711.08856, Fig. 4) the Fisher trace "sharply increases" early in training and then decreases during "consolidation" while accuracy still improves, so one task gives different Fisher magnitudes at different checkpoints.

## 3. Computable relatives of $K$ and how each relates to it

$K$ itself is upper semicomputable (approximable from above) but not computable. It has no computable lower bound and no computable convergence rate (Vitányi 2020, *Entropy* 22:408, Thms. 1–2 and Sect. 3).

| Measure | Computable? | Exact? | Relation to $K$ | Source |
|---|---|---|---|---|
| Real compressor length $C(x)$ (gzip, zstd) | yes | exact for its compressor | $K(x)\le C(x)+O(1)$, gap unknown **[proven]** | Vitányi 2020 Sect. 5 |
| Levin's $Kt(x)=\min\{\lvert d\rvert+\log t: U(d)=x \text{ in } \le t \text{ steps}\}$ | yes (finite search) | yes | $C(x)\le Kt(x)$ directly from the definitions; can be much larger **[proven, elementary]** | Allender–Buhrman–Koucký, Def. 1 |
| Two-part MDL (minimum description length) code with a computable model class | yes | for its class | $K(x)\le K(S)+\log\lvert S\rvert+O(1)$ **[proven]** | Grünwald–Vitányi Sect. 5.2.4; Grünwald 2007 |
| Prequential (online) code $\sum_t -\log p_{\hat\theta_t}(y_t\mid x_t)$ | yes, given a deterministic learner | for that learner | upper bound up to $K(\text{learner})+O(1)$ (conditional analogue of G–V Thm. 2.8; a printed conditional statement was not checked) **[proven, standard]** | Blier–Ollivier Def. 3; Voita–Titov Sect. 2.2 |
| Variational (bits-back) code, $\beta=1$ | yes | for $(P,Q)$ | upper bound; looser than prequential in practice **[proven]** | Blier–Ollivier Sect. 3.3, Table 1 |
| Coding Theorem Method (CTM): $-\log_2$ of output frequency of small Turing machines | yes, only for very short strings | no | motivated by $K=-\log m+O(1)$, but the $O(1)$ is ignored; Vitányi: "only suggestive and cannot be proved correct" **[unsupported]** as an estimate of $K$ | Soler-Toscano et al. 2014; Vitányi 2020 Sects. 4, 8 |
| Block Decomposition Method (BDM): CTM on blocks plus $\log$ multiplicity | yes | no | $K(X)\le \mathrm{BDM}+O(\log^2\lvert A\rvert)+\epsilon$ and $\mathrm{BDM}\le \lvert\mathrm{Adj}\rvert K(X)+\dots$, where $\epsilon$ is the unknown total CTM error; tends to block Shannon entropy when CTM runs out **[proven bounds, uncomputable error term]** | Zenil et al. 2018, Prop. 4.1, Sect. 4 |
| Kolmogorov structure functions $h_x,\lambda_x,\beta_x$; algorithmic minimal sufficient statistic | $h_x,\lambda_x$ upper semicomputable; $\beta_x$ neither upper nor lower semicomputable | no | defined through $K$; "no algorithm that given $x$ and $K(x)$ finds a minimal sufficient statistic" **[proven]** | Vereshchagin–Vitányi 2004, Sects. II.F, VI.B, VII |
| Sophistication (Koppel) | no | no | regularity part of $K$; such extensions "inherit inherent uncomputability" | Shalizi–Crutchfield Sect. II.A |
| Effective complexity (Gell-Mann–Lloyd): $\inf K(E)$ over ensembles $E$ making $x$ typical with $K(E)+H(E)\le K(x)+\Delta$ | not as defined (built from $K$; no algorithm found) | no | random strings are "effectively simple" (Thm. 10); tied to minimal sufficient statistics (Sect. VI) | Ay–Müller–Szkoła, Def. 4 |
| Statistical complexity $C_\mu=H[\mathcal S]$, the entropy of the causal states (Crutchfield) | yes, given a finite ε-machine; otherwise estimated | for the model | a different object: memory of a stationary process, not the length of a program; an i.i.d. coin has one causal state, so $C_\mu=0$, while a typical sample has near-maximal $K$ | Shalizi–Crutchfield Def. 12 |
| Minimal deterministic finite automaton (DFA) state count $k$ | yes (DFA minimization) | yes, by Myhill–Nerode | after a bit encoding, $K\le O(k\lvert\Sigma\rvert\log k)+c$ **[proven, elementary]**. It can be exponentially loose: "length divisible by $n$" needs $n$ states but has $K\approx\log n$ | Sipser 2nd ed., Problem 1.52 |
| Shortest program in a fixed domain-specific language (DSL) $\mathcal L$ | yes (exhaustive search) | yes, for $\mathcal L$ | $K\le \ell_{\mathcal L}+c_{\mathcal L}$ when $\ell$ is measured in bits (invariance/optimality theorem) **[proven]** | Sipser Sect. 6.4, "Optimality of the definition" (theorem number not verified) |
| Normalized compression distance (NCD) | yes | no | compressor stand-in for the normalized information distance, which is defined with $K$; accuracy depends on the compressor **[approximate]** | Cilibrasi–Vitányi Sect. 3, eq. 3.1 |
| Task2Vec norm / diagonal FIM | yes | no | none proven (Sect. 2 above) **[unsupported]** | — |

## 4. Implications

**(a) Task2Vec norm as a proxy for $K$.**

- By Task2Vec Sect. 2.2, the norm is uncertainty-weighted input energy *relative to the probe's features*. A complex task the probe represents well gets a small norm; a simple task it does not represent gets a large one. The norm therefore measures probe–task mismatch, an alignment-type quantity and the very confound this repository studies.
- The Fisher weight $p(1-p)$ rises with label noise at fixed $K(p)$, just as $C(D)\approx N H(y\mid x)+K(p)$ does (Prop. 3.3).
- A defensible Fisher-side proxy is the β = 1 variational code length $C_1(D;P,Q)$ from Task2Vec's own optimizer; it still only upper-bounds $K$, loosely. The prequential code is the tighter computable upper bound.

**(b) The diversity coefficient on real data.**

- It is $\widehat{\mathrm{div}}(D)=\mathbb E_{B_1,B_2\sim D}\, d_{\cos}(\vec f_{B_1},\vec f_{B_2})$ (2306.13840, Sect. 2.2). Cosine distance discards the norm, which is the only part Task2Vec ties to "complexity". The coefficient measures batch-to-batch *variability or coverage*, not the complexity of any single task.
- The companion paper calls it an approximation of "the effective number of tasks in a dataset" (arXiv 2306.13841, Sect. 2).
- Its stated upper reference point is uniform random tokens (2306.13840, Sect. 2.4): maximal $K$, zero learnable structure. Appendix L concedes that high diversity "could simply be due to uniform noise".
- It saturates as latent concepts increase (Appendix K.3) and grows with batch size (Appendix H).
- Toy prediction (our analysis, untested): single-target datasets score about the same whatever their $K$; mixtures of many simple targets score high.

**(c) Task2Vec distance as an alignment measure.**

- With the weights held fixed, the FIM is linear in the data distribution. A mixture with a fraction $\alpha$ of target data therefore has $F_{\text{mix}}=\alpha F_T+(1-\alpha)F_O$.
- Because diagonal-FIM entries are non-negative, the cosine distance to $F_T$ then falls monotonically as $\alpha$ rises **[approximate: derived here, assumes a fixed head]**. The curve is nonlinear, and its shape depends on $\cos(F_T,F_O)$, so it is not calibrated across task pairs. Retraining the head per batch breaks the linearity.
- The [alignment-scaling paper with Varun](https://github.com/madanva/alignment-scaling-laws) (Sect. 3, eqs. 2–3; the repository is private) defines $\alpha(S)=\sum_{x\in S} t(x)s(x)/\sum_{x\in S} t(x)$ with $s=\tfrac12\tilde s_{\text{SBERT}}+\tfrac12\tilde s_{\text{ZIP}}$. Here SBERT is Sentence-BERT maximum cosine similarity, and $\tilde s_{\text{ZIP}}$ is compression-ratio similarity under a zstd dictionary, adapted from ZIP-FIT (Obbad, Mlauzi, Miranda, Schaeffer, Obbad, Bedi, Koyejo; arXiv 2410.18194). ZIP-FIT itself uses gzip NCD (Algorithm 1).
- Both $\alpha$ components are per-example relevance scores; the compression one relates to $K$ only through a real compressor (NCD row above). Task2Vec distance is batch-level and probe-relative. None of the three is guaranteed linear in true $\alpha$, so Exp 15 must fit a calibration curve rather than assume one.

**(d) Confounds of Fisher-based measures.**

- **Probe:** Task2Vec Table 1 shows that results depend on the probe architecture; 2306.13840 Appendix H finds random probes diverge from pretrained ones.
- **Parametrization:** "the parameters of the network have different scales" (Task2Vec Sect. 3.1). Rescaling symmetries in rectifier networks change curvature without changing the function (Dinh et al., arXiv 1703.04933).
- **Dataset size:** the norm depends on "the number of samples" (Task2Vec Sect. 3.1).
- **Label noise and entropy:** the $p(1-p)$ weight. Hard tasks and noisy tasks both give large norms.
- **Training state:** a 2-epoch head (Appendix C.2), and the rise-then-fall of Fisher during training (Critical Periods, Fig. 4).
- **Prior scale:** $\lambda^2$ is learned per layer.

## Verdict in five lines

1. Task2Vec proves nothing about Kolmogorov complexity: it has one sentence citing [3] (arXiv 1810.02440), and its "complexity" claim for the norm is a two-layer derivation of predictive uncertainty plus a correlation with test error.
2. The formal link is in Achille, Paolini, Mbeng, Soatto (2021): true $K$ defines task complexity and distance, with real theorems, but those quantities are uncomputable.
3. Their computable version swaps $K$ for a KL "information in the weights"; Kolmogorov (Prop. 5.2), Shannon (Prop. 5.3) and Fisher (Thm. 5.4, second-order, at a local minimum) are three separate prior choices, and no result connects the Fisher case to $K$.
4. Task2Vec optimizes that same Gaussian Lagrangian but reports a filter-averaged diagonal precision and its L1 norm, not the $\tfrac12\log\lvert F\rvert$ term; only the β = 1 variational code length upper-bounds $K$, and loosely.
5. So "Task2Vec is a computable proxy for Kolmogorov complexity" is too generous: Task2Vec is a computable cousin of a Gaussian-code complexity, not a validated proxy for $K$, and the diversity coefficient measures variability, not complexity.

## What this means for Experiments 14–15

- **Exp 14:** test the Task2Vec L1 norm, $\tfrac12\sum\log\Lambda_{ii}$, the β = 1 variational code length, and the prequential code length against exact $K_{\mathcal L}$ (DFA states, DSL program bits).
  - Add a label-noise arm at fixed $K$; the analysis predicts the norm rises while $K(p)$ stays fixed.
  - Add a probe-swap arm.
  - Record the head's training state.
  - Convert DFA state counts to bits before calling them an upper bound on $K$.
  - Expect the diversity coefficient to track the number of mixture components, not $K$.
- **Exp 15:** treat Task2Vec cosine distance, SBERT, and compression scores as uncalibrated monotone transforms of $\alpha$ at best. Fit and report the calibration curve for each held-out (target, other) pair before reading $\hat\alpha$ as $\alpha$.

## References

- Achille, Lam, Tewari, Ravichandran, Maji, Fowlkes, Soatto, Perona. Task2Vec: Task Embedding for Meta-Learning. ICCV 2019. <https://arxiv.org/abs/1902.03545>
- Achille, Paolini, Mbeng, Soatto. The Information Complexity of Learning Tasks, their Structure and their Distance. *Information and Inference* 10(1):51–72, 2021. <https://arxiv.org/abs/1904.03292>
- Achille, Paolini, Soatto. Where is the Information in a Deep Neural Network? <https://arxiv.org/abs/1905.12213>
- Achille, Mbeng, Soatto. Dynamics and Reachability of Learning Tasks (v1: The Dynamics of Differential Learning I). <https://arxiv.org/abs/1810.02440v1>, <https://arxiv.org/abs/1810.02440>
- Achille, Rovere, Soatto. Critical Learning Periods in Deep Networks. ICLR 2019. <https://arxiv.org/abs/1711.08856>
- Miranda, Lee, Sundar, Casasola, Schaeffer, Obbad, Koyejo. Beyond Scale: The Diversity Coefficient as a Data Quality Metric for Variability in Natural Language Data. ICLR 2024 DMLR workshop. <https://arxiv.org/abs/2306.13840>
- Miranda, Yu, Goyal, Wang, Koyejo. Is Pre-training Truly Better Than Meta-Learning? ICML 2023 DMLR workshop. <https://arxiv.org/abs/2306.13841>
- Obbad, Mlauzi, Miranda, Schaeffer, Obbad, Bedi, Koyejo. ZIP-FIT: Embedding-Free Data Selection via Compression-Based Alignment. <https://arxiv.org/abs/2410.18194>
- [The alignment-scaling paper with Varun](https://github.com/madanva/alignment-scaling-laws).
- Blier, Ollivier. The Description Length of Deep Learning Models. NeurIPS 2018. <https://arxiv.org/abs/1802.07044>
- Voita, Titov. Information-Theoretic Probing with Minimum Description Length. EMNLP 2020. <https://aclanthology.org/2020.emnlp-main.14/>
- Grünwald, Vitányi. Shannon Information and Kolmogorov Complexity. <https://arxiv.org/abs/cs/0410002>
- Grünwald. The Minimum Description Length Principle. MIT Press, 2007.
- Rissanen. Modeling by shortest data description. *Automatica* 14(5), 1978.
- Vereshchagin, Vitányi. Kolmogorov's Structure Functions and Model Selection. *IEEE Trans. Inf. Theory* 50(12), 2004. <https://arxiv.org/abs/cs/0204037>
- Vitányi. How Incomputable Is Kolmogorov Complexity? *Entropy* 22:408, 2020. <https://homepages.cwi.nl/~paulv/papers/incomp.pdf>
- Soler-Toscano, Zenil, Delahaye, Gauvrit. Calculating Kolmogorov Complexity from the Output Frequency Distributions of Small Turing Machines. *PLoS ONE* 9(5):e96223, 2014. <https://doi.org/10.1371/journal.pone.0096223>
- Zenil, Hernández-Orozco, Kiani, Soler-Toscano, Rueda-Toicen, Tegnér. A Decomposition Method for Global Evaluation of Shannon Entropy and Local Estimations of Algorithmic Complexity. *Entropy* 20(8):605, 2018. <https://arxiv.org/abs/1609.00110>
- Allender, Buhrman, Koucký. Power from Random Strings (Levin's $Kt$, Def. 1). <https://eccc.weizmann.ac.il/report/2002/028/>
- Ay, Müller, Szkoła. Effective Complexity and its Relation to Logical Depth. <https://arxiv.org/abs/0810.5663>; Gell-Mann, Lloyd. Information Measures, Effective Complexity, and Total Information. *Complexity* 2:44–52, 1996.
- Shalizi, Crutchfield. Computational Mechanics: Pattern and Prediction, Structure and Simplicity. <https://arxiv.org/abs/cond-mat/9907176>; Crutchfield, Young. Inferring Statistical Complexity. *Phys. Rev. Lett.* 63:105, 1989.
- Cilibrasi, Vitányi. Clustering by Compression. <https://arxiv.org/abs/cs/0312044>
- Dinh, Pascanu, Bengio, Bengio. Sharp Minima Can Generalize for Deep Nets. <https://arxiv.org/abs/1703.04933>
- Sipser. Introduction to the Theory of Computation, 2nd ed. (Problem 1.52, Myhill–Nerode; Sect. 6.4, A Definition of Information).

**TLDR-end:** [tdc: Task2Vec↔Kolmogorov note] Task2Vec itself proves nothing about Kolmogorov complexity: the formal link is in Achille et al. 2021, where $K$ is swapped for a KL "information in the weights" whose Fisher form is a second-order, local-minimum special case not connected to $K$ by any theorem. The Task2Vec norm tracks probe-relative uncertainty and the diversity coefficient tracks variability, so both must be validated against exact $K$ (Exp 14) and known $\alpha$ (Exp 15) before being used on real data.
