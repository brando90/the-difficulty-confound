# Experiment 18: vision replication of complexity versus alignment and hard-to-easy transfer (small CNNs, CPU)

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/18_vision_cnn_complexity_vs_alignment/README.md>

Experiment 18 asks Experiment 13-small's two questions in a second modality. Small convolutional networks classify pairs of images (MNIST, Fashion-MNIST, KMNIST, CIFAR-10) under label rules of exactly known complexity (1 to 6.64 bits: parity, thresholds, digit identity, sums and differences of the two digits). Alignment is the share of training pairs drawn from the test distribution; the rest are rotated (or, in a sweep, rescaled, thickened, thinned, or swapped for another dataset). It measures whether alignment dominates complexity (elasticity ratio $E$; Experiment 13-small found 0.80, below 1) and, as Brando's Putnam-versus-GSM8K question, whether training only on hard tasks carries easy ones at matched compute.

Status: FROZEN (10-19-2026) after two disclosed calibration pilots (PROTOCOL Section 6.1); running on Sherlock's CPU partition. Design and predictions: [PROTOCOL.md](PROTOCOL.md). Code: [`expt_v1/`](expt_v1/). Sherlock CPU batch launch: [`launch/`](launch/README.md).

**TLDR-end:** [tdc: Expt 18 readme] Small CNNs on MNIST-family and CIFAR-10 image pairs with exactly counted label complexity and a train/test-shift alignment share test whether alignment dominates complexity (Experiment 13-small's E = 0.80 direction) and whether hard-only training carries easy targets at matched compute; 1,072 CPU runs, pre-registered gates.
