"""Pair classifier: one small CNN tower shared by the two images, a 128-unit trunk layer and one linear head per task."""
import torch
import torch.nn as nn

WIDTH = {'mnist': 16, 'fashion': 16, 'kmnist': 16, 'cifar10': 32}


class Tower(nn.Module):
    def __init__(self, cin, w):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, w, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(w, 2 * w, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(2 * w, 4 * w, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(2), nn.Flatten(), nn.Linear(16 * w, 64), nn.ReLU())

    def forward(self, x):
        return self.net(x)


class PairNet(nn.Module):
    def __init__(self, cin, w, heads, seed):
        """heads: {task_id: n_classes}. The tower and trunk are initialized from `seed` alone, so every run of a
        (dataset, seed) starts from the same weights; each head from (seed, task) alone."""
        super().__init__()
        torch.manual_seed(seed)
        self.tower = Tower(cin, w)
        self.trunk = nn.Sequential(nn.Linear(128, 128), nn.ReLU())
        self.heads = nn.ModuleDict()
        for t, c in sorted(heads.items()):
            from data import stable
            torch.manual_seed(stable('head', seed, t) % (2 ** 31))
            self.heads[t] = nn.Linear(128, c)

    def features(self, x):
        b = x.shape[0]
        f = self.tower(x.flatten(0, 1)).view(b, 128)
        return self.trunk(f)

    def forward(self, x):
        z = self.features(x)
        return {t: h(z) for t, h in self.heads.items()}


def n_params(m, trunk_only=False):
    ps = list(m.tower.parameters()) + list(m.trunk.parameters()) if trunk_only else list(m.parameters())
    return sum(p.numel() for p in ps)
