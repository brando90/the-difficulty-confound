"""Small pre-LN transformers trained from scratch (PROTOCOL Section 2.6): a ViT for 32x32 images and a word-level
text encoder. Both classify from a [CLS] token."""
import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, d, heads, mlp, drop):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, mlp), nn.GELU(), nn.Dropout(drop), nn.Linear(mlp, d), nn.Dropout(drop))

    def forward(self, x, pad_mask=None):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, key_padding_mask=pad_mask, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


def _init(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class ViT(nn.Module):
    def __init__(self, n_classes, patch=4, dim=192, depth=6, heads=3, mlp=768, drop=0.0, img=32):
        super().__init__()
        self.patch = nn.Conv2d(3, dim, patch, stride=patch)
        n = (img // patch) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n + 1, dim))
        self.blocks = nn.ModuleList(Block(dim, heads, mlp, drop) for _ in range(depth))
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_classes)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.apply(_init)

    def features(self, x):
        x = self.patch(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], 1) + self.pos
        for b in self.blocks:
            x = b(x)
        return self.ln(x[:, 0])

    def forward(self, x):
        return self.head(self.features(x))


class TextTransformer(nn.Module):
    PAD, UNK, CLS = 0, 1, 2

    def __init__(self, vocab, n_classes, max_len=256, dim=256, depth=4, heads=4, mlp=1024, drop=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim, padding_idx=self.PAD)
        self.pos = nn.Parameter(torch.zeros(1, max_len, dim))
        self.drop = nn.Dropout(drop)
        self.blocks = nn.ModuleList(Block(dim, heads, mlp, drop) for _ in range(depth))
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_classes)
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.apply(_init)
        nn.init.normal_(self.emb.weight, std=0.02)
        with torch.no_grad():
            self.emb.weight[self.PAD].zero_()

    def features(self, ids):
        pad = ids == self.PAD
        x = self.drop(self.emb(ids) + self.pos[:, :ids.shape[1]])
        for b in self.blocks:
            x = b(x, pad)
        return self.ln(x[:, 0])

    def forward(self, ids):
        return self.head(self.features(ids))


def n_params(m):
    return sum(p.numel() for p in m.parameters())
