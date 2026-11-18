"""
X4: character-level CNN on the raw URL string, trained (a) normally and (b) with
format-randomised training (FRT). Shows whether FRT also helps a deep model that reads the URL
text directly (and can therefore see "https://www." as a shortcut).

    python ml/char_cnn.py            # seed 42, ~30 min on CPU
Each variant is trained on the same number of URLs (the size of the clean training set); the mixed
variant samples them from clean URLs + two randomised copies (about 1/3 clean, 2/3 copies).
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

import extended as X

torch.set_num_threads(2)
MAXLEN = 160
VOCAB = {c: i + 2 for i, c in enumerate(
    "abcdefghijklmnopqrstuvwxyz0123456789-._~:/?#[]@!$&'()*+,;=%")}  # 0 pad, 1 unknown


def encode(urls) -> torch.Tensor:
    out = np.zeros((len(urls), MAXLEN), dtype=np.int64)
    for i, u in enumerate(urls):
        s = u.lower()[:MAXLEN]
        out[i, :len(s)] = [VOCAB.get(ch, 1) for ch in s]
    return torch.from_numpy(out)


class CharCNN(nn.Module):
    def __init__(self, emb=32, ch=128):
        super().__init__()
        self.emb = nn.Embedding(len(VOCAB) + 2, emb, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb, ch, k, padding=k // 2) for k in (3, 5, 7)])
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(3 * ch, 1)

    def forward(self, x):
        e = self.emb(x).transpose(1, 2)
        h = torch.cat([torch.relu(c(e)).amax(dim=2) for c in self.convs], dim=1)
        return self.fc(self.drop(h)).squeeze(1)


def train(urls, y, epochs=2, seed=42, batch=256):
    torch.manual_seed(seed)
    X_ = encode(urls)
    Y = torch.tensor(y, dtype=torch.float32)
    net = CharCNN()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    lossf = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(X_))
        tot = 0.0
        for s in range(0, len(X_), batch):
            idx = perm[s:s + batch]
            opt.zero_grad()
            loss = lossf(net(X_[idx]), Y[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        print(f"  epoch {ep + 1}: loss {tot / len(X_):.4f}", flush=True)
    return net


def predict(net, urls):
    net.eval()
    X_ = encode(urls)
    with torch.no_grad():
        return torch.cat([torch.sigmoid(net(X_[s:s + 4096])) for s in range(0, len(X_), 4096)]).numpy()


class Wrapped:
    """Gives the CNN the predict_proba(features) interface used by extended.py (features = URLs)."""

    def __init__(self, net):
        self.net = net

    def predict_proba(self, urls):
        p = predict(self.net, list(urls))
        return np.c_[1 - p, p]


def main():
    U, y = X.load_legitphish()
    S = X.prepare(U, y, 42)
    PU, Py = X.load_phiusiil(set(U))
    groups = X.phi_groups(PU)
    # make extended.feat() return the raw URLs for this "view"
    X.feat = lambda urls, view: np.asarray(list(urls), dtype=object)
    rows = []
    variants = {
        "Char-CNN, standard training": (S["U_tr"], S["y_tr"]),
        "Char-CNN, FRT (randomised copies only)": (S["aug_urls"], S["aug_y"]),
        "Char-CNN, mixed FRT (clean + 2 copies)": (list(S["U_tr"]) + list(S["aug_urls"]),
                                                   np.r_[S["y_tr"], S["aug_y"]]),
    }
    # every variant gets the same number of training URLs (= size of the clean training set)
    rng = np.random.RandomState(42)
    n = len(S["U_tr"])
    for name, (urls, labels) in variants.items():
        urls, labels = np.asarray(list(urls), dtype=object), np.asarray(labels)
        if len(urls) > n:
            idx = rng.choice(len(urls), n, replace=False)
            urls, labels = urls[idx], labels[idx]
        t0 = time.time()
        print(name, len(urls), "training URLs", flush=True)
        net = train(list(urls), np.asarray(labels))
        m = Wrapped(net)
        thr = float(np.quantile(m.predict_proba(S["U_val_legit"])[:, 1], 1 - X.FA_BUDGET))
        b = X.bench_metrics(m, "raw", thr, S["tests"])
        ph, _ = X.phi_metrics(m, "raw", thr, PU, Py, groups)
        rates = {k: (m.predict_proba(v[0])[:, 1] > thr).mean() * 100 for k, v in S["tests"].items()}
        rows.append({"Model": name, **b, **ph, **{f"Rate | {k}": v for k, v in rates.items()},
                     "Seconds": round(time.time() - t0)})
        print({k: round(v, 2) for k, v in rows[-1].items() if isinstance(v, float)}, flush=True)
        pd.DataFrame(rows).to_csv(X.OUT / "x4_char_cnn.csv", index=False)


if __name__ == "__main__":
    main()
