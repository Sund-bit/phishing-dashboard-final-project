"""
Extended validation of the robustness novelty (for publication).

  X1  Shortcut audit of a second public dataset (PhiUSIIL, Prasad & Chandra 2024)
  X2  Cross-dataset test: every defence trained on LegitPhish, tested on real, unseen PhiUSIIL URLs
  X3  Explanation (TreeSHAP): which features the standard and the proposed XGBoost rely on
  X5  Repeated runs with different seeds + paired significance tests (Wilcoxon signed-rank)

    python ml/extended.py --part x2                  # X1 + X2 + X3 (seed 42), ~20 min
    python ml/extended.py --part seeds --seeds 1 2 3 4   # X5 extra seeds, ~15 min per seed
    python ml/extended.py --part stats               # significance tests from the seed runs

PhiUSIIL file expected at data/PhiUSIIL_Phishing_URL_Dataset.csv
(https://archive.ics.uci.edu/dataset/967/phiusil-phishing-url-dataset). Label 1 = legitimate, 0 = phishing.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.parse import urlsplit

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

import robustness as R
from run_experiments import NEW_MODELS, PAPER_MODELS, TEST_SIZE, build_model

ROOT = Path(__file__).resolve().parents[1]
PHI = ROOT / "data" / "PhiUSIIL_Phishing_URL_Dataset.csv"
OUT = ROOT / "public" / "research-data"
FIG = OUT / "figures"
MODELS = PAPER_MODELS + NEW_MODELS
FA_BUDGET = 0.01

# D0-D4 come from robustness.py; D5/D6 are the mixed format-randomised training proposed after the
# cross-dataset test: clean training URLs + k randomised copies of each (k = 1 or 2).
DEFENCES = dict(R.DEFENCES)
DEFENCES["D5 Mixed FRT (clean + 1 copy) + suspicious-domain"] = ("full+novel", "mixed1")
DEFENCES["D6 Proposed: mixed FRT (clean + 2 copies) + suspicious-domain"] = ("full+novel", "mixed2")

# ------------------------------------------------------------------------------------------
# URL-level feature cache (the novel features are the slow part)
_CACHE: dict = {}
_COLS: dict = {}


def feat(urls, view) -> np.ndarray:
    c = _CACHE.setdefault(view, {})
    missing = [u for u in dict.fromkeys(urls) if u not in c]
    for s in range(0, len(missing), 50000):
        part = missing[s:s + 50000]
        F = R.featurize(part, view)
        _COLS[view] = list(F.columns)
        for u, row in zip(part, F.to_numpy(float)):
            c[u] = row
    return np.vstack([c[u] for u in urls])


def cols(view):
    if view not in _COLS:
        feat(["http://example.com"], view)
    return _COLS[view]


# ------------------------------------------------------------------------------------------
def load_legitphish():
    df = pd.read_csv(R.DATA).dropna(subset=["ClassLabel"]).drop_duplicates("URL")
    df = df[df.URL.str.contains("://")].reset_index(drop=True)
    y = (df.ClassLabel == 0).astype(int).to_numpy()  # 1 = phishing
    return df.URL.to_numpy(), y


def load_phiusiil(exclude: set):
    d = pd.read_csv(PHI, usecols=["URL", "label"]).drop_duplicates("URL")
    n0 = len(d)
    d = d[~d.URL.isin(exclude) & d.URL.str.contains("://")].reset_index(drop=True)
    y = (d.label == 0).astype(int).to_numpy()  # 1 = phishing (PhiUSIIL: label 0 = phishing)
    print(f"PhiUSIIL: {n0} unique URLs, {len(d)} after removing LegitPhish overlap; phishing={y.sum()}")
    return d.URL.to_numpy(), y


def prepare(U, y, seed: int):
    """Same protocol as robustness.py; seed 42 reproduces its numbers exactly."""
    off = seed - 42
    tr, te = train_test_split(np.arange(len(U)), test_size=TEST_SIZE, stratify=y, random_state=seed)
    tr_fit, val = train_test_split(tr, test_size=0.10, stratify=y[tr], random_state=seed)
    U_tr, y_tr = list(U[tr_fit]), y[tr_fit]
    ph_tr = [u for u, l in zip(U_tr, y_tr) if l == 1]
    lg_tr = [u for u, l in zip(U_tr, y_tr) if l == 0]
    ph_te = [u for u, l in zip(U[te], y[te]) if l == 1]
    lg_te = [u for u, l in zip(U[te], y[te]) if l == 0]
    names_tr, tlds_tr = R.attacker_pool(ph_tr)
    names_te, tlds_te = R.attacker_pool(ph_te)
    names_te = [n for n in names_te if n not in set(names_tr)] or names_te
    A_te = R.Attacker(names_te, tlds_te, seed=101 + off)
    B_te = R.Benign(R.TEST_PATHS, R.TEST_WORDS, seed=202 + off)
    tests = {
        "P0 Clean phishing": (ph_te, 1),
        "P1 Free HTTPS": ([A_te.https(u) for u in ph_te], 1),
        "P2 Domain instead of IP": ([A_te.domain(u) for u in ph_te], 1),
        "P3 Clean landing path": ([A_te.clean_path(u) for u in ph_te], 1),
        "P4 HTTPS + domain + www": ([A_te.www(A_te.https(A_te.domain(u))) for u in ph_te], 1),
        "P5 Full mimicry (P4 + clean path)": ([A_te.clean_path(A_te.www(A_te.https(A_te.domain(u)))) for u in ph_te], 1),
        "L0 Clean legitimate": (lg_te, 0),
        "L1 Inner page": ([B_te.inner_page(u) for u in lg_te], 0),
        "L2 HTTP (no TLS)": ([B_te.http(u) for u in lg_te], 0),
        "L3 Without www": ([B_te.no_www(u) for u in lg_te], 0),
        "L4 Inner page over HTTP": ([B_te.http(B_te.inner_page(u)) for u in lg_te], 0),
    }
    A_tr = R.Attacker(names_tr, tlds_tr, seed=7 + off)
    B_tr = R.Benign(None, R.TRAIN_WORDS, seed=8 + off)
    aug_urls, aug_y = [], []
    for _ in range(2):
        aug_urls += [A_tr.random_mix(u) for u in ph_tr] + [B_tr.random_mix(u) for u in lg_tr]
        aug_y += [1] * len(ph_tr) + [0] * len(lg_tr)
    return {"U_tr": U_tr, "y_tr": y_tr, "U_val_legit": [u for u, l in zip(U[val], y[val]) if l == 0],
            "tests": tests, "aug_urls": aug_urls, "aug_y": np.asarray(aug_y), "train_set": set(U[tr])}


def fit(defence, model_name, S):
    view, mode = DEFENCES[defence]
    if mode == "MODE":
        mode = "random"
    n = len(S["U_tr"])
    if mode == "random":
        X, Y = feat(S["aug_urls"], view), S["aug_y"]
    elif mode in ("mixed1", "mixed2"):
        k = n if mode == "mixed1" else 2 * n
        X = np.vstack([feat(S["U_tr"], view), feat(S["aug_urls"][:k], view)])
        Y = np.r_[S["y_tr"], S["aug_y"][:k]]
    else:
        X, Y = feat(S["U_tr"], view), S["y_tr"]
    m = build_model(model_name).fit(X, Y)
    p_val = m.predict_proba(feat(S["U_val_legit"], view))[:, 1]
    return m, view, float(np.quantile(p_val, 1 - FA_BUDGET))


def bench_metrics(m, view, thr, tests):
    probs, rates = {}, {}
    for k, (urls, _) in tests.items():
        probs[k] = m.predict_proba(feat(urls, view))[:, 1]
        rates[k] = (probs[k] > thr).mean() * 100
    P = [k for k in tests if k[:2] in ("P1", "P2", "P3", "P4", "P5")]
    L = [k for k in tests if k[:2] in ("L1", "L2", "L3", "L4")]
    det = np.mean([rates[k] for k in P])
    fa = np.mean([rates[k] for k in L])

    def auc(pos, neg):
        sp = np.concatenate([probs[k] for k in pos])
        sn = np.concatenate([probs[k] for k in neg])
        return roc_auc_score(np.r_[np.ones(len(sp)), np.zeros(len(sn))], np.r_[sp, sn]) * 100

    yc = np.r_[np.ones(len(probs["P0 Clean phishing"])), np.zeros(len(probs["L0 Clean legitimate"]))]
    pc = np.r_[probs["P0 Clean phishing"], probs["L0 Clean legitimate"]]
    return {"Clean_F1": f1_score(yc, pc > thr) * 100,
            "Avg_detection_under_attack": det, "Avg_false_alarm_on_variations": fa,
            "Robust_balanced_accuracy": (det + 100 - fa) / 2,
            "Robust_AUC": auc(P, L),
            "Mimicry_AUC": auc(["P5 Full mimicry (P4 + clean path)"],
                               ["L0 Clean legitimate", "L2 HTTP (no TLS)", "L3 Without www"])}


def phi_groups(urls):
    p = [urlsplit(u) for u in urls]
    https = np.array([x.scheme == "https" for x in p])
    www = np.array([(x.hostname or "").startswith("www.") for x in p])
    home = np.array([x.path in ("", "/") and not x.query for x in p])
    return {"Looks like legitimate format (HTTPS + www + homepage)": https & www & home,
            "HTTPS, other formats": https & ~(www & home),
            "Plain HTTP": ~https}


def phi_metrics(m, view, thr, U, y, groups):
    p = m.predict_proba(feat(list(U), view))[:, 1]
    pred = p > thr
    out = {"PhiUSIIL_Accuracy": accuracy_score(y, pred) * 100,
           "PhiUSIIL_Precision": precision_score(y, pred, zero_division=0) * 100,
           "PhiUSIIL_Recall": recall_score(y, pred) * 100,
           "PhiUSIIL_F1": f1_score(y, pred) * 100,
           "PhiUSIIL_ROC_AUC": roc_auc_score(y, p) * 100,
           "PhiUSIIL_Balanced_accuracy": (recall_score(y, pred) + (1 - pred[y == 0].mean())) / 2 * 100,
           "PhiUSIIL_False_alarm": pred[y == 0].mean() * 100}
    for g, mask in groups.items():
        sel = mask & (y == 1)
        out[f"Detect | {g}"] = pred[sel].mean() * 100 if sel.any() else np.nan
    # Format-matched test on REAL data: phishing URLs written exactly like the legitimate ones
    # (HTTPS + www + homepage) against the legitimate URLs (which all have that format).
    fm = groups["Looks like legitimate format (HTTPS + www + homepage)"]
    sel = (fm & (y == 1)) | (y == 0)
    out["PhiUSIIL_Format_matched_AUC"] = roc_auc_score(y[sel], p[sel]) * 100
    return out, p


# ------------------------------------------------------------------------------------------
def part_x2():
    FIG.mkdir(parents=True, exist_ok=True)
    U, y = load_legitphish()
    S = prepare(U, y, 42)
    PU, Py = load_phiusiil(set(U))

    # X1: shortcut audit of PhiUSIIL (80/20 split of PhiUSIIL itself)
    pdf = pd.DataFrame({"URL": PU})
    ptr, pte = train_test_split(np.arange(len(PU)), test_size=TEST_SIZE, stratify=Py, random_state=42)
    audit = R.shortcut_audit(pdf, Py, ptr, pte)
    audit.insert(0, "Dataset", "PhiUSIIL")
    leg = pd.read_csv(OUT / "robust_shortcut_audit.csv")
    leg.insert(0, "Dataset", "LegitPhish")
    pd.concat([leg, audit]).to_csv(OUT / "x1_shortcut_audit_two_datasets.csv", index=False)
    print(audit.round(2).to_string())

    groups = phi_groups(PU)
    gsize = {g: int((m & (Py == 1)).sum()) for g, m in groups.items()}
    rows, shap_rows = [], []
    for d in DEFENCES:
        for mn in MODELS:
            t0 = time.time()
            m, view, thr = fit(d, mn, S)
            b = bench_metrics(m, view, thr, S["tests"])
            ph, _ = phi_metrics(m, view, thr, PU, Py, groups)
            rows.append({"Seed": 42, "Defence": d, "Model": mn, **b, **ph, "Seconds": round(time.time() - t0)})
            print(f"[x2] {d:48s} {mn:18s} RBA={b['Robust_balanced_accuracy']:.1f} "
                  f"PhiAUC={ph['PhiUSIIL_ROC_AUC']:.2f} PhiF1={ph['PhiUSIIL_F1']:.2f} "
                  f"PhiFA={ph['PhiUSIIL_False_alarm']:.2f}", flush=True)
            if mn == "XGBoost" and d[:2] in ("D0", "D4", "D6"):
                shap_rows += shap_table(m, view, d, S, PU)
            pd.DataFrame(rows).to_csv(OUT / "x2_cross_dataset_phiusiil.csv", index=False)
    pd.DataFrame(shap_rows).to_csv(OUT / "x3_shap_xgboost.csv", index=False)
    (OUT / "x2_meta.json").write_text(json.dumps({
        "phiusiil_rows": int(len(PU)), "phiusiil_phishing": int(Py.sum()),
        "phiusiil_legitimate": int(len(Py) - Py.sum()), "phishing_group_sizes": gsize,
        "fa_budget": FA_BUDGET}, indent=2))
    pd.DataFrame(rows).to_csv(OUT / "seeds_raw.csv", index=False)
    fig_x2()
    fig_shap()


# ------------------------------------------------------------------------------------------
FORMAT_FEATURES = {"https_flag", "path_length", "query_param_count", "suspicious_file_extension",
                   "url_length", "token_count", "dot_count", "subdomain_count", "url_entropy",
                   "number_of_digits", "percentage_numeric_chars"}


def shap_table(m, view, defence, S, PU):
    import xgboost as xgb
    rng = np.random.RandomState(0)
    rows = []
    sets = {"LegitPhish test": S["tests"]["P0 Clean phishing"][0] + S["tests"]["L0 Clean legitimate"][0],
            "PhiUSIIL": list(PU)}
    for name, urls in sets.items():
        idx = rng.choice(len(urls), size=min(4000, len(urls)), replace=False)
        X = feat([urls[i] for i in idx], view)
        contrib = m.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
        imp = np.abs(contrib).mean(0)
        share = imp / imp.sum()
        for c, v, s in zip(cols(view), imp, share):
            group = "Format / structure" if c in FORMAT_FEATURES else (
                "Suspicious-domain (novel)" if c not in R.ORIGINAL else "Domain name")
            rows.append({"Defence": defence, "Data": name, "Feature": c, "Group": group,
                         "Mean_abs_SHAP": v, "Share_pct": s * 100})
    return rows


def fig_shap():
    d = pd.read_csv(OUT / "x3_shap_xgboost.csv")
    d = d[d.Data == "PhiUSIIL"]
    fig, axes = plt.subplots(1, len(d.Defence.unique()), figsize=(5.5 * len(d.Defence.unique()), 4.2))
    colors = {"Format / structure": "#f87171", "Domain name": "#60a5fa", "Suspicious-domain (novel)": "#34d399"}
    for ax, dname in zip(axes, sorted(d.Defence.unique())):
        t = d[d.Defence == dname].sort_values("Share_pct", ascending=False).head(12)[::-1]
        ax.barh(t.Feature, t.Share_pct, color=[colors[g] for g in t.Group])
        ax.set_title(dname[:44], fontsize=9)
        ax.set_xlabel("Share of total |SHAP| on PhiUSIIL (%)")
        ax.tick_params(axis="y", labelsize=7)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    fig.legend(handles, colors.keys(), loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG / "x3_shap_xgboost.png", dpi=200)
    plt.close(fig)


def fig_x2():
    d = pd.read_csv(OUT / "x2_cross_dataset_phiusiil.csv")
    piv = d.pivot_table(index="Model", columns="Defence", values="PhiUSIIL_ROC_AUC", sort=False)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    w = 0.8 / len(piv.columns)
    for i, c in enumerate(piv.columns):
        ax.bar(np.arange(len(piv)) + i * w, piv[c], w, label=c[:48])
    ax.set_xticks(np.arange(len(piv)) + w * (len(piv.columns) - 1) / 2, piv.index)
    ax.set_ylabel("ROC-AUC on PhiUSIIL (%)")
    ax.set_ylim(40, 100)
    ax.axhline(50, color="k", ls="--", lw=1)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("Trained on LegitPhish, tested on unseen PhiUSIIL URLs", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "x2_cross_dataset_auc.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------------------------------
SEED_DEFENCES = ["D0 Standard (16 features)", "D1 + Suspicious-domain features",
                 "D4 Pure FRT + suspicious-domain",
                 "D6 Proposed: mixed FRT (clean + 2 copies) + suspicious-domain"]


def part_seeds(seeds):
    U, y = load_legitphish()
    PU, Py = load_phiusiil(set(U))
    groups = phi_groups(PU)
    raw_p = OUT / "seeds_raw.csv"
    rows = pd.read_csv(raw_p).to_dict("records") if raw_p.exists() else []
    for seed in seeds:
        S = prepare(U, y, seed)
        for d in SEED_DEFENCES:
            for mn in MODELS:
                m, view, thr = fit(d, mn, S)
                b = bench_metrics(m, view, thr, S["tests"])
                ph, _ = phi_metrics(m, view, thr, PU, Py, groups)
                rows = [r for r in rows if not (r["Seed"] == seed and r["Defence"] == d and r["Model"] == mn)]
                rows.append({"Seed": seed, "Defence": d, "Model": mn, **b, **ph})
                print(f"[seed {seed}] {d:48s} {mn:18s} RBA={b['Robust_balanced_accuracy']:.1f} "
                      f"PhiAUC={ph['PhiUSIIL_ROC_AUC']:.2f}", flush=True)
                pd.DataFrame(rows).to_csv(raw_p, index=False)
        # free memory of this seed's augmented URLs
        for v in list(_CACHE):
            for u in S["aug_urls"]:
                _CACHE[v].pop(u, None)
    part_stats()


def part_stats():
    d = pd.read_csv(OUT / "seeds_raw.csv")
    d = d[d.Defence.isin(SEED_DEFENCES)]
    metrics = ["Robust_balanced_accuracy", "Robust_AUC", "Mimicry_AUC", "Clean_F1",
               "PhiUSIIL_ROC_AUC", "PhiUSIIL_F1", "PhiUSIIL_Balanced_accuracy", "PhiUSIIL_Format_matched_AUC"]
    agg = d.groupby(["Defence", "Model"])[metrics].agg(["mean", "std", "count"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg.reset_index().to_csv(OUT / "x5_seed_summary.csv", index=False)
    tests = []
    for a, b in [("D6", "D0"), ("D6", "D1"), ("D6", "D4"), ("D1", "D0")]:
        A = d[d.Defence.str.startswith(a)].set_index(["Seed", "Model"])
        B = d[d.Defence.str.startswith(b)].set_index(["Seed", "Model"])
        idx = A.index.intersection(B.index)
        for mt in metrics:
            diff = (A.loc[idx, mt] - B.loc[idx, mt]).to_numpy()
            p = wilcoxon(diff).pvalue if len(diff) >= 6 and np.any(diff != 0) else np.nan
            tests.append({"Comparison": f"{a} vs {b}", "Metric": mt, "Pairs (seed x model)": len(diff),
                          "Mean_difference": diff.mean(), "Pairs_improved": int((diff > 0).sum()),
                          "Wilcoxon_p_value": p})
    t = pd.DataFrame(tests)
    t.to_csv(OUT / "x5_significance.csv", index=False)
    print(t.round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["x2", "seeds", "stats"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3, 4])
    a = ap.parse_args()
    if a.part == "x2":
        part_x2()
    elif a.part == "seeds":
        part_seeds(a.seeds)
    else:
        part_stats()
