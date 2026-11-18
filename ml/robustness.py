"""
Main novelty: shortcut learning and evasion robustness of URL phishing detectors.

The LegitPhish paper (Potpelwar et al., Data in Brief 63, 2025) states as a limitation that the
dataset "doesn't include phishing URLs that have been intentionally modified or disguised to
trick detection systems" and lists robustness against evasion as future work. This script does
that work:

  R1  Shortcut audit       - how far can single "shortcut" attributes (HTTPS, www, path) go?
  R2  Robustness benchmark - realistic, cheap attacker edits of TEST phishing URLs and normal
                             variations of TEST legitimate URLs (nothing from the training set)
  R3  Defences             - 5 training strategies x 4 models, compared on clean and robust metrics

    python ml/robustness.py            # full run (~25-40 min)
    python ml/robustness.py --fast     # 20 % sample

All original features are recomputed from the URL with orig_features() (a re-implementation of the
LegitPhish extractor, validated against the published values) so that edited URLs get valid features.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from novelty_features import FREE_HOSTING, SDRS_WEIGHTS, extract_url_features
from run_experiments import NEW_MODELS, PAPER_MODELS, SEED, TEST_SIZE, build_model

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "LegitPhish_url_features.csv"
OUT = ROOT / "public" / "research-data"
FIG = OUT / "figures"

ORIGINAL = ["url_length", "has_ip_address", "dot_count", "https_flag", "url_entropy",
            "token_count", "subdomain_count", "query_param_count", "tld_length", "path_length",
            "has_hyphen_in_domain", "number_of_digits", "tld_popularity",
            "suspicious_file_extension", "domain_name_length", "percentage_numeric_chars"]

# --------------------------------------------------------------------------------------------
# Re-implementation of the 16 LegitPhish features (Table 1 / Algorithm 1 of the paper)
# --------------------------------------------------------------------------------------------
POPULAR_TLDS = {"com", "org", "net", "edu", "gov"}
SUSP_EXT = ("exe", "zip", "bin", "apk", "cmd", "js", "rar", "scr", "bat", "msi")
_IP_RE = re.compile(r"(\d{1,3}\.){3}\d{1,3}")
_EXT_RE = re.compile(r"\.(%s)$" % "|".join(SUSP_EXT))


def _entropy(s: str) -> float:
    n = len(s)
    return -sum(v / n * math.log2(v / n) for v in Counter(s).values()) if n else 0.0


def orig_features(url: str) -> dict:
    p = urlsplit(url)
    h = p.netloc
    labels = h.split(".") if h else [""]
    nd = sum(ch.isdigit() for ch in url)
    tld = labels[-1] if len(labels) > 1 else ""
    return {
        "url_length": len(url),
        "has_ip_address": int(bool(_IP_RE.search(h))),
        "dot_count": url.count("."),
        "https_flag": int(url.lower().startswith("https")),
        "url_entropy": _entropy(url),
        "token_count": len(re.split(r"[/.]", url)),
        "subdomain_count": max(len(labels) - 2, 0),
        "query_param_count": len(p.query.split("&")) if p.query else 1,
        "tld_length": len(tld),
        "path_length": len(p.path),
        "has_hyphen_in_domain": int("-" in h),
        "number_of_digits": nd,
        "tld_popularity": int(tld.split(":")[0] in POPULAR_TLDS),
        "suspicious_file_extension": int(bool(_EXT_RE.search(p.path.lower()))),
        "domain_name_length": len(labels[-2]) if len(labels) > 1 else len(h),
        "percentage_numeric_chars": nd / len(url) * 100 if url else 0.0,
    }


# --------------------------------------------------------------------------------------------
# Feature views
# --------------------------------------------------------------------------------------------
PATH_NOVEL = {"login_in_path_flag", "path_has_password_word", "server_script_form_flag",
              "brand_in_path_flag", "percent_encoding_count", "path_double_slash_flag",
              "login_keyword_count", "credential_harvest_combo", "sdrs_score"}
DOMAIN_DROP = ["https_flag", "path_length", "query_param_count", "suspicious_file_extension"]


def _novel(urls) -> pd.DataFrame:
    N = pd.DataFrame([extract_url_features(u) for u in urls])
    N["sdrs_score"] = sum(N[c] * w for c, w in SDRS_WEIGHTS.items()) / sum(SDRS_WEIGHTS.values()) * 100
    return N


def _host_url(u: str) -> str:
    h = (urlsplit(u).netloc or "").lower()
    return "http://" + (h[4:] if h.startswith("www.") else h)


def featurize(urls, view: str) -> pd.DataFrame:
    """view: 'full' (16 original), 'full+novel' (16+29), 'domain+novel' (host only, no https/path)."""
    urls = list(urls)
    if view == "domain+novel":
        hosts = [_host_url(u) for u in urls]
        O = pd.DataFrame([orig_features(u) for u in hosts]).drop(columns=DOMAIN_DROP)
        N = _novel(hosts)
        N = N[[c for c in N.columns if c not in PATH_NOVEL]]
        return pd.concat([O, N], axis=1)
    O = pd.DataFrame([orig_features(u) for u in urls])
    if view == "full":
        return O
    return pd.concat([O, _novel(urls)], axis=1)


# --------------------------------------------------------------------------------------------
# Attacker edits (phishing) and benign variations (legitimate)
# --------------------------------------------------------------------------------------------
_IPHOST = re.compile(r"^[\d.]+$")


def _is_ip(h: str) -> bool:
    return bool(_IPHOST.match(h or ""))


def attacker_pool(urls):
    """Attacker-chosen names (label + TLD) used by domain-based phishing URLs. On free-hosting
    platforms (x.weebly.com) the attacker's name is the subdomain, not the platform."""
    names, tlds = [], []
    for u in urls:
        h = (urlsplit(u).hostname or "").lower()
        if _is_ip(h) or h.count(".") < 1:
            continue
        parts = [p for p in h.split(".") if p != "www"]
        if any(h.endswith("." + f) for f in FREE_HOSTING) or "ipfs" in h:
            if len(parts) >= 3 and len(parts[0]) >= 4 and re.fullmatch(r"[a-z0-9-]+", parts[0]):
                names.append(parts[0])
            continue
        if len(parts) >= 2 and len(parts[-2]) >= 4:
            names.append(parts[-2])
            tlds.append(parts[-1])
    # unique names (a few domains host hundreds of URLs); TLDs keep their real frequency
    return sorted(set(names)), tlds


def randomize_format(u: str, r: random.Random, inner_page) -> str:
    """Format-randomised training (FRT): the SAME distribution of scheme / www / path form is
    applied to both classes, so these attributes stop carrying label information."""
    u = re.sub(r"^https?://", "https://" if r.random() < 0.5 else "http://", u)
    p = urlsplit(u)
    host = p.netloc
    if not _is_ip(p.hostname or ""):
        bare = host[4:] if host.lower().startswith("www.") else host
        host = ("www." + bare) if r.random() < 0.5 else bare
    k = r.random()
    if k < 1 / 3:
        return urlunsplit((p.scheme, host, "", "", ""))
    if k < 0.5:
        return urlunsplit((p.scheme, host, "/", "", ""))
    u = urlunsplit((p.scheme, host, p.path, p.query, p.fragment))
    return inner_page(u) if inner_page else u


class Attacker:
    """Cheap, realistic edits: buy a domain, get free TLS, add www, use a clean landing path."""

    def __init__(self, names, tlds, seed):
        self.names, self.tlds, self.r = names, tlds, random.Random(seed)

    def domain(self, u):
        p = urlsplit(u)
        if not _is_ip(p.hostname or ""):
            return u
        host = f"{self.r.choice(self.names)}.{self.r.choice(self.tlds)}"
        return urlunsplit((p.scheme, host, p.path, p.query, p.fragment))

    @staticmethod
    def https(u):
        return re.sub(r"^http://", "https://", u)

    @staticmethod
    def www(u):
        p = urlsplit(u)
        h = p.hostname or ""
        if _is_ip(h) or h.startswith("www."):
            return u
        return urlunsplit((p.scheme, "www." + p.netloc, p.path, p.query, p.fragment))

    @staticmethod
    def clean_path(u, slash: bool = False):
        """Bare homepage. Without the slash it copies the exact format of 98.8 % of the
        legitimate URLs (https://www.site.com), the hardest case for a lexical detector."""
        p = urlsplit(u)
        return urlunsplit((p.scheme, p.netloc, "/" if slash else "", "", ""))

    def random_mix(self, u):
        """Training augmentation: the attacker moves to a registered domain (an IP host cannot be
        disguised), then the format is randomised exactly like the legitimate class."""
        u = self.domain(u) if self.r.random() < 0.8 else u
        return randomize_format(u, self.r, inner_page=None)


# Benign variations. Training uses a RANDOM path generator over TRAIN_WORDS; the test uses fixed
# templates over a disjoint vocabulary (TEST_WORDS), so test variations are never seen in training.
TRAIN_WORDS = ["security", "travel", "music", "health", "python", "market", "report", "guide",
               "support", "world", "energy", "design", "about", "contact", "products", "blog",
               "user", "login", "signin", "search", "docs", "category", "page", "item", "profile",
               "settings", "video", "photos", "articles", "forum", "download", "faq", "team"]
TEST_PATHS = ["/account/login", "/news/{y}/{m}/{v}-{v}-{v}-{n}", "/help/{v}", "/shop/item?id={n}",
              "/en/{v}/{v}", "/wiki/{v}_{v}", "/questions/{n}/{v}-{v}", "/events/{y}/{v}"]
TEST_WORDS = ["history", "science", "football", "recipes", "weather", "finance", "careers",
              "museum", "garden", "climate", "library", "students"]


def random_train_path(r: random.Random) -> str:
    segs = []
    for _ in range(r.randint(1, 4)):
        kind = r.random()
        if kind < 0.55:
            seg = r.choice("-_").join(r.choice(TRAIN_WORDS) for _ in range(r.randint(1, 4)))
        elif kind < 0.75:
            seg = str(r.randint(1, 9999999))
        elif kind < 0.9:
            seg = r.choice(TRAIN_WORDS) + str(r.randint(1, 999))
        else:
            seg = "".join(r.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(r.randint(4, 12)))
        segs.append(seg)
    path = "/" + "/".join(segs)
    if r.random() < 0.2:
        path += r.choice([".html", ".htm", ".aspx", ".php", "/"])
    if r.random() < 0.25:
        path += "?" + "&".join(f"{r.choice(TRAIN_WORDS)}={r.randint(1, 999)}" for _ in range(r.randint(1, 3)))
    return path


class Benign:
    def __init__(self, paths, words, seed):
        self.paths, self.words, self.r = paths, words, random.Random(seed)

    def _fill(self, t):
        r = self.r
        for tok, gen in [("{w}", lambda: r.choice(self.words)), ("{v}", lambda: r.choice(self.words)),
                         ("{n}", lambda: str(r.randint(10, 999999))),
                         ("{y}", lambda: str(r.randint(2015, 2026))), ("{m}", lambda: f"{r.randint(1, 12):02d}")]:
            while tok in t:
                t = t.replace(tok, gen(), 1)
        return t

    def inner_page(self, u):
        tail = random_train_path(self.r) if self.paths is None else self._fill(self.r.choice(self.paths))
        return re.sub(r"/+$", "", u) + tail

    @staticmethod
    def http(u):
        return re.sub(r"^https://", "http://", u)

    @staticmethod
    def no_www(u):
        return u.replace("://www.", "://", 1)

    def random_mix(self, u):
        return randomize_format(u, self.r, inner_page=self.inner_page)


def build_benchmark(ph_te, lg_te, names, tlds):
    A = Attacker(names, tlds, seed=101)
    B = Benign(TEST_PATHS, TEST_WORDS, seed=202)
    tests = {
        "P0 Clean phishing": (ph_te, 1),
        "P1 Free HTTPS": ([A.https(u) for u in ph_te], 1),
        "P2 Domain instead of IP": ([A.domain(u) for u in ph_te], 1),
        "P3 Clean landing path": ([A.clean_path(u) for u in ph_te], 1),
        "P4 HTTPS + domain + www": ([A.www(A.https(A.domain(u))) for u in ph_te], 1),
        "P5 Full mimicry (P4 + clean path)": ([A.clean_path(A.www(A.https(A.domain(u)))) for u in ph_te], 1),
        "L0 Clean legitimate": (lg_te, 0),
        "L1 Inner page": ([B.inner_page(u) for u in lg_te], 0),
        "L2 HTTP (no TLS)": ([B.http(u) for u in lg_te], 0),
        "L3 Without www": ([B.no_www(u) for u in lg_te], 0),
        "L4 Inner page over HTTP": ([B.http(B.inner_page(u)) for u in lg_te], 0),
    }
    return tests


DEFENCES = {
    # name: (feature view, training data: "clean" | "mixed" (clean + randomised) | "random" (randomised only))
    "D0 Standard (16 features)": ("full", "clean"),
    "D1 + Suspicious-domain features": ("full+novel", "clean"),
    "D2 Domain-only view": ("domain+novel", "clean"),
    "D3 Format-randomised training (16)": ("full", "MODE"),
    "D4 Pure FRT + suspicious-domain": ("full+novel", "MODE"),
}


def shortcut_audit(df, y, tr, te):
    rows = []
    attrs = {
        "Uses HTTPS": df.URL.str.lower().str.startswith("https").astype(int),
        "Starts with www.": df.URL.str.contains(r"://www\.", regex=True).astype(int),
        "Has a path (not homepage)": (df.URL.map(lambda u: len(urlsplit(u).path)) > 1).astype(int),
        "Path is completely empty (no '/')": (df.URL.map(lambda u: urlsplit(u).path) == "").astype(int),
        "Raw IP host": df.URL.map(lambda u: int(_is_ip(urlsplit(u).hostname or ""))),
        "Non-standard port": df.URL.map(lambda u: int(":" in (urlsplit(u).netloc or ""))),
    }
    for name, s in attrs.items():
        s = s.to_numpy()
        tree = DecisionTreeClassifier(max_depth=1).fit(s[tr].reshape(-1, 1), y[tr])
        acc = (tree.predict(s[te].reshape(-1, 1)) == y[te]).mean()
        rows.append({"Attribute": name,
                     "Pct_of_phishing": s[y == 1].mean() * 100,
                     "Pct_of_legitimate": s[y == 0].mean() * 100,
                     "Accuracy_using_only_this_attribute": acc * 100})
    # two shortcuts together
    combo = np.c_[attrs["Uses HTTPS"], attrs["Starts with www."], attrs["Has a path (not homepage)"]]
    tree = DecisionTreeClassifier(max_depth=3).fit(combo[tr], y[tr])
    rows.append({"Attribute": "HTTPS + www + path together (3 yes/no rules)",
                 "Pct_of_phishing": np.nan, "Pct_of_legitimate": np.nan,
                 "Accuracy_using_only_this_attribute": (tree.predict(combo[te]) == y[te]).mean() * 100})
    return pd.DataFrame(rows)


AUG_MODE = "random"
FA_BUDGET = 0.01


def main(fast: bool):
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA).dropna(subset=["ClassLabel"]).drop_duplicates("URL")
    n_all = len(df)
    df = df[df.URL.str.contains("://")].reset_index(drop=True)  # 0.4 % have no scheme
    if fast:
        df = df.sample(frac=0.2, random_state=SEED).reset_index(drop=True)
    y = (df.ClassLabel == 0).astype(int).to_numpy()
    tr, te = train_test_split(np.arange(len(df)), test_size=TEST_SIZE, stratify=y, random_state=SEED)
    print(f"rows {len(df)} (of {n_all}), train {len(tr)}, test {len(te)}")

    # validation of the feature re-implementation
    R = pd.DataFrame([orig_features(u) for u in df.URL])
    match = {c: float(np.isclose(R[c], df[c], atol=0.01).mean()) for c in ORIGINAL}
    pd.DataFrame({"Feature": list(match), "Match_rate": list(match.values())}).to_csv(
        OUT / "robust_feature_reimplementation_check.csv", index=False)
    print("re-implementation mean match:", round(np.mean(list(match.values())), 4))

    audit = shortcut_audit(df, y, tr, te)
    audit.to_csv(OUT / "robust_shortcut_audit.csv", index=False)
    print(audit.round(2).to_string())

    # 10 % of the training portion is held out only to set each model's decision threshold at a
    # fixed false-alarm budget (FA_BUDGET of clean legitimate validation URLs). Test data is untouched.
    tr_fit, val = train_test_split(tr, test_size=0.10, stratify=y[tr], random_state=SEED)
    U = df.URL.to_numpy()
    U_tr, y_tr = list(U[tr_fit]), y[tr_fit]
    U_val_legit = [u for u, l in zip(U[val], y[val]) if l == 0]
    ph_tr = [u for u, l in zip(U_tr, y_tr) if l == 1]
    lg_tr = [u for u, l in zip(U_tr, y_tr) if l == 0]
    ph_te = [u for u, l in zip(U[te], y[te]) if l == 1]
    lg_te = [u for u, l in zip(U[te], y[te]) if l == 0]

    names_tr, tlds_tr = attacker_pool(ph_tr)
    names_te, tlds_te = attacker_pool(ph_te)
    names_te = [n for n in names_te if n not in set(names_tr)] or names_te  # unseen names at test
    tests = build_benchmark(ph_te, lg_te, names_te, tlds_te)
    examples = [{"Test": k, "Example_URL": v[0][i]} for k, v in tests.items() for i in range(2)]
    pd.DataFrame(examples).to_csv(OUT / "robust_benchmark_examples.csv", index=False)

    # training augmentation: one edited copy of every training URL (training URLs/templates only)
    A_tr = Attacker(names_tr, tlds_tr, seed=7)
    B_tr = Benign(None, TRAIN_WORDS, seed=8)
    # two randomised copies of every training URL; the format-randomised defences train ONLY on
    # these copies, so the original format shortcut is absent from their training data
    aug_urls, aug_y = [], []
    for _ in range(2):
        aug_urls += [A_tr.random_mix(u) for u in ph_tr] + [B_tr.random_mix(u) for u in lg_tr]
        aug_y += [1] * len(ph_tr) + [0] * len(lg_tr)
    aug_y = np.asarray(aug_y, dtype=int)

    cache: dict = {}

    def feats(key, urls, view):
        k = (key, view)
        if k not in cache:
            cache[k] = featurize(urls, view)
        return cache[k]

    rows, summary = [], []
    for dname, (view, aug) in DEFENCES.items():
        if aug == "MODE":
            aug = AUG_MODE
        if aug == "random":
            X_tr, Y = feats("aug", aug_urls, view), aug_y
        elif aug == "mixed":
            X_tr = pd.concat([feats("train", U_tr, view), feats("aug1", aug_urls[:len(U_tr)], view)],
                             ignore_index=True)
            Y = np.r_[y_tr, aug_y[:len(U_tr)]]
        else:
            X_tr, Y = feats("train", U_tr, view), y_tr
        for mname in PAPER_MODELS + NEW_MODELS:
            t0 = time.time()
            model = build_model(mname).fit(X_tr.to_numpy(float), Y)
            p_val = model.predict_proba(feats("val_legit", U_val_legit, view).to_numpy(float))[:, 1]
            thr = float(np.quantile(p_val, 1 - FA_BUDGET))
            res, probs = {}, {}
            for tname, (urls, lab) in tests.items():
                Xt = feats(tname, urls, view).to_numpy(float)
                probs[tname] = model.predict_proba(Xt)[:, 1]
                pred = (probs[tname] > thr).astype(int)
                rate = pred.mean() * 100  # detection rate for P*, false-alarm rate for L*
                res[tname] = rate
                rows.append({"Model": mname, "Defence": dname, "Test": tname,
                             "Kind": "Detection rate" if lab == 1 else "False-alarm rate",
                             "Rate": rate})
            # clean metrics
            Xc = np.r_[feats("P0 Clean phishing", tests["P0 Clean phishing"][0], view).to_numpy(float),
                       feats("L0 Clean legitimate", tests["L0 Clean legitimate"][0], view).to_numpy(float)]
            yc = np.r_[np.ones(len(ph_te)), np.zeros(len(lg_te))]
            prob = model.predict_proba(Xc)[:, 1]
            cpred = prob > thr
            det = np.mean([res[k] for k in tests if k.startswith("P") and k != "P0 Clean phishing"])
            fa = np.mean([res[k] for k in tests if k.startswith("L") and k != "L0 Clean legitimate"])
            worst_det = min(res[k] for k in tests if k.startswith("P"))
            worst_fa = max(res[k] for k in tests if k.startswith("L"))
            def auc(pos, neg):
                sp = np.concatenate([probs[k] for k in pos]); sn = np.concatenate([probs[k] for k in neg])
                return roc_auc_score(np.r_[np.ones(len(sp)), np.zeros(len(sn))], np.r_[sp, sn]) * 100
            P_att = [k for k in tests if k[:2] in ("P1", "P2", "P3", "P4", "P5")]
            L_var = [k for k in tests if k[:2] in ("L1", "L2", "L3", "L4")]
            summary.append({
                "Model": mname, "Defence": dname,
                "Threshold": thr,
                "Clean_accuracy": (cpred == yc).mean() * 100,
                "Clean_F1": f1_score(yc, cpred) * 100,
                "Clean_ROC_AUC": roc_auc_score(yc, prob) * 100,
                "Avg_detection_under_attack": det,
                "Worst_case_detection": worst_det,
                "Avg_false_alarm_on_variations": fa,
                "Worst_case_false_alarm": worst_fa,
                "Robust_balanced_accuracy": (det + (100 - fa)) / 2,
                "Robust_AUC": auc(P_att, L_var),
                "Mimicry_AUC": auc(["P5 Full mimicry (P4 + clean path)"],
                                   ["L0 Clean legitimate", "L2 HTTP (no TLS)", "L3 Without www"]),
                "Train_seconds": round(time.time() - t0, 1),
            })
            s = summary[-1]
            print(f"{dname:48s} {mname:18s} clean F1={s['Clean_F1']:.2f} "
                  f"det={det:6.2f} FA={fa:6.2f} RBA={s['Robust_balanced_accuracy']:.2f} "
                  f"rAUC={s['Robust_AUC']:.2f} mAUC={s['Mimicry_AUC']:.2f}",
                  flush=True)

    long = pd.DataFrame(rows)
    long.to_csv(OUT / "robust_benchmark_results.csv", index=False)
    summ = pd.DataFrame(summary)
    summ.to_csv(OUT / "robust_summary.csv", index=False)

    # heat-map figure for the paper (rows = defence x model, columns = tests)
    piv = long.pivot_table(index=["Defence", "Model"], columns="Test", values="Rate", sort=False)
    fig, ax = plt.subplots(figsize=(13, 0.32 * len(piv) + 2))
    shown = piv.copy()
    good = shown.copy()
    for c in shown.columns:  # colour = "goodness": detection high is good, false alarm low is good
        good[c] = shown[c] if c.startswith("P") else 100 - shown[c]
    ax.imshow(good.to_numpy(), cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    for (i, j), v in np.ndenumerate(shown.to_numpy()):
        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(shown.columns)), [c.split(" ", 1)[0] for c in shown.columns])
    ax.set_yticks(range(len(shown)), [f"{d.split(' ', 1)[0]} {m}" for d, m in shown.index], fontsize=7)
    ax.set_title("Detection rate (P*) and false-alarm rate (L*), % - green is better", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "robustness_heatmap.png", dpi=200)
    plt.close(fig)

    meta_p = OUT / "experiment_meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    meta.update({"robust_rows": int(len(df)), "robust_train": int(len(tr_fit)), "robust_val": int(len(val)),
                 "robust_test": int(len(te)), "robust_fa_budget": FA_BUDGET, "robust_aug_mode": AUG_MODE,
                 "robust_aug_rows": int(len(aug_urls)),
                 "robust_reimpl_match": round(float(np.mean(list(match.values()))), 4),
                 "robust_fast_mode": fast})
    meta_p.write_text(json.dumps(meta, indent=2))
    print(summ.round(2).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--aug-mode", default="random", choices=["mixed", "random"])
    a = ap.parse_args()
    globals()['AUG_MODE'] = a.aug_mode
    main(a.fast)
