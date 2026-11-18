"""
LegitPhish phishing-URL detection - full reproducible experiment pipeline.

    python ml/run_experiments.py                 # full run
    python ml/run_experiments.py --fast          # quick check (20% sample)

Protocol (same for every model):
  * Dataset   : LegitPhish (Potpelwar et al., Data in Brief 2025), 17 lexical features
  * Cleaning  : drop rows with missing label, drop exact duplicate URLs
  * Split     : stratified 80 % train / 20 % test, random_state = 42
  * Positive  : phishing (original label 0 is mapped to 1 = phishing)
  * Metrics   : Accuracy, Precision, Recall, F1-score, ROC-AUC (+ MCC, confusion matrix)

Experiments:
  E1  Two baseline models from the reference paper       (original 17 features)
  E2  Two new models not used in the reference paper     (original 17 features)
  E3  Novelty: same 4 models with 17 + novel suspicious-domain features
  E4  Novelty: feature-family ablation (which new family helps)
  E5  Novelty: leakage-aware domain-grouped 80/20 split (no domain in both train and test)
  E6  5-fold stratified CV on the training set (mean +- std) for the 4 models
  E7  Novelty: low-data learning curve (0.5 % ... 100 % of the training set)
  E8  Novelty: detection rate per phishing attack category
  E9  Novelty: IP-free hard benchmark (all raw-IP URLs removed)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import (GroupShuffleSplit, StratifiedGroupKFold,
                                     StratifiedKFold,
                                     cross_validate, train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from novelty_features import (FEATURE_FAMILIES, NOVEL_FEATURES,
                              add_novel_features, registered_domain)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "LegitPhish_url_features.csv"
OUT = ROOT / "public" / "research-data"
FIG = OUT / "figures"
SEED = 42
TEST_SIZE = 0.20

# ---------------------------------------------------------------------------
# >>> EDIT HERE: the two models used in the reference research paper <<<
PAPER_MODELS = ["Random Forest", "Gradient Boosting"]
# >>> the two NEW models (not used in the reference paper) <<<
NEW_MODELS = ["XGBoost", "LightGBM"]
# ---------------------------------------------------------------------------

ORIGINAL_FEATURES = [
    "url_length", "has_ip_address", "dot_count", "https_flag", "url_entropy",
    "token_count", "subdomain_count", "query_param_count", "tld_length",
    "path_length", "has_hyphen_in_domain", "number_of_digits", "tld_popularity",
    "suspicious_file_extension", "domain_name_length", "percentage_numeric_chars",
]  # 16 numeric features + URL string = the 17 features of LegitPhish


def build_model(name: str):
    models = {
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=SEED),
        "Gradient Boosting": lambda: GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.1, max_depth=3, random_state=SEED),
        "XGBoost": lambda: XGBClassifier(
            n_estimators=400, learning_rate=0.1, max_depth=6, subsample=0.9,
            colsample_bytree=0.9, eval_metric="logloss", n_jobs=-1, random_state=SEED),
        "LightGBM": lambda: LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.9,
            subsample_freq=1, colsample_bytree=0.9, random_state=SEED, verbose=-1),
        "Decision Tree": lambda: DecisionTreeClassifier(random_state=SEED),
        "Extra Trees": lambda: ExtraTreesClassifier(
            n_estimators=300, n_jobs=-1, random_state=SEED),
        "AdaBoost": lambda: AdaBoostClassifier(n_estimators=300, random_state=SEED),
        "Logistic Regression": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000)),
        "KNN": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "Naive Bayes": lambda: GaussianNB(),
    }
    return models[name]()


ALL_MODELS = ["Random Forest", "Gradient Boosting", "XGBoost", "LightGBM",
              "Decision Tree", "Extra Trees", "AdaBoost", "Logistic Regression",
              "KNN", "Naive Bayes"]


def evaluate(model, X_te, y_te) -> tuple[dict, np.ndarray]:
    prob = model.predict_proba(X_te)[:, 1]
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
    m = {
        "Accuracy": accuracy_score(y_te, pred),
        "Precision": precision_score(y_te, pred, zero_division=0),
        "Recall": recall_score(y_te, pred, zero_division=0),
        "F1": f1_score(y_te, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_te, prob),
        "MCC": matthews_corrcoef(y_te, pred),
        "Specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }
    return m, prob


def fit_eval(name, X_tr, y_tr, X_te, y_te):
    model = build_model(name)
    t0 = time.time()
    model.fit(X_tr, y_tr)
    secs = time.time() - t0
    m, prob = evaluate(model, X_te, y_te)
    m["Train_seconds"] = round(secs, 2)
    return model, m, prob


def importance(model, cols) -> pd.DataFrame | None:
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    if not hasattr(est, "feature_importances_"):
        return None
    imp = np.asarray(est.feature_importances_, dtype=float)
    imp = imp / imp.sum() if imp.sum() else imp
    return (pd.DataFrame({"Feature": cols, "Importance": imp})
            .sort_values("Importance", ascending=False).reset_index(drop=True))


def load_data(fast: bool) -> pd.DataFrame:
    df = pd.read_csv(DATA)
    n_raw = len(df)
    df = df.dropna(subset=["ClassLabel"])
    df = df.drop_duplicates(subset=["URL"]).reset_index(drop=True)
    # original: 0 = phishing, 1 = legitimate  ->  model target: 1 = phishing
    df["phishing"] = (df["ClassLabel"] == 0).astype(int)
    if fast:
        df = df.sample(frac=0.2, random_state=SEED).reset_index(drop=True)
    print(f"rows raw={n_raw}  after cleaning={len(df)}  "
          f"phishing={df.phishing.sum()}  legitimate={(1 - df.phishing).sum()}")
    df = add_novel_features(df)
    df["group_domain"] = df["URL"].map(registered_domain)
    df.attrs["n_raw"] = n_raw
    return df


def plot_roc(curves: dict, path: Path, title: str):
    plt.figure(figsize=(6, 5))
    for name, (fpr, tpr, auc) in curves.items():
        plt.plot(fpr, tpr, lw=1.8, label=f"{name} (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(title)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_cm(m: dict, name: str, path: Path):
    cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=11)
    ax.set_xticks([0, 1], ["Legitimate", "Phishing"])
    ax.set_yticks([0, 1], ["Legitimate", "Phishing"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(name, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)



def hard_subset_experiment(df: pd.DataFrame):
    """E9: IP-free benchmark. 77 % of LegitPhish phishing URLs use a raw IP host, which makes
    the task almost trivial. Here every IP-host URL is removed, so only domain-based phishing
    remains - the setting the suspicious-domain features are designed for."""
    main_models = PAPER_MODELS + NEW_MODELS
    sub = df[df["ip_literal_host"] == 0].reset_index(drop=True)
    y = sub["phishing"].to_numpy()
    print(f"IP-free subset: {len(sub)} rows, phishing={y.sum()}, legitimate={len(y) - y.sum()}")
    tr, te = train_test_split(np.arange(len(sub)), test_size=TEST_SIZE, stratify=y,
                              random_state=SEED)
    enhanced = ORIGINAL_FEATURES + NOVEL_FEATURES
    fsets = [("Original (16)", ORIGINAL_FEATURES), (f"Enhanced (16+{len(NOVEL_FEATURES)})", enhanced)]
    models = main_models + ["Logistic Regression"]
    rows, curves = [], {}
    # (a) stratified 80/20 - the main protocol
    for fset_name, cols in fsets:
        X = sub[cols].to_numpy(dtype=float)
        for name in models:
            _, m, prob = fit_eval(name, X[tr], y[tr], X[te], y[te])
            rows.append({"Model": name, "Feature_set": fset_name, "Split": "Stratified 80/20",
                         "Fold": 0, **m})
            print(f"[IP-free strat] [{fset_name:15s}] {name:19s} F1={m['F1']:.4f} "
                  f"FN={m['FN']} FP={m['FP']}")
            if name in main_models:
                fpr, tpr, _ = roc_curve(y[te], prob)
                curves.setdefault(fset_name, {})[name] = (fpr, tpr, m["ROC_AUC"])
    # (b) 5-fold domain-grouped CV: every fold is an 80/20 split with no shared domain
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for k, (a, b) in enumerate(sgkf.split(sub, y, groups=sub["group_domain"])):
        for fset_name, cols in fsets:
            X = sub[cols].to_numpy(dtype=float)
            for name in models:
                _, m, _ = fit_eval(name, X[a], y[a], X[b], y[b])
                rows.append({"Model": name, "Feature_set": fset_name,
                             "Split": "Domain-grouped 5-fold", "Fold": k + 1, **m})
        print(f"IP-free grouped fold {k + 1}/5 done")
    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "novelty_ip_free_benchmark_raw.csv", index=False)
    agg_cols = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "MCC", "FN", "FP"]
    res = raw.groupby(["Model", "Feature_set", "Split"], sort=False)[agg_cols].agg(["mean", "std"])
    res.columns = [f"{a}_{b}" for a, b in res.columns]
    res = res.reset_index()
    res["Errors_mean"] = res["FN_mean"] + res["FP_mean"]
    res.to_csv(OUT / "novelty_ip_free_benchmark.csv", index=False)
    for fset_name, c in curves.items():
        tag = "original" if fset_name.startswith("Original") else "enhanced"
        plot_roc(c, FIG / f"roc_ip_free_{tag}.png", f"ROC - IP-free subset, {fset_name}")
    meta_p = OUT / "experiment_meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    meta.update({"ip_free_rows": int(len(sub)), "ip_free_phishing": int(y.sum()),
                 "ip_free_legitimate": int(len(y) - y.sum())})
    meta_p.write_text(json.dumps(meta, indent=2))
    return res


def main(fast: bool, skip_pool: bool):
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = load_data(fast)
    y = df["phishing"].to_numpy()
    enhanced = ORIGINAL_FEATURES + NOVEL_FEATURES
    main_models = PAPER_MODELS + NEW_MODELS
    role = {m: "Reference-paper model" for m in PAPER_MODELS}
    role.update({m: "New model (this work)" for m in NEW_MODELS})

    # ---------------- stratified 80/20 split (the main protocol)
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=TEST_SIZE,
                                      stratify=y, random_state=SEED)
    # ---------------- domain-grouped 80/20 split (leakage-aware)
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    g_tr, g_te = next(gss.split(df, y, groups=df["group_domain"]))
    splits = {"Stratified 80/20": (idx_tr, idx_te), "Domain-grouped 80/20": (g_tr, g_te)}

    rows, roc_json, rocs_plot = [], {}, {}
    for split_name, (tr, te) in splits.items():
        for fset_name, cols in [("Original (16)", ORIGINAL_FEATURES),
                                (f"Enhanced (16+{len(NOVEL_FEATURES)})", enhanced)]:
            X = df[cols].to_numpy(dtype=float)
            for name in main_models:
                model, m, prob = fit_eval(name, X[tr], y[tr], X[te], y[te])
                rows.append({"Model": name, "Role": role[name], "Feature_set": fset_name,
                             "Split": split_name, **m})
                print(f"[{split_name:20s}] [{fset_name:15s}] {name:18s} "
                      f"Acc={m['Accuracy']:.4f} F1={m['F1']:.4f} AUC={m['ROC_AUC']:.4f}")
                fpr, tpr, _ = roc_curve(y[te], prob)
                key = f"{name} | {fset_name} | {split_name}"
                sel = np.unique(np.linspace(0, len(fpr) - 1, 200).astype(int))
                roc_json[key] = {"fpr": fpr[sel].round(5).tolist(),
                                 "tpr": tpr[sel].round(5).tolist(), "auc": m["ROC_AUC"]}
                rocs_plot.setdefault((split_name, fset_name), {})[name] = (fpr, tpr, m["ROC_AUC"])
                if split_name == "Stratified 80/20":
                    tag = name.lower().replace(" ", "_")
                    fs = "original" if fset_name.startswith("Original") else "enhanced"
                    plot_cm(m, f"{name} ({fs})", FIG / f"cm_{tag}_{fs}.png")
                    imp = importance(model, cols)
                    if imp is not None:
                        imp.to_csv(OUT / f"feature_importance_{tag}_{fs}.csv", index=False)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "all_results.csv", index=False)

    # E1 + E2 : main table (original features, stratified 80/20)
    main_tbl = res[(res.Split == "Stratified 80/20") & res.Feature_set.str.startswith("Original")]
    main_tbl.to_csv(OUT / "main_four_models.csv", index=False)
    # E3 : novelty table (original vs enhanced, both splits)
    piv = res.pivot_table(index=["Model", "Split"], columns="Feature_set",
                          values=["Accuracy", "F1", "ROC_AUC", "Recall", "FN", "FP"])
    piv.columns = [f"{a}__{b}" for a, b in piv.columns]
    piv = piv.reset_index()
    o, e = "Original (16)", f"Enhanced (16+{len(NOVEL_FEATURES)})"
    for k in ["Accuracy", "F1", "ROC_AUC", "Recall"]:
        piv[f"{k}_delta"] = piv[f"{k}__{e}"] - piv[f"{k}__{o}"]
    piv.to_csv(OUT / "novelty_original_vs_enhanced.csv", index=False)

    for (split_name, fset_name), curves in rocs_plot.items():
        tag = ("strat" if split_name.startswith("Strat") else "grouped") + "_" + \
              ("original" if fset_name.startswith("Original") else "enhanced")
        plot_roc(curves, FIG / f"roc_{tag}.png", f"ROC - {split_name}, {fset_name}")
    (OUT / "roc_curves.json").write_text(json.dumps(roc_json))

    # E4 : feature-family ablation (each family added alone to the original features)
    abl = []
    for split_name, (tr, te) in splits.items():
        for fam, fcols in [("None (original)", [])] + list(FEATURE_FAMILIES.items()) + \
                          [("All families", NOVEL_FEATURES)]:
            cols = ORIGINAL_FEATURES + fcols
            X = df[cols].to_numpy(dtype=float)
            for name in NEW_MODELS:
                _, m, _ = fit_eval(name, X[tr], y[tr], X[te], y[te])
                abl.append({"Model": name, "Split": split_name, "Added_family": fam,
                            "n_features": len(cols), **m})
    abl = pd.DataFrame(abl)
    base = abl[abl.Added_family == "None (original)"].set_index(["Model", "Split"])["F1"]
    abl["F1_delta_vs_original"] = abl.apply(lambda r: r.F1 - base[(r.Model, r.Split)], axis=1)
    abl.to_csv(OUT / "novelty_family_ablation.csv", index=False)

    # E6 : 5-fold CV on the training portion
    cv_rows = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fset_name, cols in [("Original (16)", ORIGINAL_FEATURES), (e, enhanced)]:
        X = df[cols].to_numpy(dtype=float)
        for name in main_models:
            cv = cross_validate(build_model(name), X[idx_tr], y[idx_tr], cv=skf, n_jobs=1,
                                scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])
            r = {"Model": name, "Feature_set": fset_name}
            for s in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
                r[f"{s}_mean"] = cv[f"test_{s}"].mean()
                r[f"{s}_std"] = cv[f"test_{s}"].std()
            cv_rows.append(r)
            print(f"CV {fset_name:15s} {name:18s} F1={r['f1_mean']:.4f}+-{r['f1_std']:.4f}")
    pd.DataFrame(cv_rows).to_csv(OUT / "cv5_train_results.csv", index=False)

    # E7 : low-data learning curve (does the novel knowledge help when labels are scarce?)
    lc = []
    rng = np.random.RandomState(SEED)
    for frac in [0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.00]:
        n = max(50, int(len(idx_tr) * frac))
        for rep in range(3 if frac < 0.25 else 1):
            sub_idx, _ = (train_test_split(idx_tr, train_size=n, stratify=y[idx_tr],
                                           random_state=SEED + rep)
                          if frac < 1 else (idx_tr, None))
            for fset_name, cols in [("Original (16)", ORIGINAL_FEATURES), (e, enhanced)]:
                X = df[cols].to_numpy(dtype=float)
                for name in NEW_MODELS + PAPER_MODELS:
                    _, m, _ = fit_eval(name, X[sub_idx], y[sub_idx], X[idx_te], y[idx_te])
                    lc.append({"Model": name, "Feature_set": fset_name, "Train_fraction": frac,
                               "Train_rows": len(sub_idx), "Repeat": rep, **m})
        print(f"learning curve: fraction {frac} done")
    lc = pd.DataFrame(lc)
    lc.to_csv(OUT / "novelty_learning_curve_raw.csv", index=False)
    lc_s = (lc.groupby(["Model", "Feature_set", "Train_fraction", "Train_rows"])
            [["Accuracy", "F1", "ROC_AUC", "FN", "FP"]].mean().reset_index())
    lc_s.to_csv(OUT / "novelty_learning_curve.csv", index=False)
    fig, axes = plt.subplots(1, len(main_models), figsize=(4 * len(main_models), 3.4), sharey=True)
    for ax, name in zip(np.atleast_1d(axes), main_models):
        for fset_name, style in [("Original (16)", "o--"), (e, "s-")]:
            d = lc_s[(lc_s.Model == name) & (lc_s.Feature_set == fset_name)]
            ax.plot(d.Train_rows, d.F1, style, label=fset_name)
        ax.set_xscale("log")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Training URLs (log scale)")
    np.atleast_1d(axes)[0].set_ylabel("Test F1-score")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "learning_curve.png", dpi=200)
    plt.close(fig)

    # E8 : detection rate per phishing attack category (stratified test set)
    cats = {
        "IP-address host": df["ip_literal_host"] == 1,
        "Fake-login pattern": (df["login_in_host_flag"] + df["login_in_path_flag"]) > 0,
        "Misleading subdomain": (df["tld_token_in_subdomain"] + df["full_domain_in_subdomain"]
                                 + df["random_looking_subdomain"]) > 0,
        "Brand impersonation / typosquat": (df["brand_domain_mismatch_flag"]
                                            + df["typosquat_flag"]) > 0,
        "Free-hosting platform": df["free_hosting_flag"] == 1,
        "HTTPS phishing (looks safe)": df["https_flag"] == 1,
        "Short URL (< 30 chars)": df["url_length"] < 30,
    }
    te_mask = np.zeros(len(df), bool)
    te_mask[idx_te] = True
    cat_rows = []
    for fset_name, cols in [("Original (16)", ORIGINAL_FEATURES), (e, enhanced)]:
        X = df[cols].to_numpy(dtype=float)
        for name in main_models:
            model = build_model(name).fit(X[idx_tr], y[idx_tr])
            pred = model.predict(X)
            for cname, cmask in cats.items():
                sel = te_mask & cmask.to_numpy() & (y == 1)
                leg = te_mask & cmask.to_numpy() & (y == 0)
                cat_rows.append({
                    "Model": name, "Feature_set": fset_name, "Category": cname,
                    "Phishing_in_test": int(sel.sum()),
                    "Detected": int(pred[sel].sum()),
                    "Detection_rate": pred[sel].mean() if sel.sum() else np.nan,
                    "Legit_in_test": int(leg.sum()),
                    "False_alarms": int(pred[leg].sum())})
    pd.DataFrame(cat_rows).to_csv(OUT / "novelty_attack_category_recall.csv", index=False)

    # Extra: all other classifiers on the same split (context for the paper)
    if not skip_pool:
        pool = []
        for fset_name, cols in [("Original (16)", ORIGINAL_FEATURES), (e, enhanced)]:
            X = df[cols].to_numpy(dtype=float)
            for name in ALL_MODELS:
                _, m, _ = fit_eval(name, X[idx_tr], y[idx_tr], X[idx_te], y[idx_te])
                pool.append({"Model": name, "Feature_set": fset_name, **m})
                print(f"pool {fset_name:15s} {name:20s} F1={m['F1']:.4f}")
        pd.DataFrame(pool).to_csv(OUT / "all_classifiers.csv", index=False)

    # Novel-feature prevalence by class (for the paper's novelty table)
    prev = []
    for c in NOVEL_FEATURES:
        prev.append({"Feature": c,
                     "Family": next(f for f, cs in FEATURE_FAMILIES.items() if c in cs),
                     "Mean_phishing": df.loc[df.phishing == 1, c].mean(),
                     "Mean_legitimate": df.loc[df.phishing == 0, c].mean(),
                     "Nonzero_phishing_pct": (df.loc[df.phishing == 1, c] > 0).mean() * 100,
                     "Nonzero_legitimate_pct": (df.loc[df.phishing == 0, c] > 0).mean() * 100})
    pd.DataFrame(prev).to_csv(OUT / "novel_feature_statistics.csv", index=False)

    df.drop(columns=["phishing"]).to_csv(OUT / "legitphish_enhanced_dataset.csv", index=False)

    meta = {
        "rows_raw": int(df.attrs["n_raw"]), "rows_clean": int(len(df)),
        "phishing": int(y.sum()), "legitimate": int(len(y) - y.sum()),
        "train_rows": int(len(idx_tr)), "test_rows": int(len(idx_te)),
        "grouped_train_rows": int(len(g_tr)), "grouped_test_rows": int(len(g_te)),
        "unique_domains": int(df.group_domain.nunique()),
        "paper_models": PAPER_MODELS, "new_models": NEW_MODELS,
        "n_original_features": len(ORIGINAL_FEATURES), "n_novel_features": len(NOVEL_FEATURES),
        "seed": SEED, "test_size": TEST_SIZE, "fast_mode": fast,
    }
    (OUT / "experiment_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))

    # E9 : IP-free hard benchmark
    hard_subset_experiment(df)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="20%% sample for a quick check")
    ap.add_argument("--skip-pool", action="store_true", help="skip the extra 10-classifier table")
    ap.add_argument("--only-ip-free", action="store_true", help="run only E9 (IP-free benchmark)")
    a = ap.parse_args()
    if a.only_ip_free:
        FIG.mkdir(parents=True, exist_ok=True)
        hard_subset_experiment(load_data(a.fast))
    else:
        main(a.fast, a.skip_pool)
