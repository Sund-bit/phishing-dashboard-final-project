# Beyond 99.9%: Shortcut-Robust Phishing URL Detection on LegitPhish

**Author:** Sundas Maria Hayat (Roll No. 232015)

**Problem.** Phishing URL detectors trained on LegitPhish (Potpelwar et al., *Data in Brief* 63, 111972, 2025)
reach about 99.9 % accuracy, but they learn format shortcuts of the dataset instead of phishing behaviour.
Cheap attacker edits (free HTTPS, a registered domain instead of an IP, a bare homepage URL) evade them,
and normal legitimate pages (inner pages, plain-HTTP sites) are flagged as phishing. The LegitPhish paper
lists exactly this as a limitation (no "intentionally modified or disguised" phishing URLs) and names
robustness against evasion as future work.

**What this repository contains**

* **Baseline:** Random Forest and Gradient Boosting, the standard classifiers for this task. The LegitPhish
  paper is a dataset paper and trains no models itself.
* **New models:** XGBoost and LightGBM.
* **Protocol:** stratified 80 % / 20 % train/test split, `random_state = 42`. Metrics: Accuracy, Precision,
  Recall, F1-score, ROC-AUC (+ MCC, confusion matrix).
* **Main novelty:**
  1. a shortcut audit of LegitPhish
  2. a robustness benchmark of attacker edits and normal page variations, built only from test URLs
  3. **mixed format-randomised training (FRT)** combined with **29 suspicious-domain features** and an
     interpretable risk score (SDRS), validated on a second dataset (PhiUSIIL), with a character-level
     CNN and with significance tests

## Project structure

```
data/LegitPhish_url_features.csv     raw dataset (101,219 URLs, 16 numeric features + URL + label)
ml/novelty_features.py               the 29 novel features + SDRS score + domain grouping
ml/run_experiments.py                standard benchmark E1-E9 (edit PAPER_MODELS / NEW_MODELS at the top)
ml/robustness.py                     main novelty R1-R3: shortcut audit, robustness benchmark, defences
ml/requirements.txt                  Python dependencies
public/research-data/                all results (CSV / JSON) + figures/ (PNG, 200 dpi, for the paper)
src/app/                             Angular dashboard that reads public/research-data/
.github/workflows/deploy.yml         builds the dashboard and deploys it to GitHub Pages
```

## Running the experiments

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
pip install -r ml/requirements.txt

python ml/run_experiments.py --fast      # quick test on a 20 % sample (~2 min)
python ml/run_experiments.py             # full run (~25 min), writes public/research-data/
python ml/run_experiments.py --only-ip-free   # only the IP-free benchmark (E9)
python ml/robustness.py --fast           # main novelty, quick test (~5 min)
python ml/robustness.py                  # main novelty, full run (~25 min)
```

## Running the dashboard locally

```bash
npm ci
npm start            # http://localhost:4200
```

---

## Main novelty: shortcut audit and robustness

### R1. Shortcut audit (`robust_shortcut_audit.csv`)

| Attribute | % of phishing | % of legitimate | Accuracy using only this attribute |
|---|---|---|---|
| Uses HTTPS | 5.65 | 99.99 | 96.50 |
| Starts with `www.` | 0.53 | 99.24 | 99.39 |
| Has a path (not a homepage) | 99.49 | 1.05 | 99.29 |
| Path completely empty (no `/`) | 0.00 | 98.82 | 99.55 |
| Raw IP host | 78.62 | 0.00 | 86.52 |
| HTTPS + www + path (3 yes/no rules) | | | **99.58** |

Three yes/no rules about the URL's format reach 99.58 %, without looking at the website name at all.

### R2. Robustness benchmark (built only from test-set URLs)

| Test | What changes | Example |
|---|---|---|
| P1 | Free HTTPS | `https://123.190.79.28:46286/bin.sh` |
| P2 | Registered domain instead of an IP (attacker names from unseen test phishing) | `http://tqshoes.com/bin.sh` |
| P3 | Bare homepage, formatted exactly like legitimate URLs (no `/`) | `http://123.190.79.28:46286` |
| P4 | HTTPS + domain + www | `https://www.limacamposadv.pm/bin.sh` |
| P5 | Full mimicry (P4 + bare homepage) | `https://www.djtool-for-spotify.org` |
| L1 | Legitimate inner page (templates and words never used in training) | `https://www.greekislandhouse.com/questions/429336/library-museum` |
| L2 | Legitimate site over plain HTTP | `http://www.greekislandhouse.com` |
| L3 | Legitimate site without www | `https://greekislandhouse.com` |
| L4 | Inner page over HTTP | `http://www.greekislandhouse.com/help/museum` |

Every model's decision threshold is set on a held-out validation set so that **1 %** of clean
legitimate URLs are flagged. All defences are therefore compared at the same false-alarm budget.

### R3. Defences

* **D0 Standard:** 16 original features, clean data.
* **D1:** D0 + 29 suspicious-domain features.
* **D2 Domain-only view:** the host name only (ignores HTTPS and the path).
* **D3 Format-randomised training (FRT):** the same random scheme / www / path changes are applied to
  *both* classes (phishing additionally moves from an IP to a registered domain), so the format carries
  no label information. Only training URLs and training vocabularies are used.
* **D4 Pure FRT:** FRT + 29 suspicious-domain features.

### Results (full data, test set, 1 % false-alarm budget)

RBA = robust balanced accuracy = mean of (average detection under attack, 100 − average false alarm on variations).

| Model | Strategy | Clean F1 % | Avg detection under attack % | Worst-case detection % | Avg false alarm on variations % | RBA % | Robust AUC % | Mimicry AUC % |
|---|---|---|---|---|---|---|---|---|
| Random Forest | D0 Standard (16 features) | 99.69 | 81.9 | 11.9 | 95.60 | **43.2** | 67.3 | 22.3 |
| Gradient Boosting | D0 Standard (16 features) | 99.69 | 78.9 | 0.0 | 76.86 | **51.0** | 69.7 | 25.7 |
| XGBoost | D0 Standard (16 features) | 99.71 | 80.3 | 9.6 | 68.52 | **55.9** | 68.9 | 30.5 |
| LightGBM | D0 Standard (16 features) | 99.73 | 81.4 | 9.6 | 90.28 | **45.5** | 69.6 | 25.2 |
| Random Forest | D1 + Suspicious-domain features | 99.72 | 83.5 | 18.1 | 95.82 | **43.9** | 69.1 | 24.9 |
| Gradient Boosting | D1 + Suspicious-domain features | 99.73 | 79.5 | 2.4 | 76.57 | **51.4** | 70.0 | 29.1 |
| XGBoost | D1 + Suspicious-domain features | 99.76 | 82.3 | 18.1 | 79.70 | **51.3** | 68.8 | 32.2 |
| LightGBM | D1 + Suspicious-domain features | 99.71 | 82.0 | 13.4 | 95.39 | **43.3** | 70.0 | 25.5 |
| Random Forest | D2 Domain-only view | 98.52 | 52.9 | 22.9 | 1.39 | **75.8** | 83.7 | 72.9 |
| Gradient Boosting | D2 Domain-only view | 97.78 | 55.7 | 28.3 | 0.62 | **77.5** | 85.3 | 76.0 |
| XGBoost | D2 Domain-only view | 98.44 | 57.9 | 30.9 | 1.24 | **78.3** | 84.5 | 74.2 |
| LightGBM | D2 Domain-only view | 98.49 | 57.9 | 30.9 | 1.38 | **78.3** | 83.5 | 72.6 |
| Random Forest | D3 Format-randomised training (16) | 94.77 | 68.6 | 25.4 | 0.45 | **84.1** | 96.0 | 77.0 |
| Gradient Boosting | D3 Format-randomised training (16) | 97.15 | 76.2 | 28.2 | 0.37 | **87.9** | 96.7 | 80.8 |
| XGBoost | D3 Format-randomised training (16) | 98.50 | 80.4 | 35.4 | 0.51 | **90.0** | 97.0 | 78.8 |
| LightGBM | D3 Format-randomised training (16) | 98.51 | 80.4 | 35.1 | 0.40 | **90.0** | 97.0 | 78.7 |
| Random Forest | D4 Pure FRT + suspicious-domain | 96.18 | 69.9 | 29.4 | 0.45 | **84.7** | 96.5 | 79.2 |
| Gradient Boosting | D4 Pure FRT + suspicious-domain | 98.03 | 82.0 | 38.4 | 0.62 | **90.7** | 97.1 | 80.5 |
| XGBoost | D4 Pure FRT + suspicious-domain | 99.09 | 85.7 | 43.3 | 0.77 | **92.5** | 97.5 | 80.9 |
| LightGBM | D4 Pure FRT + suspicious-domain | 99.11 | 85.0 | 41.0 | 0.58 | **92.2** | 97.5 | 80.9 |

**Findings**

1. The standard models (D0) flag 69–96 % of normal legitimate page variations as phishing. They detect
   only 0–12 % of fully disguised phishing. A Mimicry AUC of 22–30 % means disguised phishing looks *more*
   legitimate to them than real legitimate pages do.
2. The 29 suspicious-domain features alone (D1) do not fix this, because the shortcuts dominate.
3. The domain-only view (D2) removes false alarms but misses most attacks.
4. Format-randomised training (D3) raises RBA to 84–90 % and robust AUC to 96–97 %.
5. **Proposed D4** is the best strategy for all four models (RBA 84.7–92.5 %, false alarms 0.45–0.77 %).
   Once the shortcuts are removed, the suspicious-domain features add +0.7 to +2.8 RBA points over D3.
   XGBoost (a new model) is the most robust.
6. Fully disguised phishing (P5) is still the hardest case (29–43 % detection at a 1 % false-alarm
   budget). A lexical URL alone cannot settle it; it needs page content, domain age or reputation (future work).

Figure for the paper: `public/research-data/figures/robustness_heatmap.png`.

## Extended validation for publication (`ml/extended.py`, `ml/char_cnn.py`)

Second dataset: **PhiUSIIL** (Prasad & Chandra, *Computers & Security*, 2024),
https://archive.ics.uci.edu/dataset/967/phiusil-phishing-url-dataset, saved as
`data/PhiUSIIL_Phishing_URL_Dataset.csv`. 36,935 of its URLs also appear in LegitPhish and were
removed, leaving 198,435 URLs (100,520 phishing / 97,915 legitimate). PhiUSIIL is never used for
training or threshold tuning.

```bash
python ml/extended.py --part x2                    # X1 audit + X2 cross-dataset + X3 SHAP (~25 min)
python ml/extended.py --part seeds --seeds 1 2     # X5 extra seeds + significance tests (~60 min)
python ml/char_cnn.py                              # X4 character-level CNN (~30 min)
```

**X1. The shortcut exists in both public datasets.** PhiUSIIL's legitimate URLs are also 100 % HTTPS,
100 % `www.` and 100 % bare homepages. Three yes/no format rules reach 99.44 % on PhiUSIIL (LegitPhish: 99.58 %).

**X2. Cross-dataset testing does not reveal the shortcut.** Because both datasets share it, the standard
model still scores 99.4–99.8 % AUC on PhiUSIIL. Removing the shortcut completely (pure FRT, D4) gives
the best synthetic robustness but drops PhiUSIIL AUC to about 79 %. **Mixed FRT (D6, proposed):** clean
URLs plus two randomised copies of each, with the 29 suspicious-domain features. It keeps both.

| Model | Strategy | Clean F1 % | Robust balanced acc. % | PhiUSIIL AUC % | Format-matched AUC % (real) |
|---|---|---|---|---|---|
| XGBoost | D0 Standard | 99.71 | 55.9 | 99.65 | 85.3 |
| XGBoost | D4 Pure FRT | 99.09 | 92.5 | 79.06 | 78.2 |
| XGBoost | **D6 Mixed FRT (proposed)** | 99.55 | **90.8** | **92.30** | 83.9 |
| Gradient Boosting | D0 Standard | 99.69 | 51.0 | 99.69 | 83.8 |
| Gradient Boosting | **D6 Mixed FRT (proposed)** | 99.22 | **91.2** | **93.78** | 85.4 |

"Format-matched" uses only the 1,052 **real** PhiUSIIL phishing URLs written exactly like legitimate ones
(HTTPS + www + homepage), against the legitimate URLs, so no format cue can help. All strategies stay at
or below 87 % there. This is the hard limit of URL-only detection, and the reason page content or domain
age is future work.

**X3. Explanation (TreeSHAP, XGBoost, PhiUSIIL).** The standard model takes 89 % of its decision from
format / structure features. The proposed model takes 40 % from them and 45 % from the suspicious-domain
features (`figures/x3_shap_xgboost.png`).

**X4. Character-level CNN on the raw URL.** The shortcut is not specific to tree models:

| Char-CNN | Clean F1 % | Robust balanced acc. % | PhiUSIIL AUC % |
|---|---|---|---|
| Standard training | 99.73 | 41.6 | 99.83 |
| Pure FRT | 99.06 | 91.6 | 83.25 |
| **Mixed FRT (proposed)** | 99.55 | **91.7** | **92.54** |

**X5. Repeated runs and significance.** Seeds 42, 1 and 2 × 4 models = 12 pairs, tested with the Wilcoxon
signed-rank test (`x5_significance.csv`):
* **D6 vs D0:** robust balanced accuracy +41.0 points (12 / 12 pairs, p < 0.001). The cost is clean F1 −0.30
  and PhiUSIIL AUC −6.6. PhiUSIIL shares the shortcut, so this drop is the shortcut no longer being used.
* **D6 vs D4:** about the same robustness (+1.4, not significant), but PhiUSIIL AUC +14.9 and
  F1 +21.8 (12 / 12, p < 0.001). Mixed FRT is clearly better than pure FRT.
* **D1 vs D0 (the 29 features):** format-matched real-data AUC +2.4 (12 / 12, p < 0.001). The features
  help on real phishing that copies the legitimate format.

## Experiments

| ID | Experiment | Output file |
|----|-----------|-------------|
| E1 + E2 | 4 models, original 16 features, stratified 80/20 | `main_four_models.csv` |
| E3 | Original vs enhanced features (stratified and domain-grouped 80/20) | `novelty_original_vs_enhanced.csv` |
| E4 | Feature-family ablation (IP, LOGIN, SUBDOM, BRAND, HOSTING, OBFUSC, RISK) | `novelty_family_ablation.csv` |
| E5 | Domain-grouped split (no domain in both train and test) | inside `all_results.csv` |
| E6 | 5-fold stratified CV on the training set (mean ± std) | `cv5_train_results.csv` |
| E7 | Low-data learning curve (0.5 % to 100 % of training data) | `novelty_learning_curve.csv` |
| E8 | Detection rate per phishing attack type | `novelty_attack_category_recall.csv` |
| E9 | **IP-free hard benchmark** (all raw-IP URLs removed) | `novelty_ip_free_benchmark.csv` |
| extra | 10 classifiers, original vs enhanced | `all_classifiers.csv` |
| extra | Statistics of each novel feature by class | `novel_feature_statistics.csv` |

## Main results (test set, stratified 80/20, original features)

| Model | Role | Accuracy % | Precision % | Recall % | F1 % | ROC-AUC % | FN | FP |
|---|---|---|---|---|---|---|---|---|
| Random Forest | Reference-paper model | 99.95 | 99.92 | 99.99 | 99.96 | 99.993 | 1 | 10 |
| Gradient Boosting | Reference-paper model | 99.95 | 99.95 | 99.97 | 99.96 | 99.999 | 4 | 6 |
| XGBoost | New model (this work) | 99.96 | 99.94 | 99.99 | 99.96 | 99.999 | 1 | 8 |
| LightGBM | New model (this work) | 99.96 | 99.94 | 99.98 | 99.96 | 99.999 | 2 | 7 |

## Supporting analysis: IP-free benchmark

77 % of LegitPhish phishing URLs use a raw IP host (most are URLHaus malware-distribution links), so on
the full dataset every model is above 99.9 % and new features can add little. With every IP-host URL
removed, 51,977 domain-based URLs remain (14,597 phishing / 37,380 legitimate):

| Split | Model | F1 % original | F1 % enhanced | Errors original | Errors enhanced | Change |
|---|---|---|---|---|---|---|
| Stratified 80/20 | Random Forest | 99.829 | 99.846 | 10.0 | 9.0 | −10.0 % |
| Stratified 80/20 | Gradient Boosting | 99.795 | 99.863 | 12.0 | 8.0 | −33.3 % |
| Stratified 80/20 | XGBoost | 99.812 | 99.846 | 11.0 | 9.0 | −18.2 % |
| Stratified 80/20 | LightGBM | 99.795 | 99.846 | 12.0 | 9.0 | −25.0 % |
| Stratified 80/20 | Logistic Regression | 98.774 | 99.640 | 71.0 | 21.0 | −70.4 % |
| Domain-grouped 5-fold | Random Forest | 99.324 | 99.138 | 39.8 | 51.0 | +28.1 % |
| Domain-grouped 5-fold | Gradient Boosting | 99.367 | 99.559 | 37.2 | 25.8 | −30.6 % |
| Domain-grouped 5-fold | XGBoost | 99.408 | 99.446 | 34.8 | 32.6 | −6.3 % |
| Domain-grouped 5-fold | LightGBM | 99.394 | 99.522 | 35.6 | 28.0 | −21.3 % |
| Domain-grouped 5-fold | Logistic Regression | 98.471 | 99.326 | 89.0 | 39.4 | −55.7 % |

The enhanced features reduce errors for most models. The exception is Random Forest under domain-grouped
CV, where false positives rise. Report this case in the paper as it is.

## Limitations
* The robustness benchmark is generated by code from real test URLs. Real attacker campaigns
  should be used to confirm it, e.g. a second dataset such as PhiUSIIL (cross-dataset test).
* The full run uses a single seed. Repeating it with other seeds gives mean ± std.
* The original features are re-computed from the URL with a re-implementation of the LegitPhish extractor.
  It matches the published values 99.94 % of the time (`robust_feature_reimplementation_check.csv`).

* The full dataset is close to ceiling, so differences between models are a handful of URLs.
* The legitimate class is mostly bare home pages (`https://www.example.com`) from Wikipedia and Stack Overflow
  references, which makes structural features unusually strong.
* The brand list and keyword lists are heuristic seeds. They should be validated on another dataset
  (for example PhiUSIIL) before any claim of generalisation.
