# Domestic Flight Price Prediction — Kaggle Project

This repository includes **actual historical modeling scripts**, EDA artifacts and validation tables recovered from saved experiment archives.

## Key outputs
- Highest *reported and screenshot-supported* public Kaggle R²: **0.95885**, `submission_highround_optuna_equal_dart3.csv`. **Its exact original prediction-generation source is not present**, so this repository does **not** claim to reproduce it.
- Best personally documented V3 submission with supplied code: public Kaggle R² **0.95729**, `submission_v3_interpolation_25.csv`. This includes exact-match and interpolation enhancements that were later rejected as an architectural preference.
- Preferred policy-compliant architecture: **180 Extra Trees (depth 24) + 260 LightGBM trees (depth 7)**, 60/40 weights; total **440 trees**, without lookup/interpolation. OOF CV R² **0.931192**, RMSE **10263.411**. Public Kaggle score **unverified**.

## Directory guide
- `current/compact_v4/`: bounded-depth under-500-tree generation and validation code
- `historical/baseline/`: EDA, baseline model, blend code, notebook, charts, diagnostics
- `historical/v3/`: V3 lookup/interpolation scripts and cross-validation evidence
- `submissions/`: available generated CSV files
- `leaderboard_scores.csv`: scores and provenance
- `leaderboard_screenshot_evidence.zip`: screenshot-only evidence

## Run compact model
Install Python, numpy, pandas, scikit-learn, lightgbm, catboost (optional), xgboost (optional). In `current/compact_v4`, inspect `README.md`, `compact_under500.py` and `generate_final.py`. Use train.csv, test.csv and sample_submission.csv as specified there; these datasets are deliberately omitted from the repository ZIP to keep it light and avoid unintentional duplication.

## GitHub link
Not published yet. Upload the contents of this ZIP to your own repository and replace the placeholder in the report with the real URL.

**Metric note:** Holdout and OOF R² values are different evaluations and should not be confused with Kaggle public R². Training scores are only shown when recorded. The Kaggle public test labels were not used for model fitting.
