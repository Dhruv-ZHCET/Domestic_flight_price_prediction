# Kaggle Flight Price — V3 generalization / overfitting audit

## Public scoreboard context

- Current confirmed Kaggle score (user-reported): **0.95716** from `submission_lookup_augmented.csv`.
- Goal: **0.961**. No new candidate has yet been submitted to Kaggle. The official scoring formula was not supplied, so **do not equate the Kaggle scores with local RMSE or R²**.
- No neural network was used. No test prices were available or used, and the test set was **never included in model training or hyperparameter selection**.

## Modeling strategy

The previous tree-ensemble blend remains in place: Extra Trees, Random Forest, LightGBM and CatBoost, with economy/business-specific blend weights and an exact-match lookup for 324 test rows. Rather than adding larger boosted models (which failed strict out-of-fold or unseen-flight tests), V3 adds a *small ridge-regularized local regression* using only labeled TRAINING flights. For each flight/route/departure/arrival/stops/class group, it models `price = intercept + slope * days_left` using a penalized slope (denominator `SS_x + 150`). It is applied only when:

1. The flight group has **at least three training observations**.
2. In-sample group regression residual standard deviation is **at most 800**.
3. The target flight's `days_left` lies **inside the group’s observed training-day range**: no extrapolation.
4. An original exact-price match is **not** already being used.

The new prediction is **75–85% of the earlier ensemble/lookup prediction plus 15–25% of the regularized local regression prediction**. If conditions fail, the previous prediction is unchanged.

## Submission choice

| Submission | OOF RMSE | OOF MAE | 5-fold changes versus 9,981.47 RMSE | Test rows smoothed |
| --- | ---: | ---: | --- | ---: |
| Previous exact-match model | 9,981.47 | 3,275.41 | Baseline | 0 additional |
| `submission_v3_risk_controlled.csv` (15% adjustment) | **9,956.17** | **3,270.31** | Improves 5/5 folds | 1,389 |
| `submission_v3_interpolation_25.csv` (25% adjustment) | **9,944.05** | **3,270.52** | Improves 4/5; one fold +3.33 RMSE worse | 1,389 |

Both preserve the 324 exact lookups unchanged.

## New independent checks (different random seeds)

The interpolation was evaluated with a different Extra Trees base model on **four fresh, previously unused 80/20 splits** (see `independent_split_audit.csv`); both weights improve all four splits. For weight 25%, holdout RMSE after exact lookup versus adding local interpolation:

| Holdout seed | Before | After | Improvement |
| --- | ---: | ---: | ---: |
| 113 | 9,774.5 | 9,728.7 | 45.8 |
| 399 | 10,519.3 | 10,408.7 | 110.6 |
| 901 | 10,300.6 | 10,236.4 | 64.2 |
| 1028 | 9,468.6 | 9,445.4 | 23.2 |

This is an *additional generalization diagnostic* using standalone Extra Trees, not a completely untouched test of the original full ensemble. Grouped-by-flight held-out flights cannot benefit from local price lookup, so the original ensemble remains responsible for those records.

## Models we rejected

- Adding the newly engineered CatBoost to the original blend improved the **first CV fold**, but worsened the other four. Five-fold blend RMSE did not improve. It had **14,133 grouped-flight holdout RMSE** versus **13,771** for the earlier CatBoost on the group split.
- Heavily regularized smaller Extra Trees / LightGBM models had worse initial holdout performance.
- Broad group-average price lookups and local line extrapolation were less robust than interpolation within historically observed days.
- Business-only CatBoost improved the first fold when blended, but worsened the full 5-fold OOF score.

## Overfitting cautions

The tuned Extra Trees train RMSE on independent splits was around 6,400 while validation RMSE was 9,700–10,700. That is a **real fit/generalization gap**, not proof of no overfitting. Reducing tree count alone does not solve this; depth/leaf size matter more. We therefore keep the prior stronger ensemble rather than increasing its capacity and apply only a conservatively weighted feature-specific calibration.

We investigated many local-smoother configurations, so tuning-selection bias remains possible. The flight-cluster bootstrap confidence intervals in `v3_final_report.json` are conditional on the resulting OOF predictions and do not correct for that search. Public leaderboard gains and private leaderboard behavior cannot be guaranteed.

## How to use

1. Upload **one** of the two V3 `id,price` files in the Kaggle Submissions tab (suggest trying `submission_v3_interpolation_25.csv` first; use risk-controlled version for a lower-variance alternative).
2. Record its leaderboard score against **0.95716**.
3. If there is no consistent improvement, revert to the previous exact-match submission.
4. Confirm the competition's **Evaluation** formula before further optimization.

### Reproducibility

- `build_v3_final.py` reruns leakage-safe per-fold local ridge lookup and produces both submissions using previously computed baseline out-of-fold and test predictions.
- `check_lookup.py` rebuilds the baseline lookup predictions from the included model OOF arrays and blend weight file.
- `independent_split_audit.py` performs the fresh-seed Extra Trees holdout check.
- `modeling.py` is the original training and feature-engineering script for the baseline ML models.

The source files have workspace-style absolute `/mnt/data/...` paths. Change those paths to the Kaggle notebook input/output locations when running elsewhere. `check_lookup.py` expects `modeling.py` importable from the original project folder; the included scripts are intended to show the exact applied procedure, not to run with one button on arbitrary Kaggle kernels.
