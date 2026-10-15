# Kaggle Flight Price — EDA + Tree Ensemble Regression

## Recommended submission (based on provisional RMSE)
- **`submission_BEST_RMSE.csv`** — class-specific OOF blend. 5-fold OOF RMSE 10179.77, R² 0.9323. This is **not** a hidden Kaggle leaderboard score.
- **`submission_BEST_MAE.csv`** — Extra Trees, 5-fold OOF MAE 3340.27. For MAE-scored competitions, consider this file.
- **`submission_SIMPLE_BLEND.csv`** — fixed weighted blend of Extra Trees 60%, CatBoost 23%, LightGBM 17%. 5-fold OOF RMSE 10237.90, less adapted to validation than the optimized blend.
- Multiple `submission_*.csv` alternative model submissions are also present.

## Where to start
1. Open `EDA_report.html` for 34 univariate/bivariate/multivariate figures, detailed grouped statistics and train/test shift.
2. Open `Model_validation_report.html` for CV comparison, feature importance, hyperparameter tuning, bootstrap analysis and diagnostics.
3. `Kaggle_Flight_Price_Notebook.ipynb` provides a standalone Kaggle-style workbook of Python cells for reproducing the workflow.
4. On Kaggle choose ONE of the CSV files, using columns **`id,price`** in the same order as `sample_submission.csv`.

## Data protocol
- Training rows: 15,589; prediction-only test rows: 3,898.
- No neural networks were used. Logistic regression and Naive Bayes classifiers are not appropriate for this continuous regression target.
- Train-only target usage for EDA and feature importance; test features were only examined for covariate shift and used for prediction.
- Never use `id` as a feature. Both files restart ID at 1.
- RMSE is a **provisional** metric; please confirm official Kaggle evaluation metric. RMSLE and MAE scores are included to support a different selection if needed.
- 0.90% of test rows have flight identifiers absent from training; one test airline/prefix level is unseen. Encoders are designed to handle them.

## Reproducible original scripts
- `eda.py` — full EDA and feature drift report.
- `modeling.py` — clean fold-level transformations, candidate estimators and independent OOF/test predictions.
- `blending.py` — weighted ensembles, regularized weight optimization, per-class blends, errors and bootstrap.
- `tuning_and_importance.py` — tuning + ablation exploratory script (large run; tuning CSV persisted).
- `permutation_and_residuals.py` — validation permutation importance and diagnostic plots.

These original `.py` scripts expect source files in `/mnt/data/` and write to `/mnt/data/kaggle_flight_project/` (this current runtime). On Kaggle, use the companion notebook for portable paths or edit the two `ROOT`/`OUT` definitions in each script. For best replicability, preserve seed values.

## Shortlist scores (OOF)
```
                 model    rmse    mae  rmsle    r2  rmse_economy  rmse_business
blend_class_specialist 10179.8 3453.2  0.137 0.932        1685.3        15783.1
    blend_cv_optimized 10195.6 3495.4  0.142 0.932        1799.4        15789.9
 blend_tuned_three_way 10213.5 3485.3  0.141 0.932        1756.0        15825.1
       blend_three_way 10237.9 3413.2  0.138 0.932        1704.4        15871.8
        blend_four_way 10271.1 3385.8  0.138 0.931        1684.3        15927.2
       blend_extra_cat 10280.4 3482.4  0.140 0.931        1703.2        15939.0
     blend_extra_light 10325.4 3360.2  0.139 0.930        1747.1        16003.0
     extra_trees_tuned 10415.8 3409.2  0.139 0.929        1735.2        16147.3
        blend_extra_rf 10462.3 3380.5  0.139 0.928        1725.8        16222.2
           extra_trees 10512.8 3340.3  0.138 0.928        1723.3        16302.1
              lightgbm 10645.9 3762.5  0.185 0.926        2528.5        16361.4
              catboost 10675.2 4169.0  0.199 0.926        2316.5        16453.1
         random_forest 10714.9 3603.7  0.148 0.925        1825.5        16604.7
          bagged_trees 10865.1 3723.0  0.152 0.923        1914.5        16827.3
```

## Final blend weight protocol
OOF predictions arise from base models each fitted on other folds. A constrained convex optimization on those OOF labels chooses separate nonnegative weights for Business and Economy; these weights are applied to fold-averaged unlabeled test predictions. The optimized weights have some selection optimism; a 5-split meta-weight resampling check gave RMSE 10213.3.

## Test submission verification
- All submission variants tested for `3898` rows, columns exactly `id,price`, nonmissing finite positive prices and ID equality to `sample_submission.csv`.
- See `submission_validation.csv` for exact row counts and prediction ranges.
