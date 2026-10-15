# Kaggle Flight Price Prediction — Compact V4 (<500 Trees Total)

## Objective
Build bounded-depth tree regressors **without price lookups, exact matching, local interpolation, neural networks, price-based target encoding or test-label training**.

## Recommended submission
`submission_compact_recommended_440.csv`: one ExtraTrees ensemble of **180 trees, max_depth=24, min_samples_leaf=3, max_features=0.85**, weighted 0.60; one LightGBM booster of **260 trees, max_depth=7, num_leaves=25, min_child_samples=35, learning_rate=0.085, reg_lambda=10, reg_alpha=0.2, subsample=0.85, subsample_freq=1, colsample_bytree=0.85**, weighted 0.40. **Total trained trees used for inference = 440**.

## Additional submissions
- `submission_catboost_420.csv`: CatBoost 420 trees, depth 7, lr=0.075, l2_leaf_reg=10, random_strength=1.2. Stronger on two grouped-flight stress comparisons against our chosen blend, but weaker on ordinary random CV.
- `submission_randomforest_220.csv`: RandomForest 220 trees, max_depth=15, min_samples_leaf=3, max_features=0.8, bootstrap=True.
- `submission_rf_lightgbm_480.csv`: RandomForest220 + LightGBM260, 50/50 blend = 480 trees total.

## Leakage prevention
- All engineered features come from flight details, never the target `price`.
- The `price` column is used only to fit models within train partitions.
- The supplied test.csv does NOT have price labels and is not used for fitting or tuning.
- Five-fold stratified CV gives OOF metrics in final_cv_comparison.csv; validation splits use the target only for stratification.
- Other evidence in independent_holdouts.csv (four fresh train/valid splits) and group_blend_checks.csv (two held-out flight-code groups).
- The final prediction uses **one full-training fit per component**, NOT averaging all five CV fits. Thus the total deployed tree count remains below 500.
- `id` is never an input feature. Missing duration is handled within estimators and represented by a missing-value flag. Airline name is not used as input; flight prefix is used.

## Caveats
- Bounded depth / fewer trees do NOT mathematically eliminate overfitting. Some training-validation gaps remain.
- Grouped-flight validation is a harsher shift than the provided test, because most test flight codes appear in train.
- Earlier lookup-based model achieved Kaggle 0.95729; none of these new files has a confirmed Kaggle score. Better model simplicity may require accepting a leaderboard trade-off.
- Competition score metric still unverified; OOF RMSE used as evaluation.

## Reproduction
1. Place train.csv, test.csv, sample_submission.csv in /mnt/data, and compact_under500.py at /mnt/data/kaggle_flight_project/.
2. Run `python compact_under500.py cv et180 lgb260 rf220 cat420` for five-fold CV.
3. Run `python compact_v4/generate_final.py` to fit once on full train and create submissions.
4. See exact_hyperparameters.json for all configuration values.
