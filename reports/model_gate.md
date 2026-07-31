# Model Gate Evidence — Metro Flow Forecasting

## 1. Baseline
Historical Average (mean inflow per station × hour × weekend-flag).
Validation MAE: 29.34 | Test MAE: 37.55

## 2. Controlled runs (≥2)
Three models compared under identical split and features:
- Historical Average — val MAE 29.34
- Ridge regression — val MAE 29.29
- XGBoost — val MAE 22.42
All logged to MLflow (experiment: metro-flow-forecasting, 3 runs).

## 3. Split & metric protocol
Chronological split, no shuffling:
- Train: Jan 1–18 | Validation: Jan 19–20 | Test: Jan 21–25
Metrics: MAE (primary), RMSE (secondary). Scalers fit on train only.

## 4. Candidate selection
Selected on VALIDATION MAE (not test). XGBoost chosen (22.42, lowest).

## 5. Protected test status
Test set touched once, after selection. Test MAE: 25.75.

## 6. Baseline comparison
XGBoost beats Historical Average by ~24% on validation MAE (22.42 vs 29.34)
and on test MAE (25.75 vs 37.55).

## 7. Error analysis
[fill in your peak/off-peak and busy/quiet findings — see notebook 04]
Morning peak hardest (MAE 46); quiet stations highest relative error (~16%).

## 8. Inference artifact
artifacts/metro_xgb_model.joblib — bundles model + feature list.

## 9. Reload proof
Model reloaded from disk reproduces identical predictions without refitting
(np.allclose = True). See notebook 03.

## 10. Known limitations
- 25 days: no seasonal coverage.
- Hangzhou model does not transfer directly to Tashkent.