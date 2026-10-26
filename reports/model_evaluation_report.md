# Real Estate Price Prediction — Model Evaluation Report

**Report date:** 2026-10-26

## 1. Project Objective

Develop a machine learning model to estimate residential property
prices in Gurgaon using property characteristics, location information,
and engineered accessibility features.

## 2. Dataset

- Processed modeling dataset: 7,457 properties
- Training samples: 5,965
- Test samples: 1,492
- Numerical features: 9
- Categorical features: 6
- Total model features: 15

The target variable is `log_price`, derived using `log1p(price_lakhs)`.
Predictions are transformed back to lakhs of INR using `expm1`.

Identifiers, split labels, and target-derived price information are
excluded from the model features.

## 3. Models Evaluated

The following regression models were evaluated:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor

Hyperparameter tuning was conducted using training data and
cross-validation. The held-out test set was reserved for final evaluation.

## 4. Selected Model

**Model:** Tuned Random Forest Regressor

**Model file:** `models/selected_final_model.joblib`

The selected model was saved and reloaded successfully. The reloaded
model reproduced the original predictions within floating-point precision.

## 5. Final Test Results

| Metric | Result |
|---|---:|
| R² on log scale | 0.9175 |
| MAE on log scale | 0.1169 |
| RMSE on log scale | 0.1772 |
| R² on price scale | 0.9010 |
| MAE on price scale | ₹43.37 lakhs |
| RMSE on price scale | ₹100.81 lakhs |
| Median absolute error | ₹19.63 lakhs |
| Median absolute percentage error | 7.55% |

## 6. Error Analysis

The largest prediction errors occurred mainly among expensive properties.
The model tended to underpredict some premium properties.

### Performance by actual property price

| Price range | Samples | MAE (lakhs) | Median AE (lakhs) | Mean absolute percentage error |
|---|---:|---:|---:|---:|
| Up to ₹1 crore | 9 | 27.95 | 27.89 | 27.95% |
| ₹1–3 crore | 892 | 22.73 | 13.95 | 12.13% |
| ₹3–6 crore | 426 | 48.26 | 32.33 | 11.29% |
| ₹6–10 crore | 114 | 93.80 | 71.04 | 12.17% |
| Above ₹10 crore | 51 | 253.52 | 135.57 | 15.07% |

The model performs best in absolute-error terms for the ₹1–3 crore segment,
which contains most of the test observations. Absolute errors increase
substantially for expensive properties.

The lowest price segment contains only nine observations, so its results
should be interpreted cautiously.

## 7. Limitations

- Premium properties are less represented than mid-range properties.
- Some property characteristics that influence prices may not be available
  in the dataset.
- Large individual errors increase RMSE above MAE.
- Results on this test set do not guarantee the same accuracy for future
  listings or other real-estate markets.
- Predictions should be treated as estimates, not official valuations.

## 8. Conclusion

The tuned Random Forest Regressor achieved a test R² of 0.9010 and an
MAE of ₹43.37 lakhs. It provides promising predictive performance on the
Gurgaon dataset, particularly for the well-represented ₹1–3 crore segment.

Further improvement could focus on premium-property data coverage,
additional property-quality features, and validation on newer listings.

## 9. Generated Evaluation Artifacts

- `reports/model_tuning_results.csv`
- `reports/test_evaluation_results.csv`
- `reports/prediction_error_analysis.csv`
- `reports/largest_prediction_errors.csv`
- `reports/performance_by_price_range.csv`
- `models/selected_final_model.joblib`
- `models/selected_model_metadata.json`
