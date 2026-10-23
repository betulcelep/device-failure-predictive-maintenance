# Device Failure Prediction – Predictive Maintenance for a Delivery Fleet

Binary classification on a **predictive maintenance** dataset of daily device telemetry (124,494 device-days from 1,168 devices, nine anonymised sensor attributes): predicting device failures so that maintenance can be scheduled before a breakdown.

## The challenge: extreme class imbalance
Only 106 of 1,168 devices failed, giving **106 failure days among 124,494 rows (0.09%)**. A model that never predicts a failure scores 99.9% accuracy, so models were judged on **recall, precision and PR-AUC** instead.

## Approach
- **EDA:** attribute8 duplicated attribute7 and was dropped; record counts per device were excluded because 95% of failed devices stop reporting on the day of failure (target leakage); attributes 2, 4 and 7 are zero on most normal days but non-zero on 37–58% of failure days.
- **Feature engineering:** day-to-day and **7-day changes** per device, non-zero flags and an "any recent change" indicator.
- **Device-grouped splits** (`StratifiedGroupKFold`) so that no device appears in both training and test data.
- **Imbalance handling compared:** class weights, SMOTE (full and mild ratios), threshold tuning; Random Forest, Logistic Regression and LightGBM compared with 5-fold device-grouped cross-validation.
- **Simple rule benchmark** built from the key insight, to test whether the model adds value.

## Results
| Approach (unseen test devices) | Failures caught | Precision | Devices flagged per day |
|---|---|---|---|
| Random Forest, default threshold | 1 of 21 | 0.11 | – |
| **Random Forest, class weights, 500 trees, threshold 0.03** | **13 of 21 (62%)** | 0.014 | 3.7% |
| **Rule: change in attribute 2, 4 or 7 within 7 days** | **14 of 21 (67%)** | 0.020 | 2.8% |

- **A recent change in attribute2, attribute4 or attribute7 is the strongest early-warning signal.**
- **Full SMOTE balancing hurt performance** (PR-AUC fell from 0.061 to 0.031); a milder ratio helped but did not beat class weighting.
- **The simple rule performs as well as the model with fewer false alarms.**

## Recommendation
Deploy the rule immediately – it is simple, explainable and needs no infrastructure – and use the model's probabilities to prioritise inspections, starting with the highest-risk devices. Next steps: collect more failure cases, learn what each attribute measures, and estimate the cost of a missed failure versus an unnecessary inspection to set the alarm threshold.

## Files
- `device-failure-predictive-maintenance.ipynb` – full notebook with EDA, feature engineering, model comparison and conclusions

## Tools
Python · pandas · scikit-learn · imbalanced-learn (SMOTE) · LightGBM · matplotlib · seaborn

---
Part of my [data science portfolio](https://betulcelep.github.io/projects.html), completed during the *Mastering Applied Data Science with Deep Learning* programme.
