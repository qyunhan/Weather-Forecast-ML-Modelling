# Forecasting Relative Humidity

Multi-horizon relative humidity forecasting using interpretable machine learning and deep learning models, built as part of NUS IT1244.

> **Note:** The dataset (~800k rows) and trained LSTM / Random Forest model files are excluded from this repository due to file size constraints.

---

## Overview

Accurate short-term humidity forecasting supports real-time decision-making in agriculture, disaster preparedness, and public health. This project benchmarks tree-based ML models against deep learning approaches across three forecast horizons: **1h, 6h, and 24h ahead**.

---

## Key Skills & Technologies

`Python` `Machine Learning` `Deep Learning` `Time Series Forecasting` `Feature Engineering` `Hyperparameter Tuning` `XGBoost` `Random Forest` `Decision Trees` `LSTM` `scikit-learn` `TensorFlow/Keras` `Optuna` `GridSearchCV` `Cross-Validation` `Data Preprocessing` `Exploratory Data Analysis` `Statistical Testing`

---

## Models

| Model | Description |
|---|---|
| Decision Tree | Baseline interpretable model; tuned with GridSearchCV + TimeSeriesSplit |
| Random Forest | Ensemble model with feature engineering and permutation-based pruning |
| XGBoost | Gradient-boosted trees with Bayesian hyperparameter optimisation via Optuna |
| LSTM | Sequence model with Fourier time features and multi-step prediction variants |

---

## Results (Test Set R²)

| Horizon | Decision Tree | Random Forest | XGBoost | Best LSTM |
|---|---|---|---|---|
| 1h | 0.7385 | 0.8122 | **0.8387** | 0.8279 |
| 6h | 0.5635 | 0.6483 | **0.7023** | 0.7146 |
| 24h | 0.4854 | 0.5279 | **0.5280** | 0.5197 |

**Final model: XGBoost** — chosen for its balance of accuracy and inference speed (LSTM took >1 hour on the test set).

---

## Key Takeaways

- **Horizon-specific lag tuning** significantly outperforms fixed lag windows; optimal lag N varied from 3 (DT, 1h) to 72 (XGB, 6h)
- **Feature pruning** reduced dimensionality by >60% in Random Forest with minimal performance loss, improving interpretability and training efficiency
- **Fourier time features** (encoding daily/yearly cycles) became increasingly valuable at longer horizons (24h), where recent lagged data alone is insufficient
- **XGBoost marginally edged Random Forest** across all horizons; feature pruning that helped RF actually hurt XGB, suggesting low-importance features contribute to XGB's sequential error-correction
- **Hyperparameter tuning had an outsized impact on Decision Trees** — R² for 24h forecasts improved by 306% over the untuned baseline
- **LSTM with multi-step prediction (LSTM3)** was best for 24h despite underperforming at shorter horizons, highlighting a trade-off between model complexity and forecast horizon

---

## Methodology

- **Data:** 100,057 hourly weather records (Jan 2010 – Jun 2021), Monash University, Australia. 7 features, no missing values.
- **Train/Val/Test split:** 64% / 16% / 20%, chronologically ordered to prevent data leakage
- **Stationarity:** ADF test confirmed all features are stationary
- **Preprocessing:** Standardisation, dummy variables for anomalies (nighttime, pressure outliers), rolling averages, hour-of-day and day/night indicators
- **Evaluation metrics:** R² (primary), MAE, MSE

---

## Repo Structure

```
├── decision_tree/
├── random_forest/
├── xgboost/
├── lstm/
├── notebooks/          # EDA and visualisations
└── README.md
```

---
