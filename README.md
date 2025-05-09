#  Corporación Favorita Grocery Sales Forecasting

Welcome to the **Grocery Sales Forecasting** project, built using real-world data from the Ecuadorian retailer *Corporación Favorita*. This repository features robust **time series modeling** using both **XGBoost** and **LSTM (Long Short-Term Memory)** to predict unit sales for the next three months.  

---

## 📌 Project Summary

- **Goal:** Forecast daily grocery sales at the store-item level.
- **Dataset:** Provided by Corporación Favorita (via Kaggle), includes multi-year transactional data.
- **Models Used:**
  - XGBoost Regressor (Gradient Boosted Trees)
  - LSTM Neural Network (Sequence-based Deep Learning)
- **Evaluation Metrics:**  
  - R² Score  
  - MAE (Mean Absolute Error)  
  - MSE (Mean Squared Error)  
  - MAPE (Mean Absolute Percentage Error)

---

##  📈 Key Features & Engineering

The dataset was enriched with domain-specific and time-based features to enhance predictive power:

###  Time Features
- `year`, `month`, `day`, `day_of_week`
- `week_of_year`, `is_weekend`, `is_holiday`
- `is_bridge_day`, `day_before_holiday`, `is_day_before_holiday`

###  Lag & Rolling Features
- `lag_1`, `lag_7`, `lag_14`, `lag_30`
- `rolling_mean_7d`, `rolling_mean_14d`, `rolling_mean_30d`
- `rolling_std_7d`, `rolling_std_14d`, `rolling_std_30d`
- `expanding_mean`

###  Promotions & External Data
- `onpromotion`: whether the item was on promo
- `oil_price`: national oil price as a macroeconomic indicator

###  Cleaned Features
- Missing values handled appropriately (e.g., forward fill for oil, 0 for promotions)
- Outliers were detected using z-scores and smoothed

---

##  Model Performance

###  XGBoost Results (After Hyperparameter Tuning)
- **✅ R² Score:** `0.9782`
- **✅ MAE:** `0.646`
- **✅ MSE:** `3.02`
- **✅ MAPE:** `12.61%`

###  LSTM Results
- **✅ R² Score:** `0.6336`
- **✅ MAE:** `4.75`
- **✅ MSE:** `107.76`

XGBoost currently outperforms LSTM due to its handling of structured tabular data, though LSTM holds potential for capturing seasonality and long-term patterns with further tuning.

---

##  Visualizations

Key visual comparisons between predicted and actual sales were plotted using Matplotlib and Seaborn:
- Forecast vs. Actual for both validation and test sets
- Residual error plots
- Feature importance for XGBoost

---

##  Folder Structure

