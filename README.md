
# 🛒 Corporación Favorita Grocery Sales Forecasting App

[![Streamlit App](https://img.shields.io/badge/launch-app-green?logo=streamlit)](https://forecastingapp-grocery.streamlit.app/)

🔮 Try the app now: **[forecastingapp-grocery.streamlit.app](https://forecastingapp-grocery.streamlit.app/)**  
No installation, no configuration — just click and forecast.

<h3>App Preview</h3>
<img src="https://drive.google.com/uc?id=1njj2PSV7kki5tlzzESMWytj5hQY4vFlZ" width="800"/>

---

## 🚀 Overview

This interactive forecasting app predicts daily item sales across stores for **Corporación Favorita**, a large Ecuadorian grocery chain. Powered by an XGBoost model and feature-rich engineering, it allows users to:

- Explore sales forecasts for any store and item combo  
- Analyze the next 7, 14, or 30 days  
- Visualize results with interactive charts and tables  
- Take advantage of holiday, oil price, and outlier-aware forecasting

---

## 🔧 Features

- **Top 10 Best Forecasting Candidates**: Automatically selects store-item pairs with the richest sales history
- **Manual selection** of any store and item
- **Dynamic item filtering** based on selected store
- **Flexible forecasting range**: Choose 7, 14, or 30 days
- 📈 **Line chart** and **data table** output
-  Fully integrated with Google Drive for model + dataset loading

---

## How to Use

1. Select a forecast target using:
   - 🔘 The Top 10 dropdown
   - 🔘 Store + item combo of your choice
2. Choose a forecast start date and horizon
3. Click **🔮 Forecast**
4. View the predicted sales over time and analyze the output

---

## 🧠 How It Works

- ** Input:** Historical unit sales, promotions, dates, oil prices, and holidays
- ** Model:** A trained XGBoost regressor using engineered time series features
- ** Output:** Forecasted sales for the selected item/store and timeframe

No setup needed. Just go to the link and click:

👉 [https://forecastingapp-grocery.streamlit.app/](https://forecastingapp-grocery.streamlit.app/)

---

## ⚙️ Features

-  Forecasting for 7, 14, or 30 days  
-  Select any store/item or use the top 10 best forecast-ready combos  
-  Visual output as both a chart and a table  
-  Powered by lag features, rolling statistics, holidays, and more  
-  Outlier smoothing via Z-score logic  
- ☁ Model + data integration via Google Drive and gdown

---

## Recreate Locally (Optional)

Want to run the app on your own machine?

```bash
# 1. Clone the repo
git clone https://github.com/torifinch/Forecasting_app.git

# 2. Navigate into the directory
cd Forecasting_app

# 3. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # or venv\\Scripts\\activate on Windows

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the app
streamlit run app/app.py
```

#  Corporación Favorita Grocery Sales Models

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


