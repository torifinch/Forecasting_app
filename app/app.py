# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gdown
import os
from datetime import datetime, timedelta

# --- File Setup ---
DATA_PATH = "data/"
MODEL_PATH = "model/"
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# --- Google Drive File IDs ---
DRIVE_FILES = {
    "train": ("1fmwyk3G50Iqmb9ywRlynaGrrC1njWBGx", os.path.join(DATA_PATH, "train.csv")),
    "oil": ("1YF4BZvIM1W5b5jAauyKIlDEiYayN6rk1", os.path.join(DATA_PATH, "oil.csv")),
    "holidays": ("1-77ktGfQ7h7lb0U9Pp3GRTjl0jchpWVL", os.path.join(DATA_PATH, "holidays.csv")),
    "model": ("11qoHWIDLEM1GDVtGS2MnwpjPsKfIHazs", os.path.join(MODEL_PATH, "xgb_model.pkl"))
}

# --- Download Function ---
def download_files():
    for _, (file_id, path) in DRIVE_FILES.items():
        if not os.path.exists(path):
            gdown.download(id=file_id, output=path, quiet=False)

# --- Preprocessing Function ---
def preprocess_input(store_id, item_id, date, df_train, df_oil, df_holidays):
    date = pd.to_datetime(date)
    history = df_train[(df_train.store_nbr == store_id) & (df_train.item_nbr == item_id)]
    history = history[history.date < date].sort_values("date")

    if len(history) < 7:
        st.warning(f"⚠️ Not enough history for store {store_id}, item {item_id} on {date.date()}. Found only {len(history)} records.")

    unit_sales = history.unit_sales.tail(30).tolist()
    last_row = history.tail(1)

    lag_1 = last_row.unit_sales.values[0] if not last_row.empty else 0
    mean_30 = pd.Series(unit_sales[-30:]).mean() if len(unit_sales) >= 30 else 0
    std_30 = pd.Series(unit_sales[-30:]).std() if len(unit_sales) >= 30 else 0

    # Z-score and smoothing
    z_score = (lag_1 - mean_30) / std_30 if std_30 else 0
    is_outlier = z_score > 5
    if is_outlier:
        lag_1 = pd.Series(unit_sales[-7:]).mean() if len(unit_sales) >= 7 else mean_30

    # Oil price fill
    oil_df = df_oil.set_index("date").sort_index()
    oil_df = oil_df.reindex(pd.date_range(oil_df.index.min(), oil_df.index.max())).ffill().bfill().fillna(0)
    oil_price = oil_df.loc[date]["dcoilwtico"] if date in oil_df.index else 0

    # Holiday checks
    holiday_dates = pd.to_datetime(df_holidays.date)
    is_holiday = date in holiday_dates
    is_day_before = (date + pd.Timedelta(days=1)) in holiday_dates
    is_bridge = (date.weekday() == 0 and (date - pd.Timedelta(days=3)) in holiday_dates)

    return pd.DataFrame([{
        "id": 1,
        "store_nbr": store_id,
        "item_nbr": item_id,
        "onpromotion": 0,
        "year": date.year,
        "month": date.month,
        "day": date.day,
        "day_of_week": date.dayofweek,
        "week_of_year": date.isocalendar().week,
        "is_weekend": date.weekday() >= 5,
        "is_holiday": is_holiday,
        "is_bridge_day": is_bridge,
        "is_day_before_holiday": is_day_before,
        "expanding_mean": history.unit_sales.expanding().mean().iloc[-1] if not history.empty else 0,
        "oil_price": oil_price,
        "rolling_mean_7d": pd.Series(unit_sales[-7:]).mean() if len(unit_sales) >= 7 else 0,
        "rolling_mean_14d": pd.Series(unit_sales[-14:]).mean() if len(unit_sales) >= 14 else 0,
        "rolling_mean_30d": mean_30,
        "rolling_std_7d": pd.Series(unit_sales[-7:]).std() if len(unit_sales) >= 7 else 0,
        "rolling_std_14d": pd.Series(unit_sales[-14:]).std() if len(unit_sales) >= 14 else 0,
        "rolling_std_30d": std_30,
        "lag_1": lag_1,
        "lag_7": unit_sales[-7] if len(unit_sales) >= 7 else 0,
        "lag_14": unit_sales[-14] if len(unit_sales) >= 14 else 0,
        "z_score": z_score,
        "is_outlier": is_outlier
    }])

# --- Streamlit App ---
def main():
    st.title("🛒 Grocery Sales Forecast - Corporación Favorita")
    download_files()

    df_train = pd.read_csv(DRIVE_FILES["train"][1], parse_dates=["date"])
    df_oil = pd.read_csv(DRIVE_FILES["oil"][1], parse_dates=["date"])
    df_holidays = pd.read_csv(DRIVE_FILES["holidays"][1], parse_dates=["date"])

    with open(DRIVE_FILES["model"][1], "rb") as f:
        model = pickle.load(f)

    st.sidebar.header("📊 Forecast Settings")

    use_top_10 = st.sidebar.checkbox("Use Top 10 Best Forecasting Candidates", value=True) # Top 10 Checkbox

    if use_top_10:
        combo_counts = df_train.groupby(['store_nbr', 'item_nbr']).size().reset_index(name='count')
        top_combos = combo_counts.sort_values('count', ascending=False).head(10)
        top_combo_labels = top_combos.apply(lambda row: f"Store {int(row['store_nbr'])} - Item {int(row['item_nbr'])}", axis=1)
        selected_label = st.sidebar.selectbox("Top 10 Combos", top_combo_labels)
        selected_store = int(selected_label.split()[1])
        selected_item = int(selected_label.split()[-1])
    else:
        selected_store = st.sidebar.selectbox("Select Store", sorted(df_train.store_nbr.unique()))
        store_items = df_train[df_train.store_nbr == selected_store].item_nbr.unique()
        selected_item = st.sidebar.selectbox("Select Item", sorted(store_items))

    start_date = st.sidebar.date_input("Forecast Start Date", datetime(2014, 3, 1))
    days = st.sidebar.selectbox("Forecast Horizon", [7, 14, 30])

    if st.button("🔮 Forecast"):
        forecast_dates = pd.date_range(start=start_date, periods=days)
        inputs = pd.concat([
            preprocess_input(selected_store, selected_item, d, df_train, df_oil, df_holidays)
            for d in forecast_dates
        ], ignore_index=True)

        preds = model.predict(inputs)
        forecast_df = pd.DataFrame({"date": forecast_dates, "predicted_sales": preds})

        st.subheader("📈 Forecast Chart")
        st.line_chart(forecast_df.set_index("date"))

        st.subheader("📋 Forecast Table")
        st.dataframe(forecast_df.style.format({"predicted_sales": "{:.2f}"}))

if __name__ == "__main__":
    main()
