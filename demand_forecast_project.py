"""
Advanced Demand Forecasting Project Script
Files:
- advanced_demand_dataset.csv (expects columns: date, store, item, sales, price, promotion, holiday, temperature)
This script trains RandomForest and XGBoost models and fits Prophet for time series forecasting.
Outputs:
- forecast_results_prophet.csv
- model_evaluation.csv
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from prophet import Prophet
import os

# Load data
csv_file = "advanced_demand_dataset.csv"
df = pd.read_csv(csv_file, parse_dates=["date"])

# Basic checks
print("Rows, cols:", df.shape)
print(df.head())

# Aggregate to daily total sales (option: per item/store if desired)
daily = df.groupby("date").agg({"sales":"sum", "promotion":"sum", "holiday":"max", "temperature":"mean"}).reset_index()
daily = daily.sort_values("date")
daily["date"] = pd.to_datetime(daily["date"])

# Feature engineering
daily["year"] = daily["date"].dt.year
daily["month"] = daily["date"].dt.month
daily["day"] = daily["date"].dt.day
daily["dayofweek"] = daily["date"].dt.dayofweek
daily["is_weekend"] = (daily["dayofweek"]>=5).astype(int)
daily["lag_1"] = daily["sales"].shift(1).fillna(method="bfill")
daily["lag_7"] = daily["sales"].shift(7).fillna(method="bfill")
daily["rolling_mean_7"] = daily["sales"].rolling(7, min_periods=1).mean().shift(1).fillna(method="bfill")
daily["rolling_std_7"] = daily["sales"].rolling(7, min_periods=1).std().shift(1).fillna(0)

# Train/test split (time-based)
features = ["year","month","day","dayofweek","is_weekend","promotion","temperature","lag_1","lag_7","rolling_mean_7","rolling_std_7"]
X = daily[features]
y = daily["sales"]
train_size = int(len(daily)*0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# Random Forest
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_rmse = mean_squared_error(y_test, y_pred_rf, squared=False)
rf_r2 = r2_score(y_test, y_pred_rf)

# XGBoost
xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, verbosity=0)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
xgb_mae = mean_absolute_error(y_test, y_pred_xgb)
xgb_rmse = mean_squared_error(y_test, y_pred_xgb, squared=False)
xgb_r2 = r2_score(y_test, y_pred_xgb)

# Save evaluation
eval_df = pd.DataFrame({
    "model":["RandomForest","XGBoost"],
    "MAE":[rf_mae, xgb_mae],
    "RMSE":[rf_rmse, xgb_rmse],
    "R2":[rf_r2, xgb_r2]
})
eval_df.to_csv("model_evaluation.csv", index=False)
print("Saved model_evaluation.csv")

# Plot actual vs predicted (XGBoost)
import matplotlib.pyplot as plt
plt.figure(figsize=(12,5))
plt.plot(daily["date"].iloc[train_size:], y_test.values, label="Actual", linewidth=2)
plt.plot(daily["date"].iloc[train_size:], y_pred_xgb, label="XGBoost Pred", linestyle="--")
plt.title("Actual vs Predicted (XGBoost) - Daily Total Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("actual_vs_predicted_xgb.png")
print("Saved actual_vs_predicted_xgb.png")

# Prophet forecasting (on daily aggregated series)
prophet_df = daily[["date","sales"]].rename(columns={"date":"ds","sales":"y"})
m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
m.fit(prophet_df)
future = m.make_future_dataframe(periods=30)
forecast = m.predict(future)
forecast[["ds","yhat","yhat_lower","yhat_upper"]].to_csv("forecast_results_prophet.csv", index=False)
print("Saved forecast_results_prophet.csv")

# Save feature importances
importances = pd.Series(xgb.feature_importances_, index=features).sort_values(ascending=False)
importances.to_csv("feature_importances.csv")
print("Saved feature_importances.csv")

print("All done. Files created in the current directory:")
for f in ["model_evaluation.csv","actual_vs_predicted_xgb.png","forecast_results_prophet.csv","feature_importances.csv"]:
    print("-", f)
