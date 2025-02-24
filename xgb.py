import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt
import json
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

# Load dataset
df = pd.read_csv("food.csv")

# Convert 'date' column to datetime
df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=False)

# Aggregate sales by date
df = df.groupby('date', as_index=False).agg({'transaction_amount': 'sum'})

# Sort and set date as index
df = df.sort_values(by="date").reset_index(drop=True)
df.set_index("date", inplace=True)
df = df.asfreq('D')  # Ensure daily frequency

# Handle missing values
df['transaction_amount'] = df['transaction_amount'].interpolate(method="linear")

# Feature Engineering
df["year"] = df.index.year
df["month"] = df.index.month
df["day"] = df.index.day
df["weekday"] = df.index.weekday

# Create lag features
for lag in range(1, 8):  # Use past 7 days as features
    df[f"lag_{lag}"] = df["transaction_amount"].shift(lag)

df.dropna(inplace=True)  # Remove rows with NaN due to lagging

# Prepare input features and target
features = ["year", "month", "day", "weekday"] + [f"lag_{lag}" for lag in range(1, 8)]
X, y = df[features], df["transaction_amount"]

# Split data into training and testing
train_size = len(df) - 7  # Use last 7 days as test set
X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
X_test, y_test = X.iloc[train_size:], y.iloc[train_size:]

# Train XGBoost Regressor
xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=310,
    learning_rate=0.0300009,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8
)

xgb_model.fit(X_train, y_train)

# Predict the test period (last 7 days)
test_predictions = xgb_model.predict(X_test)

# Generate Future Dates for Forecasting
forecast_days = 30
future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='D')

# Prepare Future Data for Prediction
future_df = pd.DataFrame(index=future_dates)
future_df["year"] = future_df.index.year
future_df["month"] = future_df.index.month
future_df["day"] = future_df.index.day
future_df["weekday"] = future_df.index.weekday

# Initialize Lags with Last Available Data
last_known_data = df.iloc[-7:]["transaction_amount"].values.tolist()

# Iteratively Predict Future Values
predicted_values = []
for date in future_df.index:
    future_features = {
        "year": date.year,
        "month": date.month,
        "day": date.day,
        "weekday": date.weekday()
    }
    for i in range(7):
        future_features[f"lag_{i+1}"] = last_known_data[-(i+1)]
    
    future_input = pd.DataFrame([future_features])
    prediction = xgb_model.predict(future_input)[0]
    predicted_values.append(prediction)
    last_known_data.append(prediction)
    last_known_data.pop(0)

# Save Future Predictions
future_df["transaction_amount"] = predicted_values

# Calculate MAE & MDAPE
mae = mean_absolute_error(y_test, test_predictions)
mdape = np.median(np.abs((y_test - test_predictions) / y_test)) * 100

# Plot Results
# Plot Results (Last 30 Days + Forecast)
plt.figure(figsize=(12, 5))

# Show last 30 days of actual data
plt.plot(df.index[-30:], df["transaction_amount"].iloc[-30:], label="Actual Data (Last 30 Days)", color="blue")

# Show test predictions for the last 7 days
plt.plot(X_test.index, test_predictions, label="Test Predictions (Last 7 Days)", linestyle="dashed", color='green')

# Show forecast for next 30 days
# plt.plot(future_df.index, future_df["transaction_amount"], label="XGBoost Forecast (Next 30 Days)", linestyle="dashed", color='orange')

plt.legend()
plt.xlabel("Date")
plt.ylabel("Transaction Amount")
plt.title("XGBoost Forecast (30Days)")
plt.savefig("xgboost_forecast.png", dpi=300)

print("✅ Forecast graph saved as 'xgboost_forecast_1month.png'")


# Save forecast to CSV
# future_df.to_csv("xgboost_forecast.csv")

result_df = pd.DataFrame([{"model": "XGBOOST", "mae": mae}])
result_df.to_csv("result.csv", mode='a', header=not pd.io.common.file_exists("result.csv"), index=False)

print(f"✅ Mean Absolute Error (MAE): {mae:.2f}")
