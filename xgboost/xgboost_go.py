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

# Train XGBoost Regressor on full dataset
xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=310,  # Increase estimators for better learning
    learning_rate=0.0300009,  # Reduce to avoid overfitting
    max_depth=7,  # Increase depth for better pattern recognition
    subsample=0.8,  # Use 80% of data per tree
    colsample_bytree=0.8  # Use 80% of features per tree
)

xgb_model.fit(X, y)

# Save model summary
model_summary = xgb_model.get_params()
with open("xgboost_summary.json", "w") as f:
    json.dump(model_summary, f, indent=4)

print("✅ XGBoost Model Summary saved to xgboost_summary.json")

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
    # Use the last 7 known or predicted values as lag features
    future_features = {
        "year": date.year,
        "month": date.month,
        "day": date.day,
        "weekday": date.weekday()
    }
    for i in range(7):
        future_features[f"lag_{i+1}"] = last_known_data[-(i+1)]  # Use latest known or predicted values

    # Convert to DataFrame and Predict
    future_input = pd.DataFrame([future_features])
    prediction = xgb_model.predict(future_input)[0]
    
    # Append Prediction and Update Lag Data
    predicted_values.append(prediction)
    last_known_data.append(prediction)
    last_known_data.pop(0)  # Keep only last 7 values

# Save Future Predictions
future_df["transaction_amount"] = predicted_values

# Calculate MAE
mae = mean_absolute_error(df["transaction_amount"].iloc[-forecast_days:], predicted_values[:forecast_days])

# Calculate MDAPE
mdape = np.median(np.abs((df["transaction_amount"].iloc[-forecast_days:] - predicted_values[:forecast_days]) / df["transaction_amount"].iloc[-forecast_days:])) * 100

# Print Results
print(f"✅ Mean Absolute Error (MAE): {mae:.2f}")
print(f"✅ Median Absolute Percentage Error (MDAPE): {mdape:.2f}%")

# Plot Results
plt.figure(figsize=(12, 5))
plt.plot(df.index, df["transaction_amount"], label="Actual Data", color="blue")
plt.plot(future_df.index, future_df["transaction_amount"], label="XGBoost Forecast", linestyle="dashed", color='orange')
plt.legend()
plt.xlabel("Date")
plt.ylabel("Transaction Amount")
plt.title("XGBoost Forecast for the Next 30 Days")
plt.savefig("xgboost_forecast.png", dpi=300)
# plt.show()

# Save forecast to CSV
future_df.to_csv("xgboost_forecast.csv")
