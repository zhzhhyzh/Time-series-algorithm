import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Ensure non-GUI environments work
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, arma_order_select_ic
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
from arch.unitroot import ADF

# Load dataset
df = pd.read_csv("food.csv")

# Convert 'date' column to datetime
df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=False)

# Aggregate sales by date to remove duplicates
df = df.groupby('date', as_index=False).agg({'transaction_amount': 'sum'})

# Sort by date
df = df.sort_values(by="date").reset_index(drop=True)

# Set 'date' as the index and ensure daily frequency
df.set_index("date", inplace=True)
df = df.asfreq('D')  # Ensure daily data

# Define target column
target_col = "transaction_amount"

# Convert to numeric & handle missing values
df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
df[target_col] = df[target_col].interpolate(method="linear")  # Better handling of missing values

# ADF Test
adf_test = ADF(df[target_col])
print(f"ADF Statistic: {adf_test.stat:.4f}, p-value: {adf_test.pvalue:.4f}")
d = 0  # Differencing order
if adf_test.pvalue > 0.05:
    print("Series is NOT stationary. Applying differencing.")
    df["diff_amount"] = df[target_col].diff().dropna()
    d = 1  # First-order differencing
else:
    print("Series is stationary.")
    df["diff_amount"] = df[target_col]

# Log transform for stabilization
df["log_amount"] = np.log1p(df[target_col])
df["diff_log_amount"] = df["log_amount"].diff().dropna()

# Train-Test Split
train_size = int(len(df) * 0.8)
train, test = df["diff_log_amount"][:train_size], df["diff_log_amount"][train_size:]

# Manually define ARIMA order to avoid auto_arima issues
manual_order = (1,1,0)  # Adjust if needed

# Fit ARIMA Model with better settings
model = ARIMA(train, order=manual_order, enforce_stationarity=False, enforce_invertibility=False)
model_fit = model.fit(method_kwargs={"maxiter": 500})  # Allow more iterations

# Print ARIMA Summary
print(model_fit.summary())

# Forecast
forecast_steps = len(test)
test_forecast = model_fit.forecast(steps=forecast_steps)
test_forecast = np.expm1(test_forecast)  

# Compute MAE
mae = mean_absolute_error(np.expm1(test.cumsum() + train.iloc[-1]), test_forecast)

# Compute MDAPE
mdape = np.median(np.abs((np.expm1(test.cumsum() + train.iloc[-1]) - test_forecast) / np.expm1(test.cumsum() + train.iloc[-1]))) * 100

print(f"Improved MAE: {mae:.2f}")
print(f"Improved MDAPE: {mdape:.2f}%")

# Save test data
test_data = pd.DataFrame({"date": test.index, "actual": np.expm1(test.cumsum() + train.iloc[-1])})
test_data.to_csv("tested_data.csv", index=False)

# Save predicted data
predicted_data = pd.DataFrame({"date": test.index, "predicted": test_forecast})
predicted_data.to_csv("predicted_data.csv", index=False)

# Forecast for the next 30 days
future_forecast = model_fit.forecast(steps=30)
future_forecast = np.expm1(future_forecast + df["log_amount"].iloc[-1])  # Reverse differencing properly

# Plot results
plt.figure(figsize=(12, 5))
plt.plot(df.index, np.expm1(df["log_amount"]), label="Actual Data")
plt.plot(test.index, test_forecast, label="Test Forecast", linestyle="dashed", color='red')
plt.plot(pd.date_range(df.index[-1], periods=31, freq='D')[1:], future_forecast, label="Future Forecast (30 days)", linestyle="dashed", color='green')
plt.legend()
plt.xlabel("Date")
plt.ylabel("Transaction Amount")
plt.title("ARIMA Forecast")

# Save figure
plt.savefig("arima.png", dpi=300)

# plt.show(block=True)  # Uncomment if running interactively
