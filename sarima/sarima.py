import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # For non-GUI environments
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, acf, pacf
from sklearn.metrics import mean_absolute_error
from itertools import product

# Load dataset
df = pd.read_csv("food.csv")

# Convert 'date' column to datetime
df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=False)

# Aggregate sales by date
df = df.groupby('date', as_index=False).agg({'transaction_amount': 'sum'})

# Sort by date and set as index
df = df.sort_values(by="date").reset_index(drop=True)
df.set_index("date", inplace=True)
df = df.asfreq('D')  # Ensure daily data

# Handle missing values
df['transaction_amount'] = df['transaction_amount'].interpolate(method="linear")

# Step 1: Determine Differencing Order (d)
def adf_test(series):
    """Perform Augmented Dickey-Fuller test to check stationarity."""
    result = adfuller(series)
    return result[1] > 0.05  # If p-value > 0.05, data is non-stationary

d = 0
df["transaction_amount"] = (df["transaction_amount"] - df["transaction_amount"].mean()) / df["transaction_amount"].std()

while adf_test(df["transaction_amount"]):
    df["transaction_amount"] = df["transaction_amount"].diff().dropna()
    d += 1

# Step 2: Grid Search for Best SARIMA Parameters
p_values = range(0, 3)
q_values = range(0, 3)
P_values = range(0, 3)
Q_values = range(0, 3)
D = 1  # Seasonal differencing
s = 7  # Weekly seasonality

best_aic = float("inf")
best_params = None

for p, q, P, Q in product(p_values, q_values, P_values, Q_values):
    try:
        model = sm.tsa.statespace.SARIMAX(
            df['transaction_amount'],
            order=(p, d, q),
            seasonal_order=(P, D, Q, s),
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
        
        if model.aic < best_aic:
            best_aic = model.aic
            best_params = (p, d, q, P, D, Q, s)
    except:
        continue

(p, d, q, P, D, Q, s) = best_params
print(f"Best SARIMA Parameters: {best_params}")

# Train-Test Split
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# Train Best SARIMA Model
sarima_model = sm.tsa.statespace.SARIMAX(
    train['transaction_amount'], 
    order=(p, d, q), 
    seasonal_order=(P, D, Q, s),
    enforce_stationarity=False,
    enforce_invertibility=False
).fit()

# Forecast on test data
test_forecast = sarima_model.predict(start=len(train), end=len(df)-1, dynamic=False)

# Compute MAE and MDAPE
mae_sarima = mean_absolute_error(test['transaction_amount'], test_forecast)
mdape_sarima = np.median(np.abs((test['transaction_amount'] - test_forecast) / test['transaction_amount'])) * 100

# Step 3: Forecast Next 30 Days
future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=30)
future_forecast = sarima_model.forecast(steps=30)

# Save Forecast with Metrics to CSV
future_df = pd.DataFrame({"date": future_dates, "predicted_sales": future_forecast})
future_df.to_csv("sarima_forecast_30_days.csv", index=False)

# Save test results
test_results = pd.DataFrame({"date": test.index, "actual": test["transaction_amount"], "predicted": test_forecast})
test_results.to_csv("sarima_test_forecast.csv", index=False)

print(f"MAE (SARIMA): {mae_sarima:.2f}")
print(f"MDAPE (SARIMA): {mdape_sarima:.2f}%")
print(sarima_model.summary())


# Plot Results
plt.figure(figsize=(12, 5))
plt.plot(df.index, df["transaction_amount"], label="Actual Data", color="blue")
plt.plot(test.index, test_forecast, label="Test Forecast (SARIMA)", linestyle="dashed", color="red")
plt.plot(future_dates, future_forecast, label="Future Forecast (SARIMA, 30 days)", linestyle="dotted", color="green")
plt.legend()
plt.xlabel("Date")
plt.ylabel("Transaction Amount")
plt.title(f"SARIMA Forecast (30-day Prediction)\nMAE: {mae_sarima:.2f}, MDAPE: {mdape_sarima:.2f}%\nBest Params: {best_params}")
plt.savefig("sarima_forecast_30_days.png", dpi=300)