import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta

warnings.filterwarnings('ignore')

df = pd.read_csv("food.csv")
df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=False)

# Ensure all dates are included
date_range = pd.date_range(start=df["date"].min(), end=df["date"].max())
complete_dates = pd.DataFrame(date_range, columns=["date"])
df_by_date = df.groupby("date").agg({"transaction_amount": ["sum"]}).reset_index()
df_by_date.columns = ["date", "transaction_name"]
df_complete = pd.merge(complete_dates, df_by_date, on="date", how="left")
df_complete.fillna(0, inplace=True)

# Train-test split
test_size = 7
train_size = df_complete.shape[0] - test_size
df_train = df_complete.iloc[:train_size]
df_test = df_complete.iloc[train_size:]

# Define SARIMA parameters
p, d, q = 5, 0, 5
P, D, Q, m = 1, 1, 1, 7

# Fit SARIMA model
model = SARIMAX(df_train['transaction_name'], order=(p, d, q), seasonal_order=(P, D, Q, m))
model_fit = model.fit()

# Forecast
test_predictions = model_fit.forecast(steps=len(df_test)).values
df_test["sarimax_pred"] = test_predictions

# **Filter last 3 months for visualization**
last_3_months = df_train["date"].max() - timedelta(days=90)
df_train_filtered = df_train[df_train["date"] >= last_3_months]
df_test_filtered = df_test[df_test["date"] >= last_3_months]

# Plot the last 3 months
plt.figure(figsize=(14, 6))
sns.lineplot(data=df_train_filtered, y="transaction_name", x="date", label="Train")
sns.lineplot(data=df_test_filtered, y="transaction_name", x="date", label="Test", color="red")
sns.lineplot(data=df_test_filtered, y="sarimax_pred", x="date", label="SARIMA Predictions", color="green")

plt.title('SARIMA - 3 Months')
plt.grid()
plt.ylim(0)
plt.savefig("sarima.png", dpi=300)

# Calculate MAE
mae = mean_absolute_error(df_test["transaction_name"], test_predictions)
print(f'Mean Absolute Error: {mae}')

# Save result to CSV 
result_df = pd.DataFrame([{"model": "SARIMA", "mae": mae}])
result_df.to_csv("result.csv", mode='a', header=not pd.io.common.file_exists("result.csv"), index=False)
