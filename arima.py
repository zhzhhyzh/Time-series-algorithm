import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv("food.csv")

# Convert date column
df['date'] = df['date'].str.replace('/', '-')
df['date'] = pd.to_datetime(df['date'], format="mixed")

# Fill missing dates
date_range = pd.date_range(start=df["date"].min(), end=df["date"].max())
complete_dates = pd.DataFrame(date_range, columns=["date"])
df_by_date = df.groupby("date").agg({"transaction_amount": ["sum"]}).reset_index()
df_by_date.columns = ["date", "transaction_amount"]
df_complete = pd.merge(complete_dates, df_by_date, on="date", how="left").fillna(0)

# Define test size
test_size = 7
last_test_date = df_complete.iloc[-test_size]["date"]
one_month_before = last_test_date - pd.DateOffset(months=1)

# Keep only one month of training data
df_train = df_complete[(df_complete["date"] >= one_month_before) & (df_complete["date"] < last_test_date)]
df_test = df_complete.iloc[-test_size:]

# Train ARIMA model
p, d, q = 5, 0, 5
model = ARIMA(df_train['transaction_amount'], order=(p, d, q))
model_fit = model.fit()
test_predictions = model_fit.forecast(steps=len(df_test)).values
df_test["arima_pred"] = test_predictions

# Plot results (Only Last Month + Test Data)
plt.figure(figsize=(14, 6))
sns.lineplot(data=df_train, y="transaction_amount", x="date", label="Train", color="blue")
sns.lineplot(data=df_test, y="transaction_amount", x="date", label="Test", color="orange")
sns.lineplot(data=df_test, y="arima_pred", x="date", label="Predictions", color="red", linestyle="dashed")
plt.title("ARIMA: Actual vs Predicted (One Month Training)")
plt.grid()
plt.ylim(0)
plt.xticks(rotation=45)
plt.savefig("arima_one_month.png", dpi=300)

# Calculate MAE
mae = mean_absolute_error(df_test["transaction_amount"], test_predictions)
print(f"Mean Absolute Error: {mae}")

# Save result to CSV 
result_df = pd.DataFrame([{"model": "ARIMA", "mae": mae}])
result_df.to_csv("result.csv", mode='a', header=not pd.io.common.file_exists("result.csv"), index=False)
