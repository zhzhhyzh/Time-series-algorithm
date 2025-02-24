import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from lightgbm import LGBMRegressor
warnings.filterwarnings('ignore')

df = pd.read_csv("food.csv")
df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=False)

date_range = pd.date_range(start=df["date"].min(), end=df["date"].max())
complete_dates = pd.DataFrame(date_range, columns=["date"])
df_by_date = df.groupby("date").agg({"transaction_amount": ["sum"]}).reset_index()
df_by_date.columns = ["date", "transaction_amount"]
df_complete = pd.merge(complete_dates, df_by_date, on="date", how="left")
df_complete.fillna(0, inplace=True)

# Define test size
test_size = 7
last_test_date = df_complete.iloc[-test_size]["date"]
three_months_before = last_test_date - pd.DateOffset(months=3)

# Keep only the last 3 months of training data
df_train = df_complete[(df_complete["date"] >= three_months_before) & (df_complete["date"] < last_test_date)]
df_test = df_complete.iloc[-test_size:]

metrics = []
FEATURES = []

# Create lag features
num_lags = 3
for lag in range(1, num_lags + 1):
    df_train[f'lag_{lag}'] = df_train['transaction_amount'].shift(lag)
    FEATURES.append(f'lag_{lag}')

# Drop NaN values caused by shifting
df_train.dropna(inplace=True)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    df_train.drop(columns=['transaction_amount']), df_train['transaction_amount'], test_size=0.2, shuffle=False
)

# Train LightGBM model
model = LGBMRegressor(learning_rate=0.1, num_leaves=31, n_estimators=100)
model.fit(X_train[FEATURES], y_train)

# Validation predictions
val_predictions = model.predict(X_val[FEATURES])
mae = mean_absolute_error(y_val, val_predictions)
print(f'Mean Absolute Error on validation set: {mae}')

# Forecasting for test period
X_test = y_val[-num_lags:][::-1].values
test_predictions = []

for i in range(len(df_test)):
    pred = model.predict(pd.DataFrame(X_test[:num_lags].reshape(1, -1), columns=FEATURES))[0]
    test_predictions.append(pred)
    X_test = np.array([pred] + X_test.tolist())

df_test["lgbm_pred"] = test_predictions

# Plot results (Only Last 3 Months + Test Data)
plt.figure(figsize=(14, 6))
sns.lineplot(data=df_train, y="transaction_amount", x="date", label="Train", color="blue")
sns.lineplot(data=df_test, y="transaction_amount", x="date", label="Test", color="orange")
sns.lineplot(data=df_test, y="lgbm_pred", x="date", label="LightGBM Predictions", color="red", linestyle="dashed")
plt.title("LightGBM: Actual vs Predicted (Last 3 Months)")
plt.grid()
plt.ylim(0)
plt.xticks(rotation=45)
plt.savefig("lightGBM.png", dpi=300)

# Calculate MAE
mae = mean_absolute_error(df_test["transaction_amount"], test_predictions)
metrics.append({"model": "LightGBM", "mae": mae})
print(df_test)
print(f'Mean Absolute Error: {mae}')

result_df = pd.DataFrame([{"model": "LIGHTGBM", "mae": mae}])
result_df.to_csv("result.csv", mode='a', header=not pd.io.common.file_exists("result.csv"), index=False)