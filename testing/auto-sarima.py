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
warnings.filterwarnings('ignore')

df = pd.read_csv("food.csv")
df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=False)

date_range = pd.date_range(start=df["date"].min(), end=df["date"].max())
complete_dates = pd.DataFrame(date_range, columns=["date"])
df_by_date = df.groupby("date").agg({"transaction_amount": ["count"]}).reset_index()
df_by_date.columns = ["date", "item_name"]
df_complete = pd.merge(complete_dates, df_by_date, on="date", how="left")
df_complete.fillna(0, inplace=True)

test_size = 7

train_size = df_complete.shape[0] - test_size

df_train = df_complete.iloc[:train_size]
df_test = df_complete.iloc[train_size:]

metrics = []
model = auto_arima(
    df_train['item_name'],
    seasonal=True, 
    m=7,
    trace=True, 
    error_action='ignore', 
    suppress_warnings=True,
)
test_predictions = model.predict(n_periods=len(df_test)).values
df_test["auto_sarimax_pred"] = test_predictions
plt.figure(figsize=(14, 6))
sns.lineplot(data=df_train, y="item_name", x="date", label="Train")
sns.lineplot(data=df_test, y="item_name", x="date", label="Test")
sns.lineplot(data=df_test, y="auto_sarimax_pred", x="date", label="Auto SARIMAX Predictions")
plt.title('SARIMA')
plt.grid()
plt.ylim(0)


plt.savefig("auto-sarima.png", dpi=300)
mae = mean_absolute_error(df_test["item_name"], test_predictions)
metrics.append({"model": "Auto SARIMAX", "mae": mae})
print(df_test)
print(f'Mean Absolute Error: {mae}')