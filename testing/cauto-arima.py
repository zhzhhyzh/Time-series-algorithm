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
train_data, val_data = train_test_split(df_train['item_name'], test_size=0.2, shuffle=False)

p_values = range(0, 6)
d_values = range(0, 2)
q_values = range(0, 6)

tmp_metrics = []

for p, d, q in itertools.product(p_values, d_values, q_values):
    try:
        model = ARIMA(train_data, order=(p, d, q))
        model_fit = model.fit()

        val_predictions = model_fit.forecast(steps=len(val_data)).values
        val_data_pred = pd.Series(val_predictions, index=val_data.index)

        mae = mean_absolute_error(val_data, val_data_pred)

        tmp_metrics.append({"model": f"ARIMA({p},{d},{q})", "mae": mae})
        print(f'ARIMA({p},{d},{q}) - Mean Absolute Error: {mae}')
        
    except Exception as e:
        print(f"ARIMA({p},{d},{q}) failed with error: {e}")

tmp_results_df = pd.DataFrame(tmp_metrics)

best_params = tmp_results_df.loc[tmp_results_df['mae'].idxmin()]
print(f'Best parameters: {best_params["model"]}, MAE: {best_params["mae"]}')

p_best, d_best, q_best = map(int, best_params["model"][6:-1].split(","))
best_model = ARIMA(df_train['item_name'], order=(p_best, d_best, q_best))
best_model_fit = best_model.fit()

test_predictions = model_fit.forecast(steps=len(df_test)).values
df_test["custom_auto_arima_pred"] = test_predictions
plt.figure(figsize=(14, 6))
sns.lineplot(data=df_train, y="item_name", x="date", label="Train")
sns.lineplot(data=df_test, y="item_name", x="date", label="Test")
sns.lineplot(data=df_test, y="custom_auto_arima_pred", x="date", label="custom_auto_arima_pred")
plt.title('SARIMA')
plt.grid()
plt.ylim(0)

plt.savefig("cauto-sarima.png", dpi=300)
mae = mean_absolute_error(df_test["item_name"], test_predictions)
metrics.append({"model": "Custom Auto ARIMA", "mae": mae})
print(df_test)
print(f'Mean Absolute Error: {mae}')
