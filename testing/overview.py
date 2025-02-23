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
from lightgbm import LGBMRegressor


df = pd.read_csv("food.csv")
df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=False)

date_range = pd.date_range(start=df["date"].min(), end=df["date"].max())
complete_dates = pd.DataFrame(date_range, columns=["date"])
df_by_date = df.groupby("date").agg({"transaction_amount": ["count"]}).reset_index()
df_by_date.columns = ["date", "item_type"]
df_complete = pd.merge(complete_dates, df_by_date, on="date", how="left")
df_complete.fillna(0, inplace=True)

test_size = 7

train_size = df_complete.shape[0] - test_size

df_train = df_complete.iloc[:train_size]
df_test = df_complete.iloc[train_size:]

metrics = []

plt.figure(figsize=(14, 6))
sns.lineplot(data=df_train, y="item_type", x="date", label="Train")
sns.lineplot(data=df_test, y="item_type", x="date", label="Test")
plt.grid()
plt.ylim(0)
plt.savefig("overview.png", dpi=300)

p, d, q = 5, 0, 5
model = ARIMA(df_train['item_type'], order=(p, d, q))
model_fit = model.fit()
test_predictions = model_fit.forecast(steps=len(df_test)).values
df_test["arima_pred"] = test_predictions

plt.figure(figsize=(14, 6))
sns.lineplot(data=df_train, y="item_type", x="date", label="Train")
sns.lineplot(data=df_test, y="item_type", x="date", label="Test")
sns.lineplot(data=df_test, y="arima_pred", x="date", label="Predictions")
plt.title('ARIMA')
plt.grid()
plt.ylim(0)
plt.savefig("arima.png", dpi=300)
mae = mean_absolute_error(df_test["item_type"], test_predictions)
metrics.append({"model": "ARIMA", "mae": mae})
print(f'Mean Absolute Error: {mae}')

p, d, q = 5, 0, 5
P, D, Q, m = 1, 1, 1, 7
model = SARIMAX(df_train['item_type'], order=(p, d, q), seasonal_order=(P, D, Q, m))
model_fit = model.fit()

test_predictions = model_fit.forecast(steps=len(df_test)).values
df_test["sarimax_pred"] = test_predictions
plt.figure(figsize=(14, 6))
sns.lineplot(data=df_train, y="item_type", x="date", label="Train")
sns.lineplot(data=df_test, y="item_type", x="date", label="Test")
sns.lineplot(data=df_test, y="sarimax_pred", x="date", label="SARIMA Predictions")
plt.title('SARIMA')
plt.grid()
plt.ylim(0)
plt.savefig("sarima.png", dpi=300)
mae = mean_absolute_error(df_test["item_type"], test_predictions)
metrics.append({"model": "SARIMAX", "mae": mae})
print(f'Mean Absolute Error: {mae}')

model = auto_arima(
    df_train['item_type'],
    seasonal=False, 
    trace=True, 
    error_action='ignore', 
    suppress_warnings=True,
)
test_predictions = model.predict(n_periods=len(df_test)).values
df_test["auto_arima_pred"] = test_predictions
plt.figure(figsize=(14, 6))
sns.lineplot(data=df_train, y="item_type", x="date", label="Train")
sns.lineplot(data=df_test, y="item_type", x="date", label="Test")
sns.lineplot(data=df_test, y="auto_arima_pred", x="date", label="Auto Arima Predictions")
plt.title('SARIMA')
plt.grid()
plt.ylim(0)

plt.savefig("auto-arima.png", dpi=300)
mae = mean_absolute_error(df_test["item_type"], test_predictions)
metrics.append({"model": "Auto ARIMA", "mae": mae})
print(f'Mean Absolute Error: {mae}')

model = auto_arima(
    df_train['item_type'],
    seasonal=True, 
    m=7,
    trace=True, 
    error_action='ignore', 
    suppress_warnings=True,
)
test_predictions = model.predict(n_periods=len(df_test)).values
df_test["auto_sarimax_pred"] = test_predictions
plt.figure(figsize=(14, 6))
sns.lineplot(data=df_train, y="item_type", x="date", label="Train")
sns.lineplot(data=df_test, y="item_type", x="date", label="Test")
sns.lineplot(data=df_test, y="auto_sarimax_pred", x="date", label="Auto SARIMAX Predictions")
plt.title('SARIMA')
plt.grid()
plt.ylim(0)


plt.savefig("auto-sarima.png", dpi=300)
mae = mean_absolute_error(df_test["item_type"], test_predictions)
metrics.append({"model": "Auto SARIMAX", "mae": mae})
print(f'Mean Absolute Error: {mae}')

train_data, val_data = train_test_split(df_train['item_type'], test_size=0.2, shuffle=False)

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
best_model = ARIMA(df_train['item_type'], order=(p_best, d_best, q_best))
best_model_fit = best_model.fit()

test_predictions = model_fit.forecast(steps=len(df_test)).values
df_test["custom_auto_arima_pred"] = test_predictions
plt.figure(figsize=(14, 6))
sns.lineplot(data=df_train, y="item_type", x="date", label="Train")
sns.lineplot(data=df_test, y="item_type", x="date", label="Test")
sns.lineplot(data=df_test, y="custom_auto_arima_pred", x="date", label="custom_auto_arima_pred")
plt.title('SARIMA')
plt.grid()
plt.ylim(0)

plt.savefig("cauto-sarima.png", dpi=300)
mae = mean_absolute_error(df_test["item_type"], test_predictions)
metrics.append({"model": "Custom Auto ARIMA", "mae": mae})
print(f'Mean Absolute Error: {mae}')

FEATURES = []
    
num_lags = 3
for lag in range(1, num_lags + 1):
    df_train[f'lag_{lag}'] = df_train['item_type'].shift(lag)
    FEATURES.append(f'lag_{lag}')
    X_train, X_val, y_train, y_val = train_test_split(df_train.drop(columns=['item_type']), df_train['item_type'], test_size=0.2, shuffle=False)

model = LGBMRegressor(learning_rate=0.1, num_leaves=31, n_estimators=100)
model.fit(X_train[FEATURES], y_train)

val_predictions = model.predict(X_val[FEATURES])
mae = mean_absolute_error(y_val, val_predictions)
print(f'Mean Absolute Error on validation set: {mae}')

X_test = y_val[-num_lags:][::-1].values

test_predictions = []

for i in range(len(df_test)):
    pred = model.predict(pd.DataFrame(X_test[:num_lags].reshape(1, -1), columns=FEATURES))[0]
    
    test_predictions.append(pred)
    
    X_test = np.array([pred] + X_test.tolist())

df_test["lgbm_pred"] = test_predictions

plt.figure(figsize=(14, 6))
sns.lineplot(data=df_train, y="item_type", x="date", label="Train")
sns.lineplot(data=df_test, y="item_type", x="date", label="Test")
sns.lineplot(data=df_test, y="lgbm_pred", x="date", label="lgbm_pred")
plt.title('SARIMA')
plt.grid()
plt.ylim(0)

plt.savefig("lightGBM.png", dpi=300)
mae = mean_absolute_error(df_test["item_type"], test_predictions)
metrics.append({"model": "LightGBM", "mae": mae})
print(f'Mean Absolute Error: {mae}')

df_metrics = pd.DataFrame(metrics)
plt.figure(figsize=(8, 1 * df_metrics.shape[0] // 2))
sns.barplot(data=df_metrics, y="model", x="mae")
plt.savefig("result.png", dpi=1200)
print(df_metrics)