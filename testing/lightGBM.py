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
df_by_date = df.groupby("date").agg({"transaction_amount": ["count"]}).reset_index()
df_by_date.columns = ["date", "item_name"]
df_complete = pd.merge(complete_dates, df_by_date, on="date", how="left")
df_complete.fillna(0, inplace=True)

test_size = 7

train_size = df_complete.shape[0] - test_size

df_train = df_complete.iloc[:train_size]
df_test = df_complete.iloc[train_size:]

metrics = []
FEATURES = []
    
num_lags = 3
for lag in range(1, num_lags + 1):
    df_train[f'lag_{lag}'] = df_train['item_name'].shift(lag)
    FEATURES.append(f'lag_{lag}')
    X_train, X_val, y_train, y_val = train_test_split(df_train.drop(columns=['item_name']), df_train['item_name'], test_size=0.2, shuffle=False)

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
sns.lineplot(data=df_train, y="item_name", x="date", label="Train")
sns.lineplot(data=df_test, y="item_name", x="date", label="Test")
sns.lineplot(data=df_test, y="lgbm_pred", x="date", label="lgbm_pred")
plt.title('SARIMA')
plt.grid()
plt.ylim(0)

plt.savefig("lightGBM.png", dpi=300)
mae = mean_absolute_error(df_test["item_name"], test_predictions)
metrics.append({"model": "LightGBM", "mae": mae})
print(df_test)
print(f'Mean Absolute Error: {mae}')


