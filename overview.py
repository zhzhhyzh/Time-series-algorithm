import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv("food.csv")
df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=False)

date_range = pd.date_range(start=df["date"].min(), end=df["date"].max())
complete_dates = pd.DataFrame(date_range, columns=["date"])
df_by_date = df.groupby("date").agg({"transaction_amount": ["sum"]}).reset_index()
df_by_date.columns = ["date", "transaction_amount"]
df_complete = pd.merge(complete_dates, df_by_date, on="date", how="left")
df_complete.fillna(0, inplace=True)

test_size = 7

train_size = df_complete.shape[0] - test_size

df_train = df_complete.iloc[:train_size]
df_test = df_complete.iloc[train_size:]

plt.figure(figsize=(14, 6))
sns.lineplot(data=df_train, y="transaction_amount", x="date", label="Train")
sns.lineplot(data=df_test, y="transaction_amount", x="date", label="Test")
plt.grid()
plt.ylim(0)
plt.savefig("overview.png", dpi=300)



