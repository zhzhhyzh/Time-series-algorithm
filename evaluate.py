import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load metrics CSV correctly
df_metrics = pd.read_csv("result.csv")

# Plot bar chart
plt.figure(figsize=(8, max(3, 1 * df_metrics.shape[0] // 2)))  # Avoid very small figures
sns.barplot(data=df_metrics, y="model", x="mae", palette="Reds_r")  # Improve color
plt.xlabel("Mean Absolute Error (MAE)")
plt.ylabel("Model")
plt.title("Model Performance Comparison")
plt.savefig("result.png", dpi=1200)
plt.show()  # Display the plot

# Print DataFrame
print(df_metrics)
