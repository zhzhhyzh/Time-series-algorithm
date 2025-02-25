# Time-Series Algorithm Repository

This repository contains various time-series forecasting models, each implemented in its own dedicated file. It also includes scripts for data overview and model evaluation.

## Repository Structure

time-series-algorithm/ <br>
├── overview.py        # Generates an overview plot of the dataset (overview.png) <br>
├── evaluate.py        # Evaluates the results from different models (results.csv) <br>
├── arima.py          # Implementation of ARIMA <br>
├── sarima.py          # Implementation of SARIMA <br>
├── auto-arima.py          # Implementation of Auto ARIMA <br>
├── auto-sarima.py          # Implementation of Auto SARIMA <br>
├── custom-auto-arima.py          # Implementation of Custom Auto ARIMA <br>
├── lightGBM.py          # Implementation of LightGBM <br>
├── xgb.py          # Implementation of XGBoost <br>
└── results.csv         # CSV file containing the forecasting results from each model <br>


## ⚡ Usage

1. **Generate Dataset Overview**
   - Run `overview.py` to create a dataset visualization (`overview.png`):
     ```bash
     python overview.py
     ```

2. **Run Forecasting Models**
   - Execute any of the model scripts (e.g., ARIMA, SARIMA) to train and generate results:
     ```bash
     python arima.py
     ```

3. **Evaluate Model Performance**
   - Use `evaluate.py` to compare model outputs stored in `results.csv`:
     ```bash
     python evaluate.py
     ```

---

## 📌 Models Included

- **ARIMA** – Auto-Regressive Integrated Moving Average
- **SARIMA** – Seasonal ARIMA with seasonal adjustments
- **Auto ARIMA** – Automatically selects the best ARIMA parameters
- **Auto SARIMA** – Enhanced Auto ARIMA with seasonal components
- **Custom Auto ARIMA** – Flexible tuning of ARIMA parameters
- **LightGBM** – Gradient boosting framework based on decision trees
- **XGBoost** – Optimized gradient boosting for superior performance

---

## ✅ Contribution

Feel free to fork this repository and contribute! Pull requests are welcome for bug fixes, new features, or model enhancements.

---
|
