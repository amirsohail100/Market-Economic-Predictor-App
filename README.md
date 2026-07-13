# 📈 Market & Economic Predictor App

An interactive Machine Learning web application built with **Streamlit** that predicts the **IHSG (Jakarta Composite Index)** based on various global and local macroeconomic indicators.

The model analyzes long-term stock market relationships by capturing yearly macro-trends combined with robust feature scaling on volatile market indicators.

## 🚀 Live Demo

Deploy your live Streamlit link here: `https://share.streamlit.io/your-username/your-repo-name`

---

## 🛠️ Features & Architecture

- **Target Variable (Output):** `IHSG` (Stock Market Index)
- **Engineered Input Feature:** `Year` (Extracted as an Integer to model annual temporal trends without scaling)
- **Scaled Input Features:** `OIL`, `GOLD`, `USDIDR`, `SP500`, `VIX`, `CPI`, `BI_rate`, `US_rate`
- **Preprocessing:** Features are normalized using `StandardScaler` (excluding the Year column).
- **Robust Error Handling:** Built-in validation architecture ensuring the UI renders safely even if dependencies or `.pickle` model artifacts are missing.

---

## 📂 Project Structure

```text
├── app.py               # Main Streamlit Application Script
├── model.pickle         # Trained Machine Learning Model (Joblib/Pickle)
├── scaler.pickle        # Fitted StandardScaler Instance
├── columns.pickle       # List of Trained Feature Columns
├── requirements.txt     # Python Dependencies
└── README.md            # Project Documentation

🚀 An interactive Streamlit Web App that predicts the Indonesian Stock Market Index (IHSG) based on global economic indicators (Gold, Oil, S&P 500, VIX, CPI, Interest Rates). Powered by a trained Machine Learning Regressor with annual macro-trend tracking and feature scaling. Developed for robust financial analytics. 📊📈
```
