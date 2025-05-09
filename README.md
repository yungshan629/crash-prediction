
# Crash Prediction with LSTM and Transformer

This repository contains a deep learning–based approach to predict financial market crashes using macroeconomic, market, and sentiment indicators. Two model architectures—LSTM and Transformer—are implemented and compared to evaluate their performance in forecasting major S&P 500 drawdowns.

📘 **Report**  
Full methodology, architecture, and evaluation details are provided in [`report.pdf`](./report.pdf).

---

## 📊 Project Overview

- **Objective:** Predict whether a market crash (S&P 500 drawdown > 15%) will occur using recent 60-day macro-financial data sequences.
- **Models:**  
  - Long Short-Term Memory (LSTM)  
  - Transformer with multi-head self-attention (1 encoder layer, 8 heads)
- **Target:** Binary label — `Crash_15pct` = 1 if cumulative drawdown > 15%, 0 otherwise
- **Time range:** 1990 to latest available data (over 30 years)

---

## 🗂️ Repository Structure

```text
crash-prediction/
├── data/               # Processed time-series data and label
├── notebooks/          # Model training and evaluation (LSTM, Transformer)
├── outputs/            # Performance plots (ROC, PR curves, confusion matrices)
└── report.pdf          # Full written report for review
````

---

## 📈 Key Results

| Model       | ROC AUC | Average Precision (AP) |
| ----------- | ------- | ---------------------- |
| LSTM        | 0.8423  | 0.17                   |
| Transformer | 0.8430  | 0.22                   |

* Transformer showed marginal improvement in handling imbalanced classification.
* LSTM achieved higher sensitivity; Transformer reduced false positives.
* Confusion matrices and evaluation metrics are available in the report and output plots.

---

## 🧠 Features Used

* **Market Indices:** S\&P 500, VIX
* **Yield Spreads:** 10Y–2Y, term spreads
* **Credit Risk:** HY–IG spread, CDS
* **Macro Indicators:** ISM, core CPI, unemployment
* **Sentiment & Liquidity:** Financial Stress Indices, TED Spread
* **Derived Features:** Copper–Gold ratio, CPI vs. Core CPI differential

---

## 💡 Future Directions

* Introduce lead-time forecasts (e.g., 30-day crash warning)
* Add technical indicators (e.g., RSI, LPPLS signal)
* Explore deeper Transformer architectures and attention-guided models

---

## 🔗 Reproducibility

Code and data are prepared for reproducibility. Full source code:
[https://github.com/yungshan629/crash-prediction](https://github.com/yungshan629/crash-prediction)

---

*This project was developed as part of an academic application portfolio to demonstrate deep learning applications in financial risk forecasting.*

