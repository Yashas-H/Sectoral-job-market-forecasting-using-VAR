# 📈 Forecasting U.S. Job Openings Using Macroeconomic Indicators

This project forecasts U.S. job openings using macroeconomic time series data and advanced statistical modeling. By leveraging datasets from **FRED** and **BLS**, it provides accurate and scalable predictions of labor demand across sectors using a **Vector Autoregression (VAR)** framework. Designed for economists, data scientists, and policymakers, the pipeline emphasizes **efficiency, adaptability**, and **real-time forecasting**.

---

## 🔍 Problem Overview

Understanding job openings is key to gauging labor market health and economic momentum. This project:

- Forecasts sector-level U.S. job openings.
- Integrates data from FRED and BLS (JOLTS).
- Uses VAR models to capture interdependencies between variables like:
  - GDP
  - Unemployment rate
  - CPI
  - Interest rates
  - S&P 500
  - Initial jobless claims

---

## 📊 Data Sources

- **FRED API** – GDP, interest rates, unemployment, CPI, SP500
- **BLS JOLTS** – Monthly job openings by industry

Data is aligned to a monthly frequency using interpolation and smoothing.

---

## 🧪 Methodology

### 🔄 Preprocessing

- Merged multi-source datasets with outer joins on `date`.
- Filled missing values using forward/backward fill and cubic interpolation (`scipy.interpolate.interp1d`).
- Performed outlier clipping using IQR.
- Applied log transformation for high-variance features.
- Injected Gaussian noise for zero-variance columns to avoid transformation errors.

### 📐 Stationarity Enforcement

- Conducted ADF tests.
- Used differencing and seasonal differencing (lags of 3, 6, 12 months).
- Ensured reversibility using NumPy vectorization (`np.cumsum`, slicing).

---

## 🔁 Rolling Forecasting

- **Rolling window** strategy for training/testing:
  - 60% historical window for training
  - Forecasting horizon: 1–4 months
- Recursive forecasting using previous outputs
- Forecasts reversed to original scale with exponential and cumulative transformations

---

## 📈 Evaluation & Visualization

- Metrics: `MAE`, `RMSE` (per sector, averaged across windows)
- Forecast vs. actual plots for all key variables
- Plots saved for headless environments (`plt.savefig()`)

---

## ⚙️ Optimizations

| Optimization | Description |
|--------------|-------------|
| 🔃 Recursive Forecast Vectorization | Replaced loops with `np.empty` and `np.cumsum` |
| 🧮 Differencing Efficiency | Vectorized differencing & inverse transforms |
| 🔊 Log Transform + Noise Injection | Improves stability and avoids `log(0)` errors |
| 🧭 Clean Window Indexing | Replaced `get_loc()` with robust index logic |
| ⚡ Parallel Grid Search | Used `joblib.Parallel(n_jobs=-1)` for faster tuning |
| 🧵 Accurate Interpolation | Used `interp1d()` over `.resample()` for quarterly → monthly |

> 💡 Total runtime reduced from **54s to 33s** (~39% speedup on 8-core CPU).

---

## 📦 Technologies Used

- Python 3.x
- pandas, numpy, statsmodels, scipy
- joblib
- matplotlib, seaborn
- FRED API, BLS API

---

## ✅ Conclusion

This project delivers an optimized, modular, and production-ready framework for forecasting labor market dynamics in the U.S. economy. Its design supports:

- Real-time forecasting with adaptive VAR models
- Fast execution through vectorization and parallelization
- Sector-specific insights into labor demand

---

## 🚀 Future Work

- Incorporate exogenous shocks (e.g., COVID-19, policy changes)
- Support for real-time streaming data
- Explore nonlinear models (LSTM, Transformers)

---

