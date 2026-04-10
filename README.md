📊 multi-model-revenue-forecasting

🚀 Overview

This project builds an end-to-end revenue forecasting system using Apple’s quarterly financial data and external demand signals from Google Trends. It combines multiple time-series and machine learning models to generate accurate forecasts and provide explainable insights.

The system predicts next 4 quarters (1-year ahead) revenue and presents results through an interactive dashboard.

⸻

🧠 Key Features
	•	Multi-model forecasting: Ridge Regression, SARIMA, Holt-Winters
	•	Hybrid modeling approach combining regression and time-series methods
	•	Feature engineering for temporal and financial signals
	•	Model evaluation using RMSE and MAPE
	•	Explainable AI using feature importance and SHAP
	•	Interactive dashboard built with Streamlit
	•	Confidence intervals for forecast uncertainty

⸻

📂 Dataset
	•	Apple Financial Data (2015–2025) – Quarterly data from SEC filings
	•	Google Trends Data – External demand signal (quarterly aggregated)

Features Used:
	•	Revenue (target)
	•	Cash
	•	Total Liabilities
	•	Trend (Google Trends)
	•	Engineered Features:
	•	Revenue Lag (t-4)
	•	Year-over-Year Growth
	•	Trend Lag
	•	Quarter (seasonality)
	•	Financial indicators

⸻

⚙️ Models Implemented
	•	Ridge Regression
	•	SARIMA (Seasonal ARIMA)
	•	Holt-Winters (Triple Exponential Smoothing)
	•	Hybrid Model (Ridge + Holt-Winters)

Model Selection

Models are evaluated using:
	•	RMSE (Root Mean Squared Error)
	•	MAPE (Mean Absolute Percentage Error)

The best-performing model is selected dynamically for forecasting.

⸻

🔮 Forecasting
	•	Forecast Horizon: Next 4 Quarters
	•	Final predictions generated using the best-performing model (typically Holt-Winters)
	•	Includes confidence intervals for uncertainty estimation

⸻

📊 Dashboard Features

Built using Streamlit, the dashboard includes:

📋 Overview
	•	Dataset preview
	•	Summary statistics
	•	Feature explanation

📊 Model Performance
	•	RMSE and MAPE comparison
	•	Best model highlighting
	•	Performance insights

📈 Model Visualization
	•	Interactive comparison of actual vs predicted values
	•	Model toggle functionality

🔮 Forecast
	•	Future revenue prediction (4 quarters)
	•	Confidence intervals
	•	Summary metrics (total, average)
	•	Forecast breakdown table
	•	Insight generation

🧠 Feature Importance
	•	Coefficient-based importance (Ridge)
	•	Top 3 drivers
	•	Insight-based explanation
⸻

🔍 Explainability
	•	Feature Importance: Based on Ridge coefficients
⸻

🛠️ Tech Stack
	•	Languages & Libraries:
	•	Python
	•	Pandas, NumPy
	•	Scikit-learn
	•	Statsmodels
	•	Visualization:
	•	Plotly
	•	Frontend / Dashboard:
	•	Streamlit

⸻

📈 Results
	•	Successfully modeled revenue trends using multiple approaches
	•	Hybrid and Holt-Winters models showed strong performance
	•	Achieved low MAPE (~2–5%) depending on dataset variation
	•	Built a complete pipeline from data → model → insights

⸻

💡 Key Learnings
	•	Importance of feature engineering in time-series forecasting
	•	Trade-offs between statistical and ML-based models
	•	Model evaluation in small time-series datasets
	•	Explainability in financial ML systems
	•	Building production-style ML dashboards

⸻

🚀 How to Run
git clone https://github.com/your-username/revenue-forecasting-hybrid-timeseries.git
cd revenue-forecasting-hybrid-timeseries

pip install -r requirements.txt
streamlit run app.py


⸻

📌 Future Improvements
	•	Add real-time data integration
	•	Implement LSTM / deep learning models
	•	Improve hybrid model weighting dynamically
	•	Deploy on cloud (Streamlit Cloud / AWS)

⸻

🤝 Contributing

Feel free to fork the repo and improve the system!
