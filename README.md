Mumbai Climate Forecasting with Time Series Analysis

Project Overview

This project focuses on analyzing and forecasting climate trends specifically for Mumbai. Using historical temperature data, the project aims to:

Understand long-term trends in average temperature.

Identify seasonal patterns and anomalies.

Develop predictive models using ARIMA to forecast future temperatures.

Evaluate model performance with appropriate error metrics.

The project provides insights into Mumbai's climate patterns and can aid in planning and decision-making for urban and environmental management.

Features

Data Preprocessing:

Handling missing data through interpolation.

Ensuring consistency in timestamps by addressing irregularities and duplicates.

Data Visualization:

Trend and seasonal decomposition.

Time series plots for Mumbai.

Stationarity Tests:

Conducting Augmented Dickey-Fuller (ADF) tests to check for stationarity.

ARIMA Modeling:

Building and fitting ARIMA models for time series forecasting.

Forecasting future temperature values with confidence intervals.

Model Evaluation:

Residual analysis and error metrics (MAE, MSE, RMSE).

Visualization of test data vs. forecasted values.

Dataset

The dataset includes global temperature data, with specific focus on the city of Mumbai:

Mumbai: Financial capital of India, with a coastal climate and distinct seasonal patterns.

Columns:

Year, Month, Day: Date components.

City: Name of the city (Mumbai).

AvgTemperature: Average daily temperature (in Fahrenheit).

Tools and Libraries

Python

Pandas: Data handling and preprocessing.

NumPy: Mathematical operations.

Matplotlib & Seaborn: Data visualization.

Statsmodels: ARIMA modeling and decomposition.

Sklearn: Model evaluation.

How to Run the Project

Clone the repository:

git clone https://github.com/alt-hell/climate-forecasting.git

Navigate to the project directory:

cd climate-forecasting

Install the required libraries:

pip install -r requirements.txt

Open the Jupyter Notebook:

jupyter notebook

Run the notebook climate_forecasting.ipynb step-by-step.

Results

Decomposition Analysis: Visualized trends, seasonality, and residuals for Mumbai.

Forecasts: Predicted future temperatures for Mumbai with ARIMA models.



Libraries: TensorFlow, Pandas, Statsmodels, etc.

