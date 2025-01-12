#!/usr/bin/env python
# coding: utf-8

# # 1. INTRODUCTION:

# 
# 
# ## This dataset contains global time series data representing various metrics across cities around the world. 
# The data captures information that reflects trends, seasonality, and patterns specific to each city. Among these, four prominent Indian cities—Bombay (Mumbai), Calcutta (Kolkata), Chennai (Madras), and Delhi—have been selected as the focus of this project due to their economic, cultural, and historical significance.
# 
# These cities are not only vital to India's economy but also exhibit unique environmental, social, and urban patterns. By analyzing historical data for these cities, this project aims to develop predictive models that can forecast future trends based on past observations.
# 
# ## Why Focus on These Cities?
# 
# #### Bombay (Mumbai):
# 
# Formerly known as Bombay, Mumbai is the financial capital of India.
# It is known for its bustling economy, dense population, and coastal climate.
# 
# #### Calcutta (Kolkata):
# 
# Renamed Kolkata, this city is the cultural capital of India.
# It has a rich history and is located in eastern India by the Hooghly River.
# 
# #### Chennai (Madras):
# 
# Formerly called Madras, Chennai is a major hub in southern India.
# Known for its tropical climate and thriving automotive and IT industries.
# 
# #### Delhi:
# 
# The national capital of India, Delhi is a historical and administrative center.
# It experiences an extreme climate with distinct summer, winter, and monsoon seasons.

# # 2. Problem Statement:

# The goal of this project is to build a predictive model for time series data of four key Indian cities: Bombay (Mumbai), Calcutta (Kolkata), Chennai (Madras), and Delhi. Using historical data, the project seeks to address the following:
# 
# ### Trend Analysis:
# Understand long-term trends in the data, including economic, climatic, or urbanization patterns.
# 
# ### Seasonality Exploration:
# Identify seasonal patterns (e.g., monsoon, festival periods) that influence city-specific metrics.
# 
# ### Forecasting Future Trends:
# Develop a robust model capable of predicting future values for the selected cities, aiding in better planning and decision-making.
# 
# ### Comparative Analysis:
# Analyze the similarities and differences in patterns across the four cities to gain insights into regional variations.
# 
# ### Key Challenges
# Handling a large, global dataset and isolating relevant data for the four cities.
# Ensuring stationarity of the time series data for effective modeling.
# Balancing complex patterns of trend, seasonality, and irregularity unique to each city.

# # 3.  Import Libraries

# In[1]:


# Importing Libraries
import numpy as np # mathematical library
# ..............................................................................................................................
import pandas as pd # for handling data
pd.set_option('display.precision', 2)
# ..............................................................................................................................
import matplotlib.pyplot as plt # for plotting graphs
get_ipython().run_line_magic('matplotlib', 'inline')
#...............................................................................................................................
import seaborn as sns
#...............................................................................................................................
import warnings
warnings.filterwarnings("ignore")


# # 4.  Data Acquisition 

# In[2]:


avg_temperature = pd.read_csv(r"C:\Users\HELL\Documents\avg_temperature.csv")


# In[3]:


avg_temperature


# # 5. Data Pre-processinng

# In[4]:


avg_temperature.info()


# In[5]:


list(avg_temperature[avg_temperature['Country']=='India']['City'].unique())


# In[6]:


city_of_interest = list(avg_temperature[avg_temperature['Country']=='India']['City'].unique())
print(city_of_interest)


# In[7]:


indian_temp = avg_temperature[avg_temperature['Country']=='India']


# In[8]:


indian_temp


# In[9]:


cols = ['Year','Month','Day']
indian_temp['date'] = indian_temp[cols].apply(lambda x : ('-').join(x.values.astype(str)),axis=1)


# In[10]:


indian_temp


# In[11]:


indian_temp.info()


# In[12]:


indian_temp['date']= pd.to_datetime(indian_temp['date'])


# In[13]:


indian_temp.info()


# In[14]:


delhi_temp = indian_temp[indian_temp['City']=='Delhi'][['date','AvgTemperature']]
mumbai_temp = indian_temp[indian_temp['City']=='Bombay (Mumbai)'][['date','AvgTemperature']]
chennai_temp = indian_temp[indian_temp['City']=='Chennai (Madras)'][['date','AvgTemperature']]
kolkata_temp = indian_temp[indian_temp['City']=='Calcutta'][['date','AvgTemperature']]


# In[15]:


delhi_temp.set_index('date', inplace = True)


# In[16]:


mumbai_temp.set_index('date', inplace = True)
chennai_temp.set_index('date', inplace = True)
kolkata_temp.set_index('date', inplace = True)


# In[17]:


mumbai_temp


# # 6. Visualization of Data

# In[18]:


plt.subplot(4,1,1)
plt.plot(delhi_temp)

plt.subplot(4,1,2)
plt.plot(mumbai_temp)

plt.subplot(4,1,3)
plt.plot(chennai_temp)

plt.subplot(4,1,4)
plt.plot(kolkata_temp)


# In[19]:


mumbai_temp.describe()


# - Value of all sudden drops are same -99 , which might be bacause default reading of temp sensor in case of non-functoning. 
# - These data points can be treated as missing values

# In[20]:


mumbai_temp.plot(figsize=(12,4))


# #### Filling missing values by mean of values against t-1 and t+1

# In[21]:


mumbai_temp[mumbai_temp['AvgTemperature']==-99.0]


# In[22]:


# Interpolate the missing values (for -99.0)
mumbai_temp['AvgTemperature'] = mumbai_temp['AvgTemperature'].replace(-99.0, np.nan)
print(mumbai_temp)


# In[23]:


mumbai_temp.isna().sum()


# In[24]:


mumbai_temp['1998-12-23':'1998-12-26']


# In[25]:


# Fill NaNs with the average of forward-fill and backward-fill values
mumbai_temp['AvgTemperature'] = (
    mumbai_temp['AvgTemperature'].fillna(method='ffill') + 
    mumbai_temp['AvgTemperature'].fillna(method='bfill')
) / 2


# In[26]:


mumbai_temp['1998-12-23':'1998-12-26']


# In[27]:


time_diffs = mumbai_temp.index.to_series().diff()

# Defining  expected frequency ( '1D' for daily)
expected_frequency = pd.Timedelta('1D')  # based on time series

# Check for irregularities
irregularities = time_diffs[time_diffs != expected_frequency]

# result
if not irregularities.empty:
    print("Irregular time intervals found:")
    print(irregularities)
else:
    print("Timestamps are in sequence.")


# In[28]:


duplicates = mumbai_temp.index[mumbai_temp.index.duplicated(keep=False)]
print("Duplicate timestamps found:")
print(duplicates)


# In[29]:


mumbai_temp = mumbai_temp[~mumbai_temp.index.duplicated(keep='first')]


# In[30]:


time_diffs = mumbai_temp.index.to_series().diff()
irregularities = time_diffs[time_diffs != expected_frequency]

if not irregularities.empty:
    print("Still irregular intervals:")
    print(irregularities)
else:
    print("Timestamps are now consistent.")


# In[31]:


mumbai_temp.index.duplicated().sum()


# In[32]:


mumbai_temp.plot(figsize=(12,4))


# In[33]:


from statsmodels.tsa.seasonal import seasonal_decompose

# Decompose the time series using the correct column name
decomposition = seasonal_decompose(mumbai_temp["AvgTemperature"], model="additive", period=365)

# Plot the decomposition
fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

decomposition.observed.plot(ax=axes[0], title="Observed")
decomposition.trend.plot(ax=axes[1], title="Trend")
decomposition.seasonal.plot(ax=axes[2], title="Seasonal")
decomposition.resid.plot(ax=axes[3], title="Residual")

plt.tight_layout()
plt.show()


# In[34]:


from statsmodels.tsa.stattools import adfuller, kpss
# Perform ADF Test
result_adf = adfuller(mumbai_temp["AvgTemperature"].dropna())

print("ADF Test Results:")
print(f"ADF Statistic: {result_adf[0]}")
print(f"p-value: {result_adf[1]}")
print("Critical Values:")
for key, value in result_adf[4].items():
    print(f"   {key}: {value}")

if result_adf[1] < 0.01:
    print("The series is stationary (reject H0).")
else:
    print("The series is non-stationary (fail to reject H0).")


# In[35]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# Plot ACF
plot_acf(mumbai_temp["AvgTemperature"])
plt.show()

# Plot PACF
plot_pacf(mumbai_temp["AvgTemperature"])
plt.show()


# In[37]:


train = mumbai_temp.iloc[:-12]  # All but the last 12 months
test = mumbai_temp.iloc[-12:]  # Last 12 months


# In[36]:


from statsmodels.tsa.arima.model import ARIMA


# In[38]:


# Define the ARIMA model
model = ARIMA(train["AvgTemperature"], order=(3, 0, 0))  # Order (p, d, q)

# Fit the model
arima_model = model.fit()

# Print model summary
print(arima_model.summary())


# In[39]:


forecast = arima_model.forecast(steps=10)
print(forecast)


# In[40]:


# Use get_forecast() to get both forecasted values and confidence intervals
forecast_result = arima_model.get_forecast(steps=12)

# Extract the forecasted values
forecast_values = forecast_result.predicted_mean

# Extract the confidence intervals
confidence_intervals = forecast_result.conf_int()

# Display forecasted values and confidence intervals
print("Forecasted Values:")
print(forecast_values)
print("\nConfidence Intervals:")
print(confidence_intervals)


# In[41]:


import matplotlib.pyplot as plt

# Plot the actual data
plt.plot(train.index, train["AvgTemperature"], label="Train Data")

# Forecast index (adjust frequency as needed)
forecast_index = pd.date_range(start=train.index[-1], periods=12, freq="D")

# Plot the forecasted values
plt.plot(forecast_index, forecast_values, label="Forecast", color="orange")

# Add confidence intervals
plt.fill_between(forecast_index,
                 confidence_intervals.iloc[:, 0],
                 confidence_intervals.iloc[:, 1],
                 color="lightgrey", alpha=0.5, label="Confidence Interval")

# Add labels and legend
plt.xlabel("Date")
plt.ylabel("Avg Temperature")
plt.title("ARIMA Forecast")
plt.legend()
plt.show()


# In[50]:


mumbai_temp.max() - mumbai_temp.min()


# In[54]:


forecast_steps = len(test)  # Number of steps to forecast, same as test data length
forecast_values = arima_model.forecast(steps=forecast_steps)

# Step 2: Calculate residuals (errors)
residuals = test["AvgTemperature"] - forecast_values

# 1. Line Plot for Test vs Forecasted Values
plt.figure(figsize=(12, 6))
plt.plot(test.index, test["AvgTemperature"], label="Test Data", color="green")
plt.plot(test.index, forecast_values, label="Forecasted Values", color="red", linestyle="--")
plt.title("Test Data vs. Forecasted Values")
plt.xlabel("Time")
plt.ylabel("Avg Temperature")
plt.legend()
plt.grid(True)
plt.show()

# 2. Residual Plot (Expected to be random if the model is good)
plt.figure(figsize=(12, 6))
plt.plot(test.index, residuals, label="Residuals", color="purple")
plt.axhline(y=0, color="black", linestyle="--", label="Zero Line")
plt.title("Residual Plot")
plt.xlabel("Time")
plt.ylabel("Residuals (Actual - Predicted)")
plt.legend()
plt.grid(True)
plt.show()

# 3. Histogram of Residuals (Error Distribution)
plt.figure(figsize=(12, 6))
plt.hist(residuals, bins=20, alpha=0.7, label="Residuals", color="orange")
plt.title("Distribution of Residuals")
plt.xlabel("Residuals (Actual - Predicted)")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.show()

# 4. Error Metrics 
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(test["AvgTemperature"], forecast_values)
mse = mean_squared_error(test["AvgTemperature"], forecast_values)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")


# In[ ]:




