#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### Load the Libraries and Inspect the dataset 

### Data Cleaning Steps (Data Preprocessing)
## Load the Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt

#to plot
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#setting figure size
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10

#for normalizing data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

#read the file
df = pd.read_csv('Stock Prices updated.csv')


df = df[['Symbol','Date','Open','High','Low','Close','Adj Close','Volume']]

## Data Cleaning steps(data Preprocessing)
df.isnull().sum()
df.isna().sum()

# Remove any duplicate rows (if any)
# df.drop_duplicates(inplace=True)

# Check for missing values and drop them if any
#df.dropna(inplace=True)

# check for any outliers in the data
sns.boxplot(data=df[['Open','High','Low','Close','Adj Close','Volume']])

## get summary statistics of the data
df.describe()
df


# ## As we can see here Volume has many outliers so we will be further using IQR(Inter Quartile Range) to remove the Outliers from the dataset

# Compute the IQR for each numerical column
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Filter out rows with outliers in any numerical column
# df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]


import matplotlib.pyplot as plt
import pandas as pd

# Load the data


# Get the unique symbols
symbols = df['Symbol'].unique()

# Loop over the symbols and create a plot for each one
for symbol in symbols:
    # Filter the data for the current symbol
    subset = df[df['Symbol'] == symbol]

    # Create a figure and axis object
    fig, ax = plt.subplots()

    # Plot the data
    ax.plot(subset['Date'], subset['Volume'])

    # Set the x-axis label format
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
 #   ax.xaxis.set_major_formatter(plt.DateFormatter('%Y-%m-%d'))

    # Rotate the x-axis labels
    plt.xticks(rotation=45)

    # Set the axis labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Volume')
    ax.set_title(f'{symbol} - Stocks vs. Volume')

    # Show the plot
    #plt.show()
    data_pivot = df.pivot(index='Date', columns='Symbol', values='Volume')

# Plot the data
    data_pivot.plot(kind='line')

# Set the axis labels and title
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.title('Stocks vs. Volume')

# Show the plot
    plt.show()


# ## Check for any data inconsistencies or errors, such as negative values for prices or volumes

#df[df['Open'] < 0 ]
#df[df['Close'] < 0 ]
#df[df['Volume'] < 0]

## There are no negative open, close and Volume


# ## Exploratory Data Analysis

import matplotlib.pyplot as plt


import matplotlib.dates as mdates

# Get the unique symbols from the DataFrame
symbols = df['Symbol'].unique()

# Create a new figure and axis
fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(subset['Date'], subset['Open'])

    # Set the x-axis label format 
ax.xaxis.set_major_locator(plt.MaxNLocator(8))


# Plot the open prices for each symbol
for symbol in symbols:
    data = df[df['Symbol'] == symbol]
    ax.plot(data['Date'], data['Open'], label=symbol)

# Set the title and labels
start_date = df['Date'].iloc[0]
end_date = df['Date'].iloc[-1]
ax.set_title(f"Stock Open Prices from {start_date} to {end_date}", fontsize=25)
ax.set_xlabel('Date', fontsize=20)
ax.set_ylabel('Open Price', fontsize=20)

# Set the tick font sizes
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)

# Add the legend
ax.legend(loc='upper left', fontsize=15)

# Show the plot
plt.show()


# Get the unique symbols from the DataFrame
symbols = df['Symbol'].unique()

# Create a new figure and four subplots
fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))

ax.plot(subset['Date'], subset['Open'])

    # Set the x-axis label format 
ax.xaxis.set_major_locator(plt.MaxNLocator(8))


# Plot the open prices for each symbol in each subplot
for i, symbol in enumerate(symbols):
    data = df[df['Symbol'] == symbol]
    axs[i].plot(data['Date'], data['Open'], label=symbol)
    axs[i].set_title(symbol)

    # Set the tick font sizes
    axs[i].tick_params(axis='x', labelsize=10)
    axs[i].tick_params(axis='y', labelsize=10)

# Set the title and labels for the entire figure
end_date = df['Date'].iloc[-1]
fig.suptitle(f"Stock Open Prices from 2022-02-28 to {end_date}", fontsize=25)
fig.text(0.5, 0.04, 'Date', ha='center', fontsize=20)
fig.text(0.04, 0.5, 'Open Price', va='center', rotation='vertical', fontsize=20)

# Add the legend
fig.legend(loc='upper right', fontsize=15)

# Adjust the spacing between subplots
plt.subplots_adjust(wspace=0.3)

# Show the plot
plt.show()


#df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y').dt.strftime('%Y-%m-%d')

# Get the unique symbols from the DataFrame
symbols = df['Symbol'].unique()

# Create a new figure and axis
fig, ax = plt.subplots(figsize=(16, 8))

ax.plot(subset['Date'], subset['Close'])

    # Set the x-axis label format 
ax.xaxis.set_major_locator(plt.MaxNLocator(8))


# Plot the open prices for each symbol
for symbol in symbols:
    data = df[df['Symbol'] == symbol]
    ax.plot(data['Date'], data['Close'], label=symbol)

# Set the title and labels
start_date = df['Date'].iloc[0]
end_date = df['Date'].iloc[-1]
ax.set_title(f"Stock Close Prices from {start_date} to {end_date}", fontsize=25)
ax.set_xlabel('Date', fontsize=20)
ax.set_ylabel('Close Price', fontsize=20)

# Set the tick font sizes
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)

# Add the legend
ax.legend(loc='upper left', fontsize=15)

# Show the plot
plt.show()



#df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y')

#df['Date'] = df['Date'].dt.strftime('%d-%m-%Y')

#df.set_index('Date', inplace=True)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import matplotlib.pyplot as plt
# Get the list of symbols


#df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y')

#df['Date'] = df['Date'].dt.strftime('%d-%m-%Y')

df.set_index('Date', inplace=True)

symbols = df['Symbol'].unique()


# Train a Random Forest model for each symbol
for symbol in symbols:
    # Get the data for the current symbol
    symbol_df = df[df['Symbol'] == symbol]
    
    # Define the features and target variable
    X = symbol_df.drop(['Close', 'Symbol'], axis=1)
    y = symbol_df['Close']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a Random Forest model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Predict the stock prices for the testing set
    y_pred = rf.predict(X_test)
    
    # Evaluate the model performance using mean squared error and R2 score
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Symbol: {symbol}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2 Score: {r2:.2f}")
    
    # Plot the actual and predicted stock prices for the current symbol
    # Plot the actual and predicted stock prices for the current symbol
    plt.figure(figsize=(10,6))
    sorted_test = y_test.sort_index()
    sorted_pred = pd.DataFrame(data=y_pred, index=y_test.index).sort_index()
    plt.plot(y_test.index, y_test.values, label='Actual')
    plt.plot(y_test.index, y_pred, label='Predicted')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.xticks(rotation=90)
    plt.title(f"{symbol} Actual vs. Predicted Closing Prices")
    plt.legend()
    plt.show()
    print(f"Symbol: {symbol}")
    print("Predicted prices:", y_pred)
    print("Actual prices:", y_test.values)


# # Using Random Forest to Determine Actual and Predicted Closing Prices


# df = df.drop('Symbol', axis = 1)

X = df[['Open','High', 'Low', 'Close', 'Adj Close']].values


X = X.reshape(-1, 1)


scaler=MinMaxScaler() #initialize
scaler.fit(df)
scaled_df=scaler.transform(df)

##4 steps: import, initialize, train and interpret
km4=KMeans(n_clusters=4, random_state=0)##intialize
km4.fit(scaled_df)##train: finding clusters 

km4.inertia_ ##to calculate within cluster variation
silhouette_score(scaled_df, km4.labels_)

##how many clusters
# Determine the optimal number of clusters using the elbow method

wcv=[]
silk_score=[]
for i in range(2, 15):
    km=KMeans(n_clusters=i, random_state=0)##intialize
    km.fit(scaled_df)##train: finding clusters 

    wcv.append(km.inertia_) ##to calculate within cluster variation
    silk_score.append(silhouette_score(scaled_df, km.labels_))
    
##plotting the wcv
plt.plot(range(2,15), wcv)
plt.xlabel('No of clusters')
plt.ylabel('Within cluster variation')

#plotting silk score
plt.plot(range(2,15), silk_score)
plt.xlabel('No of clusters')
plt.ylabel('Silhoutte score')

for i in range(2, 15):
    km = KMeans(n_clusters=i, random_state=0)
    km.fit(scaled_df)
    score = silhouette_score(scaled_df, km.labels_)
    print(f"Silhouette score for {i} clusters: {score}")


# # Lets Go with 5 Clusters

km3=KMeans(n_clusters=5, random_state=0)##intialize
km3.fit(scaled_df)##train: finding clusters 

km3.labels_

# Add the cluster labels to the original DataFrame

df['labels'] = km3.labels_


cluster_means = df.groupby('labels').mean()
cluster_means


# # As we can see here from the clusters 1st Label has PFE and 2nd cluster has TSLA Combined and 3rd has Walamrt 4th label has Meta and TSLA Combined and 5th Label has TSLA and META stock Prices.

# Perform Hierarchical clustering
linkage_matrix = linkage(X, method='ward')
dendrogram(linkage_matrix)
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.title('Dendrogram for Hierarchical Clustering')
plt.show()


# # For Hierarchial CLustering based on the threshold at distance 3000 I chose 3 clusters as optimum value here

hc= AgglomerativeClustering(n_clusters = 3, linkage = 'ward')# initialise
    
hc.fit(scaled_df)
# Add the cluster labels to the dataset
    
df['labels'] = hc.labels_

# Print the summary of results    
Hcmeans = df.groupby('labels').mean()
Hcmeans


# Based on the Hierarchial Clustering we can see that we have 3 different Stock Prices 1st Label Determines stock prices for Walmart and Meta and 2nd Label Determines stock prices for PFE and 3rd Label Determines Stock Prices for Tesla.


# ## We utilized the k-means clustering model, along with the elbow plot and dendrogram, to identify market trends. This involved grouping stocks together based on their similar price movements. By doing this, investors can better identify opportunities to buy or sell stocks based on the current market conditions.
# 
# ## We discovered that META and WMT had very similar price movements and were therefore grouped together as a single cluster. Meanwhile, PFE had a constant price movement and was also grouped as a single cluster. Finally, TSLA was identified as a single cluster due to its unique price movements.

import matplotlib.pyplot as plt

df = pd.read_csv('Stock Prices updated.csv')

df = df[['Symbol','Date','Open','High','Low','Close','Adj Close','Volume']]

# Group the data by symbol
groups = df.groupby('Symbol')

# Calculate the moving averages for 10, 20, and 50 days for each symbol
for name, group in groups:
    group['MA10'] = group['Adj Close'].rolling(window=10).mean()
    group['MA20'] = group['Adj Close'].rolling(window=20).mean()
    group['MA50'] = group['Adj Close'].rolling(window=50).mean()

    # Plot the data for each symbol with its moving averages
    fig, ax = plt.subplots()
    ax.plot(group['Adj Close'], label='Adj Close')
    ax.plot(group['MA10'], label='MA10')
    ax.plot(group['MA20'], label='MA20')
    ax.plot(group['MA50'], label='MA50')
    ax.set_title(f"{name} Moving Averages")
    ax.legend()
    plt.show()


# ## Moving Averages for Adj Close of META ,PFE,TSLA and WMT are shown here 
# 
# ## Moving averages are a commonly used statistical method for analyzing time series data. A moving average is a calculation of the average value of a series of data points over a specified period of time, where the calculation "moves" forward in time as new data points become available.
# 
# ## For example, a 10-day moving average for a stock price would be calculated by taking the average of the stock's closing prices over the most recent 10 days. As each new day's closing price is added to the series, the oldest closing price is dropped and the average is recalculated using the remaining 9 days' closing prices.
# 
# ## Moving averages are often used to smooth out the fluctuations in a time series, making it easier to identify trends and patterns in the data. They can also be used as a signal for buying or selling assets, with traders often looking for crossovers between different moving averages as an indication of a change in market sentiment.

import pandas as pd
from scipy.stats import norm

# Group the data by symbol
groups = df.groupby('Symbol')

portfolio_value = 1000

import pandas as pd
import matplotlib.pyplot as plt



for name, group in groups:
    group['daily_return'] = group['Adj Close'].pct_change()
    avg_return = group['daily_return'].mean()
    print(f"Average daily return for {name}: {avg_return}")

# Calculate the daily return for each stock
df['daily_return'] = df.groupby('Symbol')['Adj Close'].pct_change()

# Calculate the average daily return for each stock
avg_daily_return = df.groupby('Symbol')['daily_return'].mean()

# Plot the average daily return for each stock
fig, ax = plt.subplots(figsize=(12,8))
ax.bar(avg_daily_return.index, avg_daily_return.values)
ax.set_xlabel('Symbol')
ax.set_ylabel('Average Daily Return')
ax.set_title('Average Daily Return for Each Symbol')
plt.show()
    


# ## As we can see from here the the average daily returns for different stocks and only walmart has a positive Average Daily return when compared to other Stocks
# 
# ## Average daily return is the average percentage change in the price of a stock or portfolio of stocks over a given period, usually one trading day. It is calculated by taking the sum of the daily returns over the period and dividing it by the number of trading days in that period.
# 
# ## The average daily return is an important measure used in the stock market to evaluate the performance of a stock or portfolio of stocks. It provides investors with an indication of how much their investment is expected to grow or shrink on a daily basis. By analyzing the historical average daily returns, investors can make informed decisions about which stocks or portfolios to invest in, based on their risk tolerance and investment goals.
# 
# ## Furthermore, average daily return is often used in the calculation of other financial metrics such as volatility, which measures the degree of variation in the price of a stock or portfolio over time. Volatility is a key consideration for investors, as it helps them to understand the potential risks and rewards of an investment.
# 
# ## 

import numpy as np
import pandas as pd
from scipy.stats import norm


# Set the portfolio value (in dollars)
portfolio_value = 10000

# Group the data by symbol
groups = df.groupby('Symbol')

# Calculate the daily returns for each symbol
for name, group in groups:
    group['daily_return'] = group['Adj Close'].pct_change()
    
    # Calculate the mean and standard deviation of the daily returns
    mean_daily_return = group['daily_return'].mean()
    std_daily_return = group['daily_return'].std()
    
    # Calculate the VaR at a 95% confidence level for a time period of 1 day
    z_score = norm.ppf(0.05)
    var = (mean_daily_return - (z_score * std_daily_return)) * portfolio_value
    
    print(f"Value at Risk (VaR) for {name}: ${var:.2f}")

    


# ## Value at Risk (VaR) is a statistical measure used to estimate the amount of potential loss that an investment portfolio may incur over a given time period at a certain level of confidence. VaR is usually expressed as a dollar amount or a percentage of the portfolio's total value.
# 
# ## Dividends are a part of a company's profits that are distributed to its shareholders. When calculating VaR for a portfolio that includes dividend-paying stocks, the dividends received by the portfolio can be taken into account as a source of income. This income can offset some of the potential losses, which in turn reduces the VaR of the portfolio.
# 
# ## we can see here the Var for META will be 10,000 - 651.5 = $9,348.5 
# 
# ## We are 95% confident that portfolio value wont be below $9,348.5.
# 
# 

import matplotlib.pyplot as plt

# Create a figure with subplots for each symbol
fig, axs = plt.subplots(nrows=len(groups), figsize=(10, 8))

# Plot a histogram of daily returns for each symbol
for i, (name, group) in enumerate(groups):
    axs[i].hist(group['daily_return'], bins=50)
    plt.xlabel('Daily Return')
    plt.ylabel('Counts')
    axs[i].set_title(name)

# Add a title and adjust the layout
fig.suptitle('Histogram of Average Daily Return by Symbol')
fig.tight_layout()


import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

# Set the portfolio value (in dollars)
portfolio_value = 10000

# Group the data by symbol
groups = df.groupby('Symbol')

# Create empty lists to store expected returns and risks
exp_returns = []
risks = []

# Calculate the daily returns for each symbol
for name, group in groups:
    group['daily_return'] = group['Adj Close'].pct_change()
    
    # Calculate the mean and standard deviation of the daily returns
    mean_daily_return = group['daily_return'].mean()
    std_daily_return = group['daily_return'].std()
    
    # Calculate the VaR at a 95% confidence level for a time period of 1 day
    z_score = norm.ppf(0.05)
    var = (mean_daily_return - (z_score * std_daily_return)) * portfolio_value
    
    # Append expected return and risk to the lists
    exp_returns.append(mean_daily_return)
    risks.append(std_daily_return)

# Convert lists to arrays
exp_returns = np.array(exp_returns)
risks = np.array(risks)

# Create scatter plot
area = np.pi * 20
plt.figure(figsize=(10, 8))
plt.scatter(exp_returns, risks, s=area)
plt.xlabel('Expected Return')
plt.ylabel('Risk')

# Add annotations for each symbol
for label, x, y in zip(groups.groups.keys(), exp_returns, risks):
    plt.annotate(label, xy=(x, y), xytext=(50, 50), textcoords='offset points', ha='right', va='bottom', 
                 arrowprops=dict(arrowstyle='-', color='blue', connectionstyle='arc3,rad=-0.3'))

plt.show()


# ## As we can see here Based on the Risk and Expected return calculated from the mean_daily_return and std_daily_return WMT stock has low risk and a positive Expected Return and TSLA has high risk and a negative Expected return and PFE with low risk and negative expected return and META with high risk and no expected return.

# ## Using Decision Tree to predict MSE for all the different stocks
# 


import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


df = pd.read_csv('Stock Prices updated.csv')
X = df[['Open', 'High', 'Low', 'Volume']]
y = df['Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a decision tree regressor with max_depth=3
tree_reg = DecisionTreeRegressor(max_depth=3)

# Fit the model to the training data
tree_reg.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = tree_reg.predict(X_test)

# Calculate the mean squared error of the predictions
mse = mean_squared_error(y_test, y_pred)

print('Mean Squared Error:', mse)

from sklearn.tree import plot_tree

import matplotlib.pyplot as plt

plt.figure(figsize = (10,10))

plot_tree(tree_reg)
