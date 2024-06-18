import numpy as np
import pandas as pd
import yfinance as yf

RISK_FREE = 0.05    # Yield on the US 10Y bond

TICKERS = ['AAPL', 'SPY']
PERIOD = '5y'
INTERVAL = '1d'

# Download only the Adj Close prices
data = yf.download(TICKERS, period=PERIOD, interval=INTERVAL)['Adj Close']

# Calculate daily returns
for ticker in TICKERS:
    data[f'{ticker}_Returns'] = data[ticker].pct_change()

    # Clean data (first NaN is a 0% return)
    data[f'{ticker}_Returns'].fillna(0, inplace=True)
    data.drop(ticker, inplace=True, axis=1)

# Calculate the market risk premium
RISK_PREMIUM = (data['SPY_Returns'].mean() * 252) - RISK_FREE

# Calculate the annualized covariance matrix
covMatrix = data.cov() * 252
covariance = covMatrix.iloc[0,1]

assetVariance = covMatrix.iloc[0, 0]
assetStddev = np.sqrt(assetVariance)

marketVariance = covMatrix.iloc[1,1]

# Calculate Beta
assetBeta = covariance / marketVariance

# Calculate Expected Return
Er = RISK_FREE + assetBeta * RISK_PREMIUM

# Calculate the Sharpe Ratio
sharpe = (Er - RISK_FREE) / assetStddev

print(f'Expected Return: {Er}')
print(f'Beta: {assetBeta}')
print(f'Sharpe: {sharpe}')