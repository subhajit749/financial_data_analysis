import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('diverse_portfolio_market_data.csv')
df['Date'] = pd.to_datetime(df['Date'], utc=True)

# Parameters
num_simulations = 10000
num_days = 252
initial_value = 100

tickers = df['Ticker'].unique()
cohort_data = []

# Process each stock
for ticker in tickers:
    ticker_df = df[df['Ticker'] == ticker].sort_values('Date')

    # NumPy Log Returns
    prices = ticker_df['Close'].values
    log_returns = np.log(prices[1:] / prices[:-1])
    dates = ticker_df['Date'].iloc[1:]

    # 1. Individual Monte Carlo Plot
    mu = np.mean(log_returns)
    sigma = np.std(log_returns)
    shocks = np.random.normal(loc=mu, scale=sigma, size=(num_days, num_simulations))
    price_paths = initial_value * np.exp(np.cumsum(shocks, axis=0))

    plt.figure(figsize=(10, 6))
    plt.plot(price_paths[:, :100], alpha=0.1, color='blue')
    plt.title(f'Monte Carlo Prediction: {ticker} (10,000 Runs)')
    plt.axhline(initial_value, color='red', linestyle='--')
    plt.ylabel('Projected Value ($)')
    plt.xlabel('Days')
    plt.savefig(f'monte_carlo_{ticker.replace("^", "")}.png')
    plt.close()

    # 2. Preparation for Cohort Analysis
    # Create temporary DF for returns with Year
    temp_returns_df = pd.DataFrame({'Return': log_returns, 'Year': dates.dt.year})
    # Annualized return per year (Mean * 252)
    yearly_perf = temp_returns_df.groupby('Year')['Return'].mean() * 252

    for year, perf in yearly_perf.items():
        cohort_data.append({'Ticker': ticker, 'Year': year, 'Annualized Return': perf})

# 3. Build and Plot Cohort Analysis Heatmap
cohort_df = pd.DataFrame(cohort_data)
cohort_pivot = cohort_df.pivot(index='Year', columns='Ticker', values='Annualized Return')

plt.figure(figsize=(12, 8))
sns.heatmap(cohort_pivot, annot=True, fmt=".2%", cmap='RdYlGn', center=0)
plt.title('Stock Cohort Analysis: Annualized Returns per Year')
plt.ylabel('Performance Year (Cohort)')
plt.savefig('stock_cohort_analysis_heatmap.png')
plt.close()

print("Individual stock simulations and cohort analysis heatmap generated.")