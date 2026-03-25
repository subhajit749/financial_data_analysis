import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import skew, kurtosis

# 1. Load and Prepare the Data
df = pd.read_csv('historical_market_data.csv')
df['Date'] = pd.to_datetime(df['Date'], utc=True)

# Create a Pivot Table (Dates as index, Tickers as columns)
pivot_df = df.pivot_table(index='Date', columns='Ticker', values='Close')
returns = pivot_df.pct_change().dropna()
tickers = ['AAPL', 'MSFT', 'JPM', 'GS', 'TSLA']

# --- Part 1: Correlation Heatmap ---
plt.figure(figsize=(10, 8))
corr_matrix = returns.corr()
sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', center=0, fmt=".2f")
plt.title('Correlation Heatmap: All 5 Stocks (Daily Returns)')
plt.savefig('correlation_heatmap_all.png')

# --- Part 2: Monte Carlo Simulations (Grid View) ---
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
axes = axes.flatten()

num_simulations = 50  # Number of random price paths
num_days = 252  # Trading days in a year
colors = ['blue', 'green', 'red', 'orange', 'purple']

for i, ticker in enumerate(tickers):
    # Calculate historical stats
    ticker_returns = returns[ticker]
    mu = ticker_returns.mean()
    sigma = ticker_returns.std()
    last_price = pivot_df[ticker].iloc[-1]

    ax = axes[i]
    sim_results = []

    # Run simulation loop
    for _ in range(num_simulations):
        # Sample daily returns from a normal distribution
        daily_returns = np.random.normal(mu, sigma, num_days)
        # Calculate price trajectory
        price_path = last_price * (1 + daily_returns).cumprod()
        ax.plot(price_path, color='grey', alpha=0.1)
        sim_results.append(price_path)

    # Plot the average path for each stock
    mean_path = np.mean(sim_results, axis=0)
    ax.plot(mean_path, color=colors[i], linewidth=2, label=f'Mean Path')

    ax.set_title(f'Monte Carlo Simulation: {ticker}')
    ax.set_xlabel('Days')
    ax.set_ylabel('Price ($)')
    ax.legend()

# Remove the extra subplot (since we have 5 stocks in a 3x2 grid)
fig.delaxes(axes[5])
plt.tight_layout()
plt.savefig('monte_carlo_all_stocks.png')


# validation
validation_results = []

for ticker in tickers:
    hist_returns = returns[ticker]

    # Historical stats
    h_mean = hist_returns.mean()
    h_std = hist_returns.std()
    h_skew = skew(hist_returns)
    h_kurt = kurtosis(hist_returns)  # excess kurtosis (Normal = 0)

    # Simulated stats (Monte Carlo assumes Normal Distribution)
    # We generate a large sample to represent the model's theoretical distribution
    sim_returns = np.random.normal(h_mean, h_std, 10000)
    s_mean = np.mean(sim_returns)
    s_std = np.std(sim_returns)
    s_skew = skew(sim_returns)
    s_kurt = kurtosis(sim_returns)

    validation_results.append({
        'Ticker': ticker,
        'Hist Skew': h_skew,
        'Sim Skew': s_skew,
        'Hist Kurtosis': h_kurt,
        'Sim Kurtosis': s_kurt
    })

# Convert to DataFrame for display
val_df = pd.DataFrame(validation_results)
print(val_df)

# Plotting distribution comparison for one example (TSLA) to show the difference
ticker_to_plot = 'TSLA'
plt.figure(figsize=(12, 6))
sns.kdeplot(returns[ticker_to_plot], label='Historical Returns', fill=True, color='blue', alpha=0.3)
sns.kdeplot(np.random.normal(returns[ticker_to_plot].mean(), returns[ticker_to_plot].std(), 10000),
            label='Monte Carlo (Normal) Distribution', color='red', linestyle='--')

plt.title(f'Distribution Validation: Historical vs. Monte Carlo (Normal) - {ticker_to_plot}')
plt.xlabel('Daily Return')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('distribution_validation.png')
plt.close()