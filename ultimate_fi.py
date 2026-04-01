import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random
import os
from scipy.stats import skew, kurtosis, norm


# --- PART 1: DATA ACQUISITION ---

def fetch_robust_market_data(tickers, period="5y"):
    """
    Scraper with resilient error handling and financial adjustment logic.
    """
    all_data = []

    for ticker in tickers:
        print(f"--- Processing: {ticker} ---")
        retries = 3
        success = False

        while retries > 0 and not success:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period, interval="1d", auto_adjust=True)

                if hist.empty:
                    print(f"Warning: No data found for {ticker}")
                    break

                hist['Ticker'] = ticker
                all_data.append(hist)
                success = True
                print(f"Successfully fetched {len(hist)} rows for {ticker}.")

            except Exception as e:
                retries -= 1
                wait_time = (4 - retries) * 5 + random.uniform(1, 3)
                print(f"Error fetching {ticker}: {e}. Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)

        time.sleep(random.uniform(1, 2))

    if not all_data:
        return pd.DataFrame()

    return pd.concat(all_data)


# --- PART 2: EXECUTION & ANALYSIS ---

def main():
    # 1. Scraping Phase
    portfolio_list = ["AAPL", "MSFT", "JPM", "GS", "JNJ", "PFE", "XOM", "AMZN", "KO", "BA", "^GSPC"]
    filename = "diverse_portfolio_market_data.csv"

    print("Starting data acquisition...")
    if not os.path.exists(filename):
        df_market = fetch_robust_market_data(portfolio_list, period="5y")
        if df_market.empty:
            print("No data collected. Exiting.")
            return
        df_market.reset_index(inplace=True)
        df_market.to_csv(filename, index=False)
        print(f"\nScraping complete. Dataset saved to {filename}.")
    else:
        print(f"Using existing file: {filename}")

    # 2. Analysis Setup
    df = pd.read_csv(filename)
    df['Date'] = pd.to_datetime(df['Date'], utc=True)

    num_simulations = 10000
    num_days = 252
    initial_value = 100
    tickers = df['Ticker'].unique()

    cohort_data = []
    validation_reports = []

    for ticker in tickers:
        print(f"\nAnalyzing {ticker}...")
        ticker_df = df[df['Ticker'] == ticker].sort_values('Date')

        # Log Returns calculation
        prices = ticker_df['Close'].values
        if len(prices) < 2:
            continue

        log_returns = np.log(prices[1:] / prices[:-1])
        dates = ticker_df['Date'].iloc[1:]

        # Stats for Simulation
        mu = np.mean(log_returns)
        sigma = np.std(log_returns)

        # --- A. MONTE CARLO SIMULATION (Price Paths) ---
        shocks = np.random.normal(loc=mu, scale=sigma, size=(num_days, num_simulations))
        price_paths = initial_value * np.exp(np.cumsum(shocks, axis=0))

        plt.figure(figsize=(10, 6))
        plt.plot(price_paths[:, :100], alpha=0.1, color='blue')
        plt.title(f'Monte Carlo Prediction: {ticker} (10,000 Runs)')
        plt.axhline(initial_value, color='red', linestyle='--')
        plt.ylabel('Projected Value ($)')
        plt.xlabel('Days')

        clean_name = ticker.replace("^", "")
        plt.savefig(f'monte_carlo_paths_{clean_name}.png')
        plt.close()

        # --- B. STATISTICAL VALIDATION (Bell Curve Comparison) ---
        hist_skew = skew(log_returns)
        hist_kurt = kurtosis(log_returns)

        # Flattens all simulated daily returns to check the distribution shape
        sim_returns = shocks.flatten()
        sim_skew = skew(sim_returns)
        sim_kurt = kurtosis(sim_returns)

        validation_reports.append({
            'Ticker': ticker,
            'Hist Skew': hist_skew,
            'Sim Skew': sim_skew,
            'Hist Kurtosis': hist_kurt,
            'Sim Kurtosis': sim_kurt
        })

        plt.figure(figsize=(10, 6))
        sns.kdeplot(log_returns, label='Historical Returns', color='crimson', fill=True)
        sns.kdeplot(sim_returns, label='Simulated (Normal)', color='royalblue', linestyle='--')

        # Theoretical Bell Curve Overlay
        x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 100)
        plt.plot(x, norm.pdf(x, mu, sigma), color='black', alpha=0.3, label='Theoretical Normal')

        plt.title(f'Distribution Validation: {ticker}\nHistorical Kurtosis: {hist_kurt:.2f} (Excess)')
        plt.xlabel('Daily Log Returns')
        plt.ylabel('Density')
        plt.legend()
        plt.savefig(f'distribution_val_{clean_name}.png')
        plt.close()

        # --- C. PREPARATION FOR COHORT ANALYSIS ---
        temp_returns_df = pd.DataFrame({'Return': log_returns, 'Year': dates.dt.year})
        yearly_perf = temp_returns_df.groupby('Year')['Return'].mean() * 252

        for year, perf in yearly_perf.items():
            cohort_data.append({'Ticker': ticker, 'Year': year, 'Annualized Return': perf})

    # 3. Final Outputs

    # Validation Table
    val_df = pd.DataFrame(validation_reports)
    print("\n--- STATISTICAL VALIDATION REPORT ---")
    print(val_df.to_string(index=False))
    val_df.to_csv("model_validation_report.csv", index=False)

    # Heatmap Generation
    if cohort_data:
        cohort_df = pd.DataFrame(cohort_data)
        cohort_pivot = cohort_df.pivot(index='Year', columns='Ticker', values='Annualized Return')

        plt.figure(figsize=(12, 8))
        sns.heatmap(cohort_pivot, annot=True, fmt=".2%", cmap='RdYlGn', center=0)
        plt.title('Stock Cohort Analysis: Annualized Returns per Year')
        plt.savefig('stock_cohort_analysis_heatmap.png')
        plt.close()
        print("\nAll simulations, validations, and heatmaps generated successfully.")


if __name__ == "__main__":
    main()