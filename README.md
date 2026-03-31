# Financial_data_analysis
This repository contains a robust Python-based tool designed to scrape historical market data, perform Monte Carlo simulations for price forecasting, and generate cohort analysis heatmaps to visualize annualized returns across a diverse portfolio.
# Features
Resilient Data Scraping: Uses yfinance with built-in exponential backoff and error handling to bypass rate limits (429 errors).

Automatic Financial Adjustments: Historical prices are automatically adjusted for stock splits and dividends to ensure return accuracy.

Monte Carlo Simulations: Runs 10,000 simulations per ticker to project potential price paths over a trading year (252 days).

Cohort Analysis: Generates a heatmap of annualized returns grouped by year and ticker, allowing for quick performance comparisons.

Automated Visualization: Exports high-resolution PNG plots for every ticker and a final summary heatmap.
# Technical Workflow
The script operates in three distinct phases:
Extraction: Fetches 5 years of daily data for a predefined list of tickers (e.g., AAPL, MSFT, ^GSPC).
Visualization: Uses Matplotlib and Seaborn to render trend lines and performance matrices.
# Output Files:
diverse_portfolio_market_data.csv: The raw historical data.
monte_carlo_[TICKER].png: Individual projection charts for each asset.
stock_cohort_analysis_heatmap.png: A comprehensive performance heatmap.
## Configuration
You can customize the analysis by modifying the following variables in the main() function:
portfolio_list: Add or remove stock tickers or indices.
num_simulations: Adjust the number of Monte Carlo paths (default: 10,000).
period: Change the look-back window (default: "5y")
