import yfinance as yf
import pandas as pd
import time
from requests.exceptions import HTTPError


class FinancialDataScraper:
    def __init__(self, tickers):
        self.tickers = tickers
        self.data_store = {}

    def fetch_data(self, period="max", interval="1d"):
        """
        Fetches historical data with resilient error handling for rate limits.
        """
        for ticker in self.tickers:
            print(f"--- Fetching Data for: {ticker} ---")
            success = False
            retries = 3

            while not success and retries > 0:
                try:
                    # yf.Ticker provides access to splits and dividends
                    stock = yf.Ticker(ticker)

                    # auto_adjust=True: Handles stock splits and dividends automatically
                    # back_adjust=False: Ensures we get the standard Adjusted Close
                    hist = stock.history(period=period, interval=interval, auto_adjust=True)

                    if hist.empty:
                        print(f"Warning: No data found for {ticker}")
                        break

                    self.data_store[ticker] = hist
                    success = True
                    print(f"Successfully retrieved {len(hist)} rows for {ticker}.")

                except Exception as e:
                    retries -= 1
                    print(f"Error fetching {ticker}: {e}. Retrying in 5 seconds...")
                    time.sleep(5)  # Exponential backoff or simple sleep to respect rate limits

            # Small cooldown between different ticker requests to stay under the radar
            time.sleep(1)

    def get_combined_dataset(self):
        """
        Combines all tickers into the 'Long-Form' format we discussed.
        """
        combined_list = []
        for ticker, df in self.data_store.items():
            temp_df = df.copy()
            temp_df['Ticker'] = ticker
            temp_df.reset_index(inplace=True)
            combined_list.append(temp_df)

        return pd.concat(combined_list, ignore_index=True)


# --- Execution ---
portfolio_tickers = ["AAPL", "MSFT", "JPM", "GS", "TSLA"]
scraper = FinancialDataScraper(portfolio_tickers)

# Fetching the data
scraper.fetch_data(period="10y")  # Fetching last 10 years

# Export to CSV
final_df = scraper.get_combined_dataset()
final_df.to_csv("historical_market_data.csv", index=False)
print("\nScraping complete. File saved as historical_market_data.csv")