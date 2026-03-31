import yfinance as yf
import pandas as pd
import time
import random


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
                # 1. Initialize Ticker object
                stock = yf.Ticker(ticker)

                # 2. Fetch data with auto_adjust=True
                # This handles stock splits and dividend payouts automatically
                # by adjusting historical prices so the returns are accurate.
                hist = stock.history(period=period, interval="1d", auto_adjust=True)

                if hist.empty:
                    print(f"Warning: No data found for {ticker}")
                    break

                # Metadata for tracking
                hist['Ticker'] = ticker
                all_data.append(hist)
                success = True
                print(f"Successfully fetched {len(hist)} rows for {ticker}.")

            except Exception as e:
                retries -= 1
                wait_time = (4 - retries) * 5 + random.uniform(1, 3)  # Exponential backoff
                print(f"Error fetching {ticker}: {e}. Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)

        # 3. Mandatory sleep to avoid '429 Too Many Requests' from Yahoo Finance
        time.sleep(random.uniform(1, 2))

    if not all_data:
        return pd.DataFrame()

    return pd.concat(all_data)


# --- EXECUTION ---
portfolio_list = ["AAPL", "MSFT", "JPM", "GS", "JNJ", "PFE", "XOM", "AMZN", "KO", "BA", "^GSPC"]

# Fetching last 5 years of data
df_market = fetch_robust_market_data(portfolio_list, period="5y")

# Final Cleanup: Reset index to make 'Date' a column
df_market.reset_index(inplace=True)
df_market.to_csv("diverse_portfolio_market_data.csv", index=False)
print("\nScraping complete. Final dataset saved.")