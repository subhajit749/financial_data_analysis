#!/usr/bin/env python3
"""
AlphaPulse — Week 4: Automated Market Data Refresh
====================================================
Run this script to pull the latest stock data and update alphapulse_tableau.csv
Supports: AAPL, MSFT, JPM, GS, TSLA
Schedule: Run daily via Task Scheduler (Windows) or cron (Linux/Mac)

Usage:
    python refresh_alphapulse.py
    python refresh_alphapulse.py --tickers AAPL MSFT --start 2016-01-01
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
import sys
import argparse
import logging
from datetime import datetime, timedelta

# ── CONFIG ─────────────────────────────────────────────────────────────────
TICKERS      = ["AAPL", "MSFT", "JPM", "GS", "TSLA"]
START_DATE   = "2016-03-28"
OUTPUT_FILE  = "alphapulse_tableau.csv"
LOG_FILE     = "alphapulse_refresh.log"
SECTOR_MAP   = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "JPM":  "Financials",
    "GS":   "Financials",
    "TSLA": "Consumer Discretionary"
}
# ────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)


def fetch_data(tickers: list, start: str) -> pd.DataFrame:
    """Download OHLCV + dividends + splits from Yahoo Finance."""
    all_frames = []
    for ticker in tickers:
        log.info(f"  Fetching {ticker}...")
        try:
            t   = yf.Ticker(ticker)
            df  = t.history(start=start, auto_adjust=False)
            df  = df.reset_index()
            df.columns = [c.replace(" ", "_") for c in df.columns]
            df["Ticker"] = ticker
            # Rename to match our schema
            df = df.rename(columns={
                "Stock_Splits": "Stock Splits",
                "Capital_Gains": "Capital Gains"
            })
            all_frames.append(df)
        except Exception as e:
            log.error(f"  Failed to fetch {ticker}: {e}")

    if not all_frames:
        raise RuntimeError("No data fetched — check internet connection.")

    combined = pd.concat(all_frames, ignore_index=True)
    return combined


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add Return, Adjusted Return, Year, Sector columns."""
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # Strip timezone from Date
    if hasattr(df["Date"].dt, "tz") and df["Date"].dt.tz is not None:
        df["Date"] = df["Date"].dt.tz_convert(None)
    df["Date"] = pd.to_datetime(df["Date"])

    # Daily Return
    df["Return"] = df.groupby("Ticker")["Close"].pct_change()

    # Dividend-adjusted Return
    df["prev_Close"] = df.groupby("Ticker")["Close"].shift(1)
    df["Adjusted Return"] = (
        (df["Close"] + df["Dividends"] - df["prev_Close"]) / df["prev_Close"]
    )

    df["Year"]   = df["Date"].dt.year
    df["Sector"] = df["Ticker"].map(SECTOR_MAP)
    df["Date"]   = df["Date"].dt.strftime("%Y-%m-%d")

    # Round for clean Tableau display
    for col in ["Open", "High", "Low", "Close", "Return", "Adjusted Return"]:
        df[col] = df[col].round(6)

    return df[[
        "Date", "Year", "Ticker", "Sector",
        "Open", "High", "Low", "Close",
        "Volume", "Dividends", "Stock Splits",
        "Return", "Adjusted Return"
    ]]


def compute_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Compute VaR, CVaR, Max Drawdown, Sharpe per ticker."""
    rows = []
    for ticker, grp in df.groupby("Ticker"):
        grp = grp.dropna(subset=["Return"])
        prices = grp["Close"].values
        years  = grp["Year"].unique()
        # Annual returns
        df_grp  = grp.copy()
        df_grp["Date"] = pd.to_datetime(df_grp["Date"])
        annual = df_grp.groupby("Year")["Return"].apply(
            lambda x: (1 + x).prod() - 1
        ).values

        mu     = np.mean(np.log(1 + annual))
        sigma  = np.std(np.log(1 + annual))
        current = prices[-1]

        np.random.seed(42)
        sim1yr  = current * np.exp(np.random.normal(mu, sigma, 10000))
        var95   = current - np.percentile(sim1yr, 5)
        var99   = current - np.percentile(sim1yr, 1)
        cvar95  = current - sim1yr[sim1yr <= np.percentile(sim1yr, 5)].mean()

        peak   = prices[0]; max_dd = 0
        for p in prices:
            peak   = max(peak, p)
            dd     = (p - peak) / peak * 100
            if dd < max_dd: max_dd = dd

        avg_ann = np.mean(annual) * 100
        std_ann = np.std(annual) * 100
        sharpe  = (avg_ann - 2.5) / std_ann if std_ann > 0 else 0

        rows.append({
            "Ticker":               ticker,
            "As_Of_Date":           datetime.today().strftime("%Y-%m-%d"),
            "Current_Price":        round(current, 2),
            "VaR_95_USD":           round(var95, 2),
            "VaR_99_USD":           round(var99, 2),
            "CVaR_95_USD":          round(cvar95, 2),
            "VaR_95_Pct":           round(var95 / current * 100, 2),
            "Max_Drawdown_Pct":     round(max_dd, 2),
            "Volatility_Annual_Pct":round(sigma * 100, 2),
            "Sharpe_Ratio":         round(sharpe, 3),
            "Avg_Annual_Return_Pct":round(avg_ann, 2),
        })

    return pd.DataFrame(rows)


def verify_accuracy(kpi_df: pd.DataFrame):
    """
    Financial Accuracy Check (Week 4 requirement).
    Benchmarks VaR against industry rule-of-thumb:
    VaR_95 % should be roughly 1.65 × Daily_Volatility × sqrt(252)
    """
    log.info("=" * 60)
    log.info("FINANCIAL ACCURACY VERIFICATION")
    log.info("=" * 60)
    passed = 0; failed = 0
    for _, row in kpi_df.iterrows():
        ticker  = row["Ticker"]
        var_pct = row["VaR_95_Pct"]
        vol_pct = row["Volatility_Annual_Pct"]
        # 95% normal VaR ≈ 1.645 × σ (annual already, so just compare)
        theoretical_var = 1.645 * vol_pct
        ratio = var_pct / theoretical_var if theoretical_var > 0 else 0
        status = "PASS" if 0.7 <= ratio <= 1.5 else "REVIEW"
        if "PASS" in status: passed += 1
        else: failed += 1
        log.info(
            f"  {ticker:6s} | VaR95%={var_pct:.2f}%  "
            f"Theoretical={theoretical_var:.2f}%  "
            f"Ratio={ratio:.2f}  {status}"
        )
    log.info(f"  Summary: {passed} PASS | {failed} REVIEW")
    log.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="AlphaPulse Data Refresh")
    parser.add_argument("--tickers", nargs="+", default=TICKERS)
    parser.add_argument("--start",   default=START_DATE)
    parser.add_argument("--output",  default=OUTPUT_FILE)
    args = parser.parse_args()

    log.info("=" * 60)
    log.info(f"AlphaPulse Refresh  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)
    log.info(f"Tickers : {args.tickers}")
    log.info(f"Start   : {args.start}")
    log.info(f"Output  : {args.output}")

    # 1. Fetch
    log.info("Step 1/4  Fetching market data...")
    raw = fetch_data(args.tickers, args.start)
    log.info(f"  Rows fetched: {len(raw):,}")

    # 2. Compute metrics
    log.info("Step 2/4  Computing Return & Adjusted Return...")
    clean = compute_metrics(raw)
    log.info(f"  Rows after processing: {len(clean):,}")

    # 3. Save CSV
    log.info(f"Step 3/4  Saving to {args.output}...")
    clean.to_csv(args.output, index=False)
    log.info(f"  Saved. File size: {os.path.getsize(args.output):,} bytes")

    # 4. KPIs + Accuracy Check
    log.info("Step 4/4  Computing KPIs & running accuracy check...")
    kpi = compute_kpis(clean)
    kpi.to_csv("alphapulse_kpi.csv", index=False)
    log.info(f"  KPI file saved: alphapulse_kpi.csv")
    verify_accuracy(kpi)

    log.info("Refresh complete. Reopen Tableau and click Data > Refresh.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
