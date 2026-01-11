import json
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

"""
MA Distance Histogram + Conditional Uptrend Probabilities

Purpose:
- Validate and learn real sweet-spots for MA20, MA50, MA150/200
- For each MA type, bin % distance above MA into ranges and compute:
  P(uptrend | distance_bin) where uptrend = f(next returns)

Outputs:
- Writes ma_hist_stats.json with structure:
{
  "MA20": [{"bin": "[-5,0)", "count": 123, "uptrend_prob": 0.18}, ...],
  "MA50": [...],
  "MA200": [...],
  "meta": {"lookahead_days": 60, "uptrend_threshold_percent": 15}
}

Notes:
- This script is designed to run offline and produce a JSON file the app can consume.
- Network calls to yfinance can be slow; using a modest ticker list initially is advised.
"""

@dataclass
class Config:
    lookahead_days: int = 60  # future window to test trend
    uptrend_threshold_percent: float = 15.0  # classify as uptrend if >= this return
    period_years: int = 5  # amount of history to download
    # Define bins for % distance above MA (in percentage points)
    bins: List[Tuple[float, float]] = (
        (-20.0, -10.0), (-10.0, -5.0), (-5.0, 0.0), (0.0, 2.0), (2.0, 4.0),
        (4.0, 6.0), (6.0, 8.0), (8.0, 12.0), (12.0, 20.0)
    )
    # Tickers source: start small; user can expand to full universe later
    tickers: List[str] = (
        [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
            "TSLA", "META", "NFLX", "AMD", "INTC",
            "GSIT", "ASTS"  # include examples user mentioned
        ]
    )


def download_prices(ticker: str, years: int) -> pd.DataFrame:
    period = f"{years}y"
    df = yf.download(ticker, period=period, auto_adjust=False, progress=False)
    if df.empty:
        return df
    # Normalize columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df.columns = [c.lower() for c in df.columns]
    return df


def compute_ma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()


def pct_distance_above(price: float, ma: float) -> float:
    if ma and ma > 0:
        return (price - ma) / ma * 100.0
    return np.nan


def classify_uptrend(close: pd.Series, idx: int, lookahead_days: int, threshold_pct: float) -> int:
    # If future window incomplete, return np.nan (skip)
    if idx + lookahead_days >= len(close):
        return np.nan
    start_price = float(close.iloc[idx])
    end_price = float(close.iloc[idx + lookahead_days])
    if start_price <= 0:
        return np.nan
    ret = (end_price - start_price) / start_price * 100.0
    return 1 if ret >= threshold_pct else 0


def tally_bins(distances: List[float], labels: List[int], bins: List[Tuple[float, float]]) -> List[Dict]:
    results = []
    distances = np.array(distances, dtype=float)
    labels = np.array(labels, dtype=float)
    for lo, hi in bins:
        mask = (distances >= lo) & (distances < hi)
        count = int(np.nansum(mask.astype(int)))
        if count == 0:
            up_prob = 0.0
        else:
            selected = labels[mask]
            # remove NaNs
            selected = selected[~np.isnan(selected)]
            if selected.size == 0:
                up_prob = 0.0
            else:
                up_prob = float(np.mean(selected))
        results.append({
            "bin": f"[{lo},{hi})",
            "count": count,
            "uptrend_prob": round(up_prob, 4)
        })
    return results


def analyze_universe(cfg: Config) -> Dict:
    stats = {"MA20": [], "MA50": [], "MA200": [], "meta": {
        "lookahead_days": cfg.lookahead_days,
        "uptrend_threshold_percent": cfg.uptrend_threshold_percent,
        "period_years": cfg.period_years,
        "bins": cfg.bins
    }}

    for tkr in cfg.tickers:
        try:
            print(f"Downloading {tkr}…")
            df = download_prices(tkr, cfg.period_years)
            if df.empty:
                print(f"No data for {tkr}")
                continue
            close = df["close"].astype(float)
            if close.isna().all():
                print(f"Invalid close series for {tkr}")
                continue
            ma20 = compute_ma(close, 20)
            ma50 = compute_ma(close, 50)
            ma200 = compute_ma(close, 200)

            # Collect per-index distances and future labels
            d20, d50, d200, y = [], [], [], []
            for i in range(len(close)):
                p = float(close.iloc[i])
                d20.append(pct_distance_above(p, float(ma20.iloc[i]) if not math.isnan(ma20.iloc[i]) else np.nan))
                d50.append(pct_distance_above(p, float(ma50.iloc[i]) if not math.isnan(ma50.iloc[i]) else np.nan))
                d200.append(pct_distance_above(p, float(ma200.iloc[i]) if not math.isnan(ma200.iloc[i]) else np.nan))
                y.append(classify_uptrend(close, i, cfg.lookahead_days, cfg.uptrend_threshold_percent))

            # Tally by bins for this ticker
            ma20_stats = tally_bins(d20, y, cfg.bins)
            ma50_stats = tally_bins(d50, y, cfg.bins)
            ma200_stats = tally_bins(d200, y, cfg.bins)

            # Merge (sum counts, weighted average probabilities)
            def merge(dst: List[Dict], src: List[Dict]):
                if not dst:
                    # initialize
                    for row in src:
                        dst.append({"bin": row["bin"], "count": row["count"], "uptrend_prob": row["uptrend_prob"]})
                else:
                    for i, row in enumerate(src):
                        a = dst[i]
                        c_old, c_new = a["count"], row["count"]
                        # weighted average of probabilities
                        if (c_old + c_new) > 0:
                            p = (a["uptrend_prob"] * c_old + row["uptrend_prob"] * c_new) / (c_old + c_new)
                        else:
                            p = 0.0
                        a["count"] = c_old + c_new
                        a["uptrend_prob"] = round(float(p), 4)

            merge(stats["MA20"], ma20_stats)
            merge(stats["MA50"], ma50_stats)
            merge(stats["MA200"], ma200_stats)
            time.sleep(0.3)  # polite rate limiting
        except Exception as e:
            print(f"Error {tkr}: {e}")

    return stats


def main():
    cfg = Config()
    stats = analyze_universe(cfg)
    out_file = "ma_hist_stats.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved {out_file}")


if __name__ == "__main__":
    main()
