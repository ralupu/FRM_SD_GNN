import os
import pandas as pd
import numpy as np
import glob

def get_asset_histories(folder="data/cryptos/"):
    all_files = glob.glob(os.path.join(folder, "*.csv"))
    info = []
    for f in all_files:
        asset = os.path.basename(f).replace(".csv", "")
        df = pd.read_csv(f)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors="coerce", utc=True)
            df = df.set_index('date')
        else:
            continue
        df = df[~df.index.duplicated(keep='first')].sort_index()
        if len(df) == 0:
            continue
        first = df.index.min()
        last = df.index.max()
        n_days = (last - first).days + 1
        info.append((asset, first, last, n_days))
    return pd.DataFrame(info, columns=["Asset", "Start", "End", "Days"]).sort_values("Days", ascending=False)

def get_returns_and_volumes(top_n=15, folder="data/cryptos/", start_date='2019/01/01', freq="M"):
    # Ensure start_date is always tz-aware (UTC)
    start_date = pd.to_datetime(start_date)
    if start_date.tzinfo is None:
        start_date = start_date.tz_localize("UTC")

    asset_hist = get_asset_histories(folder)
    asset_hist["Months"] = ((asset_hist["End"] - asset_hist["Start"]).dt.days / 30.44).round(1)

    # Only keep assets with history covering start_date
    asset_hist_ok = asset_hist[asset_hist["Start"] <= start_date].copy()
    print("\nAssets with history covering start date {}:".format(start_date.date()))
    print(asset_hist_ok[["Asset", "Start", "End", "Days", "Months"]].to_string(index=False))

    # Select top_n assets
    top_assets = asset_hist_ok.head(top_n)["Asset"].tolist()
    print(f"\nSelected top {top_n} assets with history starting <= {start_date.date()}:\n{top_assets}\n")

    price_dfs = []
    vol_dfs = []

    for asset in top_assets:
        f = os.path.join(folder, asset + ".csv")
        df = pd.read_csv(f)
        if not {'date', 'prices', 'total_volumes'}.issubset(df.columns):
            print(f"Skipping {asset}: columns missing.")
            continue
        df['date'] = pd.to_datetime(df['date'], errors="coerce", utc=True)
        df = df.set_index('date')
        df = df[~df.index.duplicated(keep='first')].sort_index()
        df = df[df.index >= start_date]
        price_df = df[['prices']].rename(columns={'prices': asset})
        price_dfs.append(price_df)
        vol_df = df[['total_volumes']].rename(columns={'total_volumes': asset})
        vol_dfs.append(vol_df)

    prices = pd.concat(price_dfs, axis=1)
    prices = prices.ffill().bfill()
    volumes = pd.concat(vol_dfs, axis=1)
    volumes = volumes.fillna(0)

    # Aggregation
    if freq == "M":
        price_agg = prices.resample("M").last()
        vol_agg = volumes.resample("M").sum()
        fname_r, fname_v = "monthly_log_returns.csv", "monthly_volumes.csv"
    elif freq == "W":
        price_agg = prices.resample("W-FRI").last()
        vol_agg = volumes.resample("W-FRI").sum()
        fname_r, fname_v = "weekly_log_returns.csv", "weekly_volumes.csv"
    else:
        raise ValueError("freq must be 'M' (monthly) or 'W' (weekly)")

    log_returns = np.log(price_agg / price_agg.shift(1))
    log_returns = log_returns.loc[log_returns.index >= start_date].iloc[1:]
    vol_agg = vol_agg.loc[log_returns.index]

    # Ensure data folder exists
    out_folder = "data"
    os.makedirs(out_folder, exist_ok=True)

    # Save to CSV
    log_returns.to_csv(os.path.join(out_folder, fname_r))
    vol_agg.to_csv(os.path.join(out_folder, fname_v))

    print(f"\nSaved log-returns for top {top_n} assets to '{out_folder}/{fname_r}'.")
    print(f"Saved volumes for top {top_n} assets to '{out_folder}/{fname_v}'.")
    print(f"Shape of log-returns: {log_returns.shape}")
    print(f"Shape of volumes: {vol_agg.shape}\n")
    print("\nFirst few rows of returns:\n", log_returns.head())
    print("\nFirst few rows of volumes:\n", vol_agg.head())
    return log_returns, vol_agg, asset_hist_ok

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_n', type=int, default=15, help="Number of assets with longest history")
    parser.add_argument('--start_date', type=str, default='2019/01/01', help="Earliest date to include (YYYY/MM/DD or YYYY-MM-DD)")
    parser.add_argument('--freq', type=str, default='M', help="Aggregation frequency: 'M' for monthly, 'W' for weekly")
    args = parser.parse_args()
    get_returns_and_volumes(top_n=args.top_n, start_date=args.start_date, freq=args.freq.upper())


# Use it like this:
# python analysis/crypto_prep.py --top_n 15 --start_date 2019-01-01 --freq M