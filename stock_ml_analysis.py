"""
Supervised Machine Learning for Stock Market Analysis
======================================================

Pipeline:
1. Pull historical price data (yfinance)
2. Engineer technical features (no lookahead leakage)
3. Label: will next-day return be positive? (binary classification)
4. Walk-forward (time-respecting) train/test split
5. Train baseline (Logistic Regression) + stronger model (Gradient Boosting)
6. Evaluate with classification metrics AND a simple strategy backtest
   vs. buy-and-hold, including transaction costs

Requirements:
    pip install yfinance scikit-learn pandas numpy matplotlib

Run:
    python stock_ml_analysis.py --ticker AAPL --start 2015-01-01

    # Individual stock + specific month (trains on all prior history,
    # evaluates only on that calendar month):
    python stock_ml_analysis.py --ticker TSLA --month 2024-03
"""

import argparse
import calendar
import warnings
from datetime import date
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report


def month_to_range(month_str: str):
    """Convert 'YYYY-MM' into (start, end) ISO date strings covering that
    single calendar month. Note: one month of daily bars is a very small
    sample for training a model — see the --month help text below."""
    year, month = (int(x) for x in month_str.split("-"))
    start = date(year, month, 1)
    last_day = calendar.monthrange(year, month)[1]
    end = date(year, month, last_day)
    return start.isoformat(), end.isoformat()


# ----------------------------------------------------------------------
# 1. DATA
# ----------------------------------------------------------------------
def load_data(ticker: str, start: str, end: str = None) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    df = df.rename(columns=str.lower)
    df = df.dropna()
    return df


# ----------------------------------------------------------------------
# 2. FEATURE ENGINEERING (all features use only PAST data -> no leakage)
# ----------------------------------------------------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Returns
    df["ret_1d"] = df["close"].pct_change(1)
    df["ret_5d"] = df["close"].pct_change(5)
    df["ret_10d"] = df["close"].pct_change(10)

    # Moving averages & relative position
    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    df["sma_ratio"] = df["sma_10"] / df["sma_50"]

    # Volatility
    df["vol_10d"] = df["ret_1d"].rolling(10).std()
    df["vol_30d"] = df["ret_1d"].rolling(30).std()

    # RSI (14-day)
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # Volume trend
    df["vol_change"] = df["volume"].pct_change(5)

    # Label: next-day direction (shifted so we predict the FUTURE from TODAY's features)
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

    df = df.dropna()
    return df


FEATURE_COLS = [
    "ret_1d", "ret_5d", "ret_10d", "sma_ratio",
    "vol_10d", "vol_30d", "rsi_14", "macd", "macd_signal", "vol_change",
]


# ----------------------------------------------------------------------
# 3. WALK-FORWARD SPLIT (never shuffle time series data)
# ----------------------------------------------------------------------
def time_split(df: pd.DataFrame, train_frac: float = 0.7):
    split_idx = int(len(df) * train_frac)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    return train, test


# ----------------------------------------------------------------------
# 4. TRAIN MODELS
# ----------------------------------------------------------------------
def train_models(train: pd.DataFrame):
    X_train = train[FEATURE_COLS]
    y_train = train["target"]

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    logreg = LogisticRegression(max_iter=1000).fit(X_train_scaled, y_train)
    gbc = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
    ).fit(X_train, y_train)  # tree models don't need scaling

    return scaler, logreg, gbc


# ----------------------------------------------------------------------
# 5. EVALUATE (classification metrics + strategy backtest)
# ----------------------------------------------------------------------
def evaluate(name, y_true, y_pred, test_df):
    print(f"\n--- {name} ---")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.3f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.3f}")

    # Simple backtest: go long when model predicts "up", stay flat otherwise
    # Apply a transaction cost each time the position changes
    cost_per_trade = 0.001  # 10 bps
    position = pd.Series(y_pred, index=test_df.index)
    strategy_ret = position.shift(1).fillna(0) * test_df["ret_1d"]
    trades = position.diff().abs().fillna(0)
    strategy_ret_after_cost = strategy_ret - trades * cost_per_trade

    cum_strategy = (1 + strategy_ret_after_cost).cumprod().iloc[-1] - 1
    cum_buyhold = (1 + test_df["ret_1d"]).cumprod().iloc[-1] - 1

    sharpe = (
        strategy_ret_after_cost.mean() / (strategy_ret_after_cost.std() + 1e-9)
        * np.sqrt(252)
    )

    print(f"Strategy total return (test period): {cum_strategy:.2%}")
    print(f"Buy & hold total return:              {cum_buyhold:.2%}")
    print(f"Strategy annualized Sharpe:            {sharpe:.2f}")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="AAPL",
                         help="Ticker symbol, e.g. AAPL, TSLA, MSFT")
    parser.add_argument("--start", type=str, default="2015-01-01",
                         help="History start date (ignored if --month is set, "
                              "except as the lower bound of training history)")
    parser.add_argument("--end", type=str, default=None,
                         help="History end date (ignored if --month is set)")
    parser.add_argument("--month", type=str, default=None,
                         help="Focus evaluation on a single calendar month, "
                              "format YYYY-MM (e.g. 2024-03). The model still "
                              "trains on all history before that month (walk-"
                              "forward); only the *evaluation* window is "
                              "restricted to the chosen month.")
    args = parser.parse_args()

    print(f"Loading data for {args.ticker}...")

    if args.month:
        focus_start, focus_end = month_to_range(args.month)
        df = load_data(args.ticker, args.start, focus_end)
        df = add_features(df)
        train = df[df.index < focus_start]
        test = df[(df.index >= focus_start) & (df.index <= focus_end)]
        print(f"Focus month: {args.month}  ({focus_start} to {focus_end})")
        if len(test) < 5:
            print("Warning: very few trading days matched this month — "
                  "check the month falls within the ticker's trading history.")
    else:
        df = load_data(args.ticker, args.start, args.end)
        df = add_features(df)
        train, test = time_split(df)

    print(f"Train rows: {len(train)}  |  Test rows: {len(test)}")

    scaler, logreg, gbc = train_models(train)

    X_test = test[FEATURE_COLS]
    y_test = test["target"]

    y_pred_log = logreg.predict(scaler.transform(X_test))
    y_pred_gbc = gbc.predict(X_test)

    evaluate("Logistic Regression (baseline)", y_test, y_pred_log, test)
    evaluate("Gradient Boosting", y_test, y_pred_gbc, test)

    print("\nFeature importances (Gradient Boosting):")
    importances = pd.Series(gbc.feature_importances_, index=FEATURE_COLS)
    print(importances.sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
