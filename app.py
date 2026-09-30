from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

from stock_ml_analysis import (
    load_data,
    add_features,
    time_split,
    train_models,
    FEATURE_COLS,
)

app = Flask(__name__)


def get_scalar(value):
    """Convert a pandas/numpy value into one normal Python value."""
    if isinstance(value, pd.Series):
        value = value.iloc[0]

    if isinstance(value, np.ndarray):
        value = value.flatten()[0]

    return value


def get_column(df, column_name):
    """Safely get one column even if pandas returns a DataFrame."""
    value = df[column_name]

    if isinstance(value, pd.DataFrame):
        value = value.iloc[:, 0]

    return value


def calculate_results(name, y_true, y_pred, test_df):
    cost_per_trade = 0.001

    position = pd.Series(
        np.asarray(y_pred).flatten(),
        index=test_df.index
    )

    ret_1d = get_column(test_df, "ret_1d")

    strategy_ret = position.shift(1).fillna(0) * ret_1d

    trades = position.diff().abs().fillna(0)

    strategy_ret_after_cost = strategy_ret - trades * cost_per_trade

    cum_strategy = (
        (1 + strategy_ret_after_cost).cumprod().iloc[-1] - 1
    )

    cum_buyhold = (
        (1 + ret_1d).cumprod().iloc[-1] - 1
    )

    sharpe = (
        strategy_ret_after_cost.mean()
        / (strategy_ret_after_cost.std() + 1e-9)
        * np.sqrt(252)
    )

    return {
        "model": name,
        "accuracy": round(float(get_scalar(
            accuracy_score(y_true, y_pred)
        )), 4),

        "precision": round(float(get_scalar(
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            )
        )), 4),

        "recall": round(float(get_scalar(
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            )
        )), 4),

        "strategy_return": round(
            float(get_scalar(cum_strategy)), 4
        ),

        "buy_hold_return": round(
            float(get_scalar(cum_buyhold)), 4
        ),

        "sharpe": round(
            float(get_scalar(sharpe)), 4
        ),
    }


def run_analysis(ticker, start="2015-01-01", end=None, month=None):

    ticker = ticker.strip().upper()

    if not ticker:
        raise ValueError("Please enter a stock ticker.")

    if month:
        import calendar

        year, month_num = (
            int(x) for x in month.split("-")
        )

        focus_start = (
            f"{year:04d}-{month_num:02d}-01"
        )

        last_day = calendar.monthrange(
            year,
            month_num
        )[1]

        focus_end = (
            f"{year:04d}-{month_num:02d}-{last_day:02d}"
        )

        df = load_data(
            ticker,
            start,
            focus_end
        )

        df = add_features(df)

        train = df[df.index < focus_start]

        test = df[
            (df.index >= focus_start)
            & (df.index <= focus_end)
        ]

    else:
        df = load_data(
            ticker,
            start,
            end
        )

        df = add_features(df)

        train, test = time_split(df)

    if len(train) < 50:
        raise ValueError(
            "Not enough training data. Choose an earlier "
            "start date or a later evaluation month."
        )

    if len(test) < 2:
        raise ValueError(
            "Not enough test data for this period. "
            "Choose a wider date range or another month."
        )

    scaler, logreg, gbc = train_models(train)

    X_test = test[FEATURE_COLS]
    y_test = test["target"]

    y_pred_log = logreg.predict(
        scaler.transform(X_test)
    )

    y_pred_gbc = gbc.predict(X_test)

    logistic = calculate_results(
        "Logistic Regression",
        y_test,
        y_pred_log,
        test
    )

    gradient = calculate_results(
        "Gradient Boosting",
        y_test,
        y_pred_gbc,
        test
    )

    importances = (
        pd.Series(
            gbc.feature_importances_,
            index=FEATURE_COLS
        )
        .sort_values(ascending=False)
    )

    last_prediction = int(
        np.asarray(y_pred_gbc).flatten()[-1]
    )

    signal = (
        "UP"
        if last_prediction == 1
        else "DOWN"
    )

    close_column = get_column(test, "close")

    last_close = get_scalar(
        close_column.iloc[-1]
    )

    return {
        "ticker": ticker,

        "train_rows": int(len(train)),

        "test_rows": int(len(test)),

        "logistic": logistic,

        "gradient_boosting": gradient,

        "feature_importance": [
            {
                "feature": str(k),
                "importance": round(
                    float(get_scalar(v)),
                    4
                )
            }
            for k, v in importances.items()
        ],

        "last_date": str(
            test.index[-1].date()
        ),

        "last_close": round(
            float(last_close),
            2
        ),

        "latest_signal": signal,
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/analyze", methods=["GET"])
def analyze():

    try:

        ticker = request.args.get(
            "ticker",
            "AAPL"
        )

        start = request.args.get(
            "start",
            "2015-01-01"
        )

        end = (
            request.args.get("end")
            or None
        )

        month = (
            request.args.get("month")
            or None
        )

        result = run_analysis(
            ticker,
            start=start,
            end=end,
            month=month
        )

        return jsonify({
            "success": True,
            "data": result
        })

    except Exception as exc:

        return jsonify({
            "success": False,
            "error": str(exc)
        }), 400


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )