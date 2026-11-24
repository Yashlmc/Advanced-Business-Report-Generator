import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from analysis import clean_sales_data


# ==================================================
# 1. PREPARE MONTHLY SALES DATA
# ==================================================

def prepare_monthly_data(df):
    """
    Prepare monthly revenue data for forecasting
    and anomaly detection.

    Returns:
        DataFrame containing:
            Month (datetime64[ns])
            Revenue (float)
    """

    if df is None or df.empty:
        raise ValueError("No sales data provided.")

    df = clean_sales_data(df)

    monthly = (
        df.groupby("Month")["Revenue"]
        .sum()
        .reset_index()
        .sort_values("Month")
    )

    monthly["Month"] = pd.to_datetime(monthly["Month"], format="%Y-%m")
    monthly = monthly.set_index("Month")

    # Fill missing months with zero revenue
    monthly = (
        monthly.asfreq("MS", fill_value=0.0)
        .rename_axis("Month")
        .reset_index()
    )

    monthly["Revenue"] = pd.to_numeric(
        monthly["Revenue"], errors="coerce"
    ).fillna(0.0)

    return monthly


# ==================================================
# 2. CHRONOLOGICAL EVALUATION HELPER
# ==================================================

def _safe_mape(y_true, y_pred):
    """Calculate MAPE while safely skipping zero-revenue months."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true > 0
    if not np.any(mask):
        return None
    return round(
        float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100),
        2,
    )


def _build_lag_features(values):
    """Build lag-3 supervised features from a monthly revenue list."""
    features = []
    targets = []
    for i in range(3, len(values)):
        features.append([i, values[i - 1], values[i - 2], values[i - 3]])
        targets.append(values[i])
    return features, targets


# ==================================================
# 3. FORECAST FUTURE REVENUE + BASELINE BENCHMARK
# ==================================================

def forecast_revenue(df, periods=3):
    """
    Forecast future monthly revenue using Random Forest Regression
    and evaluate against a chronological Naive Persistence baseline.

    Parameters:
        df:
            Sales DataFrame.
        periods:
            Number of future months to forecast (1 to 12).

    Returns:
        forecast_df:
            DataFrame containing Month (YYYY-MM) and Forecast_Revenue.
        metrics:
            Dictionary comparing Naive Persistence vs Random Forest
            on chronological hold-out MAE and MAPE.
    """

    if not isinstance(periods, int) or isinstance(periods, bool):
        raise ValueError("Forecast period must be an integer.")

    if not 1 <= periods <= 12:
        raise ValueError("Forecast period must be between 1 and 12.")

    monthly = prepare_monthly_data(df)

    if len(monthly) < 8:
        raise ValueError(
            "At least 8 months of sales history are required for forecasting."
        )

    values = monthly["Revenue"].tolist()
    features, targets = _build_lag_features(values)

    # ----------------------------------
    # CHRONOLOGICAL HOLD-OUT EVALUATION
    # ----------------------------------

    test_size = min(3, max(2, len(targets) // 4))
    X_train, X_test = features[:-test_size], features[-test_size:]
    y_train, y_test = targets[:-test_size], targets[-test_size:]

    eval_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        random_state=42,
        n_jobs=-1,
    )
    eval_model.fit(X_train, y_train)
    rf_val_preds = np.maximum(0.0, eval_model.predict(X_test))

    # Naive persistence baseline: predict previous month's actual revenue
    baseline_val_preds = [row[1] for row in X_test]

    rf_mae = round(float(mean_absolute_error(y_test, rf_val_preds)), 2)
    baseline_mae = round(float(mean_absolute_error(y_test, baseline_val_preds)), 2)
    rf_mape = _safe_mape(y_test, rf_val_preds)
    baseline_mape = _safe_mape(y_test, baseline_val_preds)

    metrics = {
        "holdout_months": test_size,
        "rf_mae": rf_mae,
        "rf_mape": rf_mape,
        "baseline_mae": baseline_mae,
        "baseline_mape": baseline_mape,
        "beats_baseline": rf_mae <= baseline_mae,
    }

    # ----------------------------------
    # RETRAIN ON FULL HISTORY & FORECAST
    # ----------------------------------

    full_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        random_state=42,
        n_jobs=-1,
    )
    full_model.fit(features, targets)

    predictions = []
    history = values.copy()

    for _ in range(periods):
        i = len(history)
        input_data = [[i, history[-1], history[-2], history[-3]]]
        prediction = max(0.0, float(full_model.predict(input_data)[0]))
        predictions.append(prediction)
        history.append(prediction)

    last_month = monthly["Month"].max()
    future_dates = pd.date_range(
        start=last_month + pd.offsets.MonthBegin(1),
        periods=periods,
        freq="MS",
    )

    forecast_df = pd.DataFrame(
        {
            "Month": future_dates.strftime("%Y-%m"),
            "Forecast_Revenue": [round(value, 2) for value in predictions],
        }
    )

    return forecast_df, metrics


# ==================================================
# 4. DETECT REVENUE ANOMALIES
# ==================================================

def detect_anomalies(df):
    """
    Detect unusually high or low monthly revenue using Isolation Forest.

    Returns:
        monthly:
            Complete monthly revenue DataFrame with Month (YYYY-MM),
            Revenue, Anomaly, and Status.
        anomalies:
            DataFrame containing only anomalous months.
    """

    monthly = prepare_monthly_data(df)

    if len(monthly) < 8:
        raise ValueError(
            "At least 8 months of sales history are required for anomaly detection."
        )

    model = IsolationForest(
        n_estimators=100,
        contamination=0.10,
        random_state=42,
    )

    monthly["Anomaly"] = model.fit_predict(monthly[["Revenue"]])
    monthly["Status"] = monthly["Anomaly"].map({1: "Normal", -1: "Anomaly"})
    monthly["Revenue"] = monthly["Revenue"].round(2)
    monthly["Month"] = monthly["Month"].dt.strftime("%Y-%m")

    anomalies = monthly[monthly["Anomaly"] == -1].copy().reset_index(drop=True)

    return monthly, anomalies