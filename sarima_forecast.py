import os
import re
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")


# -------- CONFIG --------
INPUT_FILE = "Sales Data - Historical.xlsx"
INPUT_SHEET = "2020-today"           # based on your file
OUTPUT_FILE = "Sales_Forecast_SARIMA_13m.xlsx"
FORECAST_STEPS = 13
MIN_HISTORY_POINTS = 12              # below this, fallback method is used
SEASONAL_PERIOD = 12                 # monthly seasonality
# ------------------------


def parse_period_to_date(label: str):
    """
    Convert column header like 'January  (2020) - Quantity'
    into a Timestamp('2020-01-01').
    """
    if not isinstance(label, str):
        return None

    # Extract: MonthName (Year)
    m = re.search(r'([A-Za-z]+)\s*\((\d{4})\)', label)
    if not m:
        return None

    month_name = m.group(1)
    year = int(m.group(2))

    # Try full month name, then short
    try:
        month_num = pd.to_datetime(month_name, format="%B").month
    except ValueError:
        month_num = pd.to_datetime(month_name, format="%b").month

    return pd.Timestamp(year=year, month=month_num, day=1)


def prepare_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melt the wide monthly columns into:
    Item No. | Item Description | Date | Quantity
    """
    id_cols = ["Item No.", "Item Description"]

    # Any column that ends with 'Quantity' is treated as a time-series column
    value_cols = [c for c in df.columns if "Quantity" in str(c)]

    # Melt to long
    long_df = df.melt(
        id_vars=id_cols,
        value_vars=value_cols,
        var_name="Period",
        value_name="Quantity"
    )

    # Drop empty
    long_df = long_df.dropna(subset=["Quantity"])

    # Parse to real dates
    long_df["Date"] = long_df["Period"].apply(parse_period_to_date)
    long_df = long_df.dropna(subset=["Date"])

    # Sort nicely
    long_df = long_df.sort_values(["Item No.", "Date"]).reset_index(drop=True)

    return long_df


def get_global_last_date(df: pd.DataFrame) -> pd.Timestamp:
    """
    Use the headers to determine the last available month in the dataset.
    """
    value_cols = [c for c in df.columns if "Quantity" in str(c)]
    dates = []

    for c in value_cols:
        d = parse_period_to_date(c)
        if d is not None:
            dates.append(d)

    if not dates:
        raise ValueError("No valid monthly quantity columns found.")

    return max(dates)


def forecast_series_sarima(ts: pd.Series, steps: int) -> np.ndarray:
    """
    Fit SARIMA on a single-item monthly series.
    Fallbacks:
      - if not enough points: use mean
      - if model fails: use mean
    """
    ts = ts.sort_index().asfreq("MS")  # ensure monthly start frequency

    valid_ts = ts.dropna()
    if len(valid_ts) < MIN_HISTORY_POINTS:
        mean_val = float(valid_ts.mean() if len(valid_ts) > 0 else 0.0)
        return np.full(steps, max(mean_val, 0.0))

    try:
        model = SARIMAX(
            ts,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, SEASONAL_PERIOD),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False)
        fc = result.forecast(steps=steps)
        fc = fc.fillna(valid_ts.mean())  # just in case
        return np.array([max(float(v), 0.0) for v in fc])
    except Exception:
        # Safe fallback = flat mean
        mean_val = float(valid_ts.mean() if len(valid_ts) > 0 else 0.0)
        return np.full(steps, max(mean_val, 0.0))


def build_forecast_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a wide forecast table:
    Item No. | Item Description | [Next 13 months as Forecast columns...]
    """
    long_df = prepare_long(raw_df)
    last_date = get_global_last_date(raw_df)

    # Forecast horizon: next 13 months after the last column month
    future_index = pd.date_range(
        last_date + pd.offsets.MonthBegin(1),
        periods=FORECAST_STEPS,
        freq="MS"
    )

    rows = []

    for (item_no), grp in long_df.groupby("Item No."):
        desc = grp["Item Description"].iloc[0]
        ts = grp.set_index("Date")["Quantity"].astype(float)

        fc_vals = forecast_series_sarima(ts, FORECAST_STEPS)

        row = {
            "Item No.": item_no,
            "Item Description": desc,
        }

        for dt, v in zip(future_index, fc_vals):
            col_name = f"{dt.strftime('%B')} ({dt.year}) - Forecast"
            row[col_name] = round(float(v), 2)

        rows.append(row)

    forecast_df = pd.DataFrame(rows)
    forecast_df = forecast_df.sort_values("Item No.").reset_index(drop=True)
    return forecast_df


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, INPUT_FILE)
    output_path = os.path.join(script_dir, OUTPUT_FILE)

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Input file not found at: {input_path}\n"
            f"Make sure '{INPUT_FILE}' is saved in: {script_dir}"
        )

    print(f"Loading data from: {input_path}")
    raw_df = pd.read_excel(input_path, sheet_name=INPUT_SHEET)

    print("Building SARIMA forecasts...")
    forecast_df = build_forecast_table(raw_df)

    print(f"Saving forecast to: {output_path}")
    forecast_df.to_excel(output_path, index=False)

    print("Done. Open the output file to review your 13-month forecasts.")


if __name__ == "__main__":
    main()
