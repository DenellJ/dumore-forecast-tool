import io
import re
import warnings

import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# ------------ CONFIG ------------
FORECAST_STEPS = 13
MIN_HISTORY_POINTS = 12       # below this, fallback to mean
SEASONAL_PERIOD = 12          # monthly seasonality
ROUND_DECIMALS = 2            # forecasts rounded to 2dp

# Brand colors
DUMORE_BLUE = "#1B2A7A"
DUMORE_YELLOW = "#F8C700"
# -------------------------------

# -------------- STYLING --------------

st.set_page_config(
    page_title="Dumore Demand Forecasting Tool",
    page_icon="📈",
    layout="centered",
)

st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: #ffffff;
            font-family: -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
        }}

        .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
        }}

        /* Global text blue */
        h1, h2, h3, h4, h5, h6,
        label,
        p,
        .stMarkdown,
        .stSelectbox label {{
            color: {DUMORE_BLUE} !important;
        }}

        /* Primary buttons (Generate / Download) */
        .stButton>button,
        .stDownloadButton>button {{
            background: {DUMORE_BLUE} !important;
            color: #ffffff !important;
            border-radius: 999px !important;
            padding: 0.6rem 1.6rem !important;
            font-weight: 500 !important;
            border: none !important;
            box-shadow: 0 4px 10px rgba(0,0,0,0.10);
            transition: all 0.25s ease-in-out;
        }}

        /* Hover gradient, keep white text */
        .stButton>button:hover,
        .stDownloadButton>button:hover {{
            background: linear-gradient(90deg, {DUMORE_YELLOW}, {DUMORE_BLUE}) !important;
            color: #ffffff !important;
            transform: translateY(-1px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.16);
        }}

        /* File uploader label */
        .stFileUploader label div {{
            color: {DUMORE_BLUE} !important;
            font-weight: 500 !important;
        }}

        /* Spinner text */
        div[data-testid="stSpinner"] p {{
            color: {DUMORE_BLUE} !important;
            font-weight: 600 !important;
        }}

        /* Success / error text */
        div[data-testid="stSuccess"] p,
        div[data-testid="stError"] p {{
            color: {DUMORE_BLUE} !important;
        }}

        /* DataFrame frame */
        .stDataFrame {{
            border: 1px solid {DUMORE_BLUE};
            border-radius: 8px;
            overflow: hidden;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------- HELPERS --------------

def parse_period_to_date(label: str):
    """
    Convert header text into a Timestamp.
    Robust to patterns like:
      'January  (2020) - Quantity'
      'Jan 2020 - Quantity'
      'Jan-20'
    Returns None if no obvious month/year.
    """
    if not isinstance(label, str):
        return None

    s = label.strip()

    # 1) 'MonthName (YYYY)'
    m = re.search(r'([A-Za-z]+)\s*\((\d{4})\)', s)
    if m:
        month_name, year_str = m.group(1), m.group(2)
    else:
        # 2) 'MonthName YYYY'
        m = re.search(r'([A-Za-z]+)\s+(\d{4})', s)
        if m:
            month_name, year_str = m.group(1), m.group(2)
        else:
            # 3) 'MonthName-YY' or 'MonthName-YYYY'
            m = re.search(r'([A-Za-z]+)[\s\-_]+(\d{2,4})', s)
            if m:
                month_name, year_str = m.group(1), m.group(2)
            else:
                return None

    # Normalize year
    year = int(year_str)
    if year < 100:  # handle YY format
        year = 2000 + year if year < 50 else 1900 + year

    # Normalize month
    try:
        month_num = pd.to_datetime(month_name, format="%B").month
    except ValueError:
        try:
            month_num = pd.to_datetime(month_name, format="%b").month
        except ValueError:
            return None

    return pd.Timestamp(year=year, month=month_num, day=1)


def prepare_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melt wide table into long:
    Item No. | Item Description | Date | Quantity
    """
    id_candidates = ["Item No.", "Item No", "Item", "Item Code"]
    desc_candidates = ["Item Description", "Description"]

    id_col = next((c for c in id_candidates if c in df.columns), None)
    desc_col = next((c for c in desc_candidates if c in df.columns), None)

    if id_col is None:
        raise ValueError("Could not find an 'Item No.' column.")
    if desc_col is None:
        raise ValueError("Could not find an 'Item Description' column.")

    id_cols = [id_col, desc_col]

    # Time-series columns = contain 'Quantity'
    value_cols = [c for c in df.columns if "Quantity" in str(c) and c not in id_cols]

    # Fallback: any column (not ID) that can be parsed as Month/Year
    if not value_cols:
        for c in df.columns:
            if c in id_cols:
                continue
            if parse_period_to_date(c) is not None:
                value_cols.append(c)

    if not value_cols:
        raise ValueError(
            "No valid monthly columns found. "
            "Ensure your headers include a month & year, e.g. 'January  (2020) - Quantity'."
        )

    long_df = df.melt(
        id_vars=id_cols,
        value_vars=value_cols,
        var_name="Period",
        value_name="Quantity"
    )

    long_df = long_df.dropna(subset=["Quantity"])
    long_df["Date"] = long_df["Period"].apply(parse_period_to_date)
    long_df = long_df.dropna(subset=["Date"])
    long_df = long_df.sort_values([id_col, "Date"]).reset_index(drop=True)

    return long_df.rename(columns={id_col: "Item No.", desc_col: "Item Description"})


def forecast_series_sarima(ts: pd.Series, steps: int) -> np.ndarray:
    """
    Fit SARIMA to single-item series with safe fallbacks.
    """
    ts = ts.sort_index().asfreq("MS")
    valid_ts = ts.dropna()

    if len(valid_ts) == 0:
        return np.zeros(steps)

    if len(valid_ts) < MIN_HISTORY_POINTS:
        mean_val = float(valid_ts.mean())
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
        fc = fc.fillna(valid_ts.mean())
        return np.array([max(float(v), 0.0) for v in fc])
    except Exception:
        mean_val = float(valid_ts.mean())
        return np.full(steps, max(mean_val, 0.0))


def build_forecast_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return wide table:
    Item No. | Item Description | [Next 13 months forecast columns...]
    """
    long_df = prepare_long(raw_df)

    if long_df.empty or long_df["Date"].isna().all():
        raise ValueError("Unable to detect any valid dates from your column headers.")

    last_date = long_df["Date"].max()

    future_index = pd.date_range(
        last_date + pd.offsets.MonthBegin(1),
        periods=FORECAST_STEPS,
        freq="MS"
    )

    rows = []

    for item_no, grp in long_df.groupby("Item No."):
        desc = grp["Item Description"].iloc[0]
        ts = grp.set_index("Date")["Quantity"].astype(float)

        fc_vals = forecast_series_sarima(ts, FORECAST_STEPS)

        row = {
            "Item No.": item_no,
            "Item Description": desc,
        }

        for dt, v in zip(future_index, fc_vals):
            col_name = f"{dt.strftime('%B')} ({dt.year}) - Forecast"
            row[col_name] = round(float(v), ROUND_DECIMALS)

        rows.append(row)

    forecast_df = pd.DataFrame(rows)
    return forecast_df.sort_values("Item No.").reset_index(drop=True)

# -------------- APP LAYOUT --------------

def main():
    cols = st.columns([1, 3])
    with cols[0]:
        try:
            st.image("DUMORE ENT FC STACKED.png", use_container_width=True)
        except Exception:
            st.write("")
    with cols[1]:
        st.markdown(
            f"""
            <h1 style="margin-bottom:0;">Dumore Demand Forecasting Tool</h1>
            <p style="font-size:0.95rem; margin-top:0.2rem;">
            SARIMA-based 13-month forecast per item.
            </p>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"<hr style='border:1px solid {DUMORE_BLUE}; margin-top:0.8rem; margin-bottom:1.2rem;'>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <p style="font-size:0.9rem;">
        Upload your historical sales file and generate an Excel report with
        the next {FORECAST_STEPS} months of forecasted quantities for every item.
        </p>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload Excel file (.xlsx) with historical sales",
        type=["xlsx"],
    )

    if uploaded_file is not None:
        try:
            xls = pd.ExcelFile(uploaded_file)
            sheet_name = st.selectbox(
                "Select the sheet with historical data",
                xls.sheet_names,
            )

            if st.button("Generate Forecast"):
                with st.spinner("Calculating SARIMA forecasts..."):
                    raw_df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
                    forecast_df = build_forecast_table(raw_df)

                st.success("Forecast generated successfully.")
                st.dataframe(forecast_df.head(20))

                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                    forecast_df.to_excel(
                        writer,
                        index=False,
                        sheet_name="Forecast_13m",
                    )
                output.seek(0)

                st.download_button(
                    label="Download Forecast Excel",
                    data=output,
                    file_name="Sales_Forecast_SARIMA_13m.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
