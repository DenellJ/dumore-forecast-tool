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

# -------------- STYLING & PAGE --------------

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

        h1, h2, h3, h4, h5, h6,
        label,
        p,
        .stMarkdown,
        .stSelectbox label {{
            color: {DUMORE_BLUE} !important;
        }}

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

        .stButton>button:hover,
        .stDownloadButton>button:hover {{
            background: linear-gradient(90deg, {DUMORE_YELLOW}, {DUMORE_BLUE}) !important;
            color: #ffffff !important;
            transform: translateY(-1px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.16);
        }}

        .stFileUploader label div {{
            color: {DUMORE_BLUE} !important;
            font-weight: 500 !important;
        }}

        div[data-testid="stSpinner"] p {{
            color: {DUMORE_BLUE} !important;
            font-weight: 600 !important;
        }}

        div[data-testid="stSuccess"] p,
        div[data-testid="stError"] p {{
            color: {DUMORE_BLUE} !important;
        }}

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
    Parse headers like 'January  (2020) - Quantity' into Timestamp('2020-01-01').
    Adjust this if your actual headers differ.
    """
    if not isinstance(label, str):
        return None

    s = label.strip()
    m = re.search(r'([A-Za-z]+)\s*\((\d{4})\)', s)
    if not m:
        return None

    month_name, year_str = m.group(1), m.group(2)
    year = int(year_str)

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
    Reshape wide table to:
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

    value_cols = [c for c in df.columns if "Quantity" in str(c) and c not in id_cols]
    if not value_cols:
        raise ValueError(
            "No monthly 'Quantity' columns found. "
            "Expected headers like 'January  (2020) - Quantity'."
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
    Faster SARIMA per item with safe fallbacks.
    """
    ts = ts.sort_index().asfreq("MS")
    valid_ts = ts.dropna()

    if len(valid_ts) == 0:
        return np.zeros(steps)

    if len(valid_ts) < MIN_HISTORY_POINTS:
        mean_val = float(valid_ts.mean())
        return np.full(steps, max(mean_val, 0.0))

    try:
        # Slightly lighter seasonal spec + maxiter cap for speed
        model = SARIMAX(
            valid_ts,
            order=(1, 1, 1),
            seasonal_order=(0, 1, 1, SEASONAL_PERIOD),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False, maxiter=40)
        fc = result.forecast(steps=steps)
        fc = fc.fillna(valid_ts.mean())
        return np.array([max(float(v), 0.0) for v in fc])
    except Exception:
        mean_val = float(valid_ts.mean())
        return np.full(steps, max(mean_val, 0.0))


def build_forecast_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build:
    Item No. | Item Description | [Next 13 months of forecasts...]
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

    # Progress bar for UX
    items = list(long_df["Item No."].unique())
    progress = st.progress(0.0, text="Running forecasts per item...")

    for i, item_no in enumerate(items, start=1):
        grp = long_df[long_df["Item No."] == item_no]
        desc = grp["Item Description"].iloc[0]
        ts = grp.set_index("Date")["Quantity"].astype(float)

        fc_vals = forecast_series_sarima(ts, FORECAST_STEPS)

        row = {"Item No.": item_no, "Item Description": desc}
        for dt, v in zip(future_index, fc_vals):
            col_name = f"{dt.strftime('%B')} ({dt.year}) - Forecast"
            row[col_name] = round(float(v), ROUND_DECIMALS)

        rows.append(row)

        progress.progress(i / len(items), text=f"Running forecasts... ({i}/{len(items)})")

    progress.empty()

    forecast_df = pd.DataFrame(rows)
    return forecast_df.sort_values("Item No.").reset_index(drop=True)


# ---------- CACHE WRAPPER (KEY FOR SPEED) ----------

@st.cache_data(show_spinner=False)
def build_forecast_table_cached(file_bytes: bytes, sheet_name: str) -> pd.DataFrame:
    """
    Cached wrapper:
    - Reads Excel from bytes
    - Builds forecast table
    Cache key = file content + sheet name.
    """
    raw_df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name)
    return build_forecast_table(raw_df)


# -------------- APP LAYOUT --------------

def main():
    # Logo + title
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
                    file_bytes = uploaded_file.getvalue()
                    forecast_df = build_forecast_table_cached(file_bytes, sheet_name)
                st.session_state["forecast_df"] = forecast_df

        except Exception as e:
            st.error(f"Error: {e}")

    # Show results if available
    if "forecast_df" in st.session_state:
        forecast_df = st.session_state["forecast_df"]

        st.success("Forecast generated successfully.")
        st.dataframe(forecast_df.head(50))

        # Build Excel in memory
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

if __name__ == "__main__":
    main()
