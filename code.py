import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import plotly.express as px

# ================================================================
# 0. PAGE CONFIG + GLOBAL THEME
# ================================================================
st.set_page_config(
    page_title="Enchanto – AI Demand & Inventory Dashboard",
    page_icon="💜",
    layout="wide",
)

# White–purple theme + header spacing fix
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f3ff;
    }
    /* more top padding so header is not cut by browser bar */
    .block-container {
        padding-top: 2.5rem !important;
        padding-bottom: 2rem !important;
    }
    .enchanto-header {
        background: linear-gradient(90deg,#4c1d95,#7c3aed);
        color: #f9fafb;
        padding: 10px 18px;
        border-radius: 18px;
        box-shadow: 0 4px 16px rgba(15,23,42,0.25);
        margin-top: 0.2rem;
        margin-bottom: 1rem;
    }
    .enchanto-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 3px 9px;
        border-radius: 999px;
        background: rgba(243,244,246,0.18);
        font-size: 0.75rem;
        margin-bottom: 4px;
    }
    .enchanto-tag-icon {
        width: 18px;
        height: 18px;
        border-radius: 999px;
        background: #f97316;
        color: white;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
    }
    .enchanto-title {
        font-size: 1.35rem;
        font-weight: 600;
        margin-bottom: 1px;
    }
    .enchanto-subtitle {
        font-size: 0.83rem;
        opacity: 0.95;
    }
    .kpi-card {
        border-radius: 12px;
        background-color: #faf5ff;
        border: 1px solid #e5e7eb;
        padding: 0.6rem 0.85rem;
        box-shadow: 0 3px 10px rgba(15,23,42,0.08);
    }
    .kpi-label {
        font-size: 0.75rem;
        color: #6b7280;
    }
    .kpi-value {
        font-size: 1.15rem;
        font-weight: 600;
        color: #4c1d95;
    }
    .kpi-note {
        font-size: 0.72rem;
        color: #6b7280;
        margin-top: 2px;
    }
    .section-title {
        font-size: 1rem;
        font-weight: 600;
        color: #111827;
        margin-top: 0.3rem;
        margin-bottom: 0.1rem;
    }
    .section-subtitle {
        font-size: 0.80rem;
        color: #6b7280;
        margin-bottom: 0.35rem;
    }
    .stock-banner {
        border-radius: 10px;
        padding: 0.6rem 0.85rem;
        font-size: 0.8rem;
        margin-bottom: 0.4rem;
    }
    .stock-high {
        background: #fef2f2;
        border: 1px solid #fecaca;
        color: #991b1b;
    }
    .stock-medium {
        background: #fffbeb;
        border: 1px solid #facc15;
        color: #92400e;
    }
    .stock-low {
        background: #ecfdf3;
        border: 1px solid #bbf7d0;
        color: #166534;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ================================================================
# 1. LOAD DATA
# ================================================================
SALES_FILE = Path("enchanto_sales_simulated_2024.xlsx")
INVENTORY_FILE = Path("enchanto_inventory_solver_template.xlsx")  # not used directly now

@st.cache_data(show_spinner=True)
def load_sales_data(path: Path):
    xls = pd.ExcelFile(path)
    all_months = [pd.read_excel(path, sheet_name=s) for s in xls.sheet_names]
    sales = pd.concat(all_months, ignore_index=True)

    sales["Order DateTime"] = pd.to_datetime(sales["Order DateTime"])
    sales["Order Date"] = pd.to_datetime(sales["Order Date"])

    if "Festival Season" not in sales.columns:
        sales["Festival Season"] = "No"
    else:
        sales["Festival Season"] = sales["Festival Season"].fillna("No")

    daily = (
        sales
        .groupby(["Order Date", "SKU", "Region", "Category", "Product Name", "Festival Season"])
        ["Quantity Sold"]
        .sum()
        .reset_index()
        .rename(columns={"Order Date": "date", "Quantity Sold": "daily_demand"})
    )

    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values(["SKU", "Region", "date"]).reset_index(drop=True)
    daily["Year"] = daily["date"].dt.year
    daily["Month"] = daily["date"].dt.month
    daily["Month_Name"] = daily["date"].dt.strftime("%b")

    # 3-day smoothing
    daily["demand_smooth"] = (
        daily
        .groupby(["SKU", "Region"])["daily_demand"]
        .transform(lambda x: x.rolling(window=3, min_periods=1, center=True).mean())
    )

    return sales, daily

sales, daily_demand = load_sales_data(SALES_FILE)

# ================================================================
# 2. HELPER FUNCTIONS
# ================================================================
def get_current_stock(sales: pd.DataFrame, sku: str, region: str) -> int:
    df = (
        sales[(sales["SKU"] == sku) & (sales["Region"] == region)]
        .sort_values("Order DateTime")
    )
    if df.empty or "Stock After Sale" not in df.columns:
        return 0
    stock_series = df["Stock After Sale"].dropna()
    if stock_series.empty:
        return 0
    return int(round(stock_series.iloc[-1]))

def build_features_for_sku_region(daily_demand, sku, region, min_days=40):
    df = daily_demand[(daily_demand["SKU"] == sku) & (daily_demand["Region"] == region)].copy()
    df = df.sort_values("date").reset_index(drop=True)
    history_len = len(df)

    if history_len < min_days:
        return None, None, None, None, history_len

    df["day_of_week"] = df["date"].dt.weekday
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["festival_flag"] = (df["Festival Season"] == "Yes").astype(int)

    df["lag_1"] = df["demand_smooth"].shift(1)
    df["lag_7"] = df["demand_smooth"].shift(7)
    df["rolling_7"] = df["demand_smooth"].shift(1).rolling(window=7).mean()

    df_model = df.dropna().reset_index(drop=True)

    feature_cols = [
        "day_of_week",
        "month",
        "is_weekend",
        "festival_flag",
        "lag_1",
        "lag_7",
        "rolling_7",
    ]

    X = df_model[feature_cols]
    y = df_model["demand_smooth"]
    return df_model, X, y, feature_cols, history_len

def forecast_next_n_days(model, df_history, n_days=30):
    history = df_history.copy().sort_values("date").reset_index(drop=True)
    last_date = history["date"].iloc[-1]
    preds = []

    for i in range(1, n_days + 1):
        next_date = last_date + pd.Timedelta(days=i)
        dow = next_date.weekday()
        month = next_date.month
        is_weekend = int(dow in [5, 6])
        festival_flag = 1 if month in [10, 11, 12] else 0

        lag_1 = history["demand_smooth"].iloc[-1]
        if len(history) >= 7:
            lag_7 = history["demand_smooth"].iloc[-7]
            rolling_7 = history["demand_smooth"].iloc[-7:].mean()
        else:
            lag_7 = lag_1
            rolling_7 = history["demand_smooth"].mean()

        x_future = pd.DataFrame([{
            "day_of_week": dow,
            "month": month,
            "is_weekend": is_weekend,
            "festival_flag": festival_flag,
            "lag_1": lag_1,
            "lag_7": lag_7,
            "rolling_7": rolling_7,
        }])

        y_future = model.predict(x_future)[0]
        preds.append({"date": next_date, "forecast_demand": y_future})

        new_row = {
            "date": next_date,
            "daily_demand": y_future,
            "demand_smooth": y_future,
            "Festival Season": "Yes" if festival_flag == 1 else "No",
        }
        history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)

    return pd.DataFrame(preds)

def compute_lt_stats(forecast_df, lead_time_days=7):
    lt = forecast_df.head(lead_time_days)["forecast_demand"]
    mean_daily = lt.mean()
    total_mean = lt.sum()
    daily_std = lt.std(ddof=0)
    total_std = daily_std * np.sqrt(max(lead_time_days, 1))
    return mean_daily, total_mean, daily_std, total_std

def approx_z(service_level):
    table = {
        0.80: 0.84,
        0.85: 1.04,
        0.90: 1.28,
        0.95: 1.65,
        0.97: 1.88,
        0.98: 2.05,
        0.99: 2.33,
    }
    keys = list(table.keys())
    closest = min(keys, key=lambda k: abs(k - service_level))
    return table[closest]

def optimize_reorder_policy(
    forecast_df,
    current_stock,
    holding_cost_per_unit,
    stockout_cost_per_unit,
    lead_time_days=7,
):
    mean_daily, total_mean, daily_std, total_std = compute_lt_stats(
        forecast_df, lead_time_days
    )
    service_levels = [0.80, 0.85, 0.90, 0.95, 0.97, 0.98, 0.99]

    rows = []
    best_row = None
    for sl in service_levels:
        z = approx_z(sl)
        safety_stock = z * total_std
        reorder_level = total_mean + safety_stock
        holding_cost = holding_cost_per_unit * reorder_level
        expected_stockout_units = (1 - sl) * total_mean
        stockout_cost = stockout_cost_per_unit * expected_stockout_units
        total_cost = holding_cost + stockout_cost
        reorder_qty = max(0, reorder_level - current_stock)

        row = {
            "Service_Level": sl,
            "Safety_Stock": safety_stock,
            "Reorder_Level": reorder_level,
            "Holding_Cost": holding_cost,
            "Stockout_Cost": stockout_cost,
            "Total_Cost": total_cost,
            "Reorder_Qty": reorder_qty,
        }
        rows.append(row)
        if best_row is None or total_cost < best_row["Total_Cost"]:
            best_row = row

    result_df = pd.DataFrame(rows).sort_values("Total_Cost")
    return result_df, best_row, mean_daily, daily_std, total_mean

def optimize_logistics_distribution(daily_demand, sales, sku, inbound_units, lead_time_days=7):
    regions = (
        daily_demand[daily_demand["SKU"] == sku]["Region"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    rows = []
    remaining = inbound_units

    for region in regions:
        hist = daily_demand[(daily_demand["SKU"] == sku) & (daily_demand["Region"] == region)]
        if hist.empty:
            continue

        current_stock = get_current_stock(sales, sku, region)
        hist = hist.sort_values("date")
        last_30 = hist.tail(30)
        mean_daily = last_30["daily_demand"].mean()
        lt_total = mean_daily * lead_time_days

        shortage_before = max(0.0, lt_total - current_stock)
        rows.append({
            "Region": region,
            "Current_Stock": current_stock,
            "LT_Demand_7d": lt_total,
            "Shortage_Before": shortage_before,
            "Inbound_Allocated": 0.0,
        })

    rows = sorted(rows, key=lambda r: r["Shortage_Before"], reverse=True)
    for row in rows:
        if remaining <= 0:
            break
        need = row["Shortage_Before"]
        give = min(remaining, need)
        row["Inbound_Allocated"] = give
        remaining -= give

    for row in rows:
        final_stock = row["Current_Stock"] + row["Inbound_Allocated"]
        shortage_after = max(0.0, row["LT_Demand_7d"] - final_stock)
        excess_after = max(0.0, final_stock - row["LT_Demand_7d"])
        row["Shortage_After"] = shortage_after
        excess_after = max(0.0, final_stock - row["LT_Demand_7d"])
        row["Excess_After"] = excess_after

    return pd.DataFrame(rows)

# ================================================================
# 3. HEADER
# ================================================================
st.markdown(
    """
    <div class="enchanto-header">
      <div class="enchanto-tag">
        <span class="enchanto-tag-icon">AI</span>
        Enchanto – Demand Forecasting & Inventory Optimization Agent
      </div>
      <div class="enchanto-title">
        Enchanto Perfumes – AI Inventory Control Dashboard
      </div>
      <div class="enchanto-subtitle">
        56 SKUs • 5 Regions • ML Forecasting • Reorder Optimization • Logistics Distribution
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ================================================================
# 4. SIDEBAR – CONTROLS
# ================================================================
with st.sidebar:
    st.markdown("### Scenario Selection")

    sku_options = (
        daily_demand[["SKU", "Product Name"]]
        .drop_duplicates()
        .sort_values("SKU")
    )
    sku_list = sku_options["SKU"].tolist()
    sku = st.selectbox(
        "Select SKU",
        options=sku_list,
        format_func=lambda s: f"{s} – {sku_options.set_index('SKU').loc[s, 'Product Name']}",
    )

    regions_for_sku = (
        daily_demand[daily_demand["SKU"] == sku]["Region"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    region = st.selectbox("Select Region", options=regions_for_sku)

    computed_stock = get_current_stock(sales, sku, region)
    st.metric(
        "Current stock (units)",
        value=int(computed_stock),
        help="Pulled from latest 'Stock After Sale' in sales data.",
    )
    current_stock = float(computed_stock)

    inbound_units = st.number_input(
        "Inbound stock for this SKU (across all regions)",
        min_value=0.0,
        value=250.0,
        step=10.0,
    )

    st.markdown("### Cost Assumptions")
    logistics_cost = st.number_input(
        "Logistics cost per unit (₹)",
        min_value=1.0,
        value=50.0,
        step=1.0,
        help="Used to derive holding cost (2% of logistics cost).",
    )
    holding_cost_per_unit = logistics_cost * 0.02
    stockout_cost_per_unit = holding_cost_per_unit * 8

    st.write(f"- Holding cost per unit ≈ **₹{holding_cost_per_unit:.2f}**")
    st.write(f"- Stockout cost per unit ≈ **₹{stockout_cost_per_unit:.2f}**")

    run_btn = st.button("▶ Run Analysis")

if "run_done" not in st.session_state:
    st.session_state["run_done"] = False

if run_btn or not st.session_state["run_done"]:
    st.session_state["run_done"] = True

    df_model, X, y, feature_cols, history_len = build_features_for_sku_region(
        daily_demand, sku, region, min_days=40
    )

    if df_model is None:
        st.warning(
            f"Not enough history for {sku} in {region} "
            f"(have {history_len} days; need ≥ 40). Please choose another region or SKU."
        )
    else:
        # ============================================================
        # 5. TRAIN MODEL & FORECAST
        # ============================================================
        split_idx = int(len(df_model) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        model = RandomForestRegressor(
            n_estimators=400,
            max_depth=12,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_test_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_test_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        future_forecast = forecast_next_n_days(
            model,
            df_model[["date", "daily_demand", "demand_smooth", "Festival Season"]],
            n_days=30,
        )

        mean_daily_lt, total_mean_lt, daily_std_lt, total_std_lt = compute_lt_stats(
            future_forecast, lead_time_days=7
        )

        # ============================================================
        # 6. TABS LAYOUT
        # ============================================================
        tab_overview, tab_forecast, tab_inventory, tab_logistics = st.tabs(
            ["📊 Overview", "📈 Demand Forecasting", "📦 Inventory & Reorder", "🚚 Logistics & Risk"]
        )

        # -------------------- OVERVIEW TAB -------------------------
        with tab_overview:
            st.markdown("<div class='section-title'>Model Overview</div>", unsafe_allow_html=True)
            st.markdown(
                "<div class='section-subtitle'>Quick snapshot of forecasting quality, demand level, "
                "and inventory situation for the selected SKU–Region.</div>",
                unsafe_allow_html=True,
            )

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
                st.markdown("<div class='kpi-label'>Model MAE (smoothed)</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='kpi-value'>{mae:.2f}</div>", unsafe_allow_html=True)
                st.markdown("<div class='kpi-note'>Average absolute error</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with k2:
                st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
                st.markdown("<div class='kpi-label'>Model RMSE (smoothed)</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='kpi-value'>{rmse:.2f}</div>", unsafe_allow_html=True)
                st.markdown("<div class='kpi-note'>Penalises large errors</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with k3:
                st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
                st.markdown("<div class='kpi-label'>Lead-time demand (7 days)</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='kpi-value'>{total_mean_lt:.1f}</div>", unsafe_allow_html=True)
                st.markdown("<div class='kpi-note'>Forecasted units for 7 days</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with k4:
                st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
                st.markdown("<div class='kpi-label'>Current stock (units)</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='kpi-value'>{current_stock:.0f}</div>", unsafe_allow_html=True)
                st.markdown("<div class='kpi-note'>Fixed from transaction data</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        # -------------------- FORECAST TAB -------------------------
        with tab_forecast:
            st.markdown("<div class='section-title'>Demand Forecasting</div>", unsafe_allow_html=True)
            st.markdown(
                "<div class='section-subtitle'>Smoothed historical demand and 30-day forecast, "
                "plus model fit on a hold-out period and regional / monthly patterns.</div>",
                unsafe_allow_html=True,
            )

            c1, c2 = st.columns(2)

            # Smoothed history + forecast
            with c1:
                hist = df_model.copy()
                hist["type"] = "Smoothed history"
                fc = future_forecast.copy()
                fc["type"] = "Forecast"
                fc = fc.rename(columns={"forecast_demand": "demand_smooth"})

                combined = pd.concat([
                    hist[["date", "demand_smooth", "type"]],
                    fc[["date", "demand_smooth", "type"]],
                ])

                fig_forecast = px.line(
                    combined,
                    x="date",
                    y="demand_smooth",
                    color="type",
                    color_discrete_map={
                        "Smoothed history": "#6366f1",
                        "Forecast": "#a855f7",
                    },
                    labels={"demand_smooth": "Smoothed demand (units)", "date": "Date"},
                    title=f"Smoothed demand & 30-day forecast – {sku} / {region}",
                )
                fig_forecast.update_layout(
                    legend_title_text="",
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(fig_forecast, use_container_width=True)
                st.caption(
                    "The purple forecast line continues the smoothed demand trend into the next 30 days, "
                    "helping the manager see upcoming rises or dips clearly."
                )

            # Actual vs Predicted
            with c2:
                df_fit = pd.DataFrame({
                    "date": df_model["date"].iloc[split_idx:],
                    "Actual": y_test.values,
                    "Predicted": y_test_pred,
                })
                df_fit_melt = df_fit.melt("date", var_name="Series", value_name="Demand")
                fig_fit = px.line(
                    df_fit_melt,
                    x="date",
                    y="Demand",
                    color="Series",
                    color_discrete_map={
                        "Actual": "#9ca3af",
                        "Predicted": "#6366f1",
                    },
                    title="Model fit – Actual vs Predicted (test period)",
                )
                fig_fit.update_layout(
                    legend_title_text="",
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(fig_fit, use_container_width=True)
                st.caption(
                    "The predicted line closely follows the actual sales pattern, "
                    "showing that lag features and seasonality are captured well (low RMSE)."
                )

            # Demand by region & month
            st.markdown("<div class='section-title'>Demand by Region & Month (SKU level)</div>", unsafe_allow_html=True)
            rd, md = st.columns(2)

            sku_hist = daily_demand[daily_demand["SKU"] == sku].copy()
            region_totals = (
                sku_hist.groupby("Region")["daily_demand"]
                .sum()
                .reset_index()
                .sort_values("daily_demand", ascending=False)
            )
            month_totals = (
                sku_hist.groupby("Month_Name")["daily_demand"]
                .sum()
                .reindex(index=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
                .reset_index()
                .rename(columns={"Month_Name": "Month"})
            )

            with rd:
                fig_reg = px.bar(
                    region_totals,
                    x="Region",
                    y="daily_demand",
                    title="Total demand by region",
                    labels={"daily_demand": "Units sold"},
                    color="Region",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig_reg.update_layout(margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_reg, use_container_width=True)

            with md:
                fig_mon = px.bar(
                    month_totals,
                    x="Month",
                    y="daily_demand",
                    title="Total demand by month (all regions)",
                    labels={"daily_demand": "Units sold"},
                    color="Month",
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                )
                fig_mon.update_layout(showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_mon, use_container_width=True)

            # Short interpretation for these bars
            if not region_totals.empty and not month_totals.empty:
                peak_region = region_totals.iloc[0]["Region"]
                peak_month_row = month_totals.sort_values("daily_demand", ascending=False).iloc[0]
                peak_month = peak_month_row["Month"]

                st.caption(
                    f"**Region view:** {peak_region} contributes the highest demand for this SKU, so it "
                    f"should receive priority when allocating limited stock. "
                    f"**Month view:** demand peaks in {peak_month}, linking closely to festive or promotion periods."
                )

        # -------------------- INVENTORY TAB -------------------------
        with tab_inventory:
            st.markdown("<div class='section-title'>Reorder Optimization</div>", unsafe_allow_html=True)
            st.markdown(
                "<div class='section-subtitle'>Optimal service level, safety stock, reorder "
                "level, and reorder quantity based on forecasted demand and cost assumptions.</div>",
                unsafe_allow_html=True,
            )

            reorder_df, best_policy, mean_daily, daily_std, total_mean = optimize_reorder_policy(
                future_forecast,
                current_stock=current_stock,
                holding_cost_per_unit=holding_cost_per_unit,
                stockout_cost_per_unit=stockout_cost_per_unit,
                lead_time_days=7,
            )

            r1, r2, r3, r4 = st.columns(4)
            with r1:
                st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
                st.markdown("<div class='kpi-label'>Optimal service level</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='kpi-value'>{best_policy['Service_Level']*100:.0f}%</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with r2:
                st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
                st.markdown("<div class='kpi-label'>Reorder level (units)</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='kpi-value'>{best_policy['Reorder_Level']:.1f}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with r3:
                st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
                st.markdown("<div class='kpi-label'>Recommended reorder qty</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='kpi-value'>{best_policy['Reorder_Qty']:.1f}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with r4:
                st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
                st.markdown("<div class='kpi-label'>Minimum total cost (₹)</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='kpi-value'>{best_policy['Total_Cost']:.1f}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            df_display = reorder_df.copy()
            df_display["Service_Level"] = (df_display["Service_Level"] * 100).round(0).astype(int).astype(str) + "%"
            st.dataframe(
                df_display.style.highlight_min(subset=["Total_Cost"], color="#bbf7d0"),
                use_container_width=True,
            )

            st.caption(
                f"The table compares different service levels (80–99%). The highlighted row gives "
                f"the lowest total cost, recommending a service level of {best_policy['Service_Level']*100:.0f}%, "
                f"reorder level of {best_policy['Reorder_Level']:.1f} units and reorder quantity of "
                f"about {best_policy['Reorder_Qty']:.1f} units for {sku} in {region}."
            )

        # -------------------- LOGISTICS TAB -------------------------
        with tab_logistics:
            st.markdown("<div class='section-title'>Logistics Distribution & Stockout Risk</div>", unsafe_allow_html=True)
            st.markdown(
                "<div class='section-subtitle'>Inbound stock is allocated first to regions with the highest "
                "forecasted shortage, and a colour-coded banner shows risk for the selected region.</div>",
                unsafe_allow_html=True,
            )

            dist_df = optimize_logistics_distribution(
                daily_demand, sales, sku, inbound_units=inbound_units, lead_time_days=7
            )

            if dist_df.empty:
                st.info("No demand history found across regions for this SKU – cannot run logistics optimization.")
            else:
                st.dataframe(dist_df, use_container_width=True)

                selected_row = dist_df[dist_df["Region"] == region].iloc[0]
                final_stock = selected_row["Current_Stock"] + selected_row["Inbound_Allocated"]
                coverage_ratio = final_stock / (selected_row["LT_Demand_7d"] or 1)
                shortage_after = selected_row["Shortage_After"]

                if shortage_after > 0:
                    extra_reorder = shortage_after * 1.2
                    banner_class = "stock-banner stock-high"
                    banner_text = (
                        f"🛑 <strong>HIGH STOCKOUT RISK</strong> in <strong>{region}</strong>. "
                        f"7-day demand ≈ <strong>{selected_row['LT_Demand_7d']:.1f}</strong> units, "
                        f"but stock after allocation is only <strong>{final_stock:.1f}</strong> units. "
                        f"Expected shortfall ≈ <strong>{shortage_after:.1f}</strong> units. "
                        f"→ Consider an extra reorder of ~<strong>{extra_reorder:.0f} units</strong>."
                    )
                elif coverage_ratio < 1.2:
                    extra_reorder = selected_row["LT_Demand_7d"] * 0.2
                    banner_class = "stock-banner stock-medium"
                    banner_text = (
                        f"⚠️ <strong>WATCH LEVELS</strong> in <strong>{region}</strong>. "
                        f"Stock after allocation just covers the 7-day demand "
                        f"(coverage ratio ≈ <strong>{coverage_ratio:.2f}</strong>). "
                        f"→ You may top up by ~<strong>{extra_reorder:.0f} units</strong> as a buffer."
                    )
                else:
                    banner_class = "stock-banner stock-low"
                    banner_text = (
                        f"✅ <strong>LOW STOCKOUT RISK</strong> in <strong>{region}</strong>. "
                        f"Stock after allocation comfortably covers the 7-day forecast "
                        f"(coverage ratio ≈ <strong>{coverage_ratio:.2f}</strong>). "
                        f"No urgent extra reorder is required for this region."
                    )

                st.markdown(
                    f'<div class="{banner_class}">{banner_text}</div>',
                    unsafe_allow_html=True,
                )

                worst_row = dist_df.sort_values("Shortage_After", ascending=False).iloc[0]
                best_row_log = dist_df.sort_values("Excess_After", ascending=False).iloc[0]

                st.caption(
                    f"Inbound stock of {inbound_units:.0f} units is pushed first to regions with the biggest gaps. "
                    f"{worst_row['Region']} remains the tightest region, while {best_row_log['Region']} has "
                    f"the most comfortable coverage. This shows how AI-driven allocation reduces stockouts "
                    f"and improves delivery speed for Enchanto perfumes."
                )
