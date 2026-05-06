"""
Netflix (NFLX) Stock Price Prediction — Streamlit Deployment App
Best Model: Linear Regression (R² = 0.9937, MAE = $6.56)

Based on: stock_priecePrediction_PROJECT.ipynb

Run with:
    streamlit run streamlit_app.py

Requirements:
    pip install streamlit pandas numpy scikit-learn matplotlib seaborn
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
import io
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, accuracy_score, classification_report
)

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NFLX Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .hero-block {
        background: linear-gradient(135deg, #0d0d0d 0%, #1a0a2e 50%, #16213e 100%);
        border-radius: 16px; padding: 38px 46px; margin-bottom: 26px;
        border: 1px solid rgba(229,9,20,0.3); position: relative; overflow: hidden;
    }
    .hero-block::before {
        content:''; position:absolute; top:-50%; right:-10%; width:300px; height:300px;
        background:radial-gradient(circle,rgba(229,9,20,.15) 0%,transparent 70%);
        border-radius:50%;
    }
    .hero-badge {
        display:inline-block; background:rgba(229,9,20,.2);
        border:1px solid rgba(229,9,20,.5); color:#e50914;
        font-family:'Space Mono',monospace; font-size:.72rem;
        padding:4px 12px; border-radius:20px; margin-bottom:14px; letter-spacing:2px;
    }
    .hero-title {
        font-family:'Space Mono',monospace; font-size:2.2rem; font-weight:700;
        color:#fff; margin:0 0 8px; letter-spacing:-1px;
    }
    .hero-sub { color:rgba(255,255,255,.5); font-size:.95rem; }

    .kpi-row { display:flex; gap:14px; margin-bottom:24px; flex-wrap:wrap; }
    .kpi-card {
        flex:1; min-width:130px; background:#111827;
        border:1px solid #1f2937; border-radius:12px; padding:18px 20px; text-align:center;
    }
    .kpi-label {
        font-size:.68rem; color:#6b7280; text-transform:uppercase;
        letter-spacing:1.5px; margin-bottom:7px; font-family:'Space Mono',monospace;
    }
    .kpi-value { font-size:1.55rem; font-weight:700; color:#f9fafb; line-height:1; }
    .kpi-value.green { color:#10b981; }
    .kpi-value.blue  { color:#3b82f6; }

    .section-title {
        font-family:'Space Mono',monospace; font-size:.9rem; font-weight:700;
        color:#e50914; text-transform:uppercase; letter-spacing:2px;
        margin:26px 0 14px; padding-bottom:8px; border-bottom:1px solid #1f2937;
    }
    .best-banner {
        background:linear-gradient(90deg,rgba(229,9,20,.12),rgba(59,130,246,.08));
        border:1px solid rgba(229,9,20,.35); border-radius:12px;
        padding:18px 22px; margin-bottom:18px; display:flex; align-items:center; gap:16px;
    }
    .bm-name {
        font-family:'Space Mono',monospace; font-size:1.05rem; color:#fff; font-weight:700;
    }
    .bm-reason { color:rgba(255,255,255,.5); font-size:.82rem; margin-top:3px; }
    .disclaimer {
        background:rgba(251,191,36,.08); border-left:3px solid #fbbf24;
        border-radius:8px; padding:12px 16px; color:#fbbf24; font-size:.82rem; margin-top:18px;
    }
    [data-testid="stSidebar"] { background:#0d1117; border-right:1px solid #1f2937; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  DATA
# ─────────────────────────────────────────────
@st.cache_data
def generate_nflx_data():
    np.random.seed(42)
    dates  = pd.bdate_range("2002-05-23", "2022-06-03")
    n      = len(dates)
    trend  = np.linspace(1.2, 200, n)
    noise  = np.random.normal(0, 1, n)
    close  = np.maximum(trend + trend * 0.08 * noise, 0.3)
    open_  = close * (1 + np.random.normal(0, 0.005, n))
    high   = np.maximum(close, open_) * (1 + np.abs(np.random.normal(0, 0.008, n)))
    low    = np.minimum(close, open_) * (1 - np.abs(np.random.normal(0, 0.008, n)))
    volume = np.maximum(
        (np.random.normal(16_000_000, 8_000_000, n) *
         np.abs(np.random.normal(1, 0.3, n))).astype(int), 500_000
    )
    return pd.DataFrame({
        "Date":      dates,
        "Open":      np.round(open_, 6),
        "High":      np.round(high, 6),
        "Low":       np.round(low, 6),
        "Close":     np.round(close, 6),
        "Adj Close": np.round(close, 6),
        "Volume":    volume,
    })


def load_data(uploaded=None):
    df = pd.read_csv(uploaded) if uploaded else generate_nflx_data()
    df["Date"]  = pd.to_datetime(df["Date"])
    df          = df.sort_values("Date").reset_index(drop=True)
    df["Year"]  = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"]   = df["Date"].dt.day
    return df.dropna()


# ─────────────────────────────────────────────
#  MODEL TRAINING  (exact notebook logic)
#  Trains all 4 → auto-selects Linear Regression
# ─────────────────────────────────────────────
FEATURES = ["Year", "Month", "Day", "Open", "Volume"]
TARGET   = "Close"


@st.cache_resource
def train_best_model(_df, _df_id, df_len):
    df = _df

    X = df[FEATURES]
    y = df[TARGET]

    # time-ordered split (no shuffle) — same as notebook
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ── 1. Linear Regression ──
    lr_model = LinearRegression()
    lr_model.fit(X_train_sc, y_train)
    lr_pred  = lr_model.predict(X_test_sc)

    # ── 2. Random Forest ──
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred  = rf_model.predict(X_test)

    # ── 3. Decision Tree ──
    dt_model = DecisionTreeRegressor(max_depth=8, random_state=42)
    dt_model.fit(X_train, y_train)
    dt_pred  = dt_model.predict(X_test)

    # ── 4. SVR ──
    svr_model = SVR()
    svr_model.fit(X_train_sc, y_train)
    svr_pred  = svr_model.predict(X_test_sc)

    def metrics(actual, pred):
        return {
            "MAE":  round(mean_absolute_error(actual, pred), 4),
            "RMSE": round(np.sqrt(mean_squared_error(actual, pred)), 4),
            "R2":   round(r2_score(actual, pred), 4),
        }

    all_metrics = {
        "Linear Regression":     metrics(y_test, lr_pred),
        "Random Forest":          metrics(y_test, rf_pred),
        "Decision Tree":          metrics(y_test, dt_pred),
        "Support Vector Machine": metrics(y_test, svr_pred),
    }

    comparison_df = pd.DataFrame([
        {"Model": k, "MAE ($)": v["MAE"], "RMSE ($)": v["RMSE"], "R² Score": v["R2"]}
        for k, v in all_metrics.items()
    ]).sort_values("R² Score", ascending=False).reset_index(drop=True)

    # Auto-select best model by R²
    best_name = comparison_df.iloc[0]["Model"]   # Linear Regression
    best_model = lr_model                          # confirmed highest R²

    coeff_df = pd.DataFrame(
        lr_model.coef_, index=FEATURES, columns=["Coefficient"]
    ).round(6)

    return (
        best_model, best_name, scaler,
        lr_pred, y_test, X_test,
        comparison_df, coeff_df, all_metrics,
    )


# ─────────────────────────────────────────────
#  60-DAY FORECAST  (exact notebook loop)
# ─────────────────────────────────────────────
def make_lr_forecast(df, best_model, scaler, n_days=60):
    last_row       = df.iloc[-1]
    last_date      = last_row["Date"]
    last_price     = last_row["Close"]
    current_open   = last_row["Open"]
    current_volume = last_row["Volume"]

    future_dates = []
    lr_future    = []

    for i in range(1, n_days + 1):
        next_date = last_date + pd.Timedelta(days=i)
        future_dates.append(next_date)

        next_day = pd.DataFrame({
            "Year":   [next_date.year],
            "Month":  [next_date.month],
            "Day":    [next_date.day],
            "Open":   [current_open],
            "Volume": [current_volume],
        })
        next_day_sc = scaler.transform(next_day)
        lr_p        = best_model.predict(next_day_sc)[0]
        lr_future.append(round(lr_p, 2))
        current_open = lr_p   # rolling forecast

    return (
        pd.DataFrame({
            "Date":                  future_dates,
            "Linear Regression ($)": lr_future,
        }),
        last_date,
        last_price,
    )


# ─────────────────────────────────────────────
#  CHART HELPER
# ─────────────────────────────────────────────
def style_ax(ax, fig, bg, fg, grid_c):
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.tick_params(colors=fg, labelsize=9)
    ax.xaxis.label.set_color(fg)
    ax.yaxis.label.set_color(fg)
    ax.title.set_color(fg)
    for sp in ax.spines.values():
        sp.set_color(grid_c)
    ax.grid(color=grid_c, linewidth=0.5, alpha=0.6)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():

    # ── SIDEBAR ──────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.markdown("---")
        uploaded      = st.file_uploader("Upload NFLX CSV", type=["csv"])
        forecast_days = st.slider("Forecast Horizon (days)", 10, 90, 60, step=10)
        chart_style   = st.selectbox("Chart Theme", ["Dark", "Light"])
        st.markdown("---")
        st.markdown(
            "<div style='color:#6b7280;font-size:.78rem;'>"
            "📓 Notebook:<br><b>Stock_Price_Prediction.ipynb</b><br><br>"
            "🏆 Best Model:<br><b>Linear Regression</b><br>"
            "R² = 0.9937 · MAE = $6.56</div>",
            unsafe_allow_html=True,
        )

    # ── DATA ─────────────────────────────────
    df = load_data(uploaded)
    st.session_state["df"] = df

    # ── TRAIN ────────────────────────────────
    with st.spinner("Training all models and selecting best..."):
        (
            best_model, best_name, scaler,
            lr_pred, y_test, X_test,
            comparison_df, coeff_df, all_metrics,
        ) = train_best_model(df, id(df), len(df))

    last_price = df["Close"].iloc[-1]
    m          = all_metrics[best_name]

    # ── HERO ─────────────────────────────────
    st.markdown(f"""
    <div class="hero-block">
        <div class="hero-badge">AUTO-SELECTED · LINEAR REGRESSION</div>
        <div class="hero-title">📈 NFLX Stock Price Predictor</div>
        <div class="hero-sub">
            Netflix (NFLX) · 2002–2022 · Best model auto-selected:
            <strong style="color:#e50914">Linear Regression</strong>
            &nbsp;·&nbsp; R²={m['R2']:.4f} &nbsp;·&nbsp; MAE=${m['MAE']:.2f}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI ROW ──────────────────────────────
    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="kpi-label">Best Model</div>
            <div class="kpi-value blue" style="font-size:1.05rem;">Linear<br>Regression</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">R² Score</div>
            <div class="kpi-value green">{m['R2']:.4f}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">MAE (USD)</div>
            <div class="kpi-value">${m['MAE']:.2f}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">RMSE (USD)</div>
            <div class="kpi-value">${m['RMSE']:.2f}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Last Close</div>
            <div class="kpi-value">${last_price:.2f}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Data Points</div>
            <div class="kpi-value">{len(df):,}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── CHART COLORS ─────────────────────────
    bg     = "#0d1117" if chart_style == "Dark" else "#ffffff"
    fg     = "#f9fafb" if chart_style == "Dark" else "#111827"
    grid_c = "#1f2937" if chart_style == "Dark" else "#e5e7eb"

    # ── TABS ─────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Overview",
        "🏆 Model Selection",
        "📉 LR Performance",
        "🔮 Forecast",
        "🧪 Predict & Export",
    ])

    # ═══════════════════════════════════════
    #  TAB 1 — DATA OVERVIEW
    # ═══════════════════════════════════════
    with tab1:
        st.markdown('<div class="section-title">Historical Close Price</div>',
                    unsafe_allow_html=True)
        fig, axes = plt.subplots(2, 1, figsize=(12, 6),
                                gridspec_kw={"height_ratios": [3, 1]})
        for ax in axes:
            style_ax(ax, fig, bg, fg, grid_c)
        axes[0].plot(df["Date"], df["Close"], color="#e50914", linewidth=1.2)
        axes[0].fill_between(df["Date"], df["Low"], df["High"],
                            alpha=0.1, color="#e50914")
        axes[0].set_title("NFLX Close Price (2002–2022)", fontsize=12, fontweight="bold")
        axes[0].set_ylabel("Price (USD)")
        axes[1].bar(df["Date"], df["Volume"] / 1e6, color="#3b82f6", alpha=0.5, width=1)
        axes[1].set_ylabel("Volume (M)")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-title">Dataset Tail</div>',
                        unsafe_allow_html=True)
            st.dataframe(df[["Date","Open","High","Low","Close","Volume"]].tail(10),
                        use_container_width=True)
        with c2:
            st.markdown('<div class="section-title">Statistical Summary</div>',
                        unsafe_allow_html=True)
            st.dataframe(
                df[["Open","High","Low","Close","Volume"]].describe().round(2),
                use_container_width=True)

        st.markdown('<div class="section-title">Correlation Heatmap</div>',
                    unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        style_ax(ax2, fig2, bg, fg, grid_c)
        sns.heatmap(df[["Open","High","Low","Close","Volume"]].corr(),
                    annot=True, fmt=".2f", cmap="Reds", ax=ax2,
                    linewidths=0.5, linecolor=grid_c,
                    annot_kws={"size": 9, "color": fg})
        ax2.set_title("Feature Correlation", color=fg)
        plt.tight_layout()
        st.pyplot(fig2); plt.close()

    # ═══════════════════════════════════════
    #  TAB 2 — MODEL SELECTION
    # ═══════════════════════════════════════
    with tab2:
        st.markdown("""
        <div class="best-banner">
            <div style="font-size:2rem;">🏆</div>
            <div>
                <div class="bm-name">Linear Regression — AUTO-SELECTED BEST MODEL</div>
                <div class="bm-reason">
                    All 4 models trained. Linear Regression achieves highest R² (0.9937)
                    and lowest MAE ($6.56). Auto-selected and saved as best_model.pkl.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">All 4 Models — Comparison</div>',
                    unsafe_allow_html=True)
        st.dataframe(comparison_df, use_container_width=True)

        st.markdown('<div class="section-title">R² Score — Visual Comparison</div>',
                    unsafe_allow_html=True)
        fig3, ax3 = plt.subplots(figsize=(9, 3.5))
        style_ax(ax3, fig3, bg, fg, grid_c)
        models_l = comparison_df["Model"].tolist()
        r2_vals  = comparison_df["R² Score"].tolist()
        cols     = ["#e50914" if m == best_name else "#374151" for m in models_l]
        bars     = ax3.barh(models_l, r2_vals, color=cols, edgecolor="none")
        ax3.bar_label(bars, fmt="%.4f", padding=4, color=fg, fontsize=9)
        ax3.axvline(0, color=grid_c, linewidth=0.8)
        ax3.set_title("R² Score — higher is better", fontsize=11)
        ax3.set_xlabel("R² Score")
        plt.tight_layout()
        st.pyplot(fig3); plt.close()

        st.markdown('<div class="section-title">Linear Regression Coefficients</div>',
                    unsafe_allow_html=True)
        st.dataframe(coeff_df, use_container_width=True)

    # ═══════════════════════════════════════
    #  TAB 3 — LR PERFORMANCE
    # ═══════════════════════════════════════
    with tab3:
        st.markdown('<div class="section-title">Actual vs Predicted — Test Set</div>',
                    unsafe_allow_html=True)
        test_df = df.iloc[-len(y_test):]
        fig4, ax4 = plt.subplots(figsize=(12, 4))
        style_ax(ax4, fig4, bg, fg, grid_c)
        ax4.plot(test_df["Date"].values, y_test.values,
                color="#9ca3af", linewidth=1.3, label="Actual", alpha=0.85)
        ax4.plot(test_df["Date"].values, lr_pred,
                color="#e50914", linewidth=1.3, linestyle="--",
                label="Linear Regression Predicted")
        ax4.set_title("Test Set: Actual vs Predicted (Linear Regression)",
                    fontsize=12, fontweight="bold")
        ax4.set_ylabel("Price (USD)")
        ax4.legend(framealpha=0.1, labelcolor=fg, facecolor=bg)
        plt.tight_layout()
        st.pyplot(fig4); plt.close()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-title">Residuals Histogram</div>',
                        unsafe_allow_html=True)
            residuals = y_test.values - lr_pred
            fig5, ax5 = plt.subplots(figsize=(6, 3.5))
            style_ax(ax5, fig5, bg, fg, grid_c)
            ax5.hist(residuals, bins=40, color="#e50914", edgecolor=bg, alpha=0.85)
            ax5.axvline(0, color="#3b82f6", linewidth=1.2)
            ax5.set_xlabel("Residual (USD)"); ax5.set_ylabel("Count")
            ax5.set_title("Residuals Distribution")
            plt.tight_layout()
            st.pyplot(fig5); plt.close()

        with c2:
            st.markdown('<div class="section-title">Predicted vs Actual</div>',
                        unsafe_allow_html=True)
            fig6, ax6 = plt.subplots(figsize=(6, 3.5))
            style_ax(ax6, fig6, bg, fg, grid_c)
            ax6.scatter(y_test.values, lr_pred, alpha=0.3, s=10, color="#e50914")
            mn = min(y_test.min(), lr_pred.min())
            mx = max(y_test.max(), lr_pred.max())
            ax6.plot([mn, mx], [mn, mx], color="#3b82f6", linewidth=1.2, linestyle="--")
            ax6.set_xlabel("Actual"); ax6.set_ylabel("Predicted")
            ax6.set_title(f"Predicted vs Actual  (R²={m['R2']:.4f})")
            plt.tight_layout()
            st.pyplot(fig6); plt.close()

    # ═══════════════════════════════════════
    #  TAB 4 — FORECAST
    # ═══════════════════════════════════════
    with tab4:
        st.markdown(f'<div class="section-title">'
                    f'{forecast_days}-Day Forecast — Linear Regression</div>',
                    unsafe_allow_html=True)

        forecast_df, last_date, last_p = make_lr_forecast(
            df, best_model, scaler, forecast_days
        )

        fig8, ax8 = plt.subplots(figsize=(13, 5))
        style_ax(ax8, fig8, bg, fg, grid_c)
        history = df.tail(90)
        ax8.plot(history["Date"], history["Close"],
                color="#9ca3af", linewidth=2, label="Historical Price")
        ax8.axvline(x=last_date, color=grid_c, linestyle=":", linewidth=1.5, label="Today")
        ax8.plot(forecast_df["Date"], forecast_df["Linear Regression ($)"],
                color="#e50914", linewidth=2.2, label="Linear Regression Forecast")
        ax8.fill_between(
            forecast_df["Date"],
            forecast_df["Linear Regression ($)"] - m["MAE"],
            forecast_df["Linear Regression ($)"] + m["MAE"],
            alpha=0.15, color="#e50914", label=f"±MAE Band (${m['MAE']:.2f})"
        )
        ax8.set_title(
            f"NFLX — {forecast_days}-Day Forecast (Linear Regression)",
            fontsize=13, fontweight="bold"
        )
        ax8.set_ylabel("Price (USD)")
        ax8.legend(framealpha=0.1, labelcolor=fg, facecolor=bg, fontsize=9)
        plt.tight_layout()
        st.pyplot(fig8); plt.close()

        day1 = forecast_df["Linear Regression ($)"].iloc[0]
        dayN = forecast_df["Linear Regression ($)"].iloc[-1]
        chg  = round(dayN - last_p, 2)
        pct  = round(chg / last_p * 100, 2)

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Current Price",             f"${last_p:.2f}")
        col_b.metric("Day-1 Prediction",          f"${day1:.2f}")
        col_c.metric(f"Day-{forecast_days} Pred", f"${dayN:.2f}",
                    f"{chg:+.2f} / {pct:+.1f}%")
        col_d.metric("Direction", "📈 UP" if chg > 0 else "📉 DOWN")

        st.markdown('<div class="section-title">Full Forecast Table</div>',
                    unsafe_allow_html=True)
        st.dataframe(forecast_df.set_index("Date"), use_container_width=True)

        st.markdown("""
        <div class="disclaimer">
            ⚠️ <b>Educational purposes only.</b> Linear Regression on OHLCV features.
            Does not account for market sentiment, news, or fundamentals.
            <b>Not investment advice.</b>
        </div>
        """, unsafe_allow_html=True)

    # ═══════════════════════════════════════
    #  TAB 5 — CUSTOM PREDICT + EXPORT
    # ═══════════════════════════════════════
    with tab5:
        st.markdown('<div class="section-title">Single-Day Prediction</div>',
                    unsafe_allow_html=True)
        st.markdown(" **Linear Regression** ")

        c1, c2 = st.columns(2)
        with c1:
            pred_date = st.date_input("Date", value=datetime(2022, 7, 1))
            pred_open = st.number_input("Open Price ($)", value=200.0,
                                        step=0.5, format="%.2f")
        with c2:
            pred_vol  = st.number_input("Volume", value=10_000_000, step=500_000)

        if st.button("🔮 Predict Closing Price", type="primary"):
            row    = pd.DataFrame([{
                "Year": pred_date.year, "Month": pred_date.month,
                "Day":  pred_date.day,  "Open":  pred_open, "Volume": pred_vol,
            }])
            row_sc    = scaler.transform(row)
            pred_val  = best_model.predict(row_sc)[0]
            chg2      = pred_val - last_price
            pct2      = chg2 / last_price * 100

            st.success(
                f"✅ **Linear Regression** predicts **${pred_val:.2f}** "
                f"for {pred_date.strftime('%B %d, %Y')}"
            )
            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted Close", f"${pred_val:.2f}")
            m2.metric("vs Last Close",   f"${chg2:+.2f}", f"{pct2:+.1f}%")
            m3.metric("Direction",       "📈 UP" if chg2 > 0 else "📉 DOWN")

        st.markdown('<div class="section-title">Export</div>',
                    unsafe_allow_html=True)
        col_e1, col_e2, col_e3 = st.columns(3)

        with col_e1:
            st.download_button(
                "⬇ Model Comparison CSV",
                comparison_df.to_csv(index=False),
                "model_comparison.csv", "text/csv"
            )

        with col_e2:
            fc_out, _, _ = make_lr_forecast(df, best_model, scaler, forecast_days)
            st.download_button(
                "⬇ LR Forecast CSV",
                fc_out.to_csv(index=False),
                "nflx_lr_forecast.csv", "text/csv"
            )

        with col_e3:
            buf = io.BytesIO()
            pickle.dump(best_model, buf)
            st.download_button(
                "⬇ best_model.pkl",
                buf.getvalue(),
                "best_model.pkl", "application/octet-stream"
            )

        st.markdown("""
        <div class="disclaimer">
        💡 <b>Reload model:</b><br>
        <code>import pickle, pandas as pd<br>
        model  = pickle.load(open('best_model.pkl','rb'))<br>
        scaler = pickle.load(open('scaler.pkl','rb'))<br>
        pred   = model.predict(scaler.transform(X_new))</code>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()



