import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import json
from datetime import date
import matplotlib.pyplot as plt

# =========================================================
# Helper utilities (robust + fail-safe)
# =========================================================

def _clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a clean, numeric price DataFrame with date index."""
    if df is None or len(df) == 0:
        return pd.DataFrame()

    # If MultiIndex columns (e.g., yf with multiple tickers), select price level
    if isinstance(df.columns, pd.MultiIndex):
        for lvl in ["Adj Close", "Close"]:
            if lvl in df.columns.get_level_values(0):
                df = df[lvl].copy()
                break
        else:
            df = df.xs(df.columns.levels[0][-1], level=0, axis=1).copy()

    # Keep only numeric columns
    df = df.apply(pd.to_numeric, errors="coerce")
    # Drop columns that are entirely NaN
    df = df.dropna(axis=1, how="all")
    # Fill small gaps and drop remaining NaNs
    df = df.ffill().bfill().dropna(how="any")
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
    df = df.sort_index()
    return df


def _safe_yf_download(tickers, start=None, end=None, period=None) -> pd.DataFrame:
    """Download prices safely and return a cleaned DataFrame or empty if failed."""
    try:
        if period:
            raw = yf.download(tickers, period=period, auto_adjust=False, progress=False)
        else:
            raw = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)
    except Exception:
        return pd.DataFrame()

    if isinstance(raw, pd.DataFrame) and not raw.empty:
        if "Adj Close" in raw.columns:
            out = raw["Adj Close"].copy()
        elif "Close" in raw.columns:
            out = raw["Close"].copy()
        else:
            if isinstance(raw.columns, pd.MultiIndex):
                for lvl in ["Adj Close", "Close"]:
                    try:
                        out = raw.xs(lvl, level=0, axis=1)
                        break
                    except Exception:
                        out = raw.copy()
            else:
                out = raw.copy()
        return _clean_prices(out)
    return pd.DataFrame()


def simulate_stock_price(S0: float, mu: float, sigma: float, days: int):
    """Simulate stock price using geometric Brownian motion (Euler)."""
    dt = 1 / 252
    prices = [S0]
    for _ in range(days - 1):
        S_prev = prices[-1]
        dS = S_prev * (mu * dt + sigma * np.sqrt(dt) * np.random.normal())
        prices.append(S_prev + dS)
    return prices


def simulate_universe(names, days=252, seed=42) -> pd.DataFrame:
    np.random.seed(seed)
    data = {}
    for name in names:
        S0 = np.random.randint(50, 150)
        mu = np.random.uniform(0.05, 0.2)
        sigma = np.random.uniform(0.1, 0.4)
        data[name] = simulate_stock_price(S0, mu, sigma, days)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    return pd.DataFrame(data, index=idx)


def normalize_to_100(df: pd.DataFrame) -> pd.DataFrame:
    return 100 * df.div(df.iloc[0])


def calculate_portfolio_value(weights: np.ndarray, prices_df: pd.DataFrame) -> pd.Series:
    if prices_df.empty:
        return pd.Series(dtype=float)
    norm = normalize_to_100(prices_df)
    weighted = norm.mul(weights, axis=1)
    return weighted.sum(axis=1)


def calculate_risk_metrics(value_index: pd.Series):
    if value_index is None or value_index.empty or len(value_index) < 2:
        return (np.nan, np.nan, np.nan, np.nan)
    rets = value_index.pct_change().dropna()
    if len(rets) == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    vol = rets.std() * np.sqrt(252)
    exp_ret = rets.mean() * 252
    sharpe = exp_ret / vol if vol != 0 else np.nan
    roll_max = value_index.cummax()
    drawdown = value_index / roll_max - 1.0
    max_dd = drawdown.min()
    return (float(exp_ret), float(vol), float(sharpe), float(max_dd))

# =========================================================
# UI – resilient data pipeline with graceful fallbacks
# =========================================================

st.set_page_config(page_title="Portfolio Simulator Pro", layout="wide")
st.title("📈 Portfolio Simulator & Risk Lab — Pro Edition")
st.caption("Live data • Uploads • Simulation • Optimization • Monte Carlo • Heatmaps • Correlations • Benchmarks • Dividends • Forecast • Sentiment • Multi-user")

# Session init
if "saved_portfolios" not in st.session_state:
    st.session_state.saved_portfolios = {}

# Sidebar controls
st.sidebar.header("⚙️ Data & Modes")
source = st.sidebar.selectbox("Primary data source", ["Yahoo Finance (live)", "Upload file", "Simulated data"]) 

# Yahoo inputs
if source == "Yahoo Finance (live)":
    tickers_in = st.sidebar.text_input("Tickers (space/comma)", "AAPL MSFT TSLA AMZN GOOGL")
    tickers = [t.strip().upper() for t in tickers_in.replace(",", " ").split() if t.strip()]
    period = st.sidebar.selectbox("History period", ["6mo", "1y", "2y", "5y"]) 
else:
    tickers, period = [], None

# Upload option is always available as override
uploaded = st.sidebar.file_uploader("Upload CSV/XLSX (Date index, columns=tickers)", type=["csv", "xlsx"]) 

# Fallback toggle
use_backup_if_empty = st.sidebar.checkbox("Auto-fallback to simulated data", value=True)

# Dividends toggle
st.sidebar.markdown("---")
use_dividends = st.sidebar.checkbox("Apply dividend yield & reinvest", value=False)

# Educational mode
edu_mode = st.sidebar.checkbox("Educational tooltips", value=False)

# Fetch / clean data
prices_df = pd.DataFrame()
if uploaded is not None:
    try:
        if uploaded.name.lower().endswith(".csv"):
            temp = pd.read_csv(uploaded, index_col=0, parse_dates=True)
        else:
            temp = pd.read_excel(uploaded, index_col=0, parse_dates=True)
        prices_df = _clean_prices(temp)
        if prices_df.empty:
            st.warning("Uploaded file had no usable numeric price data. Falling back…")
    except Exception as e:
        st.error(f"Failed to read the uploaded file: {e}")

if prices_df.empty and source == "Yahoo Finance (live)" and tickers:
    dl = _safe_yf_download(tickers, period=period)
    if dl.empty:
        st.warning("Yahoo Finance returned no usable data. Will try fallback.")
    prices_df = dl

if prices_df.empty:
    if use_backup_if_empty or source == "Simulated data":
        st.info("Using simulated prices as fallback.")
        prices_df = simulate_universe(["TechCorp", "HealthInc", "FinBank", "EnergyCo", "RetailMart"], days=252)
    else:
        st.error("No data available. Upload a file or choose live data.")
        st.stop()

# ===== Main layout
left, right = st.columns([2, 1])

with left:
    st.subheader("📊 Price History")
    st.line_chart(prices_df)

with right:
    st.subheader("🎯 Portfolio Allocation")
    alloc = {}
    for col in prices_df.columns:
        alloc[col] = st.slider(f"{col} %", 0, 100, 20)
    total = sum(alloc.values())
    if total == 0:
        st.warning("All allocations are zero. Using equal weights.")
        weights = np.array([1.0 / len(alloc)] * len(alloc))
    else:
        weights = np.array([w / total for w in alloc.values()])

    # Save / Load portfolios (JSON)
    st.markdown("**Save / Load Portfolio**")
    pname = st.text_input("Name this portfolio", "MyPortfolio")
    if st.button("💾 Save weights"):
        st.session_state.saved_portfolios[pname] = dict(alloc)
        st.success(f"Saved as '{pname}'.")
    if st.session_state.saved_portfolios:
        pick = st.selectbox("Load saved portfolio", list(st.session_state.saved_portfolios.keys()))
        if st.button("📂 Load weights"):
            loaded = st.session_state.saved_portfolios.get(pick, {})
            st.info(f"Loaded '{pick}'. Move any slider to refresh weights.")
            # NOTE: Streamlit sliders can't be programmatically set after creation without rerun/state pattern.

# Apply dividends (simple model): constant annual yield per ticker
if use_dividends:
    st.subheader("💸 Dividend Yield (annual, %) per asset")
    dy = {}
    for col in prices_df.columns:
        dy[col] = st.slider(f"{col} dividend %", 0.0, 10.0, 2.0, 0.1)
else:
    dy = {c: 0.0 for c in prices_df.columns}

# Compute portfolio value index with optional dividend reinvestment
norm_prices = normalize_to_100(prices_df)
# Approximate daily dividend return as annual_yield/252
div_matrix = pd.DataFrame({c: (dy[c] / 100.0) / 252 for c in prices_df.columns}, index=norm_prices.index)
# Total return index: price return + dividend yield compounding
price_ret = prices_df.pct_change().fillna(0)
tri = (1 + price_ret + div_matrix).cumprod()
tri_norm = 100 * tri.div(tri.iloc[0])

value_index = tri_norm.mul(weights, axis=1).sum(axis=1)

st.subheader("📈 Portfolio Performance (with dividends if enabled)")
st.line_chart(value_index)

exp_ret, vol, sharpe, max_dd = calculate_risk_metrics(value_index)
cols = st.columns(4)
cols[0].metric("Expected Return (ann.)", f"{exp_ret:.2%}" if pd.notna(exp_ret) else "–")
cols[1].metric("Volatility (ann.)", f"{vol:.2%}" if pd.notna(vol) else "–")
cols[2].metric("Sharpe Ratio", f"{sharpe:.2f}" if pd.notna(sharpe) else "–")
cols[3].metric("Max Drawdown", f"{max_dd:.2%}" if pd.notna(max_dd) else "–")

# =========================================================
# Benchmarks (SPY/QQQ/VOO + S&P 500 index)
# =========================================================

st.subheader("📌 Benchmarks Comparison")
bench_tickers = ["^GSPC", "SPY", "QQQ", "VOO"]
bench = _safe_yf_download(bench_tickers, start=prices_df.index.min(), end=prices_df.index.max())
if bench.empty:
    st.warning("Benchmark data unavailable. Skipping.")
else:
    bench_norm = normalize_to_100(bench)
    comp = pd.concat([value_index.rename("Portfolio"), bench_norm.reindex(value_index.index, method="ffill")], axis=1)
    st.line_chart(comp)

# =========================================================
# Optimization (Max Sharpe) + Risk/Return Heatmap
# =========================================================

st.subheader("🔍 Optimization & Heatmap")
opt_cols = st.columns([1,1,2])
with opt_cols[0]:
    n_portfolios = st.number_input("Random portfolios", min_value=500, max_value=10000, value=3000, step=500)
with opt_cols[1]:
    risk_free = st.number_input("Risk-free rate (ann.)", min_value=0.0, max_value=0.1, value=0.0, step=0.001)

returns = prices_df.pct_change().dropna()
if not returns.empty and returns.shape[1] > 0:
    mean_ret = returns.mean() * 252
    cov = returns.cov() * 252

    best = {"sharpe": -np.inf, "w": None, "ret": None, "vol": None}
    pts = []  # (vol, ret, sharpe)
    ws = []
    for _ in range(int(n_portfolios)):
        w = np.random.random(len(mean_ret))
        w /= w.sum()
        pr = float(np.dot(w, mean_ret))
        pv = float(np.sqrt(np.dot(w.T, np.dot(cov, w))))
        sh = (pr - risk_free) / pv if pv != 0 else -np.inf
        pts.append((pv, pr, sh))
        ws.append(w)
        if sh > best["sharpe"]:
            best = {"sharpe": sh, "w": w, "ret": pr, "vol": pv}

    # Scatter heatmap (Sharpe color)
    fig, ax = plt.subplots(figsize=(6,4))
    sc = ax.scatter([p[0] for p in pts], [p[1] for p in pts], c=[p[2] for p in pts])
    ax.set_xlabel("Volatility (ann.)")
    ax.set_ylabel("Return (ann.)")
    ax.set_title("Risk/Return – Random Portfolios (color = Sharpe)")
    plt.colorbar(sc, ax=ax, label="Sharpe")
    st.pyplot(fig)

    st.success("✅ Optimal portfolio (Max Sharpe)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Return", f"{best['ret']:.2%}")
    c2.metric("Volatility", f"{best['vol']:.2%}")
    c3.metric("Sharpe", f"{best['sharpe']:.2f}")
    st.write("**Weights:**")
    for name, w in zip(prices_df.columns, best["w"]):
        st.write(f"- {name}: {w:.2%}")
else:
    st.warning("Not enough data for optimization.")

# =========================================================
# Correlation Matrix
# =========================================================

st.subheader("🧩 Correlation Matrix (daily returns)")
if returns.empty or returns.shape[1] < 2:
    st.info("Need at least 2 assets for correlation.")
else:
    corr = returns.corr()
    st.dataframe(corr.style.format("{:.2f}"))
    fig2, ax2 = plt.subplots(figsize=(5,4))
    im = ax2.imshow(corr.values, aspect='auto')
    ax2.set_xticks(range(len(corr.columns)))
    ax2.set_yticks(range(len(corr.columns)))
    ax2.set_xticklabels(corr.columns, rotation=45, ha='right')
    ax2.set_yticklabels(corr.columns)
    ax2.set_title("Correlation Heatmap")
    plt.colorbar(im, ax=ax2)
    st.pyplot(fig2)

# =========================================================
# Monte Carlo Simulation (Portfolio future paths)
# =========================================================

st.subheader("🎲 Monte Carlo Simulation (1-year forward)")
mc_cols = st.columns([1,1,1,1])
with mc_cols[0]:
    mc_days = st.number_input("Days ahead", min_value=60, max_value=756, value=252, step=21)
with mc_cols[1]:
    mc_paths = st.number_input("Paths", min_value=100, max_value=5000, value=1000, step=100)
with mc_cols[2]:
    start_value = st.number_input("Start value", min_value=10.0, value=float(value_index.iloc[-1]), step=10.0)
with mc_cols[3]:
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)

np.random.seed(int(seed))
mu_vec = returns.mean().values  # daily mean
cov_mat = returns.cov().values  # daily cov
w_vec = weights
port_mu = float(np.dot(mu_vec, w_vec))
port_var = float(np.dot(w_vec.T, np.dot(cov_mat, w_vec)))
port_sigma = np.sqrt(port_var)

# Simulate geometric Brownian portfolio index (using portfolio stats)
# dS/S = mu*dt + sigma*sqrt(dt)*Z
if not np.isnan(port_sigma):
    dt = 1/252
    sims = np.zeros((int(mc_days)+1, int(mc_paths)))
    sims[0,:] = start_value
    for t in range(1, int(mc_days)+1):
        z = np.random.normal(size=int(mc_paths))
        sims[t,:] = sims[t-1,:] * (1 + port_mu*dt + port_sigma*np.sqrt(dt)*z)
    # Percentiles
    p5 = np.percentile(sims, 5, axis=1)
    p50 = np.percentile(sims, 50, axis=1)
    p95 = np.percentile(sims, 95, axis=1)
    mc_df = pd.DataFrame({"p05": p5, "median": p50, "p95": p95})
    st.line_chart(mc_df)
    st.caption("Shaded percentiles show downside/median/upside scenarios.")
else:
    st.warning("Monte Carlo not available due to insufficient variance data.")

# =========================================================
# Simple Forecast (no external ML) – linear trend extrapolation
# =========================================================

st.subheader("🔮 Simple Price Trend Forecast (no-ML)")
try:
    horizon = st.slider("Forecast days", 10, 90, 30)
    last_prices = prices_df.iloc[-30:]  # last 30 days
    forecast = {}
    idx_future = pd.bdate_range(prices_df.index[-1] + pd.Timedelta(days=1), periods=horizon)
    for c in prices_df.columns:
        y = last_prices[c].values
        x = np.arange(len(y))
        # linear fit
        a, b = np.polyfit(x, y, 1)
        yhat = a * (len(y) + np.arange(horizon)) + b
        forecast[c] = yhat
    fc_df = pd.DataFrame(forecast, index=idx_future)
    combo = pd.concat([prices_df.tail(60), fc_df])
    st.line_chart(combo)
except Exception as e:
    st.warning(f"Forecast skipped: {e}")

# =========================================================
# Ultra-simple Sentiment (manual paste of headlines)
# =========================================================

st.subheader("📰 Sentiment Sandbox (paste headlines – demo)")
text = st.text_area("Paste recent headlines here (one per line)")
if text.strip():
    pos_words = {"beat","surge","gain","growth","record","upgrade","strong","bull"}
    neg_words = {"miss","drop","fall","loss","downgrade","weak","bear","fraud","probe"}
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    score = 0
    for ln in lines:
        lw = ln.lower()
        score += sum(w in lw for w in pos_words)
        score -= sum(w in lw for w in neg_words)
    label = "Positive" if score>0 else ("Negative" if score<0 else "Neutral")
    st.metric("Sentiment", f"{label}")

# =========================================================
# Multi-user Competition (local session)
# =========================================================

st.subheader("🏆 Competition Mode (this session)")
comp_name = st.text_input("Competitor name", "Team A")
if st.button("Add competitor using current weights"):
    st.session_state.saved_portfolios[comp_name] = dict(alloc)
    st.success(f"Added {comp_name}")

if len(st.session_state.saved_portfolios) >= 2:
    st.write("Comparing saved teams (normalized to 100)")
    comp_df = pd.DataFrame()
    for name, wdict in st.session_state.saved_portfolios.items():
        w_vec_comp = np.array([wdict.get(col, 0) for col in prices_df.columns])
        if w_vec_comp.sum() == 0:
            w_vec_comp = np.array([1/len(prices_df.columns)]*len(prices_df.columns))
        else:
            w_vec_comp = w_vec_comp / w_vec_comp.sum()
        vi = normalize_to_100(prices_df).mul(w_vec_comp, axis=1).sum(axis=1)
        comp_df[name] = vi
    st.line_chart(comp_df)
    # Leaderboard by final value
    last_vals = comp_df.iloc[-1].sort_values(ascending=False)
    st.write("**Leaderboard (final index value):**")
    for i, (n, v) in enumerate(last_vals.items(), start=1):
        st.write(f"{i}. {n}: {v:.2f}")

# =========================================================
# Educational tooltips
# =========================================================

if edu_mode:
    with st.expander("What is a portfolio?"):
        st.write("A portfolio is a collection of assets. Your weights decide how much each asset contributes to total performance.")
    with st.expander("Sharpe Ratio"):
        st.write("Return per unit of risk (volatility). Higher is better, assuming a non-negative risk-free rate.")
    with st.expander("Correlation"):
        st.write("Measures how assets move together. Lower/negative correlations can improve diversification.")

# Downloads
st.subheader("⬇️ Download Portfolio Values")
export_df = pd.DataFrame({"Portfolio": value_index})
st.download_button("Download CSV", export_df.to_csv().encode("utf-8"), file_name="portfolio_values.csv", mime="text/csv")

st.success("All set! Explore modes, optimize, simulate, compare benchmarks, and more.")
