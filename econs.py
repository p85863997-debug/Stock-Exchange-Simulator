import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import json
from datetime import date, datetime, timedelta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

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
            # Take the last level if Adj Close/Close not found
            try:
                df = df.xs(df.columns.levels[0][-1], level=0, axis=1).copy()
            except:
                df = df.iloc[:, :5].copy()  # Take first 5 columns as fallback

    # Keep only numeric columns
    numeric_cols = []
    for col in df.columns:
        try:
            pd.to_numeric(df[col], errors='raise')
            numeric_cols.append(col)
        except:
            continue
    
    if not numeric_cols:
        return pd.DataFrame()
    
    df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    
    # Drop columns that are entirely NaN
    df = df.dropna(axis=1, how="all")
    
    if df.empty:
        return pd.DataFrame()
    
    # Fill small gaps and drop remaining NaNs
    df = df.ffill().bfill().dropna(how="any")
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            # Create a default date range if index conversion fails
            df.index = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq='D')
    
    df = df.sort_index()
    return df


def _safe_yf_download(tickers, start=None, end=None, period=None) -> pd.DataFrame:
    """Download prices safely and return a cleaned DataFrame or empty if failed."""
    if not tickers:
        return pd.DataFrame()
    
    try:
        # Ensure tickers is a list
        if isinstance(tickers, str):
            tickers = [tickers]
        
        # Download data
        if period:
            raw = yf.download(tickers, period=period, auto_adjust=True, progress=False, threads=True)
        else:
            raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, threads=True)
        
        if raw is None or raw.empty:
            return pd.DataFrame()
            
    except Exception as e:
        st.warning(f"Yahoo Finance download failed: {str(e)}")
        return pd.DataFrame()

    try:
        if isinstance(raw, pd.DataFrame) and not raw.empty:
            # Handle single vs multiple tickers
            if len(tickers) == 1:
                # Single ticker - raw might not have MultiIndex
                if isinstance(raw.columns, pd.MultiIndex):
                    if "Adj Close" in raw.columns.get_level_values(0):
                        out = raw["Adj Close"].copy()
                    elif "Close" in raw.columns.get_level_values(0):
                        out = raw["Close"].copy()
                    else:
                        out = raw.iloc[:, -1:].copy()  # Take last column
                else:
                    # For single ticker, take the raw data if it's already price-like
                    out = raw.copy()
                    
                # Ensure it's a DataFrame with proper column name
                if isinstance(out, pd.Series):
                    out = out.to_frame(tickers[0])
                elif out.shape[1] == 1 and out.columns[0] != tickers[0]:
                    out.columns = [tickers[0]]
            else:
                # Multiple tickers
                if "Adj Close" in raw.columns.get_level_values(0):
                    out = raw["Adj Close"].copy()
                elif "Close" in raw.columns.get_level_values(0):
                    out = raw["Close"].copy()
                else:
                    # Try to extract price data from MultiIndex
                    try:
                        out = raw.xs("Close", level=0, axis=1)
                    except:
                        out = raw.copy()
            
            return _clean_prices(out)
    except Exception as e:
        st.warning(f"Data processing failed: {str(e)}")
        return pd.DataFrame()
    
    return pd.DataFrame()


def simulate_stock_price(S0: float, mu: float, sigma: float, days: int):
    """Simulate stock price using geometric Brownian motion (Euler)."""
    if days <= 0:
        return [S0]
    
    dt = 1 / 252
    prices = [S0]
    
    for _ in range(days - 1):
        S_prev = prices[-1]
        if S_prev <= 0:
            S_prev = 0.01  # Prevent negative prices
        dS = S_prev * (mu * dt + sigma * np.sqrt(dt) * np.random.normal())
        new_price = max(S_prev + dS, 0.01)  # Ensure positive prices
        prices.append(new_price)
    
    return prices


def simulate_universe(names, days=252, seed=42) -> pd.DataFrame:
    """Create simulated price data for given asset names."""
    if not names:
        names = ["Asset1", "Asset2", "Asset3"]
    
    np.random.seed(seed)
    data = {}
    
    for name in names:
        S0 = np.random.randint(50, 150)
        mu = np.random.uniform(0.05, 0.2)
        sigma = np.random.uniform(0.1, 0.4)
        data[name] = simulate_stock_price(S0, mu, sigma, days)
    
    # Create business day index ending today
    try:
        idx = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=days)
    except:
        idx = pd.date_range(end=pd.Timestamp.now(), periods=days, freq='D')
    
    return pd.DataFrame(data, index=idx)


def normalize_to_100(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize prices to start at 100."""
    if df.empty:
        return df
    return 100 * df.div(df.iloc[0])


def calculate_portfolio_value(weights: np.ndarray, prices_df: pd.DataFrame) -> pd.Series:
    """Calculate portfolio value time series."""
    if prices_df.empty or len(weights) != len(prices_df.columns):
        return pd.Series(dtype=float)
    
    norm = normalize_to_100(prices_df)
    weighted = norm.mul(weights, axis=1)
    return weighted.sum(axis=1)


def calculate_risk_metrics(value_index: pd.Series):
    """Calculate key risk metrics for a return series."""
    if value_index is None or value_index.empty or len(value_index) < 2:
        return (np.nan, np.nan, np.nan, np.nan)
    
    rets = value_index.pct_change().dropna()
    if len(rets) == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    
    # Annualized metrics
    vol = rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    exp_ret = rets.mean() * 252
    sharpe = exp_ret / vol if vol != 0 else 0
    
    # Maximum drawdown
    try:
        roll_max = value_index.cummax()
        drawdown = value_index / roll_max - 1.0
        max_dd = drawdown.min()
    except:
        max_dd = np.nan
    
    return (float(exp_ret), float(vol), float(sharpe), float(max_dd))


def safe_optimization(returns_df: pd.DataFrame, n_portfolios: int = 1000, risk_free: float = 0.0):
    """Perform portfolio optimization safely."""
    if returns_df.empty or returns_df.shape[1] < 2:
        return None, [], []
    
    try:
        mean_ret = returns_df.mean() * 252
        cov = returns_df.cov() * 252
        
        # Check for valid covariance matrix
        if np.any(np.isnan(cov.values)) or np.any(np.isinf(cov.values)):
            return None, [], []
        
        best = {"sharpe": -np.inf, "w": None, "ret": None, "vol": None}
        pts = []  # (vol, ret, sharpe)
        
        for _ in range(int(n_portfolios)):
            # Generate random weights
            w = np.random.random(len(mean_ret))
            w /= w.sum()  # Normalize to sum to 1
            
            try:
                pr = float(np.dot(w, mean_ret))
                pv = float(np.sqrt(np.dot(w.T, np.dot(cov, w))))
                
                if pv == 0:
                    sh = -np.inf
                else:
                    sh = (pr - risk_free) / pv
                
                if np.isnan(sh) or np.isinf(sh):
                    continue
                    
                pts.append((pv, pr, sh))
                
                if sh > best["sharpe"]:
                    best = {"sharpe": sh, "w": w.copy(), "ret": pr, "vol": pv}
                    
            except:
                continue
        
        return best, pts, (mean_ret, cov)
        
    except Exception as e:
        st.warning(f"Optimization failed: {str(e)}")
        return None, [], []


# =========================================================
# UI – resilient data pipeline with graceful fallbacks
# =========================================================

st.set_page_config(page_title="Portfolio Simulator Pro", layout="wide")
st.title("📈 Portfolio Simulator & Risk Lab — Pro Edition")
st.caption("Live data • Uploads • Simulation • Optimization • Monte Carlo • Heatmaps • Correlations • Benchmarks • Dividends • Forecast • Sentiment • Multi-user")

# Initialize session state
if "saved_portfolios" not in st.session_state:
    st.session_state.saved_portfolios = {}

# Sidebar controls
st.sidebar.header("⚙️ Data & Modes")
source = st.sidebar.selectbox("Primary data source", ["Yahoo Finance (live)", "Upload file", "Simulated data"]) 

# Yahoo inputs
tickers = []
period = None
if source == "Yahoo Finance (live)":
    tickers_in = st.sidebar.text_input("Tickers (space/comma separated)", "AAPL MSFT TSLA AMZN GOOGL")
    if tickers_in.strip():
        tickers = [t.strip().upper() for t in tickers_in.replace(",", " ").split() if t.strip()]
    period = st.sidebar.selectbox("History period", ["6mo", "1y", "2y", "5y"]) 

# Upload option
uploaded = st.sidebar.file_uploader("Upload CSV/XLSX (Date index, columns=tickers)", type=["csv", "xlsx"]) 

# Fallback toggle
use_backup_if_empty = st.sidebar.checkbox("Auto-fallback to simulated data", value=True)

# Additional options
st.sidebar.markdown("---")
use_dividends = st.sidebar.checkbox("Apply dividend yield & reinvestment", value=False)
edu_mode = st.sidebar.checkbox("Educational tooltips", value=False)

# =========================================================
# Data Loading Pipeline
# =========================================================

prices_df = pd.DataFrame()
data_source_used = "None"

# Try uploaded file first
if uploaded is not None:
    try:
        if uploaded.name.lower().endswith(".csv"):
            temp = pd.read_csv(uploaded, index_col=0, parse_dates=True)
        else:
            temp = pd.read_excel(uploaded, index_col=0, parse_dates=True)
        
        prices_df = _clean_prices(temp)
        if not prices_df.empty:
            data_source_used = f"Uploaded file ({uploaded.name})"
        else:
            st.warning("Uploaded file had no usable numeric price data.")
    except Exception as e:
        st.error(f"Failed to read the uploaded file: {str(e)}")

# Try Yahoo Finance if no upload or upload failed
if prices_df.empty and source == "Yahoo Finance (live)" and tickers:
    with st.spinner("Downloading data from Yahoo Finance..."):
        dl = _safe_yf_download(tickers, period=period)
        if not dl.empty:
            prices_df = dl
            data_source_used = f"Yahoo Finance ({', '.join(tickers)})"
        else:
            st.warning("Yahoo Finance returned no usable data.")

# Fallback to simulated data
if prices_df.empty:
    if use_backup_if_empty or source == "Simulated data":
        st.info("Using simulated data.")
        asset_names = tickers if tickers else ["TechCorp", "HealthInc", "FinBank", "EnergyCo", "RetailMart"]
        prices_df = simulate_universe(asset_names, days=252)
        data_source_used = "Simulated data"
    else:
        st.error("❌ No data available. Please upload a file, enter valid tickers, or enable auto-fallback.")
        st.stop()

# Display data info
st.success(f"✅ Data loaded: {data_source_used} | {len(prices_df)} days | {len(prices_df.columns)} assets")

# =========================================================
# Main Dashboard Layout
# =========================================================

left, right = st.columns([2, 1])

with left:
    st.subheader("📊 Price History")
    if not prices_df.empty:
        st.line_chart(prices_df)
    else:
        st.error("No price data to display")

with right:
    st.subheader("🎯 Portfolio Allocation")
    
    # Dynamic weight sliders
    alloc = {}
    total_weight = 0
    
    if not prices_df.empty:
        # Create sliders for each asset
        for i, col in enumerate(prices_df.columns):
            default_weight = 100 // len(prices_df.columns) if i < len(prices_df.columns) - 1 else 100 - total_weight
            alloc[col] = st.slider(f"{col} %", 0, 100, default_weight, key=f"weight_{col}")
            total_weight += alloc[col] if i < len(prices_df.columns) - 1 else 0
        
        # Display total allocation
        total = sum(alloc.values())
        if total == 0:
            st.warning("⚠️ All allocations are zero. Using equal weights.")
            weights = np.array([1.0 / len(alloc)] * len(alloc))
        else:
            st.info(f"Total allocation: {total}%")
            weights = np.array([w / total for w in alloc.values()])
        
        # Portfolio saving/loading
        st.markdown("**💾 Save / Load Portfolio**")
        pname = st.text_input("Portfolio name", "MyPortfolio")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save"):
                st.session_state.saved_portfolios[pname] = dict(alloc)
                st.success(f"✅ Saved '{pname}'")
        
        with col2:
            if st.session_state.saved_portfolios:
                pick = st.selectbox("Load saved", list(st.session_state.saved_portfolios.keys()))
                if st.button("📂 Load"):
                    loaded = st.session_state.saved_portfolios.get(pick, {})
                    st.info(f"📂 Loaded '{pick}'. Adjust sliders to apply.")
    else:
        st.error("No assets available for allocation")
        weights = np.array([])

# =========================================================
# Dividend Configuration
# =========================================================

if not prices_df.empty and use_dividends:
    st.subheader("💸 Dividend Yields (annual %)")
    div_cols = st.columns(min(len(prices_df.columns), 4))
    dy = {}
    
    for i, col in enumerate(prices_df.columns):
        with div_cols[i % len(div_cols)]:
            dy[col] = st.number_input(f"{col}", 0.0, 15.0, 2.0, 0.1, key=f"div_{col}")
else:
    dy = {c: 0.0 for c in prices_df.columns} if not prices_df.empty else {}

# =========================================================
# Portfolio Performance Calculation
# =========================================================

if not prices_df.empty and len(weights) == len(prices_df.columns):
    # Calculate total return index (price + dividends)
    if use_dividends and any(dy.values()):
        norm_prices = normalize_to_100(prices_df)
        # Simple dividend model: add daily dividend yield
        div_matrix = pd.DataFrame({c: (dy[c] / 100.0) / 252 for c in prices_df.columns}, index=norm_prices.index)
        price_ret = prices_df.pct_change().fillna(0)
        # Compound total return
        tri = (1 + price_ret + div_matrix).cumprod()
        tri_norm = 100 * tri.div(tri.iloc[0])
        value_index = tri_norm.mul(weights, axis=1).sum(axis=1)
        performance_label = "📈 Portfolio Performance (including dividends)"
    else:
        norm_prices = normalize_to_100(prices_df)
        value_index = norm_prices.mul(weights, axis=1).sum(axis=1)
        performance_label = "📈 Portfolio Performance"
    
    # Display performance chart
    st.subheader(performance_label)
    st.line_chart(value_index)
    
    # Calculate and display metrics
    exp_ret, vol, sharpe, max_dd = calculate_risk_metrics(value_index)
    
    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("📊 Annual Return", f"{exp_ret:.2%}" if pd.notna(exp_ret) else "N/A")
    with metric_cols[1]:
        st.metric("📈 Volatility", f"{vol:.2%}" if pd.notna(vol) else "N/A")
    with metric_cols[2]:
        st.metric("⚡ Sharpe Ratio", f"{sharpe:.2f}" if pd.notna(sharpe) else "N/A")
    with metric_cols[3]:
        st.metric("📉 Max Drawdown", f"{max_dd:.2%}" if pd.notna(max_dd) else "N/A")

else:
    st.warning("⚠️ Cannot calculate portfolio performance - check data and weights")
    value_index = pd.Series(dtype=float)

# =========================================================
# Benchmark Comparison
# =========================================================

if not prices_df.empty:
    st.subheader("📌 Benchmark Comparison")
    
    benchmark_options = st.multiselect(
        "Select benchmarks to compare",
        ["^GSPC", "SPY", "QQQ", "VOO", "VTI", "IWM"],
        default=["^GSPC", "SPY"]
    )
    
    if benchmark_options:
        with st.spinner("Loading benchmark data..."):
            bench = _safe_yf_download(
                benchmark_options, 
                start=prices_df.index.min(), 
                end=prices_df.index.max()
            )
        
        if not bench.empty and not value_index.empty:
            bench_norm = normalize_to_100(bench)
            # Align dates
            common_dates = value_index.index.intersection(bench_norm.index)
            if len(common_dates) > 0:
                comp_data = pd.concat([
                    value_index.loc[common_dates].rename("Portfolio"),
                    bench_norm.loc[common_dates]
                ], axis=1)
                st.line_chart(comp_data)
                
                # Benchmark performance table
                bench_metrics = []
                for col in bench_norm.columns:
                    ret, vol, sharpe, dd = calculate_risk_metrics(bench_norm[col])
                    bench_metrics.append({
                        "Benchmark": col,
                        "Return": f"{ret:.2%}" if pd.notna(ret) else "N/A",
                        "Volatility": f"{vol:.2%}" if pd.notna(vol) else "N/A",
                        "Sharpe": f"{sharpe:.2f}" if pd.notna(sharpe) else "N/A"
                    })
                
                if bench_metrics:
                    st.dataframe(pd.DataFrame(bench_metrics), hide_index=True)
            else:
                st.warning("No overlapping dates with benchmark data")
        else:
            st.info("Benchmark data not available")

# =========================================================
# Portfolio Optimization
# =========================================================

if not prices_df.empty and len(prices_df.columns) > 1:
    st.subheader("🎯 Portfolio Optimization")
    
    opt_cols = st.columns(3)
    with opt_cols[0]:
        n_portfolios = st.number_input("Random portfolios", 500, 10000, 2000, 500)
    with opt_cols[1]:
        risk_free = st.number_input("Risk-free rate", 0.0, 0.1, 0.02, 0.001, format="%.3f")
    with opt_cols[2]:
        if st.button("🚀 Optimize"):
            with st.spinner("Optimizing portfolio..."):
                returns = prices_df.pct_change().dropna()
                best, pts, extras = safe_optimization(returns, n_portfolios, risk_free)
                
                if best is not None and pts:
                    # Create efficient frontier plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    vols = [p[0] for p in pts]
                    rets = [p[1] for p in pts]
                    sharpes = [p[2] for p in pts]
                    
                    scatter = ax.scatter(vols, rets, c=sharpes, cmap='viridis', alpha=0.6)
                    ax.scatter(best["vol"], best["ret"], color='red', s=100, marker='*', 
                              label=f'Optimal (Sharpe={best["sharpe"]:.2f})')
                    
                    ax.set_xlabel("Annual Volatility")
                    ax.set_ylabel("Annual Return")
                    ax.set_title("Efficient Frontier - Risk vs Return")
                    ax.legend()
                    plt.colorbar(scatter, label="Sharpe Ratio")
                    st.pyplot(fig)
                    
                    # Display optimal weights
                    st.success("🏆 Optimal Portfolio (Maximum Sharpe)")
                    opt_metrics = st.columns(3)
                    with opt_metrics[0]:
                        st.metric("Return", f"{best['ret']:.2%}")
                    with opt_metrics[1]:
                        st.metric("Volatility", f"{best['vol']:.2%}")
                    with opt_metrics[2]:
                        st.metric("Sharpe", f"{best['sharpe']:.2f}")
                    
                    st.write("**Optimal Weights:**")
                    opt_weights_df = pd.DataFrame({
                        "Asset": prices_df.columns,
                        "Weight": [f"{w:.1%}" for w in best["w"]]
                    })
                    st.dataframe(opt_weights_df, hide_index=True)
                else:
                    st.error("Optimization failed - insufficient data or numerical issues")

# =========================================================
# Correlation Analysis
# =========================================================

if not prices_df.empty and len(prices_df.columns) > 1:
    st.subheader("🔗 Correlation Analysis")
    
    returns = prices_df.pct_change().dropna()
    if not returns.empty:
        corr = returns.corr()
        
        # Correlation matrix as heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(corr.values, cmap='RdYlBu', vmin=-1, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha='right')
        ax.set_yticklabels(corr.columns)
        
        # Add correlation values to cells
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                text = ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                              ha="center", va="center", color="black" if abs(corr.iloc[i, j]) < 0.5 else "white")
        
        ax.set_title("Asset Correlation Matrix")
        plt.colorbar(im, ax=ax, label="Correlation")
        st.pyplot(fig)
        
        # Display correlation table
        st.dataframe(corr.style.format("{:.3f}"))

# =========================================================
# Monte Carlo Simulation
# =========================================================

if not prices_df.empty and not value_index.empty:
    st.subheader("🎲 Monte Carlo Portfolio Simulation")
    
    mc_cols = st.columns(4)
    with mc_cols[0]:
        mc_days = st.number_input("Days to simulate", 30, 756, 252, 30)
    with mc_cols[1]:
        mc_paths = st.number_input("Number of paths", 100, 5000, 1000, 100)
    with mc_cols[2]:
        start_value = st.number_input("Starting value", 50.0, 200.0, float(value_index.iloc[-1]), 10.0)
    with mc_cols[3]:
        mc_seed = st.number_input("Random seed", 0, 999, 42, 1)
    
    if st.button("🎯 Run Monte Carlo"):
        with st.spinner("Running Monte Carlo simulation..."):
            returns = prices_df.pct_change().dropna()
            
            if not returns.empty:
                np.random.seed(int(mc_seed))
                
                # Portfolio-level statistics
                port_returns = (returns * weights).sum(axis=1)
                mu = port_returns.mean()
                sigma = port_returns.std()
                
                if not np.isnan(sigma) and sigma > 0:
                    # Generate paths
                    dt = 1
                    paths = np.zeros((int(mc_days) + 1, int(mc_paths)))
                    paths[0, :] = start_value
                    
                    for t in range(1, int(mc_days) + 1):
                        z = np.random.normal(0, 1, int(mc_paths))
                        paths[t, :] = paths[t-1, :] * (1 + mu + sigma * z)
                    
                    # Calculate percentiles
                    p5 = np.percentile(paths, 5, axis=1)
                    p25 = np.percentile(paths, 25, axis=1)
                    p50 = np.percentile(paths, 50, axis=1)
                    p75 = np.percentile(paths, 75, axis=1)
                    p95 = np.percentile(paths, 95, axis=1)
                    
                    # Create results dataframe
                    mc_df = pd.DataFrame({
                        "P05": p5,
                        "P25": p25,
                        "Median": p50,
                        "P75": p75,
                        "P95": p95
                    })
                    
                    st.line_chart(mc_df)
                    
                    # Summary statistics
                    final_values = paths[-1, :]
                    st.write("**Final Value Statistics:**")
                    summary_cols = st.columns(5)
                    with summary_cols[0]:
                        st.metric("5th %ile", f"{np.percentile(final_values, 5):.1f}")
                    with summary_cols[1]:
                        st.metric("25th %ile", f"{np.percentile(final_values, 25):.1f}")
                    with summary_cols[2]:
                        st.metric("Median", f"{np.percentile(final_values, 50):.1f}")
                    with summary_cols[3]:
                        st.metric("75th %ile", f"{np.percentile(final_values, 75):.1f}")
                    with summary_cols[4]:
                        st.metric("95th %ile", f"{np.percentile(final_values, 95):.1f}")
                else:
                    st.error("Insufficient data for Monte Carlo simulation")
            else:
                st.error("No return data available for simulation")

# =========================================================
# Simple Price Forecasting
# =========================================================

if not prices_df.empty:
    st.subheader("🔮 Simple Price Forecast")
    
    forecast_cols = st.columns(2)
    with forecast_cols[0]:
        forecast_days = st.slider("Forecast horizon (days)", 5, 90, 30)
    with forecast_cols[1]:
        lookback_days = st.slider("Lookback period (days)", 10, 100, 30)
    
    try:
        # Simple linear trend extrapolation
        recent_prices = prices_df.tail(lookback_days)
        forecasts = {}
        
        future_dates = pd.bdate_range(
            start=prices_df.index[-1] + pd.Timedelta(days=1),
            periods=forecast_days
        )
        
        for col in prices_df.columns:
            y = recent_prices[col].values
            x = np.arange(len(y))
            
            # Linear regression
            try:
                coeffs = np.polyfit(x, y, 1)
                future_x = np.arange(len(y), len(y) + forecast_days)
                forecast_values = np.polyval(coeffs, future_x)
                # Ensure positive prices
                forecast_values = np.maximum(forecast_values, 0.01)
                forecasts[col] = forecast_values
            except:
                # Fallback: use last price
                forecasts[col] = [recent_prices[col].iloc[-1]] * forecast_days
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame(forecasts, index=future_dates)
        
        # Combine historical and forecast
        combined_df = pd.concat([
            prices_df.tail(60),  # Last 60 days of actual prices
            forecast_df
        ])
        
        st.line_chart(combined_df)
        st.caption("📈 Blue: Historical prices | Red: Linear trend forecast")
        
    except Exception as e:
        st.warning(f"Forecast generation failed: {str(e)}")

# =========================================================
# Sentiment Analysis Sandbox
# =========================================================

st.subheader("📰 Market Sentiment Analyzer")
st.caption("Paste news headlines to get a simple sentiment score")

sentiment_text = st.text_area("Paste news headlines (one per line)", height=100, 
                             placeholder="Example:\nApple beats earnings expectations\nTech stocks surge on AI optimism\nMarket volatility increases amid uncertainty")

if sentiment_text.strip():
    # Simple keyword-based sentiment
    positive_words = {
        'beat', 'beats', 'surge', 'surges', 'gain', 'gains', 'growth', 'record', 
        'upgrade', 'strong', 'bull', 'bullish', 'rise', 'rises', 'up', 'higher',
        'profit', 'profits', 'success', 'optimism', 'rally', 'boost', 'positive'
    }
    
    negative_words = {
        'miss', 'misses', 'drop', 'drops', 'fall', 'falls', 'loss', 'losses',
        'downgrade', 'weak', 'bear', 'bearish', 'down', 'lower', 'decline',
        'fraud', 'probe', 'concern', 'worry', 'crash', 'plunge', 'negative'
    }
    
    lines = [line.strip() for line in sentiment_text.strip().split('\n') if line.strip()]
    
    positive_count = 0
    negative_count = 0
    total_words = 0
    
    for line in lines:
        words = line.lower().split()
        total_words += len(words)
        
        for word in words:
            # Remove punctuation
            clean_word = ''.join(c for c in word if c.isalnum())
            if clean_word in positive_words:
                positive_count += 1
            elif clean_word in negative_words:
                negative_count += 1
    
    # Calculate sentiment score
    if positive_count + negative_count > 0:
        sentiment_score = (positive_count - negative_count) / (positive_count + negative_count)
        
        if sentiment_score > 0.2:
            sentiment_label = "🟢 Positive"
            sentiment_color = "green"
        elif sentiment_score < -0.2:
            sentiment_label = "🔴 Negative"
            sentiment_color = "red"
        else:
            sentiment_label = "🟡 Neutral"
            sentiment_color = "orange"
    else:
        sentiment_score = 0
        sentiment_label = "⚪ Neutral"
        sentiment_color = "gray"
    
    # Display results
    sent_cols = st.columns(4)
    with sent_cols[0]:
        st.metric("Sentiment", sentiment_label)
    with sent_cols[1]:
        st.metric("Score", f"{sentiment_score:.2f}")
    with sent_cols[2]:
        st.metric("Positive words", positive_count)
    with sent_cols[3]:
        st.metric("Negative words", negative_count)

# =========================================================
# Multi-User Competition Mode
# =========================================================

if not prices_df.empty:
    st.subheader("🏆 Portfolio Competition")
    
    comp_cols = st.columns(2)
    with comp_cols[0]:
        competitor_name = st.text_input("Team/User name", "Team Alpha")
    with comp_cols[1]:
        if st.button("🎯 Add to Competition"):
            if competitor_name.strip():
                st.session_state.saved_portfolios[competitor_name] = dict(alloc)
                st.success(f"✅ Added {competitor_name} to competition!")
    
    # Display competition leaderboard
    if len(st.session_state.saved_portfolios) >= 1:
        st.write("**🏁 Competition Dashboard**")
        
        competition_data = {}
        performance_metrics = []
        
        for name, portfolio_weights in st.session_state.saved_portfolios.items():
            # Convert portfolio weights to numpy array
            w_array = np.array([portfolio_weights.get(col, 0) for col in prices_df.columns])
            
            if w_array.sum() == 0:
                w_array = np.ones(len(prices_df.columns)) / len(prices_df.columns)
            else:
                w_array = w_array / w_array.sum()
            
            # Calculate portfolio performance
            portfolio_value = calculate_portfolio_value(w_array, prices_df)
            
            if not portfolio_value.empty:
                competition_data[name] = portfolio_value
                
                # Calculate metrics for leaderboard
                ret, vol, sharpe, dd = calculate_risk_metrics(portfolio_value)
                final_value = portfolio_value.iloc[-1] if len(portfolio_value) > 0 else 100
                
                performance_metrics.append({
                    "Team": name,
                    "Final Value": f"{final_value:.1f}",
                    "Return": f"{ret:.1%}" if pd.notna(ret) else "N/A",
                    "Volatility": f"{vol:.1%}" if pd.notna(vol) else "N/A",
                    "Sharpe": f"{sharpe:.2f}" if pd.notna(sharpe) else "N/A",
                    "Max DD": f"{dd:.1%}" if pd.notna(dd) else "N/A"
                })
        
        if competition_data:
            # Performance chart
            comp_df = pd.DataFrame(competition_data)
            st.line_chart(comp_df)
            
            # Leaderboard table
            if performance_metrics:
                metrics_df = pd.DataFrame(performance_metrics)
                # Sort by final value (convert to float for sorting)
                metrics_df['Final Value Float'] = pd.to_numeric(metrics_df['Final Value'], errors='coerce')
                metrics_df = metrics_df.sort_values('Final Value Float', ascending=False)
                metrics_df = metrics_df.drop('Final Value Float', axis=1)
                
                st.write("**🥇 Leaderboard**")
                st.dataframe(metrics_df, hide_index=True)
                
                # Winner announcement
                if len(metrics_df) > 0:
                    winner = metrics_df.iloc[0]['Team']
                    st.success(f"🏆 Current Leader: **{winner}**")

# =========================================================
# Educational Content
# =========================================================

if edu_mode:
    st.subheader("📚 Educational Resources")
    
    with st.expander("💼 What is a Portfolio?"):
        st.write("""
        A **portfolio** is a collection of financial investments like stocks, bonds, or other assets.
        
        Key concepts:
        - **Diversification**: Spreading investments across different assets to reduce risk
        - **Asset Allocation**: How you divide your money among different types of investments
        - **Risk vs Return**: Generally, higher potential returns come with higher risk
        """)
    
    with st.expander("📊 Understanding Risk Metrics"):
        st.write("""
        **Sharpe Ratio**: Measures return per unit of risk. Higher is better.
        - Formula: (Return - Risk-free rate) / Volatility
        - Good Sharpe: > 1.0, Excellent: > 2.0
        
        **Volatility**: Measures how much prices bounce around. Lower usually preferred.
        
        **Maximum Drawdown**: The largest peak-to-trough decline. Shows worst-case scenario.
        """)
    
    with st.expander("🔗 Correlation Explained"):
        st.write("""
        **Correlation** measures how assets move together:
        - **+1.0**: Perfect positive correlation (move together)
        - **0.0**: No correlation (independent movement)  
        - **-1.0**: Perfect negative correlation (move opposite)
        
        Lower correlations between assets can improve diversification benefits.
        """)
    
    with st.expander("🎯 Portfolio Optimization"):
        st.write("""
        **Modern Portfolio Theory** seeks to maximize return for a given level of risk.
        
        The **Efficient Frontier** shows the best possible risk/return combinations.
        
        **Maximum Sharpe Portfolio** offers the best risk-adjusted returns.
        """)

# =========================================================
# Export and Download Options
# =========================================================

st.subheader("💾 Export Data")

if not value_index.empty:
    # Prepare export data
    export_data = pd.DataFrame({
        "Date": value_index.index,
        "Portfolio_Value": value_index.values
    })
    
    # Add individual asset prices if available
    if not prices_df.empty:
        for col in prices_df.columns:
            export_data[f"{col}_Price"] = prices_df[col].reindex(value_index.index).values
    
    # Download buttons
    export_cols = st.columns(3)
    
    with export_cols[0]:
        csv_data = export_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Download CSV",
            data=csv_data,
            file_name=f"portfolio_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with export_cols[1]:
        # Portfolio weights as JSON
        weights_data = {
            "timestamp": datetime.now().isoformat(),
            "weights": dict(alloc) if 'alloc' in locals() else {},
            "total_allocation": sum(alloc.values()) if 'alloc' in locals() else 0
        }
        weights_json = json.dumps(weights_data, indent=2).encode('utf-8')
        st.download_button(
            label="⚙️ Download Weights",
            data=weights_json,
            file_name=f"portfolio_weights_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )
    
    with export_cols[2]:
        if st.button("🗑️ Clear Saved Portfolios"):
            st.session_state.saved_portfolios = {}
            st.success("✅ Cleared all saved portfolios")

# =========================================================
# Footer and Status
# =========================================================

st.markdown("---")

# Display system status
status_cols = st.columns(4)
with status_cols[0]:
    st.metric("📊 Data Points", len(prices_df) if not prices_df.empty else 0)
with status_cols[1]:
    st.metric("💼 Assets", len(prices_df.columns) if not prices_df.empty else 0)
with status_cols[2]:
    st.metric("💾 Saved Portfolios", len(st.session_state.saved_portfolios))
with status_cols[3]:
    data_age = (datetime.now() - prices_df.index[-1]).days if not prices_df.empty else "N/A"
    st.metric("📅 Data Age (days)", data_age)

# Success message and tips
if not prices_df.empty:
    st.success("🎉 Portfolio Simulator is fully operational!")
    
    with st.expander("💡 Pro Tips"):
        st.write("""
        **Getting Started:**
        1. 🔄 Try different data sources (Yahoo Finance, upload, or simulated)
        2. ⚖️ Experiment with different weight allocations
        3. 📊 Compare against benchmarks like S&P 500
        4. 🎯 Use optimization to find efficient portfolios
        5. 🎲 Run Monte Carlo to see potential outcomes
        
        **Advanced Features:**
        - 💸 Enable dividends for total return calculation
        - 🏆 Create competition between different strategies  
        - 📰 Use sentiment analysis for market timing ideas
        - 💾 Save and load different portfolio configurations
        
        **Data Quality:**
        - Ensure uploaded files have dates as index
        - Use standard ticker symbols for Yahoo Finance
        - Check for missing data that might affect calculations
        """)
else:
    st.error("❌ Please load valid data to use the Portfolio Simulator")

st.caption("Built with Streamlit • Data from Yahoo Finance • For educational purposes only")
st.caption("⚠️ This tool is for educational and research purposes. Not financial advice.")
