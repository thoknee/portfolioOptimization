import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import cvxpy as cp
import matplotlib.pyplot as plt

@st.cache_data(show_spinner=False)


def fetch_prices(tickers, start, end):
    df = yf.download(tickers, start=start, end=end)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame(name=tickers[0])
    return df.dropna(how="all")

def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="all")

def ann_stats(returns: pd.DataFrame):
    mu = returns.mean() * 252.0
    cov = returns.cov() * 252.0
    return mu, cov

def try_solvers(prob):
    for s in [cp.ECOS, cp.OSQP, cp.SCS]:
        try:
            prob.solve(solver=s, warm_start=True)
            if prob.status in ("optimal", "optimal_inaccurate"):
                return True
        except Exception:
            continue
    return False


# Finds weights of tickers for the highest possible return
def solve_max_return(mu, long_only=True):
    mu = np.asarray(mu)
    n = mu.size
    w = cp.Variable(n)
    cons = [cp.sum(w) == 1]
    if long_only:
        cons += [w >= 0]
    prob = cp.Problem(cp.Maximize(mu @ w), cons)
    ok = try_solvers(prob)
    return None if (not ok or w.value is None) else np.array(w.value).flatten()


# Finds the smallest possible variance
def solve_min_var(cov, target_ret, mu, long_only=True):
    mu = np.asarray(mu); cov = np.asarray(cov)
    n = mu.size
    w = cp.Variable(n)
    cons = [cp.sum(w) == 1, mu @ w >= target_ret]
    if long_only:
        cons += [w >= 0]
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cov)), cons)
    ok = try_solvers(prob)
    return None if (not ok or w.value is None) else np.array(w.value).flatten()


# Finds a tradeoff between returns and variance.
# The higher the gamma, the more variance plays a part.
# A small gamma means variance plays a small role and thus will mostly look to maximize returns
# A larger gamma will result in trying to minimize variance first.
# attempting to maximize returns - gamma*variance.
def solve_tradeoff(mu, cov, gamma=10.0, long_only=True):
    mu = np.asarray(mu); cov = np.asarray(cov)
    n = mu.size
    w = cp.Variable(n)
    cons = [cp.sum(w) == 1]
    if long_only:
        cons += [w >= 0]
    prob = cp.Problem(cp.Maximize(mu @ w - gamma * cp.quad_form(w, cov)), cons)
    ok = try_solvers(prob)
    return None if (not ok or w.value is None) else np.array(w.value).flatten()


# Uses mean variance optimization to find the most efficient frontier.
# The efficient frontier shows the set of optimal portfolios that provide 
# #the best possible expected return for the level of risk in the portfolio.
def efficient_frontier(mu, cov, n_points=30, long_only=True):
    mu_arr = np.asarray(mu)
    ret_min, ret_max = float(np.min(mu_arr)), float(np.max(mu_arr))
    targets = np.linspace(ret_min, ret_max, n_points)
    weights, pts = [], []
    for tr in targets:
        w = solve_min_var(cov, tr, mu_arr, long_only=long_only)
        if w is None:
            continue
        r, v, _ = portfolio_stats(w, mu_arr, cov)
        weights.append(w); pts.append((r, v))
    return (np.array(weights) if weights else None,
            np.array(pts) if pts else None)


# Finds stats of portfolio
def portfolio_stats(w, mu, cov, rf=0.0):
    w = np.asarray(w).reshape(-1)
    ret = float(mu @ w)
    vol = float(np.sqrt(max(0.0, w @ cov @ w)))
    sharpe = (ret - rf) / vol if vol > 0 else np.nan
    return ret, vol, sharpe

# Backtests according to the weights
def backtest_static(prices: pd.DataFrame, weights: np.ndarray):
    rets = prices.pct_change().dropna().copy()
    w = np.asarray(weights).reshape(-1)
    rets = rets.loc[:, prices.columns]
    port_rets = np.nan_to_num(rets.values) @ w
    curve = (1 + pd.Series(port_rets, index=rets.index)).cumprod()
    return pd.Series(port_rets, index=rets.index, name="port"), curve

# Finds maximum drawdown of portfolio
def max_drawdown(curve: pd.Series):
    roll_max = curve.cummax()
    drawdown = curve / roll_max - 1.0
    mdd = drawdown.min()
    return float(mdd), drawdown

# Sortino ratio
def sortino_ratio(port_rets: pd.Series, rf_annual=0.0, target_annual=0.0):
    rf_daily = rf_annual / 252.0
    mar_daily = target_annual / 252.0
    diff = port_rets - mar_daily - rf_daily*0
    downside = diff[diff < 0]
    dd = downside.std(ddof=0)
    ann_ret = port_rets.mean() * 252.0
    return ann_ret / (dd*np.sqrt(252)) if dd > 0 else np.nan

# Calmar ratio
def calmar_ratio(curve: pd.Series):
    ann_ret = (curve.iloc[-1] ** (252.0/len(curve)) - 1.0) if len(curve) > 0 else np.nan
    mdd, _ = max_drawdown(curve)
    return ann_ret / abs(mdd) if (mdd is not None and mdd < 0) else np.nan


# Value at risk and conditional value at risk calculations.
def hist_var_cvar(port_rets: pd.Series, horizon_days=1, tail_prob=0.05):
    if horizon_days <= 1:
        agg = port_rets
    else:
        agg = port_rets.rolling(horizon_days).sum().dropna()
    if len(agg) == 0:
        return np.nan, np.nan
    q = np.quantile(agg, tail_prob) 
    var_loss = -q
    cvar_loss = -agg[agg <= q].mean() if np.any(agg <= q) else np.nan
    return float(var_loss), float(cvar_loss)


# Calcultes var and cvar according to a normal distribution
def parametric_var_cvar_mc(port_mean_d, port_std_d, horizon_days=1, tail_prob=0.05, draws=100000, seed=0):
    rng = np.random.default_rng(seed)
    mu_h = port_mean_d * horizon_days
    sig_h = port_std_d * np.sqrt(horizon_days)
    samples = rng.normal(loc=mu_h, scale=sig_h, size=draws)
    q = np.quantile(samples, tail_prob)
    var_loss = -q
    cvar_loss = -samples[samples <= q].mean()
    return float(var_loss), float(cvar_loss)


# monte carlo simulation of your portfolio
def mc_sim(mu_ann, cov_ann, weights, horizon_days=126, n_paths=5000, seed=1):
    rng = np.random.default_rng(seed)
    w = np.asarray(weights).reshape(-1)
    n = len(w)

    mu_d = np.asarray(mu_ann) / 252.0
    cov_d = np.asarray(cov_ann) / 252.0

    jitter = 1e-10 * np.eye(n)
    cov_d_psd = cov_d + jitter
   
   # use cholesky factorization
    try:
        L = np.linalg.cholesky(cov_d_psd)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(cov_d)
        vals_clipped = np.clip(vals, 0, None)
        cov_d_psd = (vecs @ np.diag(vals_clipped) @ vecs.T) + jitter
        L = np.linalg.cholesky(cov_d_psd)

    # Random noise every day
    Z = rng.standard_normal((horizon_days, n_paths, n))
    # Mix this with covariance
    shocks = (Z @ L.T)

    # Use this to find daily returns
    daily_rets = shocks + mu_d
    # Find the portfolio daily return
    port_daily = daily_rets @ w 
    # Money made each day
    equity = np.cumprod(1 + port_daily, axis=0)
    # Last day is total money made by these returns
    end_vals = equity[-1]
    
    # Total returns
    horizon_rets = end_vals - 1.0
    # bottom 5%
    q = np.quantile(horizon_rets, 0.05)
    # loss level you only exceed in the worst 5% of cases.
    var5 = -q
    # average loss inside that worst 5%
    cvar5 = -horizon_rets[horizon_rets <= q].mean()

    # Find the drawdown
    roll_max = np.maximum.accumulate(equity, axis=0)
    dd = equity / roll_max - 1.0
    mdd_paths = dd.min(axis=0)

    # outputs
    out = {
        "end_vals": end_vals,
        "horizon_rets": horizon_rets,
        "VaR5": float(var5),
        "CVaR5": float(cvar5),
        "MDD_mean": float(mdd_paths.mean()),
        "MDD_p5": float(np.quantile(mdd_paths, 0.05)),
        "MDD_p50": float(np.quantile(mdd_paths, 0.50)),
        "MDD_p95": float(np.quantile(mdd_paths, 0.95)),
    }
    return out



st.set_page_config(page_title="Portfolio Optimization Dashboard", layout="wide")
st.title("Portfolio Optimization Dashboard")

st.sidebar.header("Data")
tickers_str = st.sidebar.text_input("Comma-separated tickers", value="AAPL, MSFT, GOOGL, AMZN, JNJ, PG, JPM")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2024-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.Timestamp.today().normalize())
rf = st.sidebar.number_input("Risk free rate (annual, dec.)", value=0.02, step=0.005, format="%.3f")

st.sidebar.header("Optimization")
opt_mode = st.sidebar.selectbox(
    "Mode",
    ["Max Return", "Min Variance (target μ)", "Risk/Return Tradeoff (μ − γσ²)", "Max Sharpe"],
)
long_only = st.sidebar.checkbox("Long only (no shorting)", value=True)

gamma = None
target_ret = None
if opt_mode == "Min Variance (target μ)":
    target_ret = st.sidebar.number_input("Target annual return (dec.)", value=0.10, step=0.01, format="%.3f")
elif opt_mode == "Risk–Return Tradeoff (μ − γσ²)":
    gamma = st.sidebar.slider("Risk aversion γ", min_value=1.0, max_value=100.0, value=10.0, step=1.0)

st.sidebar.header("Risk Settings")
tail_prob = st.sidebar.slider("Tail prob for VaR/CVaR", min_value=0.01, max_value=0.10, value=0.05, step=0.01)
horizon_days_var = st.sidebar.number_input("Horizon (days) for VaR/CVaR", min_value=1, value=20, step=1)
horizon_days_mc = st.sidebar.number_input("Horizon (days) for MC sim", min_value=10, value=126, step=1)
mc_paths = st.sidebar.number_input("# Monte Carlo paths", min_value=1000, value=5000, step=1000)

go = st.sidebar.button("Run")

if tickers_str:
    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
    prices = fetch_prices(tickers, start_date, end_date)
    if prices.empty:
        st.warning("No price data found for the chosen dates/tickers.")
    else:
        st.subheader("Price Data")
        st.dataframe(prices)
        rets_d = daily_returns(prices)

        st.subheader("General Stats")
        mu, cov = ann_stats(rets_d)

        c1, c2 = st.columns(2)
        with c1:
            st.write("Annualized Mean (μ)")
            st.dataframe(mu.to_frame("μ").style.format("{:.2%}"))
        with c2:
            st.write("Annualized Covariance (Σ)")
            st.dataframe(pd.DataFrame(cov, index=prices.columns, columns=prices.columns).style.format("{:.4f}"))

        st.write("Correlation Matrix")
        st.dataframe(rets_d.corr().style.format("{:.2f}"))

        weights = None
        frontier_pts = None
        chosen_label = None

        if go:
            try:
                if opt_mode == "Max Return":
                    weights = solve_max_return(mu.values, long_only=long_only)
                    chosen_label = "Max Return"

                elif opt_mode == "Min Variance (target μ)":
                    weights = solve_min_var(cov.values, target_ret, mu.values, long_only=long_only)
                    chosen_label = f"Min Var @ μ≥{target_ret:.2%}"

                elif opt_mode == "Risk/Return Tradeoff (μ − γσ²)":
                    weights = solve_tradeoff(mu.values, cov.values, gamma=gamma, long_only=long_only)
                    chosen_label = f"Tradeoff (γ={gamma:.0f})"

                elif opt_mode == "Max Sharpe":
                    W, pts = efficient_frontier(mu.values, cov.values, n_points=40, long_only=long_only)
                    if W is None or pts is None or len(W) == 0:
                        st.error("Efficient frontier failed to compute.")
                    else:
                        sharpes = (pts[:,0] - rf) / np.where(pts[:,1] > 0, pts[:,1], np.nan)
                        idx = np.nanargmax(sharpes)
                        weights = W[idx]
                        frontier_pts = pts
                        chosen_label = f"Frontier Max Sharpe (rf={rf:.2%})"

                if weights is None:
                    st.error("Optimization failed — no solution found.")
                else:
                    if long_only:
                        weights = np.maximum(weights, 0)
                    s = weights.sum()
                    if s <= 0:
                        st.error("Weights sum to zero; try different settings.")
                    else:
                        weights = weights / s

                        # Finds portfolio stats and prints them
                        r, v, s_ = portfolio_stats(weights, mu.values, cov.values, rf=rf)
                        st.success(f"Solution: {chosen_label}")
                        st.write(f"Expected μ: **{r:.2%}**  |  σ: **{v:.2%}**  |  Sharpe (rf={rf:.2%}): **{s_:.2f}**")

                        # optimized weights
                        w_df = pd.DataFrame({"Ticker": prices.columns, "Weight": weights})
                        st.write("Weights")
                        st.dataframe(w_df.style.format({"Weight": "{:.2%}"}))
                        
                        # pie chart of the weights
                        fig_w = plt.figure()
                        plt.pie(weights, labels=prices.columns, autopct='%1.1f%%')
                        plt.title("Portfolio Weights")
                        st.pyplot(fig_w)


                        # Porfolio returns and money made
                        port_rets, curve = backtest_static(prices, weights)
                        st.write("Backtest (Static Weights)")
                        st.line_chart(curve)

                        # Drawdowns
                        mdd, dd_series = max_drawdown(curve)
                        sharpe = (port_rets.mean()*252 - rf) / (port_rets.std()*np.sqrt(252)) if port_rets.std() > 0 else np.nan
                        sortino = sortino_ratio(port_rets, rf_annual=rf, target_annual=0.0)
                        calmar = calmar_ratio(curve)


                        # ratios
                        c3, c4, c5 = st.columns(3)
                        c3.metric("Sharpe", f"{sharpe:.2f}")
                        c4.metric("Sortino", f"{sortino:.2f}")
                        c5.metric("Calmar", f"{calmar:.2f}")

                        # Prints max drawdown
                        st.write(f"Max Drawdown: **{mdd:.2%}**")
                        st.line_chart(dd_series, height=180)


                        # Calculates value at risk and conditional value at risk
                        st.subheader("VaR / CVaR")
                        var_h, cvar_h = hist_var_cvar(port_rets, horizon_days=horizon_days_var, tail_prob=tail_prob)
                        port_mean_d, port_std_d = port_rets.mean(), port_rets.std(ddof=0)
                        var_p, cvar_p = parametric_var_cvar_mc(port_mean_d, port_std_d, horizon_days=horizon_days_var, tail_prob=tail_prob)

                        colv1, colv2 = st.columns(2)
                        with colv1:
                            st.markdown(f"**Historical ({int((1-tail_prob)*100)}% conf, {horizon_days_var}d)**")
                            st.write(f"VaR: **{var_h:.2%}**  |  CVaR: **{cvar_h:.2%}**")
                        with colv2:
                            st.markdown(f"**Parametric (Normal MC, {int((1-tail_prob)*100)}% conf, {horizon_days_var}d)**")
                            st.write(f"VaR: **{var_p:.2%}**  |  CVaR: **{cvar_p:.2%}**")

                        # Runs monte carlo simulation on our portfolio
                        st.subheader("Monte Carlo Simulation (Correlated, Multi-Asset)")
                        mc = mc_sim(mu, cov, weights, horizon_days=horizon_days_mc, n_paths=int(mc_paths))
                        
                        # Finds mean and std of portfolio
                        mean_ret_h = mc["horizon_rets"].mean()
                        std_ret_h = mc["horizon_rets"].std(ddof=0)

                        # Finds mean, std, and sharpe
                        ann_factor = 252 / horizon_days_mc
                        mc_mu = mean_ret_h * ann_factor
                        mc_sigma = std_ret_h * np.sqrt(ann_factor)
                        mc_sharpe = (mc_mu - rf) / mc_sigma if mc_sigma > 0 else np.nan
                        

                        # Find the value at risk, max drawdown, and the returns/sharpe
                        # through a montecarlo simulation.
                        c6, c7, c8, c9, c10, c11 = st.columns(6)
                        c6.metric(f"MC VaR5 ({horizon_days_mc}d)", f"{mc['VaR5']:.2%}")
                        c7.metric(f"MC CVaR5 ({horizon_days_mc}d)", f"{mc['CVaR5']:.2%}")
                        c8.metric("MDD (mean, MC)", f"{mc['MDD_mean']:.2%}")
                        c9.metric("MDD (5th pct, MC)", f"{mc['MDD_p5']:.2%}")
                        c10.metric("MC Exp. Return (annual)", f"{mc_mu:.2%}")
                        c11.metric("MC Sharpe (annual)", f"{mc_sharpe:.2f}")

                        # Histogram of returns
                        fig_hist = plt.figure()
                        plt.hist(mc["horizon_rets"], bins=50)
                        plt.title(f"MC Horizon Returns Distribution ({horizon_days_mc}d)")
                        plt.xlabel("Return"); plt.ylabel("Freq")
                        st.pyplot(fig_hist)

                        # Efficient frontier graph.
                        if frontier_pts is not None:
                            fig = plt.figure()
                            plt.scatter(frontier_pts[:,1], frontier_pts[:,0])
                            plt.scatter([v], [r], marker="*", s=200)
                            plt.xlabel("Volatility (σ)"); plt.ylabel("Return (μ)")
                            plt.title("Efficient Frontier (annualized)")
                            st.pyplot(fig)

            except Exception as e:
                st.error(f"Error during optimization: {e}")


