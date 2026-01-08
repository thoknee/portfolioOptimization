import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import cvxpy as cp
import matplotlib.pyplot as plt
import statsmodels.api as sm
from src.simpleStats import fetch_prices, daily_returns, ann_stats


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


# Value at risk and conditional value at risk calculations. (historical)
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


def alpha_calc(beta, portfolio: pd.Series, start, end):
    
    benchmark = daily_returns(yf.download(beta, start=start, end=end)["Close"])

    df = pd.concat([portfolio, benchmark], axis=1).dropna()
    df.columns = ["portfolio", "benchmark"]

    X = sm.add_constant(df["benchmark"])
    Y = df["portfolio"]

    regression = sm.OLS(Y, X).fit()
    
    residuals = regression.resid
    return residuals, (regression.params['const'])
    




