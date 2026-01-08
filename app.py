import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import cvxpy as cp
import matplotlib.pyplot as plt
from src.simpleStats import fetch_prices, daily_returns, ann_stats
from src.portfolioCalculations import *



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
    ["Minimum Variance","Max Return", "Risk/Return Tradeoff (return − gamma*sigma^2)", "Max Sharpe"],
)
long_only = st.sidebar.checkbox("Long only (no shorting)", value=True)



gamma = None
target_ret = None



if opt_mode == "Minimum Variance":
    min_returns = st.sidebar.checkbox("Minimum Returns", value=True)
    if min_returns:
        target_ret = st.sidebar.number_input("Target annual return", value=0.10, step=0.01, format="%.3f")
    else:
        target_ret = 0
elif opt_mode == "Risk/Return Tradeoff (return − gamma*sigma^2)":
    gamma = st.sidebar.slider("Risk aversion gamma", min_value=1.0, max_value=100.0, value=10.0, step=1.0)


st.sidebar.subheader("Alpha Calculations")
alpha = st.sidebar.checkbox( "Calculate Alpha of portfolio")
if alpha:
    benchmark = st.sidebar.text_input("Porfolio Benchmark", value="SPY")


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
            st.write("Annualized Mean")
            st.dataframe(mu.to_frame("μ").style.format("{:.2%}"))
        with c2:
            st.write("Annualized Covariance")
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

                elif opt_mode == "Minimum Variance":
                    weights = solve_min_var(cov.values, target_ret, mu.values, long_only=long_only)
                    chosen_label = f"Min Var @ return≥{target_ret:.2%}"

                elif opt_mode == "Risk/Return Tradeoff (return − gamma*sigma^2)":
                    weights = solve_tradeoff(mu.values, cov.values, gamma=gamma, long_only=long_only)
                    chosen_label = f"Tradeoff (gamma={gamma:.0f})"

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
                        st.write(f"Expected return: **{r:.2%}**  |  vol: **{v:.2%}**  |  Sharpe (rf={rf:.2%}): **{s_:.2f}**")

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

                        if alpha:
                            st.subheader("Alpha Calculation")
                            a = pd.DataFrame()
                            df, alpha = alpha_calc(benchmark, port_rets, start_date, end_date)
                            a['alpha']= df.cumsum()
                            a['port'] = port_rets.cumsum()

                            st.line_chart(a)

                            st.write(f"The alpha of this strategy is:  **{alpha:.5%}**")



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

                        st.header("Efficient Frontier of Portfolio")
                        # Efficient frontier graph.
                        if frontier_pts is not None:
                            fig = plt.figure()
                            plt.scatter(frontier_pts[:,1], frontier_pts[:,0])
                            plt.scatter([v], [r], marker="*", s=200)
                            plt.xlabel("Volatility"); plt.ylabel("Return")
                            plt.title("Efficient Frontier (annualized)")
                            st.pyplot(fig)

            except Exception as e:
                st.error(f"Error during optimization: {e}")


