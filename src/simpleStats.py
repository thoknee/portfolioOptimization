import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import cvxpy as cp
import matplotlib.pyplot as plt


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