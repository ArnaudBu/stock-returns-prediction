#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyfolio as pf
from pandas.plotting import table
from datetime import datetime

# Import prices
prices_list = [pd.read_csv(f"data/sp500/prices_daily.csv"),
               pd.read_csv(f"data/nyse/prices_daily.csv"),
               pd.read_csv(f"data/nasdaq/prices_daily.csv")
               ]
for i in range(3):
    prices_list[i]['Date'] = pd.to_datetime(prices_list[i]['Date'])
    prices_list[i].set_index(['Date'], inplace=True)
prices = prices_list[0]
for i in range(1, 3):
    cols_to_use = prices_list[i].columns.difference(prices.columns)
    prices = pd.merge(prices,
                      prices_list[i][cols_to_use],
                      left_index=True,
                      right_index=True,
                      how='outer')

# Load data
probas = pd.read_csv(f"backtest/probas.csv")
# Format date
probas['Date'] = pd.to_datetime(probas['Date'])
probas.set_index(['Date'], inplace=True)

# Filter prices
prices = prices[prices.index >= min(probas.index)]
probas[prices.isnull()] = np.nan
probas[probas == 0] = np.nan

# Returns on prices
returns = prices.pct_change()


# Softmax function
def softmax(x):
    e_x = np.exp(x - np.nanmax(x))
    return e_x / np.nansum(e_x)


# Order and select function
def orderselect(x, n=100):
    order = (-np.array(x)).argsort()
    return [1/min(n, len(x)) if i in order[:n] else np.nan for i in range(len(x))]


# nasdaq
nasdaq_returns = returns['^IXIC']
nasdaq_returns = (1 + nasdaq_returns).cumprod()
nasdaq_returns.name = "Nasdaq"

# s&p
sp_returns = returns['^GSPC']
sp_returns = (1 + sp_returns).cumprod()
sp_returns.name = "S&P 500"

# Softmax strategy
weights = probas.apply(softmax, axis=1)
softmaxstrat = returns.multiply(weights).apply(np.nansum, axis=1)
softmaxstrat = (1 + softmaxstrat).cumprod()
softmaxstrat.name = "Softmax"

# Order and select
weights = probas.apply(lambda x: orderselect(x, 100),
                       axis=1,
                       result_type="expand")
weights.columns = probas.columns
strat100 = returns.multiply(weights).apply(np.nansum, axis=1)
strat100 = (1 + strat100).cumprod()
strat100.name = "Best 100"

# Analysis on best performing stocks
stocks = returns.copy()
stocks = stocks[stocks.index > datetime(year=2020, month=1, day=1)]
stocks_full = stocks.copy()
stocks[weights.isna()] = 0
cols = (1+stocks).cumprod().iloc[-1].nlargest(10).index.tolist()
stocks = (1 + stocks[cols]).cumprod()
stocks_full = (1 + stocks_full[cols]).cumprod()

# Order and select
weights = probas.apply(lambda x: orderselect(x, 1000),
                       axis=1,
                       result_type="expand")
weights.columns = probas.columns
strat1000 = returns.multiply(weights).apply(np.nansum, axis=1)
strat1000 = (1 + strat1000).cumprod()
strat1000.name = "Best 1000"

# Plot
fig, ax = plt.subplots(figsize=(16, 8))
softmaxstrat.plot(ax=ax, color="darkorange")
strat100.plot(ax=ax, color="dodgerblue")
strat1000.plot(ax=ax, color="seagreen")
nasdaq_returns.plot(ax=ax, color="red")
sp_returns.plot(ax=ax, color="purple")
plt.legend(loc="best")
ax.set_ylabel("Cummulative return")
ax.set_title("Backtest based on the data from 2018 to 2021", fontsize=20)

# Test on the numbers
try:
    d = pd.read_csv(f"backtest/yields.csv")
    a = max(d.nb)+1
except Exception as e:
    d = pd.DataFrame()
    a = 1

for i in range(a, 2001):
    weights = probas.apply(lambda x: orderselect(x, i),
                           axis=1,
                           result_type="expand")
    weights.columns = probas.columns
    strat = returns.multiply(weights).apply(np.nansum, axis=1)
    strat = (1 + strat).cumprod()
    data = pd.DataFrame({"nb": [i], "yield": [strat.iloc[-1]]})
    d = d.append(data)
    print(f"{i}: {strat.iloc[-1]}")
    d.to_csv(f"backtest/yields.csv", index=False)

# Plot
fig, ax = plt.subplots(figsize=(16, 8))
d['yield'][d['nb'] < 10000].plot(ax=ax, color="dodgerblue")
# plt.legend(loc="best")
ax.set_ylabel("cummulative return")
ax.set_title("Returns for a strategy consisting of the x most promising stocks",
             fontsize=20)

# Tear sheet
weights = probas.apply(softmax, axis=1)
softmaxstrat = returns.multiply(weights).apply(np.nansum, axis=1)
softmax = pf.timeseries.perf_stats(softmaxstrat)
weights = probas.apply(lambda x: orderselect(x, 100),
                       axis=1,
                       result_type="expand")
weights.columns = probas.columns
top100strat = returns.multiply(weights).apply(np.nansum, axis=1)
top100 = pf.timeseries.perf_stats(top100strat)
weights = probas.apply(lambda x: orderselect(x, 1000),
                       axis=1,
                       result_type="expand")
weights.columns = probas.columns
top1000strat = returns.multiply(weights).apply(np.nansum, axis=1)
top1000 = pf.timeseries.perf_stats(top1000strat)
nasdaq = pf.timeseries.perf_stats(returns['^IXIC'])
sp500 = pf.timeseries.perf_stats(returns['^GSPC'])

tearsheet = pd.concat({'Softmax': softmax,
                       'Top 100': top100,
                       'Top 1000': top1000,
                       'Nasdaq': nasdaq,
                       'S&P 500': sp500
                       }, axis=1)
tearsheet = tearsheet.round(2)
tearsheet.fillna("", inplace=True)
ax = plt.subplot(111, frame_on=False)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
table(ax, tearsheet)

# Plot comparision

stocks.plot(ylabel="Cumulative returns", title="Cumulative returns with model")

stocks_full.plot(ylabel="Cumulative returns", title="Cumulative returns without model")

# Plot
fig, ax = plt.subplots()
stocks['OESX'].plot(ax=ax, color="darkorange", label="With model")
stocks_full['OESX'].plot(ax=ax, color="dodgerblue", label="Without model")
plt.legend(loc="best")
ax.set_ylabel("Cummulative return")
ax.set_title("Returns comparison from the use of the model for OESX stock", fontsize=20)

# Plot
fig, ax = plt.subplots()
stocks['SAVA'].plot(ax=ax, color="darkorange", label="With model")
stocks_full['SAVA'].plot(ax=ax, color="dodgerblue", label="Without model")
plt.legend(loc="best")
ax.set_ylabel("Cummulative return")
ax.set_title("Returns comparison from the use of the model for SAVA stock", fontsize=20)
