#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

from utils import process_data

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
data = pd.read_csv(f"data/data_clean.csv")
# Format date
data['date'] = pd.to_datetime(data['date'])
# Define variables
features = ['netIncome', 'grossProfit', 'ebit', 'totalRevenue', 'costOfRevenue', 'totalOtherIncomeExpenseNet', 'otherCurrentLiab', 'totalAssets', 'commonStock', 'otherLiab', 'otherAssets', 'cash', 'propertyPlantEquipment', 'accountsPayable', 'capitalSurplus', 'changeToLiabilities', 'totalCashflowsFromInvestingActivities', 'netBorrowings', 'totalCashFromFinancingActivities', 'changeInCash', 'totalCashFromOperatingActivities', 'depreciation', 'changeToNetincome', 'capitalExpenditures', 'changeToOperatingActivities', 'market_cap', 'div_percent', 'ebitAbs', 'totalRevenueAbs', 'yield', 'sector', 'outperform', 'positive' ]
target = 'outperform_next'

# Define dates
date_start = datetime(year=2018, month=1, day=1)
dates = list(set(data[data.date >= date_start]['date']))
dates.sort()
dates.append(max(dates) + relativedelta(years=10))

all_probas = []
for i in range(len(dates)-1):
    date = dates[i]
    date_next = dates[i+1]
    print(f"{date} - {date_next}")
    # Prepare model
    date_train = date - relativedelta(years=1)
    date_eval = date
    clf, X_train, y_train, X_eval, y_eval, _, _, indices = \
        process_data(data, date_train, date_eval, features, target)
    sbl = data.symbol[indices == "valid"].tolist()
    # Fit model
    clf.fit(X_train=X_train,
            y_train=y_train,
            eval_set=[(X_train, y_train)],
            eval_name=['train'],
            eval_metric=['auc'],
            max_epochs=50,
            patience=0,
            batch_size=1024,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False,
            )
    # Get relevant symbols
    prb = clf.predict_proba(X_eval)[:, 1]
    order = (-prb).argsort()
    symbols = {}
    for i in order:
        if sbl[i] not in symbols:
            symbols[sbl[i]] = prb[i]
    # Prices data for model testing
    prices_period = prices[prices.index < date_next]
    prices_period = prices_period[prices_period.index >= date]
    prices_period[~prices_period.isnull()] = 0
    prices_period.loc[:, symbols.keys()] = symbols.values()
    all_probas.append(prices_period)

probas = pd.concat(all_probas)
probas.to_csv(f"backtest/probas.csv")
