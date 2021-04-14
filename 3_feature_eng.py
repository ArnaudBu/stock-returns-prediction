#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import datetime
import numpy as np
import dtale
from tqdm import tqdm

# Load data
data = pd.concat([pd.read_csv(f"data/sp500/data.csv"),
                  pd.read_csv(f"data/nyse/data.csv"),
                  pd.read_csv(f"data/nasdaq/data.csv")
                  ]
                 ).reset_index(drop=True)

# Remove duplicate
data = data.drop_duplicates()

# Format date
data['date'] = pd.to_datetime(data['date'])
data['date_price'] = pd.to_datetime(data['date_price'])
data['date_price_previous'] = pd.to_datetime(data['date_price_previous'])
data['date_price_next'] = pd.to_datetime(data['date_price_next'])
data['date_ref'] = pd.to_datetime(data['date_ref'])
data['date_ref_previous'] = pd.to_datetime(data['date_ref_previous'])
data['date_ref_next'] = pd.to_datetime(data['date_ref_next'])

# Assess missing values
percent_missing = data.isnull().sum() * 100 / len(data)

# Remove data whose price date is too different from financial statements date
data = data[abs(data['date'] - data['date_price']) < datetime.timedelta(weeks=2)]

# Remove features that have more than 20% of missing values
col2rm = percent_missing[percent_missing > 20].index.tolist()
data = data.drop(columns=col2rm)

# Creation of new variables

# Yield for previous year
data['yield'] = (np.log(data['price']/data['price_previous']))/((data['date_price'] - data['date_price_previous']) / datetime.timedelta(weeks=52))
# Yield for next year
data['yield_next'] = (np.log(data['price_next']/data['price']))/((data['date_price_next'] - data['date_price']) / datetime.timedelta(weeks=52))
# Reference yield for previous year
data['yield_ref'] = (np.log(data['ref']/data['ref_previous']))/((data['date_ref'] - data['date_ref_previous']) / datetime.timedelta(weeks=52))
# Reference yield for next year
data['yield_ref_next'] = (np.log(data['ref_next']/data['ref']))/((data['date_ref_next'] - data['date_ref']) / datetime.timedelta(weeks=52))
# Best performance than reference for previous year
data['outperform'] = data['yield'] > data['yield_ref']
# Best performance than reference for next year
data['outperform_next'] = data['yield_next'] > data['yield_ref_next']
# Positivive performance for previous year
data['positive'] = data['yield'] > 0
# Positive performance reference for next year
data['positive_next'] = data['yield_next'] > 0
# Market capitalization
data['market_cap'] = data['price'] * data['sharesNumber']
# percent of dividends
data['div_percent'] = data['eps'] / data['price']
# group sectors
di = {'Consumer Discretionary': 'Consumer Services',
      'Consumer Non-Durables': 'Consumer Services',
      'Consumer Durables': 'Consumer Services',
      'Consumer Staples': 'Consumer Services',
      'Utilities': 'Energy',
      'Basic Industries': 'Industrials',
      'Materials': 'Industrials',
      'Information Technology': 'Technology',
      'Financials': 'Finance',
      }
data = data.replace({"sector": di})
# Selection of variables for first analysis
info = ['date', 'symbol', 'sector', ]
variables = ['netIncome',
             'grossProfit',
             'ebit',
             'totalRevenue',
             'costOfRevenue',
             'totalOtherIncomeExpenseNet',
             'otherCurrentLiab',
             'totalAssets',
             'commonStock',
             'otherLiab',
             'otherAssets',
             'cash',
             'propertyPlantEquipment',
             'accountsPayable',
             'capitalSurplus',
             'changeToLiabilities',
             'totalCashflowsFromInvestingActivities',
             'netBorrowings',
             'totalCashFromFinancingActivities',
             'changeInCash',
             'totalCashFromOperatingActivities',
             'depreciation',
             'changeToNetincome',
             'capitalExpenditures',
             'changeToOperatingActivities'
             ]
targets = ['market_cap',
           'div_percent',
           'yield',
           'yield_ref',
           'yield_next',
           'yield_ref_next',
           'outperform',
           'outperform_next',
           'positive',
           'positive_next'
           ]
data[['ebitAbs', 'totalRevenueAbs']] = data[['ebit', 'totalRevenue']]
variables_abs = ['ebitAbs', 'totalRevenueAbs']
# Normalization by market cap
data[variables] = data[variables].div(data.market_cap, axis=0)
# Data selection
data = data[info + variables + variables_abs + targets]

# Remove when next yield is not available
data = data.dropna(subset=['yield_next'])

# Includ previous values
data = data.sort_values("date")
prev_features = variables + ['div_percent', 'yield', 'market_cap']
data_prev = data[["date", "symbol"] + prev_features]
data_merged = list()
for sbl in tqdm(data.symbol.unique().tolist()):
    df1 = data[data.symbol == sbl]
    df2 = data_prev[data_prev.symbol == sbl]
    data_merged.append(pd.merge_asof(df1, df2, left_on="date", right_on="date", direction="backward", suffixes=("", "_evol"), allow_exact_matches=False))
data_evol = pd.concat(data_merged).reset_index(drop=True)
data_evol = data_evol.dropna(subset=["yield_evol"])
data_evol[[a + "_evol" for a in prev_features]] = data_evol[[a + "_evol" for a in prev_features]].values - data_evol[prev_features].values
data_evol = data_evol.drop(columns=["symbol_evol"])

# Assess missing values
percent_missing = data.isnull().sum() * 100 / len(data)
print(percent_missing.sort_values())

# Assess missing values
percent_missing = data_evol.isnull().sum() * 100 / len(data)
print(percent_missing.sort_values())

# DTale
d = dtale.show(data)

d.open_browser()

# Save data
data.to_csv("data/data_clean.csv", index=False)
data_evol.to_csv("data/data_evol_clean.csv", index=False)
