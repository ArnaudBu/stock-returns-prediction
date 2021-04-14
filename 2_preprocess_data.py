#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from tqdm import tqdm

# Modify project and reference index according to your needs
project = "nyse"
ref_index = "^GSCP"

# Load data
prices = pd.read_csv(f"data/{project}/prices_daily.csv")
dividends = pd.read_csv(f"data/{project}/dividends.csv")
income = pd.read_csv(f"data/{project}/incomeStatementHistory.csv")
balance = pd.read_csv(f"data/{project}/balanceSheetHistory.csv")
cashflow = pd.read_csv(f"data/{project}/cashflowStatementHistory.csv")
companies = pd.read_csv(f"data/{project}/{project}.csv", sep=";")
shares = pd.read_csv(f"data/{project}/shares.csv")
print("Data loaded")

# Format date
prices['Date'] = pd.to_datetime(prices['Date'])
dividends['Date'] = pd.to_datetime(dividends['Date'])
income['date'] = pd.to_datetime(income['date'])
balance['date'] = pd.to_datetime(balance['date'])
cashflow['date'] = pd.to_datetime(cashflow['date'])
shares['date'] = pd.to_datetime(shares['date'])
print("Date formatted")

# Merge financial statements
fin_stats = income.merge(balance,
                         on=['date', 'symbol'],
                         how="inner",
                         suffixes=("", "_y"))
fin_stats = fin_stats.merge(cashflow,
                            on=['date', 'symbol'],
                            how="inner",
                            suffixes=("", "_y"))
print("Statetements merged")

# Merge with price current year
fin_stats = fin_stats.sort_values("date")
prices_long = pd.melt(prices, "Date").sort_values("Date")
fin = list()
for sbl in tqdm(fin_stats.symbol.unique().tolist()):
    df1 = fin_stats[fin_stats.symbol == sbl]
    df2 = prices_long[prices_long.variable == sbl]
    fin.append(pd.merge_asof(df1,
                             df2,
                             left_on="date",
                             right_on="Date",
                             direction="backward"))
fin = pd.concat(fin).reset_index(drop=True)
fin = fin.rename(columns={"value": "price", "Date": "date_price"})
fin = fin.drop(columns=["variable"])
print("Current prices merged")

# Merge with price previous year
prices_long_previous = prices_long.copy()
prices_long_previous['Date'] = prices_long_previous['Date'] + pd.DateOffset(years=1)
fin2 = list()
for sbl in tqdm(fin.symbol.unique().tolist()):
    df1 = fin[fin.symbol == sbl]
    df2 = prices_long_previous[prices_long_previous.variable == sbl]
    fin2.append(pd.merge_asof(df1,
                              df2,
                              left_on="date",
                              right_on="Date",
                              direction="backward"))
fin = pd.concat(fin2).reset_index(drop=True)
fin['Date'] = fin['Date'] - pd.DateOffset(years=1)
fin = fin.rename(columns={"value": "price_previous",
                          "Date": "date_price_previous"})
fin = fin.drop(columns=["variable"])
print("Previous prices merged")

# Merge with price next year
prices_long_next = prices_long.copy()
prices_long_next['Date'] = prices_long_next['Date'] - pd.DateOffset(years=1)
fin2 = list()
for sbl in tqdm(fin.symbol.unique().tolist()):
    df1 = fin[fin.symbol == sbl]
    df2 = prices_long_next[prices_long_next.variable == sbl]
    fin2.append(pd.merge_asof(df1,
                              df2,
                              left_on="date",
                              right_on="Date",
                              direction="backward"))
fin = pd.concat(fin2).reset_index(drop=True)
fin['Date'] = fin['Date'] + pd.DateOffset(years=1)
fin = fin.rename(columns={"value": "price_next", "Date": "date_price_next"})
fin = fin.drop(columns=["variable"])
print("Next prices merged")

# Merge with dividends
for index, row in tqdm(fin.iterrows()):
    datemax = pd.to_datetime(row['date'])
    datemin = datemax - pd.DateOffset(years=1)
    eps = dividends[row['symbol']][(dividends['Date'] <= datemax) & (dividends['Date'] > datemin)].sum()
    fin.at[index, 'eps'] = eps
print("Dividends merged")

# Merge with sector
cpn = companies[['Symbol', 'GICS Sector']]
fin = fin.merge(cpn, left_on="symbol", right_on="Symbol")
fin = fin.rename(columns={"GICS Sector": "sector"})
fin = fin.drop(columns=["Symbol"])
print("Sector merged")

# Add reference index
df1 = prices_long[prices_long.variable == ref_index]
fin = fin.sort_values("date")
fin = fin[fin['date_price_previous'].notnull()]
fin = pd.merge_asof(fin,
                    df1,
                    left_on="date_price",
                    right_on="Date",
                    direction="nearest",
                    suffixes=("", "_ref"))
fin = pd.merge_asof(fin,
                    df1,
                    left_on="date_price_next",
                    right_on="Date",
                    direction="nearest",
                    suffixes=("", "_ref_next"))
fin = pd.merge_asof(fin,
                    df1,
                    left_on="date_price_previous",
                    right_on="Date",
                    direction="nearest",
                    suffixes=("", "_ref_previous"))
fin = fin.rename(columns={"value": "ref",
                          "Date": "date_ref",
                          "value_ref_next": "ref_next",
                          "value_ref_previous":"ref_previous",
                          "Date_ref_next": "date_ref_next",
                          "Date_ref_previous": "date_ref_previous"})
fin = fin.drop(columns=["variable", "variable_ref_next", "variable_ref_previous"])
fin = fin.sort_values(["symbol", "date"])
print("Reference index merged")

# Merge with shares
shares = shares.fillna(0)
shares['sharesNumber'] = shares['annualOrdinarySharesNumber'] + shares['annualPreferredSharesNumber']
shares = shares.drop(columns=["annualOrdinarySharesNumber", "annualPreferredSharesNumber"])
shares = shares.sort_values("date")
fin2 = list()
for sbl in tqdm(fin.symbol.unique().tolist()):
    df1 = fin[fin.symbol == sbl]
    df2 = shares[shares.symbol == sbl]
    fin2.append(pd.merge_asof(df1,
                              df2,
                              left_on="date",
                              right_on="date",
                              direction="nearest",
                              suffixes=("", "_ref")))
fin = pd.concat(fin2).reset_index(drop=True)
fin = fin.drop(columns=['symbol_ref'])

# Assess missing values
percent_missing = fin.isnull().sum() * 100 / len(fin)
print(percent_missing.sort_values())

# save
fin.to_csv(f"data/{project}/data.csv", index=False)
