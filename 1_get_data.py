#!/usr/bin/env python
# -*- coding: utf-8 -*-

from yfinance import Ticker
import pandas as pd
from yahoofinancials import YahooFinancials
import requests
from tqdm import tqdm
import time
import pickle

# with open('tmp.pickle', 'rb') as f:
#     statements, tickers_done = pickle.load(f)


# Download function
def _download_one(ticker, start=None, end=None,
                  auto_adjust=False, back_adjust=False,
                  actions=False, period="max", interval="1d",
                  prepost=False, proxy=None, rounding=False):

    return Ticker(ticker).history(period=period, interval=interval,
                                  start=start, end=end, prepost=prepost,
                                  actions=actions, auto_adjust=auto_adjust,
                                  back_adjust=back_adjust, proxy=proxy,
                                  rounding=rounding, many=True)


# Modify project and reference index according to your needs
tickers_all = []
for project in ["sp500", "nyse", "nasdaq"]:
    print(project)
    ref_index = ["^GSPC", "^IXIC"]

    # Load tickers
    companies = pd.read_csv(f"data/{project}/{project}.csv", sep=",")
    tickers = companies.Symbol.tolist()
    tickers = [a for a in tickers if a not in tickers_all and "^" not in a and r"/" not in a]
    tickers_all += tickers

    # Download prices
    full_data = {}
    for ticker in tqdm(tickers + ref_index):
        tckr = _download_one(ticker,
                             period="7y",
                             actions=True)
        full_data[ticker] = tckr
    ohlc = pd.concat(full_data.values(), axis=1,
                     keys=full_data.keys())
    ohlc.columns = ohlc.columns.swaplevel(0, 1)
    ohlc.sort_index(level=0, axis=1, inplace=True)
    prices = ohlc["Adj Close"]
    dividends = ohlc["Dividends"]
    prices.to_csv(f"data/{project}/prices_daily.csv")
    dividends.to_csv(f"data/{project}/dividends.csv")

    statements = {}
    tickers_done = []
    for ticker in tqdm(tickers):
        # Get statements
        if ticker in tickers_done:
            continue
        yahoo_financials = YahooFinancials(ticker)
        stmts_codes = ['income', 'cash', 'balance']
        all_statement_data = yahoo_financials.get_financial_stmts('annual',
                                                                  stmts_codes)
        # build statements dictionnary
        for a in all_statement_data.keys():
            if a not in statements:
                statements[a] = list()
            for b in all_statement_data[a]:
                try:
                    for result in all_statement_data[a][b]:
                        extracted_date = list(result)[0]
                        dataframe_row = list(result.values())[0]
                        dataframe_row['date'] = extracted_date
                        dataframe_row['symbol'] = b
                        statements[a].append(dataframe_row)
                except Exception as e:
                    print("Error on " + ticker + " : " + a)
        tickers_done.append(ticker)
        with open('tmp.pickle', 'wb') as f:
            pickle.dump([statements, tickers_done], f)

    # save dataframes
    for a in all_statement_data.keys():
        df = pd.DataFrame(statements[a]).set_index('date')
        df.to_csv(f"data/{project}/{a}.csv")

    # Donwload shares
    shares = []
    tickers_done = []
    for ticker in tqdm(tickers):
        if ticker in tickers_done:
            continue
        d = requests.get(f"https://query1.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/{ticker}?symbol={ticker}&padTimeSeries=true&type=annualPreferredSharesNumber,annualOrdinarySharesNumber&merge=false&period1=0&period2=2013490868")
        if not d.ok:
            time.sleep(300)
            d = requests.get(f"https://query1.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/{ticker}?symbol={ticker}&padTimeSeries=true&type=annualPreferredSharesNumber,annualOrdinarySharesNumber&merge=false&period1=0&period2=2013490868")
        ctn = d.json()['timeseries']['result']
        dct = dict()
        for n in ctn:
            type = n['meta']['type'][0]
            dct[type] = dict()
            if type in n:
                for o in n[type]:
                    if o is not None:
                        dct[type][o['asOfDate']] = o['reportedValue']['raw']
        df = pd.DataFrame.from_dict(dct)
        df['symbol'] = ticker
        shares.append(df)
        tickers_done.append(ticker)
        time.sleep(1)

    # save dataframe
    df = pd.concat(shares)
    df['date'] = df.index
    df.to_csv(f"data/{project}/shares.csv", index=False)

    # https://query1.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/MSFT?symbol=MSFT&padTimeSeries=true&type=annualTreasurySharesNumber,trailingTreasurySharesNumber,annualPreferredSharesNumber,trailingPreferredSharesNumber,annualOrdinarySharesNumber,trailingOrdinarySharesNumber,annualShareIssued,trailingShareIssued,annualNetDebt,trailingNetDebt,annualTotalDebt,trailingTotalDebt,annualTangibleBookValue,trailingTangibleBookValue,annualInvestedCapital,trailingInvestedCapital,annualWorkingCapital,trailingWorkingCapital,annualNetTangibleAssets,trailingNetTangibleAssets,annualCapitalLeaseObligations,trailingCapitalLeaseObligations,annualCommonStockEquity,trailingCommonStockEquity,annualPreferredStockEquity,trailingPreferredStockEquity,annualTotalCapitalization,trailingTotalCapitalization,annualTotalEquityGrossMinorityInterest,trailingTotalEquityGrossMinorityInterest,annualMinorityInterest,trailingMinorityInterest,annualStockholdersEquity,trailingStockholdersEquity,annualOtherEquityInterest,trailingOtherEquityInterest,annualGainsLossesNotAffectingRetainedEarnings,trailingGainsLossesNotAffectingRetainedEarnings,annualOtherEquityAdjustments,trailingOtherEquityAdjustments,annualFixedAssetsRevaluationReserve,trailingFixedAssetsRevaluationReserve,annualForeignCurrencyTranslationAdjustments,trailingForeignCurrencyTranslationAdjustments,annualMinimumPensionLiabilities,trailingMinimumPensionLiabilities,annualUnrealizedGainLoss,trailingUnrealizedGainLoss,annualTreasuryStock,trailingTreasuryStock,annualRetainedEarnings,trailingRetainedEarnings,annualAdditionalPaidInCapital,trailingAdditionalPaidInCapital,annualCapitalStock,trailingCapitalStock,annualOtherCapitalStock,trailingOtherCapitalStock,annualCommonStock,trailingCommonStock,annualPreferredStock,trailingPreferredStock,annualTotalPartnershipCapital,trailingTotalPartnershipCapital,annualGeneralPartnershipCapital,trailingGeneralPartnershipCapital,annualLimitedPartnershipCapital,trailingLimitedPartnershipCapital,annualTotalLiabilitiesNetMinorityInterest,trailingTotalLiabilitiesNetMinorityInterest,annualTotalNonCurrentLiabilitiesNetMinorityInterest,trailingTotalNonCurrentLiabilitiesNetMinorityInterest,annualOtherNonCurrentLiabilities,trailingOtherNonCurrentLiabilities,annualLiabilitiesHeldforSaleNonCurrent,trailingLiabilitiesHeldforSaleNonCurrent,annualRestrictedCommonStock,trailingRestrictedCommonStock,annualPreferredSecuritiesOutsideStockEquity,trailingPreferredSecuritiesOutsideStockEquity,annualDerivativeProductLiabilities,trailingDerivativeProductLiabilities,annualEmployeeBenefits,trailingEmployeeBenefits,annualNonCurrentPensionAndOtherPostretirementBenefitPlans,trailingNonCurrentPensionAndOtherPostretirementBenefitPlans,annualNonCurrentAccruedExpenses,trailingNonCurrentAccruedExpenses,annualDuetoRelatedPartiesNonCurrent,trailingDuetoRelatedPartiesNonCurrent,annualTradeandOtherPayablesNonCurrent,trailingTradeandOtherPayablesNonCurrent,annualNonCurrentDeferredLiabilities,trailingNonCurrentDeferredLiabilities,annualNonCurrentDeferredRevenue,trailingNonCurrentDeferredRevenue,annualNonCurrentDeferredTaxesLiabilities,trailingNonCurrentDeferredTaxesLiabilities,annualLongTermDebtAndCapitalLeaseObligation,trailingLongTermDebtAndCapitalLeaseObligation,annualLongTermCapitalLeaseObligation,trailingLongTermCapitalLeaseObligation,annualLongTermDebt,trailingLongTermDebt,annualLongTermProvisions,trailingLongTermProvisions,annualCurrentLiabilities,trailingCurrentLiabilities,annualOtherCurrentLiabilities,trailingOtherCurrentLiabilities,annualCurrentDeferredLiabilities,trailingCurrentDeferredLiabilities,annualCurrentDeferredRevenue,trailingCurrentDeferredRevenue,annualCurrentDeferredTaxesLiabilities,trailingCurrentDeferredTaxesLiabilities,annualCurrentDebtAndCapitalLeaseObligation,trailingCurrentDebtAndCapitalLeaseObligation,annualCurrentCapitalLeaseObligation,trailingCurrentCapitalLeaseObligation,annualCurrentDebt,trailingCurrentDebt,annualOtherCurrentBorrowings,trailingOtherCurrentBorrowings,annualLineOfCredit,trailingLineOfCredit,annualCommercialPaper,trailingCommercialPaper,annualCurrentNotesPayable,trailingCurrentNotesPayable,annualPensionandOtherPostRetirementBenefitPlansCurrent,trailingPensionandOtherPostRetirementBenefitPlansCurrent,annualCurrentProvisions,trailingCurrentProvisions,annualPayablesAndAccruedExpenses,trailingPayablesAndAccruedExpenses,annualCurrentAccruedExpenses,trailingCurrentAccruedExpenses,annualInterestPayable,trailingInterestPayable,annualPayables,trailingPayables,annualOtherPayable,trailingOtherPayable,annualDuetoRelatedPartiesCurrent,trailingDuetoRelatedPartiesCurrent,annualDividendsPayable,trailingDividendsPayable,annualTotalTaxPayable,trailingTotalTaxPayable,annualIncomeTaxPayable,trailingIncomeTaxPayable,annualAccountsPayable,trailingAccountsPayable,annualTotalAssets,trailingTotalAssets,annualTotalNonCurrentAssets,trailingTotalNonCurrentAssets,annualOtherNonCurrentAssets,trailingOtherNonCurrentAssets,annualDefinedPensionBenefit,trailingDefinedPensionBenefit,annualNonCurrentPrepaidAssets,trailingNonCurrentPrepaidAssets,annualNonCurrentDeferredAssets,trailingNonCurrentDeferredAssets,annualNonCurrentDeferredTaxesAssets,trailingNonCurrentDeferredTaxesAssets,annualDuefromRelatedPartiesNonCurrent,trailingDuefromRelatedPartiesNonCurrent,annualNonCurrentNoteReceivables,trailingNonCurrentNoteReceivables,annualNonCurrentAccountsReceivable,trailingNonCurrentAccountsReceivable,annualFinancialAssets,trailingFinancialAssets,annualInvestmentsAndAdvances,trailingInvestmentsAndAdvances,annualOtherInvestments,trailingOtherInvestments,annualInvestmentinFinancialAssets,trailingInvestmentinFinancialAssets,annualHeldToMaturitySecurities,trailingHeldToMaturitySecurities,annualAvailableForSaleSecurities,trailingAvailableForSaleSecurities,annualFinancialAssetsDesignatedasFairValueThroughProfitorLossTotal,trailingFinancialAssetsDesignatedasFairValueThroughProfitorLossTotal,annualTradingSecurities,trailingTradingSecurities,annualLongTermEquityInvestment,trailingLongTermEquityInvestment,annualInvestmentsinJointVenturesatCost,trailingInvestmentsinJointVenturesatCost,annualInvestmentsInOtherVenturesUnderEquityMethod,trailingInvestmentsInOtherVenturesUnderEquityMethod,annualInvestmentsinAssociatesatCost,trailingInvestmentsinAssociatesatCost,annualInvestmentsinSubsidiariesatCost,trailingInvestmentsinSubsidiariesatCost,annualInvestmentProperties,trailingInvestmentProperties,annualGoodwillAndOtherIntangibleAssets,trailingGoodwillAndOtherIntangibleAssets,annualOtherIntangibleAssets,trailingOtherIntangibleAssets,annualGoodwill,trailingGoodwill,annualNetPPE,trailingNetPPE,annualAccumulatedDepreciation,trailingAccumulatedDepreciation,annualGrossPPE,trailingGrossPPE,annualLeases,trailingLeases,annualConstructionInProgress,trailingConstructionInProgress,annualOtherProperties,trailingOtherProperties,annualMachineryFurnitureEquipment,trailingMachineryFurnitureEquipment,annualBuildingsAndImprovements,trailingBuildingsAndImprovements,annualLandAndImprovements,trailingLandAndImprovements,annualProperties,trailingProperties,annualCurrentAssets,trailingCurrentAssets,annualOtherCurrentAssets,trailingOtherCurrentAssets,annualHedgingAssetsCurrent,trailingHedgingAssetsCurrent,annualAssetsHeldForSaleCurrent,trailingAssetsHeldForSaleCurrent,annualCurrentDeferredAssets,trailingCurrentDeferredAssets,annualCurrentDeferredTaxesAssets,trailingCurrentDeferredTaxesAssets,annualRestrictedCash,trailingRestrictedCash,annualPrepaidAssets,trailingPrepaidAssets,annualInventory,trailingInventory,annualInventoriesAdjustmentsAllowances,trailingInventoriesAdjustmentsAllowances,annualOtherInventories,trailingOtherInventories,annualFinishedGoods,trailingFinishedGoods,annualWorkInProcess,trailingWorkInProcess,annualRawMaterials,trailingRawMaterials,annualReceivables,trailingReceivables,annualReceivablesAdjustmentsAllowances,trailingReceivablesAdjustmentsAllowances,annualOtherReceivables,trailingOtherReceivables,annualDuefromRelatedPartiesCurrent,trailingDuefromRelatedPartiesCurrent,annualTaxesReceivable,trailingTaxesReceivable,annualAccruedInterestReceivable,trailingAccruedInterestReceivable,annualNotesReceivable,trailingNotesReceivable,annualLoansReceivable,trailingLoansReceivable,annualAccountsReceivable,trailingAccountsReceivable,annualAllowanceForDoubtfulAccountsReceivable,trailingAllowanceForDoubtfulAccountsReceivable,annualGrossAccountsReceivable,trailingGrossAccountsReceivable,annualCashCashEquivalentsAndShortTermInvestments,trailingCashCashEquivalentsAndShortTermInvestments,annualOtherShortTermInvestments,trailingOtherShortTermInvestments,annualCashAndCashEquivalents,trailingCashAndCashEquivalents,annualCashEquivalents,trailingCashEquivalents,annualCashFinancial,trailingCashFinancial&merge=false&period1=493590046&period2=1613490868
    # https://query1.finance.yahoo.com/v8/finance/chart/MSFT?symbol=MSFT&period1=1550725200&period2=1613491890&useYfid=true&interval=1d&events=div
    # https://query1.finance.yahoo.com/v10/finance/quoteSummary/MSFT?formatted=true&crumb=2M1BZy1YB7f&lang=en-US&region=US&modules=incomeStatementHistory,cashflowStatementHistory,balanceSheetHistory,incomeStatementHistoryQuarterly,cashflowStatementHistoryQuarterly,balanceSheetHistoryQuarterly&corsDomain=finance.yahoo.com
