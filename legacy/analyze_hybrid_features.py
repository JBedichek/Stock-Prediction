"""
Analyze hybrid scraper output and generate JSON documentation of all features.
"""

import sys
sys.path.append('/home/james/Desktop/Stock-Prediction')

import json
import pandas as pd
from utils.utils import pic_load


def extract_feature_metadata(data_file='s_lot_hybrid_data.pkl'):
    """
    Extract all features from hybrid data and create metadata JSON.

    Returns:
        dict: Feature metadata with names, frequencies, and categories
    """
    # Load data
    print(f"Loading {data_file}...")
    data = pic_load(data_file)

    if not data:
        print("No data found!")
        return None

    # Get a sample stock with data
    sample_ticker = None
    for ticker in data:
        if data[ticker]:
            sample_ticker = ticker
            break

    if not sample_ticker:
        print("No valid stock data found!")
        return None

    print(f"Using sample ticker: {sample_ticker}")
    print(f"Total stocks in dataset: {len(data)}")

    # Get sample date and tensor
    sample_dates = list(data[sample_ticker].keys())
    sample_date = sample_dates[len(sample_dates) // 2]  # Middle date
    sample_tensor = data[sample_ticker][sample_date]

    total_features = sample_tensor.shape[0]
    print(f"Total features per day: {total_features}")
    print(f"Sample date: {sample_date}")

    # Define feature structure based on hybrid_scraper.py logic
    features = {
        "metadata": {
            "total_features": total_features,
            "price_features_count": 19,
            "fundamental_features_count": total_features - 19,
            "date_range_days": 9473,
            "fundamental_frequency": "quarterly_forward_filled_to_daily",
            "price_frequency": "daily",
            "normalization": "z-score",
            "data_sources": {
                "price": "yfinance",
                "fundamentals": "FMP API"
            }
        },
        "features": []
    }

    # Price features (indices 0-18)
    price_features = [
        {
            "index": 0,
            "name": "price_open",
            "description": "Opening price",
            "frequency": "daily",
            "source": "yfinance",
            "category": "price",
            "subcategory": "ohlcv"
        },
        {
            "index": 1,
            "name": "price_high",
            "description": "Daily high price",
            "frequency": "daily",
            "source": "yfinance",
            "category": "price",
            "subcategory": "ohlcv"
        },
        {
            "index": 2,
            "name": "price_low",
            "description": "Daily low price",
            "frequency": "daily",
            "source": "yfinance",
            "category": "price",
            "subcategory": "ohlcv"
        },
        {
            "index": 3,
            "name": "price_close",
            "description": "Closing price",
            "frequency": "daily",
            "source": "yfinance",
            "category": "price",
            "subcategory": "ohlcv"
        },
        {
            "index": 4,
            "name": "price_volume",
            "description": "Trading volume",
            "frequency": "daily",
            "source": "yfinance",
            "category": "price",
            "subcategory": "ohlcv"
        },
        {
            "index": 5,
            "name": "price_return_1d",
            "description": "1-day price return",
            "frequency": "daily",
            "source": "yfinance (calculated)",
            "category": "price",
            "subcategory": "returns"
        },
        {
            "index": 6,
            "name": "price_return_5d",
            "description": "5-day price return",
            "frequency": "daily",
            "source": "yfinance (calculated)",
            "category": "price",
            "subcategory": "returns"
        },
        {
            "index": 7,
            "name": "price_return_20d",
            "description": "20-day price return",
            "frequency": "daily",
            "source": "yfinance (calculated)",
            "category": "price",
            "subcategory": "returns"
        },
        {
            "index": 8,
            "name": "price_volatility_10d",
            "description": "10-day rolling volatility",
            "frequency": "daily",
            "source": "yfinance (calculated)",
            "category": "price",
            "subcategory": "volatility"
        },
        {
            "index": 9,
            "name": "price_volatility_20d",
            "description": "20-day rolling volatility",
            "frequency": "daily",
            "source": "yfinance (calculated)",
            "category": "price",
            "subcategory": "volatility"
        },
        {
            "index": 10,
            "name": "volume_change_1d",
            "description": "1-day volume change",
            "frequency": "daily",
            "source": "yfinance (calculated)",
            "category": "price",
            "subcategory": "volume"
        },
        {
            "index": 11,
            "name": "intraday_range",
            "description": "High-low range relative to close",
            "frequency": "daily",
            "source": "yfinance (calculated)",
            "category": "price",
            "subcategory": "intraday"
        },
        {
            "index": 12,
            "name": "price_sma_10",
            "description": "10-day simple moving average",
            "frequency": "daily",
            "source": "yfinance (calculated)",
            "category": "price",
            "subcategory": "moving_averages"
        },
        {
            "index": 13,
            "name": "price_sma_20",
            "description": "20-day simple moving average",
            "frequency": "daily",
            "source": "yfinance (calculated)",
            "category": "price",
            "subcategory": "moving_averages"
        },
        {
            "index": 14,
            "name": "price_sma_50",
            "description": "50-day simple moving average",
            "frequency": "daily",
            "source": "yfinance (calculated)",
            "category": "price",
            "subcategory": "moving_averages"
        },
        {
            "index": 15,
            "name": "price_sma_200",
            "description": "200-day simple moving average",
            "frequency": "daily",
            "source": "yfinance (calculated)",
            "category": "price",
            "subcategory": "moving_averages"
        },
        {
            "index": 16,
            "name": "price_vs_sma_10",
            "description": "Price relative to 10-day SMA (deviation)",
            "frequency": "daily",
            "source": "yfinance (calculated)",
            "category": "price",
            "subcategory": "relative_position"
        },
        {
            "index": 17,
            "name": "price_vs_sma_20",
            "description": "Price relative to 20-day SMA (deviation)",
            "frequency": "daily",
            "source": "yfinance (calculated)",
            "category": "price",
            "subcategory": "relative_position"
        },
        {
            "index": 18,
            "name": "price_vs_sma_50",
            "description": "Price relative to 50-day SMA (deviation)",
            "frequency": "daily",
            "source": "yfinance (calculated)",
            "category": "price",
            "subcategory": "relative_position"
        }
    ]

    features["features"].extend(price_features)

    # Fundamental features (indices 19+)
    # Note: These are extracted from actual FMP data, so they vary by stock
    # We'll describe the major categories and common features

    fundamental_categories = {
        "income": {
            "description": "Income statement metrics",
            "frequency": "quarterly_forward_filled",
            "source": "FMP",
            "category": "fundamental",
            "subcategory": "income_statement",
            "common_fields": [
                "revenue", "costOfRevenue", "grossProfit", "grossProfitRatio",
                "researchAndDevelopmentExpenses", "generalAndAdministrativeExpenses",
                "sellingAndMarketingExpenses", "sellingGeneralAndAdministrativeExpenses",
                "otherExpenses", "operatingExpenses", "costAndExpenses",
                "interestIncome", "interestExpense", "depreciationAndAmortization",
                "ebitda", "ebitdaratio", "operatingIncome", "operatingIncomeRatio",
                "totalOtherIncomeExpensesNet", "incomeBeforeTax", "incomeBeforeTaxRatio",
                "incomeTaxExpense", "netIncome", "netIncomeRatio", "eps", "epsdiluted",
                "weightedAverageShsOut", "weightedAverageShsOutDil"
            ]
        },
        "balance": {
            "description": "Balance sheet metrics",
            "frequency": "quarterly_forward_filled",
            "source": "FMP",
            "category": "fundamental",
            "subcategory": "balance_sheet",
            "common_fields": [
                "cashAndCashEquivalents", "shortTermInvestments", "cashAndShortTermInvestments",
                "netReceivables", "inventory", "otherCurrentAssets", "totalCurrentAssets",
                "propertyPlantEquipmentNet", "goodwill", "intangibleAssets",
                "goodwillAndIntangibleAssets", "longTermInvestments", "taxAssets",
                "otherNonCurrentAssets", "totalNonCurrentAssets", "otherAssets", "totalAssets",
                "accountPayables", "shortTermDebt", "taxPayables", "deferredRevenue",
                "otherCurrentLiabilities", "totalCurrentLiabilities", "longTermDebt",
                "deferredRevenueNonCurrent", "deferredTaxLiabilitiesNonCurrent",
                "otherNonCurrentLiabilities", "totalNonCurrentLiabilities",
                "otherLiabilities", "capitalLeaseObligations", "totalLiabilities",
                "preferredStock", "commonStock", "retainedEarnings",
                "accumulatedOtherComprehensiveIncomeLoss", "othertotalStockholdersEquity",
                "totalStockholdersEquity", "totalEquity", "totalLiabilitiesAndStockholdersEquity",
                "minorityInterest", "totalLiabilitiesAndTotalEquity", "totalInvestments",
                "totalDebt", "netDebt"
            ]
        },
        "cashflow": {
            "description": "Cash flow statement metrics",
            "frequency": "quarterly_forward_filled",
            "source": "FMP",
            "category": "fundamental",
            "subcategory": "cash_flow",
            "common_fields": [
                "netIncome", "depreciationAndAmortization", "deferredIncomeTax",
                "stockBasedCompensation", "changeInWorkingCapital", "accountsReceivables",
                "inventory", "accountsPayables", "otherWorkingCapital",
                "otherNonCashItems", "netCashProvidedByOperatingActivities",
                "investmentsInPropertyPlantAndEquipment", "acquisitionsNet",
                "purchasesOfInvestments", "salesMaturitiesOfInvestments",
                "otherInvestingActivites", "netCashUsedForInvestingActivites",
                "debtRepayment", "commonStockIssued", "commonStockRepurchased",
                "dividendsPaid", "otherFinancingActivites",
                "netCashUsedProvidedByFinancingActivities", "effectOfForexChangesOnCash",
                "netChangeInCash", "cashAtEndOfPeriod", "cashAtBeginningOfPeriod",
                "operatingCashFlow", "capitalExpenditure", "freeCashFlow"
            ]
        },
        "metric": {
            "description": "Key financial metrics and ratios",
            "frequency": "quarterly_forward_filled",
            "source": "FMP",
            "category": "fundamental",
            "subcategory": "key_metrics",
            "common_fields": [
                "revenuePerShare", "netIncomePerShare", "operatingCashFlowPerShare",
                "freeCashFlowPerShare", "cashPerShare", "bookValuePerShare",
                "tangibleBookValuePerShare", "shareholdersEquityPerShare",
                "interestDebtPerShare", "marketCap", "enterpriseValue",
                "peRatio", "priceToSalesRatio", "pocfratio", "pfcfRatio", "pbRatio",
                "ptbRatio", "evToSales", "enterpriseValueOverEBITDA", "evToOperatingCashFlow",
                "evToFreeCashFlow", "earningsYield", "freeCashFlowYield", "debtToEquity",
                "debtToAssets", "netDebtToEBITDA", "currentRatio", "interestCoverage",
                "incomeQuality", "dividendYield", "payoutRatio", "salesGeneralAndAdministrativeToRevenue",
                "researchAndDdevelopementToRevenue", "intangiblesToTotalAssets",
                "capexToOperatingCashFlow", "capexToRevenue", "capexToDepreciation",
                "stockBasedCompensationToRevenue", "grahamNumber", "roic",
                "returnOnTangibleAssets", "grahamNetNet", "workingCapital",
                "tangibleAssetValue", "netCurrentAssetValue", "investedCapital",
                "averageReceivables", "averagePayables", "averageInventory",
                "daysSalesOutstanding", "daysPayablesOutstanding", "daysOfInventoryOnHand",
                "receivablesTurnover", "payablesTurnover", "inventoryTurnover",
                "roe", "capexPerShare"
            ]
        },
        "ratio": {
            "description": "Financial ratios",
            "frequency": "quarterly_forward_filled",
            "source": "FMP",
            "category": "fundamental",
            "subcategory": "financial_ratios",
            "common_fields": [
                "currentRatio", "quickRatio", "cashRatio", "daysOfSalesOutstanding",
                "daysOfInventoryOutstanding", "operatingCycle", "daysOfPayablesOutstanding",
                "cashConversionCycle", "grossProfitMargin", "operatingProfitMargin",
                "pretaxProfitMargin", "netProfitMargin", "effectiveTaxRate",
                "returnOnAssets", "returnOnEquity", "returnOnCapitalEmployed",
                "netIncomePerEBT", "ebtPerEbit", "ebitPerRevenue", "debtRatio",
                "debtEquityRatio", "longTermDebtToCapitalization", "totalDebtToCapitalization",
                "interestCoverage", "cashFlowToDebtRatio", "companyEquityMultiplier",
                "receivablesTurnover", "payablesTurnover", "inventoryTurnover",
                "fixedAssetTurnover", "assetTurnover", "operatingCashFlowPerShare",
                "freeCashFlowPerShare", "cashPerShare", "payoutRatio",
                "operatingCashFlowSalesRatio", "freeCashFlowOperatingCashFlowRatio",
                "cashFlowCoverageRatios", "shortTermCoverageRatios",
                "capitalExpenditureCoverageRatio", "dividendPaidAndCapexCoverageRatio",
                "dividendPayoutRatio", "priceBookValueRatio", "priceToBookRatio",
                "priceToSalesRatio", "priceEarningsRatio", "priceToFreeCashFlowsRatio",
                "priceToOperatingCashFlowsRatio", "priceCashFlowRatio",
                "priceEarningsToGrowthRatio", "priceSalesRatio", "dividendYield",
                "enterpriseValueMultiple", "priceFairValue"
            ]
        },
        "ev": {
            "description": "Enterprise value metrics",
            "frequency": "quarterly_forward_filled",
            "source": "FMP",
            "category": "fundamental",
            "subcategory": "enterprise_value",
            "common_fields": [
                "stockPrice", "numberOfShares", "marketCapitalization", "minusCashAndCashEquivalents",
                "addTotalDebt", "enterpriseValue"
            ]
        },
        "growth": {
            "description": "Growth metrics",
            "frequency": "quarterly_forward_filled",
            "source": "FMP",
            "category": "fundamental",
            "subcategory": "financial_growth",
            "common_fields": [
                "revenueGrowth", "grossProfitGrowth", "ebitgrowth", "operatingIncomeGrowth",
                "netIncomeGrowth", "epsgrowth", "epsdilutedGrowth", "weightedAverageSharesGrowth",
                "weightedAverageSharesDilutedGrowth", "dividendsperShareGrowth",
                "operatingCashFlowGrowth", "freeCashFlowGrowth", "tenYRevenueGrowthPerShare",
                "fiveYRevenueGrowthPerShare", "threeYRevenueGrowthPerShare",
                "tenYOperatingCFGrowthPerShare", "fiveYOperatingCFGrowthPerShare",
                "threeYOperatingCFGrowthPerShare", "tenYNetIncomeGrowthPerShare",
                "fiveYNetIncomeGrowthPerShare", "threeYNetIncomeGrowthPerShare",
                "tenYShareholdersEquityGrowthPerShare", "fiveYShareholdersEquityGrowthPerShare",
                "threeYShareholdersEquityGrowthPerShare", "tenYDividendperShareGrowthPerShare",
                "fiveYDividendperShareGrowthPerShare", "threeYDividendperShareGrowthPerShare",
                "receivablesGrowth", "inventoryGrowth", "assetGrowth", "bookValueperShareGrowth",
                "debtGrowth", "rdexpenseGrowth", "sgaexpensesGrowth"
            ]
        }
    }

    # Add fundamental feature placeholders
    # Since actual feature names are dynamic, we'll create category-based documentation
    fundamental_start_idx = 19

    for category_key, category_info in fundamental_categories.items():
        for field in category_info["common_fields"]:
            feature_name = f"{category_key}_{field}"
            features["features"].append({
                "index": f"{fundamental_start_idx}+",
                "name": feature_name,
                "description": f"{category_info['description']}: {field}",
                "frequency": category_info["frequency"],
                "source": category_info["source"],
                "category": category_info["category"],
                "subcategory": category_info["subcategory"],
                "original_frequency": "quarterly",
                "transformation": "forward_filled_to_daily"
            })

    return features


def main():
    """Main execution."""
    print("="*80)
    print("HYBRID FEATURE ANALYZER")
    print("="*80)

    # Extract metadata
    metadata = extract_feature_metadata('s_lot_hybrid_data.pkl')

    if not metadata:
        print("Failed to extract metadata!")
        return

    # Save to JSON
    output_file = 'hybrid_features_documentation.json'
    with open(output_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\n✅ Saved feature documentation to: {output_file}")
    print(f"\nSummary:")
    print(f"  Total features: {metadata['metadata']['total_features']}")
    print(f"  Price features: {metadata['metadata']['price_features_count']}")
    print(f"  Fundamental features: {metadata['metadata']['fundamental_features_count']}")
    print(f"  Total documented features: {len(metadata['features'])}")


if __name__ == '__main__':
    main()
