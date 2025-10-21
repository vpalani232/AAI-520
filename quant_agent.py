from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from datetime import datetime
import json
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import requests
import time
import os
from dotenv import load_dotenv

load_dotenv()

NIM_MODEL_NAME = "meta/llama-3.1-70b-instruct"
NIM_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"

llm = LLM(model="nvidia_nim/meta/llama-3.1-405b-instruct", temperature=0.7)



# ---------- TOOL 1: FETCH STOCK DATA ----------
tool("Fetch Stock Data")
def fetch_stock_data(params: dict) -> dict:
    """
    Retrieves real-time and historical stock price and fundamental data for a given ticker.
    The input should be a JSON string with 'ticker' (string), 'period' (string, e.g., '1y'), and 'interval' (string, e.g., '1d').
    Example input: '{"ticker": "AAPL", "period": "1y", "interval": "1d"}'
    """
    ticker = params.get("ticker", "").upper()
    period = params.get("period", "1y")
    interval = params.get("interval", "1d")

    if not ticker:
        raise ValueError("Ticker symbol is required")

    max_retries = 3
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period, interval=interval)
            if hist.empty:
                raise ValueError(f"No data found for ticker: {ticker}")

            info = stock.info
            price_data = {
                "dates": hist.index.strftime("%Y-%m-%d").tolist(),
                "open": hist["Open"].tolist(),
                "high": hist["High"].tolist(),
                "low": hist["Low"].tolist(),
                "close": hist["Close"].tolist(),
                "volume": hist["Volume"].tolist(),
            }
            fundamentals = {
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "peg_ratio": info.get("pegRatio"),
                "price_to_book": info.get("priceToBook"),
                "dividend_yield": info.get("dividendYield"),
                "beta": info.get("beta"),
                "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
                "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
                "current_price": info.get("currentPrice"),
                "company_name": info.get("longName"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
            }
            return {
                "ticker": ticker,
                "price_data": price_data,
                "fundamentals": fundamentals,
                "data_points": len(hist),
                "period": period,
                "interval": interval,
                "retrieved_at": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2**attempt))
            else:
                raise Exception(
                    f"Failed to fetch data for {ticker} after {max_retries} attempts: {str(e)}"
                )


# ---------- TOOL 2: CALCULATE TECHNICAL INDICATORS ----------
@tool("Calculate Technical Indicators")
def calculate_technical_indicators(params: dict) -> dict:
    """
    Calculates standard technical indicators (e.g., SMA, RSI, MACD) on price data.
    The input should be a JSON string with 'price_data' (string/JSON of stock data) and 'indicators' (string, comma-separated, e.g., 'sma,rsi').
    Example input: '{"price_data": "...", "indicators": "sma,rsi"}'
    """
    price_data = json.loads(params.get("price_data", "{}"))
    indicators_str = params.get("indicators", "sma,ema,rsi,macd")
    indicators = [i.strip() for i in indicators_str.split(",")]

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(price_data["dates"]),
            "close": price_data["close"],
            "high": price_data["high"],
            "low": price_data["low"],
            "volume": price_data["volume"],
        }
    )
    df.set_index("date", inplace=True)

    results = {}

    if "sma" in indicators:
        results["sma_20"] = df["close"].rolling(20).mean().tolist()
        results["sma_50"] = df["close"].rolling(50).mean().tolist()
        results["sma_200"] = df["close"].rolling(200).mean().tolist()

    if "ema" in indicators:
        results["ema_12"] = df["close"].ewm(span=12, adjust=False).mean().tolist()
        results["ema_26"] = df["close"].ewm(span=26, adjust=False).mean().tolist()

    if "rsi" in indicators:
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        results["rsi"] = rsi.tolist()
        results["rsi_signal"] = (
            "overbought"
            if rsi.iloc[-1] > 70
            else "oversold"
            if rsi.iloc[-1] < 30
            else "neutral"
        )

    if "macd" in indicators:
        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        results["macd"] = macd.tolist()
        results["macd_signal"] = signal.tolist()
        results["macd_histogram"] = histogram.tolist()

    if "bollinger" in indicators:
        sma_20 = df["close"].rolling(20).mean()
        std_20 = df["close"].rolling(20).std()
        results["bollinger_upper"] = (sma_20 + 2 * std_20).tolist()
        results["bollinger_middle"] = sma_20.tolist()
        results["bollinger_lower"] = (sma_20 - 2 * std_20).tolist()

    if "atr" in indicators:
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        results["atr"] = tr.rolling(14).mean().tolist()

    results["volatility_20d"] = df["close"].pct_change().rolling(20).std().tolist()
    results["volatility_current"] = float(
        df["close"].pct_change().rolling(20).std().iloc[-1]
    )

    return {
        "indicators": results,
        "dates": df.index.strftime("%Y-%m-%d").tolist(),
        "calculated_at": datetime.utcnow().isoformat(),
    }


# ---------- TOOL 3: ANALYZE FUNDAMENTALS ----------
@tool("Analyze Stock Fundamentals")
def analyze_fundamentals(params: dict) -> dict:
    """
    Retrieves and analyzes key fundamental metrics for a stock, categorized by valuation,
    profitability, growth, and financial health.

    The input must be a JSON string with the following keys:
    - 'ticker': (string, REQUIRED) The stock ticker symbol (e.g., 'AAPL').
    - 'metrics': (string, OPTIONAL) A comma-separated list of fundamental categories to retrieve.
                 Defaults to 'valuation,profitability,growth'.
                 Available options: 'valuation', 'profitability', 'growth', 'financial_health'.

    Example input: '{"ticker": "MSFT", "metrics": "valuation,financial_health"}'

    Returns a JSON string containing the structured fundamental analysis.
    """
    ticker = params.get("ticker", "").upper()
    metrics_str = params.get("metrics", "valuation,profitability,growth")
    metrics = [m.strip() for m in metrics_str.split(",")]

    stock = yf.Ticker(ticker)
    info = stock.info

    analysis = {
        "ticker": ticker,
        "company_name": info.get("longName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
    }

    if "valuation" in metrics:
        analysis["valuation"] = {
            "market_cap": info.get("marketCap"),
            "enterprise_value": info.get("enterpriseValue"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "peg_ratio": info.get("pegRatio"),
            "price_to_sales": info.get("priceToSalesTrailing12Months"),
            "price_to_book": info.get("priceToBook"),
            "ev_to_revenue": info.get("enterpriseToRevenue"),
            "ev_to_ebitda": info.get("enterpriseToEbitda"),
        }
    if "profitability" in metrics:
        analysis["profitability"] = {
            "profit_margins": info.get("profitMargins"),
            "operating_margins": info.get("operatingMargins"),
            "return_on_assets": info.get("returnOnAssets"),
            "return_on_equity": info.get("returnOnEquity"),
            "gross_margins": info.get("grossMargins"),
        }
    if "growth" in metrics:
        analysis["growth"] = {
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "earnings_quarterly_growth": info.get("earningsQuarterlyGrowth"),
        }
    if "financial_health" in metrics:
        analysis["financial_health"] = {
            "current_ratio": info.get("currentRatio"),
            "quick_ratio": info.get("quickRatio"),
            "debt_to_equity": info.get("debtToEquity"),
            "total_cash": info.get("totalCash"),
            "total_debt": info.get("totalDebt"),
            "free_cash_flow": info.get("freeCashflow"),
        }
    return analysis


# ---------- TOOL 4: PORTFOLIO OPTIMIZATION ----------
@tool("Optimize Portfolio")
def optimize_portfolio(params: dict) -> dict:
    """
    Performs Markowitz-style portfolio optimization (Modern Portfolio Theory)
    on a list of stock tickers to find the optimal asset weights.

    The input must be a JSON string with the following required keys:
    - 'tickers': (string, REQUIRED) A comma-separated list of stock ticker symbols (MINIMUM 3).
    - 'price_data': (string/JSON, REQUIRED) The price data for all tickers, typically the output from 'fetch_stock_data'.
                    Must contain 'close' prices for each ticker.
    - 'optimization_target': (string, OPTIONAL) The target optimization goal.
                             Accepts 'max_sharpe' (default) or 'min_volatility'.

    Example input: '{"tickers": "AAPL,MSFT,GOOGL", "price_data": "{...}", "optimization_target": "max_sharpe"}'

    Returns a JSON string with the optimal weights, expected return, volatility, and Sharpe ratio.
    """
    tickers = [t.strip().upper() for t in params.get("tickers", "").split(",")]
    if len(tickers) < 3:
        raise ValueError(
            f"Minimum 3 tickers required for portfolio optimization. Got {len(tickers)}"
        )

    price_data = json.loads(params.get("price_data", "{}"))
    optimization_target = params.get("optimization_target", "max_sharpe")

    returns_data = {}
    for t in tickers:
        if t in price_data:
            prices = pd.Series(price_data[t]["close"])
            returns_data[t] = prices.pct_change().dropna()
    returns_df = pd.DataFrame(returns_data)

    mean_returns = returns_df.mean() * 252
    cov_matrix = returns_df.cov() * 252
    num_assets = len(tickers)

    def portfolio_stats(weights):
        ret = np.sum(mean_returns * weights)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = ret / vol
        return ret, vol, sharpe

    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(num_assets))
    init_guess = num_assets * [1.0 / num_assets]

    if optimization_target == "max_sharpe":
        opt_result = minimize(
            lambda w: -portfolio_stats(w)[2],
            init_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
    elif optimization_target == "min_volatility":
        opt_result = minimize(
            lambda w: portfolio_stats(w)[1],
            init_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
    else:
        opt_result = minimize(
            lambda w: -portfolio_stats(w)[2],
            init_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

    optimal_weights = opt_result.x
    opt_return, opt_volatility, opt_sharpe = portfolio_stats(optimal_weights)

    allocation = {t: float(w) for t, w in zip(tickers, optimal_weights)}

    return {
        "optimization_target": optimization_target,
        "optimal_allocation": allocation,
        "expected_annual_return": float(opt_return),
        "expected_annual_volatility": float(opt_volatility),
        "sharpe_ratio": float(opt_sharpe),
        "num_assets": num_assets,
        "diversification_score": float(1 - np.max(optimal_weights)),
        "optimized_at": datetime.utcnow().isoformat(),
    }


# ---------- TOOL 5: CALCULATE RISK METRICS ----------
@tool("Calculate Risk Metrics")
def calculate_risk_metrics(params: dict) -> dict:
    """
    Calculates key portfolio risk metrics including Value at Risk (VaR), Max Drawdown,
    Sharpe Ratio, and Volatility based on historical price data and specified weights.

    The input must be a JSON string with the following keys:
    - 'price_data': (string/JSON, REQUIRED) Historical price data (output from 'fetch_stock_data').
                    It must be a JSON string where keys are tickers and values contain a 'close' price list.
    - 'portfolio_weights': (string/JSON, OPTIONAL) A JSON string mapping tickers to their weights
                           (e.g., '{"AAPL": 0.5, "MSFT": 0.5}'). If omitted, an equally-weighted portfolio is assumed.
    - 'confidence_level': (string, OPTIONAL) The confidence level for VaR calculation (e.g., '0.95' or '0.99').
                          Defaults to '0.95'.

    Example input: '{"price_data": "{...}", "portfolio_weights": "{\"AAPL\": 0.6, \"MSFT\": 0.4}", "confidence_level": "0.99"}'

    Returns a JSON string containing the calculated risk metrics.
    """
    price_data = json.loads(params.get("price_data", "{}"))
    portfolio_weights = (
        json.loads(params.get("portfolio_weights", "{}"))
        if params.get("portfolio_weights")
        else None
    )
    confidence_level = float(params.get("confidence_level", "0.95"))

    returns_data = {}
    for t, data in price_data.items():
        returns_data[t] = pd.Series(data["close"]).pct_change().dropna()
    returns_df = pd.DataFrame(returns_data)

    if portfolio_weights:
        weights = np.array([portfolio_weights.get(t, 0) for t in returns_df.columns])
        portfolio_returns = returns_df.dot(weights)
    else:
        portfolio_returns = returns_df.mean(axis=1)

    var_1day = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
    cumulative = (1 + portfolio_returns).cumprod()
    max_drawdown = (cumulative - cumulative.expanding().max()).min()
    sharpe_ratio = (portfolio_returns.mean() * 252) / (
        portfolio_returns.std() * np.sqrt(252)
    )
    correlation_matrix = returns_df.corr().to_dict()

    return {
        "value_at_risk": {
            "confidence_level": confidence_level,
            "var_1day": float(var_1day),
            "var_1month": float(var_1day * np.sqrt(21)),
        },
        "max_drawdown": float(max_drawdown),
        "sharpe_ratio": float(sharpe_ratio),
        "volatility_annual": float(portfolio_returns.std() * np.sqrt(252)),
        "correlation_matrix": correlation_matrix,
        "downside_deviation": float(
            portfolio_returns[portfolio_returns < 0].std() * np.sqrt(252)
        ),
        "calculated_at": datetime.utcnow().isoformat(),
    }


# ---------- TOOL 6: LLM INTERPRETATION ----------
import requests
from datetime import datetime
import json

NVIDIA_NIM_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
NIM_MODEL = "meta/llama-3.1-70b-instruct"


@tool("LLM Interpretation Generator")
def interpret_metrics_with_llm(params: dict) -> dict:
    """
    Uses NVIDIA NIM LLM to provide contextual interpretation of calculated metrics.
    """
    metrics_data = params.get("metrics_data", "{}")
    market_context = params.get("market_context", "")
    analysis_focus = params.get("analysis_focus", "comprehensive")

    # Construct the prompt for the LLM
    prompt = f"""
You are a quantitative financial analyst. Analyze the following metrics:

Metrics Data:
{metrics_data}

Market Context:
{market_context}

Analysis Focus: {analysis_focus}

Provide actionable insights, key patterns, opportunities, risks, and recommendations in a concise, data-driven manner.
"""

    headers = {
        "Authorization": f"Bearer {os.getenv('NVIDIA_NIM_API_KEY')}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": NIM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 1024,
    }

    try:
        response = requests.post(NVIDIA_NIM_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()

        # Extract the generated content
        interpretation = result["choices"][0]["message"]["content"]

        return {
            "interpretation": interpretation,
            "focus": analysis_focus,
            "model_used": NIM_MODEL,
            "generated_at": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        return {
            "interpretation": "Unavailable",
            "error": str(e),
            "fallback": "Metrics calculated successfully; manual interpretation required.",
        }


# ---------- TOOL 7: DATA QUALITY VALIDATION ----------
@tool("Validate Financial Data Quality")
def validate_data_quality(params: dict) -> dict:
    """
    Analyzes historical price data for common anomalies like missing values,
    zero/negative prices, extreme daily moves (>20%), and large time gaps (>4 days).
    It returns a data quality score and a list of identified issues.

    The input must be a JSON string with the following required key:
    - 'price_data': (string/JSON, REQUIRED) The historical price data, typically
                    the 'price_data' dictionary output from 'fetch_stock_data'.
                    Must contain 'dates', 'close', 'high', 'low', and 'volume' lists.

    Example input: '{"price_data": "{...}"}'

    Returns a JSON string containing the data quality score, status, and list of issues.
    """
    price_data = json.loads(params.get("price_data", "{}"))
    df = pd.DataFrame(
        {
            "close": price_data["close"],
            "high": price_data["high"],
            "low": price_data["low"],
            "volume": price_data["volume"],
        }
    )

    issues = []
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        issues.append(f"Missing {missing_count} data points")
    if (df["close"] <= 0).any():
        issues.append("Zero or negative prices detected")
    extreme_moves = df["close"].pct_change().abs().gt(0.2).sum()
    if extreme_moves > 0:
        issues.append(f"{extreme_moves} extreme moves (>20%)")
    gaps = pd.to_datetime(price_data["dates"]).diff().gt(pd.Timedelta(days=4)).sum()
    if gaps > 0:
        issues.append(f"{gaps} data gaps detected")

    score = max(0, 1.0 - len(issues) * 0.15)

    return {
        "quality_score": score,
        "issues": issues,
        "status": "good" if score > 0.8 else "acceptable" if score > 0.6 else "poor",
        "validated_at": datetime.utcnow().isoformat(),
    }


# =======================
# WRAP TOOLS INTO CREWAI
# =======================

tools = [
    fetch_stock_data,
    calculate_technical_indicators,
    analyze_fundamentals,
    optimize_portfolio,
    calculate_risk_metrics,
    interpret_metrics_with_llm,
    validate_data_quality,
]

quantitative_analysis_agent = Agent(
    role="Quantitative Financial Analyst",
    backstory="""You are a seasoned, data-driven financial expert proficient in quantitative modeling and 
    algorithmic trading principles.Your mission is to interpret complex data and metrics into clear, 
    actionable financial strategies.""",
    goal="""
    You are a quantitative financial analyst agent. A user will provide a query such as:
    - "Analyze AAPL, MSFT for the last 1 year"
    - "Optimize portfolio: AAPL, MSFT, GOOGL, max Sharpe"

    Based on the query, you will:
    1. Identify stock tickers, period, and any optimization preferences.
    2. Retrieve stock data using fetch_stock_data.
    3. Validate data quality using validate_data_quality.
    4. Calculate technical indicators using calculate_technical_indicators.
    5. Analyze fundamentals using analyze_fundamentals.
    6. Optimize the portfolio if 3 or more tickers are given.
    7. Compute risk metrics.
    8. Generate LLM interpretation of all metrics.

    Return a single JSON object containing:
    {
        "tickers": [...],
        "data_quality": {...},
        "price_data": {...},
        "technical_indicators": {...},
        "fundamental_analysis": {...},
        "portfolio_optimization": {...},
        "risk_metrics": {...},
        "llm_interpretation": {...},
        "timestamp": "ISO-8601"
    }
    """,
    instructions="""
    - Parse user queries automatically to extract tickers, periods, and optimization targets.
    - Use all tools responsibly.
    - Validate data and handle errors.
    - Include confidence scores and flag anomalies.
    - Return structured JSON outputs.
    """,
    tools=tools,
    llm=llm,
)
user_query = "Optimize portfolio: AAPL, MSFT, GOOGL, max Sharpe"

quantitative_analysis_task = Task(
    description=f"Perform the full financial analysis as per the user's query: '{user_query}'",
    agent=quantitative_analysis_agent,  # Assign the agent to the task
    expected_output=quantitative_analysis_agent.goal,  # Use the agent's goal as the expected output structure
)

financial_crew = Crew(
    agents=[quantitative_analysis_agent],
    tasks=[quantitative_analysis_task],
    verbose=True,
)

result = financial_crew.kickoff()
print(result)
