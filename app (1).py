import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests, json, re
from scipy.stats import norm

st.set_page_config(page_title="MCP Quant Dashboard", layout="wide")

st.title("MCP Quant Dashboard")

# -----------------------------
# CONFIG
# -----------------------------
EDGE_THRESHOLD = 5
MIN_LIQUIDITY = 250
MAX_DAYS = 10
NEAR_MONEY = 10
MOMENTUM_WEIGHT = 1.0
EWMA_LAMBDA = 0.94

# -----------------------------
# ENGINE
# -----------------------------
class MCPQuantEngine:
    def get_prices(self, ticker, period="5y"):
        data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return close.dropna()

    def ewma_volatility(self, close):
        returns = np.log(close / close.shift(1)).dropna()
        variance = returns.var()
        for r in returns:
            variance = EWMA_LAMBDA * variance + (1 - EWMA_LAMBDA) * (r ** 2)
        return np.sqrt(variance) * np.sqrt(252)

    def ewma_probability(self, ticker, target, days, direction):
        close = self.get_prices(ticker, "1y")
        current = close.iloc[-1]
        vol = self.ewma_volatility(close)
        sigma = vol * np.sqrt(max(days, 1) / 252)
        z = np.log(target / current) / sigma

        if direction == "above":
            return (1 - norm.cdf(z)) * 100
        else:
            return norm.cdf(z) * 100

    def historical_probability(self, ticker, target, days, direction, lookback=252):
        close = self.get_prices(ticker, "5y").tail(lookback)
        current = close.iloc[-1]
        required_return = target / current - 1
        future_returns = (close.shift(-days) / close - 1).dropna()

        if direction == "above":
            return (future_returns >= required_return).mean() * 100
        else:
            return (future_returns <= required_return).mean() * 100

    def momentum_score(self, ticker):
        close = self.get_prices(ticker, "1y")

        ma20 = close.rolling(20).mean()
        ma50 = close.rolling(50).mean()
        ma200 = close.rolling(200).mean()

        ret5 = (close.iloc[-1] / close.iloc[-6] - 1) * 100
        ret20 = (close.iloc[-1] / close.iloc[-21] - 1) * 100

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        rs = gain.rolling(14).mean() / loss.rolling(14).mean()
        rsi = 100 - (100 / (1 + rs))

        score = 0
        score += 2 if close.iloc[-1] > ma20.iloc[-1] else -2
        score += 3 if close.iloc[-1] > ma50.iloc[-1] else -3
        score += 5 if close.iloc[-1] > ma200.iloc[-1] else -5

        if ret5 > 5:
            score += 2
        elif ret5 < -5:
            score -= 2

        if ret20 > 10:
            score += 3
        elif ret20 < -10:
            score -= 3

        if rsi.iloc[-1] > 70:
            score += 2
        elif rsi.iloc[-1] < 30:
            score -= 2

        return score

    def score_market(self, market, ticker, target, days, direction, market_probability):
        close = self.get_prices(ticker, "1y")
        current = close.iloc[-1]

        ewma = self.ewma_probability(ticker, target, days, direction)
        hist = self.historical_probability(ticker, target, days, direction, 252)
        base = (ewma + hist) / 2

        momentum = self.momentum_score(ticker)
        mom_adj = (momentum / 20) * MOMENTUM_WEIGHT

        if direction == "below":
            mom_adj = -mom_adj

        final = max(0.01, min(99.99, base + mom_adj))
        edge = final - market_probability

        if edge >= EDGE_THRESHOLD:
            signal = "BUY YES"
        elif edge <= -EDGE_THRESHOLD:
            signal = "BUY NO"
        else:
            signal = "PASS"

        return {
            "Market": market,
            "Ticker": ticker,
            "Current Price": round(current, 4),
            "Target": target,
            "Days": days,
            "Direction": direction,
            "Market Prob %": round(market_probability, 2),
            "EWMA Prob %": round(ewma, 2),
            "Historical Prob %": round(hist, 2),
            "Base Prob %": round(base, 2),
            "Momentum": momentum,
            "Final Prob %": round(final, 2),
            "Edge %": round(edge, 2),
            "Signal": signal,
            "Position Size $": 2 if edge >= 5 and edge < 8 else 3 if edge >= 8 and edge < 12 else 5 if edge >= 12 else 0
        }

# -----------------------------
# SCREENER
# -----------------------------
asset_map = {
    "bitcoin": "BTC-USD", "btc": "BTC-USD",
    "ethereum": "ETH-USD", "eth": "ETH-USD",
    "xrp": "XRP-USD",
    "solana": "SOL-USD", "sol": "SOL-USD",
    "tesla": "TSLA", "tsla": "TSLA",
    "nvidia": "NVDA", "nvda": "NVDA",
    "silver": "SI=F",
    "gold": "GC=F",
    "oil": "CL=F", "wti": "CL=F"
}

def find_ticker(market):
    text = str(market).lower()
    for key, ticker in asset_map.items():
        if key in text:
            return ticker
    return None

def extract_target(market):
    text = str(market).replace(",", "")
    nums = re.findall(r"\$?(\d+(?:\.\d+)?)", text)
    nums = [float(x) for x in nums if float(x) < 100000 and float(x) != 2026]
    return nums[0] if nums else None

def infer_direction(market):
    text = str(market).lower()
    if "above" in text or "reach" in text or "greater" in text:
        return "above"
    if "below" in text or "dip" in text or "low" in text:
        return "below"
    return "above"

@st.cache_data(ttl=300)
def pull_markets():
    url = "https://gamma-api.polymarket.com/markets"
    params = {
        "closed": "false",
        "limit": 1000,
        "order": "volume",
        "ascending": "false"
    }

    markets_raw = requests.get(url, params=params).json()
    rows = []

    for m in markets_raw:
        try:
            outcome_prices = json.loads(m.get("outcomePrices", "[]"))
        except:
            outcome_prices = []

        yes_price = float(outcome_prices[0]) * 100 if len(outcome_prices) > 0 else None
        no_price = float(outcome_prices[1]) * 100 if len(outcome_prices) > 1 else None

        rows.append({
            "Market": m.get("question"),
            "Resolution Date": m.get("endDate"),
            "Market Prob %": yes_price,
            "No Prob %": no_price,
            "Volume": m.get("volumeNum"),
            "Liquidity": m.get("liquidityNum"),
            "clobTokenIds": m.get("clobTokenIds")
        })

    df = pd.DataFrame(rows)
    df["Resolution Date"] = pd.to_datetime(df["Resolution Date"], errors="coerce", utc=True)
    df["Days"] = (df["Resolution Date"] - pd.Timestamp.now(tz="UTC")).dt.days

    df["Ticker"] = df["Market"].apply(find_ticker)
    df["Target"] = df["Market"].apply(extract_target)
    df["Direction"] = df["Market"].apply(infer_direction)

    df = df[
        (df["Ticker"].notna()) &
        (df["Target"].notna()) &
        (df["Days"] >= 0) &
        (df["Days"] <= MAX_DAYS) &
        (df["Liquidity"] >= MIN_LIQUIDITY)
    ].copy()

    return df

# -----------------------------
# DASHBOARD
# -----------------------------
if st.button("Run MCP Screener"):
    markets_df = pull_markets()

    st.subheader("Markets Found")
    st.write(len(markets_df))

    engine = MCPQuantEngine()
    scored = []

    for _, row in markets_df.iterrows():
        try:
            scored.append(
                engine.score_market(
                    market=row["Market"],
                    ticker=row["Ticker"],
                    target=row["Target"],
                    days=int(row["Days"]),
                    direction=row["Direction"],
                    market_probability=row["Market Prob %"]
                )
            )
        except Exception as e:
            pass

    results = pd.DataFrame(scored)

    if len(results) > 0:
        results = results.sort_values("Edge %", ascending=False)

        st.subheader("Top Trade Candidates")
        st.dataframe(results, use_container_width=True)

        buys = results[results["Signal"].isin(["BUY YES", "BUY NO"])]

        st.subheader("Actionable Trades")
        st.dataframe(buys, use_container_width=True)

        results.to_csv("mcp_dashboard_results.csv", index=False)
        st.success("Saved results to mcp_dashboard_results.csv")
    else:
        st.warning("No scored markets found.")
