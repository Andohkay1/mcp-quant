import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests, json, re, os
from scipy.stats import norm
from datetime import datetime

st.set_page_config(page_title="MCP Quant Dashboard", layout="wide")
st.title("MCP Quant Dashboard")

EDGE_THRESHOLD = 5
MIN_LIQUIDITY = 250
MAX_DAYS = 10
MOMENTUM_WEIGHT = 1.0
EWMA_LAMBDA = 0.94
JOURNAL_FILE = "mcp_journal.csv"


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
        return (1 - norm.cdf(z)) * 100 if direction == "above" else norm.cdf(z) * 100

    def historical_probability(self, ticker, target, days, direction, lookback=252):
        close = self.get_prices(ticker, "5y").tail(lookback)
        current = close.iloc[-1]
        required_return = target / current - 1
        future_returns = (close.shift(-days) / close - 1).dropna()
        return (future_returns >= required_return).mean() * 100 if direction == "above" else (future_returns <= required_return).mean() * 100

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

    def range_probability(self, ticker, lower, upper, days):
        close = self.get_prices(ticker, "1y")
        current = close.iloc[-1]
        vol = self.ewma_volatility(close)
        sigma = vol * np.sqrt(max(days, 1) / 252)

        z_low = np.log(lower / current) / sigma
        z_high = np.log(upper / current) / sigma

        return (norm.cdf(z_high) - norm.cdf(z_low)) * 100

    def score_market(self, market, ticker, target, days, direction, market_probability, market_type="price", upper=None):
        close = self.get_prices(ticker, "1y")
        current = close.iloc[-1]

        if market_type == "range" and upper is not None:
            ewma = self.range_probability(ticker, target, upper, days)
            hist = ewma
        else:
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

        size = 0
        if edge >= 5 and edge < 8:
            size = 2
        elif edge >= 8 and edge < 12:
            size = 3
        elif edge >= 12:
            size = 5

        return {
            "Market": market,
            "Ticker": ticker,
            "Current Price": round(current, 4),
            "Target": target,
            "Upper": upper,
            "Days": days,
            "Type": market_type,
            "Direction": direction,
            "Market Prob %": round(market_probability, 2),
            "EWMA Prob %": round(ewma, 2),
            "Historical Prob %": round(hist, 2),
            "Base Prob %": round(base, 2),
            "Momentum": momentum,
            "Momentum Adj %": round(mom_adj, 2),
            "Final Prob %": round(final, 2),
            "Edge %": round(edge, 2),
            "Signal": signal,
            "Position Size $": size
        }


asset_map = {
    "bitcoin": "BTC-USD", "btc": "BTC-USD",
    "ethereum": "ETH-USD", "eth": "ETH-USD",
    "xrp": "XRP-USD",
    "solana": "SOL-USD", "sol": "SOL-USD",
    "tesla": "TSLA", "tsla": "TSLA",
    "nvidia": "NVDA", "nvda": "NVDA",
    "silver": "SI=F", "gold": "GC=F",
    "oil": "CL=F", "wti": "CL=F"
}


def find_ticker(market):
    text = str(market).lower()
    for key, ticker in asset_map.items():
        if key in text:
            return ticker
    return None


def classify_market(market):
    text = str(market).lower()

    non_price_words = [
        "ai model", "#1 ai", "election", "nominee", "president",
        "fed chair", "ceo", "app store", "posts", "tariff",
        "unemployment", "gdp", "cpi", "inflation", "interest rate",
        "win", "wins", "champion", "world cup", "ufc", "nba", "nfl",
        "mlb", "tennis", "candidate"
    ]

    if any(w in text for w in non_price_words):
        return "event"

    if "between" in text:
        return "range"

    if "close above" in text or "closes above" in text or "close below" in text or "closes below" in text:
        return "daily_close"

    price_words = [
        "price", "above $", "below $", "greater than $", "less than $",
        "reach $", "hit $", "dip to $"
    ]

    if any(w in text for w in price_words):
        return "price"

    return "event"


def infer_direction(market):
    text = str(market).lower()
    if "below" in text or "less than" in text or "dip" in text or "low" in text:
        return "below"
    return "above"


def extract_numbers(market):
    text = str(market).replace(",", "")
    nums = re.findall(r"\$?(\d+(?:\.\d+)?)", text)
    return [float(x) for x in nums if float(x) < 100000 and float(x) != 2026]


def extract_target(market):
    nums = extract_numbers(market)
    return nums[0] if nums else None


def extract_upper(market):
    nums = extract_numbers(market)
    return nums[1] if len(nums) > 1 else None


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
    df["Upper"] = df["Market"].apply(extract_upper)
    df["Direction"] = df["Market"].apply(infer_direction)
    df["Market Type"] = df["Market"].apply(classify_market)

    df = df[
        (df["Market Type"].isin(["price", "range", "daily_close"])) &
        (df["Ticker"].notna()) &
        (df["Target"].notna()) &
        (df["Days"] >= 0) &
        (df["Days"] <= MAX_DAYS) &
        (df["Liquidity"] >= MIN_LIQUIDITY)
    ].copy()

    return df


def save_to_journal(row):
    journal_row = row.copy()
    journal_row["Date Saved"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    journal_row["Result"] = ""
    journal_row["PnL"] = ""

    if os.path.exists(JOURNAL_FILE):
        df = pd.read_csv(JOURNAL_FILE)
        df = pd.concat([df, pd.DataFrame([journal_row])], ignore_index=True)
    else:
        df = pd.DataFrame([journal_row])

    df.to_csv(JOURNAL_FILE, index=False)


def load_journal():
    if os.path.exists(JOURNAL_FILE):
        return pd.read_csv(JOURNAL_FILE)
    return pd.DataFrame()


tab1, tab2, tab3 = st.tabs(["Dashboard", "Journal", "Analytics"])

with tab1:
    st.subheader("Run Market Screener")

    if st.button("Run MCP Screener", key="run_screener_button"):
        markets_df = pull_markets()
        st.session_state["markets_df"] = markets_df

        engine = MCPQuantEngine()
        scored = []

        for _, row in markets_df.iterrows():
            try:
                scored.append(
                    engine.score_market(
                        market=row["Market"],
                        ticker=row["Ticker"],
                        target=row["Target"],
                        upper=row["Upper"],
                        days=max(int(row["Days"]), 1),
                        direction=row["Direction"],
                        market_probability=row["Market Prob %"],
                        market_type=row["Market Type"]
                    )
                )
            except Exception:
                pass

        results = pd.DataFrame(scored)

        if len(results) > 0:
            results = results.sort_values("Edge %", ascending=False)
            st.session_state["results"] = results
            results.to_csv("mcp_dashboard_results.csv", index=False)

    if "markets_df" in st.session_state:
        markets_df = st.session_state["markets_df"]

        st.metric("Markets Found", len(markets_df))

        st.subheader("Filtered Markets")
        st.dataframe(
            markets_df[["Market", "Market Type", "Ticker", "Target", "Upper", "Direction", "Market Prob %", "Days", "Liquidity"]],
            use_container_width=True
        )

    if "results" in st.session_state:
        results = st.session_state["results"]

        st.subheader("Top Trade Candidates")
        st.dataframe(results, use_container_width=True)

        buys = results[results["Signal"].isin(["BUY YES", "BUY NO"])]

        st.subheader("Actionable Trades")
        st.dataframe(buys, use_container_width=True)

        st.markdown("---")
        st.subheader("🔍 Explain Model")

        selected_trade = st.selectbox(
            "Select a trade to explain",
            results["Market"].tolist(),
            key="explain_trade_selectbox"
        )

        explain = results[results["Market"] == selected_trade].iloc[0]

        c1, c2, c3 = st.columns(3)

        c1.metric("Current Price", explain["Current Price"])
        c1.metric("Market Probability", f"{explain['Market Prob %']}%")

        c2.metric("EWMA Probability", f"{explain['EWMA Prob %']}%")
        c2.metric("Historical Probability", f"{explain['Historical Prob %']}%")

        c3.metric("Final Probability", f"{explain['Final Prob %']}%")
        c3.metric("Edge", f"{explain['Edge %']}%")

        st.markdown("### Model Components")
        st.write(f"**Market Type:** {explain['Type']}")
        st.write(f"**Direction:** {explain['Direction']}")
        st.write(f"**Target:** {explain['Target']}")
        st.write(f"**Upper Bound:** {explain['Upper']}")
        st.write(f"**Days to Expiry:** {explain['Days']}")
        st.write(f"**Momentum Score:** {explain['Momentum']}")
        st.write(f"**Momentum Adjustment:** {explain['Momentum Adj %']}%")
        st.write(f"**Signal:** {explain['Signal']}")
        st.write(f"**Suggested Position Size:** ${explain['Position Size $']}")

        if explain["Signal"] == "BUY YES":
            st.success("✅ The model believes the true probability is higher than the market price.")
        elif explain["Signal"] == "BUY NO":
            st.error("❌ The model believes the true probability is lower than the market price.")
        else:
            st.info("⚪ The model does not see enough edge to trade.")

        st.success("Saved results to mcp_dashboard_results.csv")

        st.subheader("Save Trade to Journal")

        trade_options = results["Market"].tolist()

        selected_save = st.selectbox(
            "Select trade to save",
            trade_options,
            key="save_trade_selectbox"
        )

        if st.button("Save Selected Trade", key="save_trade_button"):
            row = results[results["Market"] == selected_save].iloc[0].to_dict()
            save_to_journal(row)
            st.success("Trade saved to journal.")


with tab2:
    st.subheader("Trade Journal")
    journal = load_journal()

    if len(journal) > 0:
        st.dataframe(journal, use_container_width=True)
    else:
        st.info("No trades saved yet.")


with tab3:
    st.subheader("Analytics")
    journal = load_journal()

    if len(journal) > 0:
        total = len(journal)
        buy_count = journal[journal["Signal"].isin(["BUY YES", "BUY NO"])].shape[0]
        pass_count = journal[journal["Signal"] == "PASS"].shape[0]
        avg_edge = journal["Edge %"].mean()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Saved Trades", total)
        c2.metric("Buy Signals", buy_count)
        c3.metric("Pass Signals", pass_count)
        c4.metric("Avg Edge", round(avg_edge, 2))

        st.subheader("Edge Distribution")
        st.bar_chart(journal["Edge %"])
    else:
        st.info("No analytics available yet.")
