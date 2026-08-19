import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import json
import re
import xml.etree.ElementTree as ET
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from datetime import datetime

import gspread
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="MCP Quant Dashboard", layout="wide")
st.title("MCP Quant Dashboard")

EDGE_THRESHOLD = 5
MIN_LIQUIDITY = 250
MAX_DAYS = 60
MOMENTUM_WEIGHT = 1.0
EWMA_LAMBDA = 0.94
SPREADSHEET_NAME = "Polymarket Journal"
WORKSHEET_NAME = "Trades"
STARTING_BANKROLL = 100
BUY_THRESHOLD = 6
MIN_ACTIONABLE_EDGE = 8.0
MIN_TRADABLE_ENTRY_PRICE_PCT = 15.0
MAX_TRADABLE_ENTRY_PRICE_PCT = 85.0
MIN_SELECTED_MODEL_PROB_PCT = 55.0
MIN_TRADING_DAYS_REMAINING = 0.01
CALIBRATION_MIN_SAMPLES = 30
CALIBRATION_BLEND = 0.75
CALIBRATION_MAX_SHIFT = 15.0


def _safe_float(value, default=0.0):
    try:
        number = float(value)
        return number if np.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def forecast_confidence(ewma_probability, historical_probability, liquidity):
    """Transparent confidence label for forecast review."""
    disagreement = abs(_safe_float(ewma_probability) - _safe_float(historical_probability))
    liquidity_value = _safe_float(liquidity)
    if disagreement <= 15 and liquidity_value >= 1000:
        return "High"
    if disagreement <= 30 and liquidity_value >= MIN_LIQUIDITY:
        return "Medium"
    return "Low"


def brier_score(probability_pct, outcome_yes):
    """Binary Brier score: 0 is perfect and 1 is the worst possible score."""
    probability = np.clip(_safe_float(probability_pct) / 100.0, 0.0, 1.0)
    outcome = 1.0 if bool(outcome_yes) else 0.0
    return float((probability - outcome) ** 2)


def calibrate_probability(raw_probability_pct, resolved_df, min_samples=CALIBRATION_MIN_SAMPLES):
    """
    Calibrate the existing model probability using resolved historical trades.

    The original model probability is preserved. Calibration is deliberately
    conservative: it requires a minimum sample, shrinks the regression output
    toward the raw probability, and caps the adjustment size.
    """
    raw = float(np.clip(_safe_float(raw_probability_pct, 50.0), 0.01, 99.99))

    if resolved_df is None or resolved_df.empty:
        return raw, "RAW (0 resolved)"

    data = resolved_df.copy()
    required = {"Final Prob %", "Result"}
    if not required.issubset(data.columns):
        return raw, "RAW (missing calibration columns)"

    data["Final Prob %"] = pd.to_numeric(data["Final Prob %"], errors="coerce")
    data["Outcome"] = (
        data["Result"].astype(str).str.strip().str.upper().map({"YES": 1, "NO": 0})
    )
    data = data.dropna(subset=["Final Prob %", "Outcome"])
    data = data[data["Final Prob %"].between(0, 100)]

    if len(data) < min_samples:
        return raw, f"RAW (<{min_samples} resolved)"

    if data["Outcome"].nunique() < 2:
        return raw, "RAW (one outcome only)"

    try:
        X = (data[["Final Prob %"]].to_numpy(dtype=float) / 100.0)
        y = data["Outcome"].to_numpy(dtype=int)

        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        model.fit(X, y)

        regression_probability = float(
            model.predict_proba(np.array([[raw / 100.0]]))[0, 1] * 100.0
        )

        # Conservative shrinkage prevents a small live sample from moving
        # probabilities too aggressively.
        calibrated = (
            CALIBRATION_BLEND * regression_probability
            + (1.0 - CALIBRATION_BLEND) * raw
        )

        shift = calibrated - raw
        if abs(shift) > CALIBRATION_MAX_SHIFT:
            calibrated = raw + np.sign(shift) * CALIBRATION_MAX_SHIFT

        calibrated = float(np.clip(calibrated, 1.0, 99.0))
        return calibrated, f"CALIBRATED ({len(data)} resolved)"

    except Exception as error:
        return raw, f"RAW (calibration error: {type(error).__name__})"


def evaluate_execution_approval(row):
    signal = str(row.get("Signal", ""))
    edge = _safe_float(row.get("Edge %"), 0.0)
    entry_price = _safe_float(row.get("Entry Price %"), 0.0)
    days_remaining = _safe_float(row.get("Days"), 0.0)
    model_yes = _safe_float(row.get("Calibrated Prob %", row.get("Final Prob %")), 0.0)
    if signal == "BUY YES":
        selected_model_probability = model_yes
    elif signal == "BUY NO":
        selected_model_probability = 100.0 - model_yes
    else:
        return False, "No BUY signal", 0.0
    reasons = []
    if edge < MIN_ACTIONABLE_EDGE:
        reasons.append(f"edge below {MIN_ACTIONABLE_EDGE:.0f}%")
    if entry_price < MIN_TRADABLE_ENTRY_PRICE_PCT:
        reasons.append(f"entry price below {MIN_TRADABLE_ENTRY_PRICE_PCT:.0f}%")
    if entry_price > MAX_TRADABLE_ENTRY_PRICE_PCT:
        reasons.append(f"entry price above {MAX_TRADABLE_ENTRY_PRICE_PCT:.0f}%")
    if selected_model_probability < MIN_SELECTED_MODEL_PROB_PCT:
        reasons.append(f"selected-outcome model probability below {MIN_SELECTED_MODEL_PROB_PCT:.0f}%")
    if days_remaining < MIN_TRADING_DAYS_REMAINING:
        reasons.append("too little trading time remaining")
    approved = not reasons
    return approved, ("Approved" if approved else "; ".join(reasons)), selected_model_probability


class MCPTradingClient:
    """Small execution client for the Moreton Capital Partners Trading Desk API."""

    def __init__(self):
        if "mcp_trading" not in st.secrets:
            raise RuntimeError("Missing [mcp_trading] in Streamlit Secrets.")

        cfg = st.secrets["mcp_trading"]
        self.base_url = str(cfg.get("base_url", "")).rstrip("/")
        self.email = str(cfg.get("email", ""))
        self.password = str(cfg.get("password", ""))
        self.live_enabled = bool(cfg.get("live_trading_enabled", False))
        self.max_order_amount = float(cfg.get("max_order_amount", 1.0))
        self.timeout = 20

        if not self.base_url or not self.email or not self.password:
            raise RuntimeError("MCP base_url, email, and password are required in Streamlit Secrets.")

    def _request(self, method, path, *, token=None, params=None, json_body=None):
        headers = {"Accept": "application/json"}
        if json_body is not None:
            headers["Content-Type"] = "application/json"
        if token:
            headers["Authorization"] = f"Bearer {token}"

        response = requests.request(
            method,
            f"{self.base_url}{path}",
            headers=headers,
            params=params,
            json=json_body,
            timeout=self.timeout,
        )

        try:
            payload = response.json()
        except ValueError:
            payload = {"detail": response.text or "Non-JSON response"}

        if not response.ok:
            detail = payload.get("detail", payload) if isinstance(payload, dict) else payload
            raise RuntimeError(f"MCP API {response.status_code}: {detail}")

        return payload

    def login(self):
        payload = self._request(
            "POST",
            "/v1/auth/login",
            json_body={"email": self.email, "password": self.password},
        )
        token = payload.get("access_token")
        if not token:
            raise RuntimeError("MCP login succeeded but no access_token was returned.")
        return token

    def balance(self, token):
        return self._request("GET", "/v1/balance", token=token)

    def price_estimate(self, token, token_id, side, amount):
        return self._request(
            "GET",
            f"/v1/markets/price-estimate/{token_id}",
            token=token,
            params={"side": side, "amount": str(amount), "order_type": "FOK"},
        )

    def place_market_order(self, token, token_id, side, amount):
        if not self.live_enabled:
            raise RuntimeError(
                "Live trading is disabled. Set live_trading_enabled = true in [mcp_trading] only when ready."
            )
        return self._request(
            "POST",
            "/v1/orders/market",
            token=token,
            json_body={
                "token_id": str(token_id),
                "side": side,
                "amount": str(amount),
                "order_type": "FOK",
            },
        )


def parse_outcome_token_ids(raw_value):
    """Return outcome token IDs in Polymarket outcome order: YES first, NO second."""
    if isinstance(raw_value, list):
        values = raw_value
    elif isinstance(raw_value, str):
        try:
            values = json.loads(raw_value)
        except json.JSONDecodeError:
            values = [x.strip() for x in raw_value.strip("[]").split(",") if x.strip()]
    else:
        values = []

    return [str(value).strip().strip('"').strip("'") for value in values]


def token_for_signal(row):
    token_ids = parse_outcome_token_ids(row.get("clobTokenIds"))
    if len(token_ids) < 2:
        raise RuntimeError("This market does not contain both YES and NO outcome token IDs.")

    signal = str(row.get("Signal", ""))
    if signal == "BUY YES":
        return token_ids[0], "YES"
    if signal == "BUY NO":
        return token_ids[1], "NO"
    raise RuntimeError("The selected market is not an actionable BUY signal.")


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

        sigma = vol * np.sqrt(max(float(days), 1 / 390) / 252)
        z = np.log(target / current) / sigma

        if direction == "above":
            return (1 - norm.cdf(z)) * 100

        return norm.cdf(z) * 100

    def historical_probability(self, ticker, target, days, direction, lookback=252):
        close = self.get_prices(ticker, "5y").tail(lookback)
        current = close.iloc[-1]

        required_return = target / current - 1
        historical_days = max(int(np.ceil(float(days))), 1)
        future_returns = (close.shift(-historical_days) / close - 1).dropna()

        if direction == "above":
            return (future_returns >= required_return).mean() * 100

        return (future_returns <= required_return).mean() * 100

    def get_ohlc(self, ticker, period="5y", interval="1d"):
        data = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data.dropna(how="all")

    def ewma_barrier_probability(self, ticker, target, days, direction):
        close = self.get_prices(ticker, "1y")
        current = float(close.iloc[-1])
        if direction == "above" and current >= float(target):
            return 100.0
        if direction == "below" and current <= float(target):
            return 100.0
        vol = float(self.ewma_volatility(close))
        horizon_days = max(float(days), 1 / 390)
        sigma_t = vol * np.sqrt(horizon_days / 252)
        if not np.isfinite(sigma_t) or sigma_t <= 0:
            return 0.0
        log_distance = abs(np.log(float(target) / current))
        probability = 2 * (1 - norm.cdf(log_distance / sigma_t))
        return float(np.clip(probability * 100, 0.0, 100.0))

    def historical_barrier_probability(self, ticker, target, days, direction, lookback=756):
        ohlc = self.get_ohlc(ticker, period="5y", interval="1d").tail(lookback)
        if ohlc.empty or not {"Close", "High", "Low"}.issubset(ohlc.columns):
            return np.nan
        current = float(ohlc["Close"].dropna().iloc[-1])
        target_ratio = float(target) / current
        horizon = max(int(np.ceil(float(days))), 1)
        hits = []
        for i in range(0, len(ohlc) - horizon):
            start_price = float(ohlc["Close"].iloc[i])
            if not np.isfinite(start_price) or start_price <= 0:
                continue
            window = ohlc.iloc[i + 1:i + horizon + 1]
            if direction == "above":
                hit = float(window["High"].max()) >= start_price * target_ratio
            else:
                hit = float(window["Low"].min()) <= start_price * target_ratio
            hits.append(bool(hit))
        return float(np.mean(hits) * 100) if hits else np.nan

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

        sigma = vol * np.sqrt(max(float(days), 1 / 390) / 252)

        z_low = np.log(lower / current) / sigma
        z_high = np.log(upper / current) / sigma

        return (norm.cdf(z_high) - norm.cdf(z_low)) * 100

    def score_market(self, row, calibration_df=None):
        market = row["Market"]
        ticker = row["Ticker"]
        target = row["Target"]
        upper = row["Upper"]
        days = max(float(row["Days"]), 1 / 390)
        direction = row["Direction"]
        market_probability = row["Market Prob %"]
        market_type = row["Market Type"]

        close = self.get_prices(ticker, "1y")
        current = close.iloc[-1]

        if market_type == "range" and pd.notna(upper):
            ewma = self.range_probability(ticker, target, upper, days)
            hist = ewma
        elif market_type == "barrier":
            ewma = self.ewma_barrier_probability(ticker, target, days, direction)
            hist = self.historical_barrier_probability(ticker, target, days, direction)
            if pd.isna(hist):
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

        calibrated_prob, calibration_status = calibrate_probability(
            final, calibration_df
        )

        model_yes = calibrated_prob
        model_no = 100 - calibrated_prob

        market_yes = market_probability
        market_no = row["No Prob %"]

        yes_edge = model_yes - market_yes
        no_edge = model_no - market_no

        if yes_edge > BUY_THRESHOLD:
            signal = "BUY YES"
            edge = yes_edge
        elif no_edge > BUY_THRESHOLD:
            signal = "BUY NO"
            edge = no_edge
        else:
            signal = "PASS"
            edge = max(yes_edge, no_edge)

        size = 0
        abs_edge = abs(edge)

        if signal != "PASS":
            if BUY_THRESHOLD < abs_edge < 8:
                size = 2
            elif 8 <= abs_edge < 12:
                size = 3
            elif abs_edge >= 12:
                size = 5

        if signal == "BUY YES":
            entry_side = "YES"
            entry_price = market_yes
        elif signal == "BUY NO":
            entry_side = "NO"
            entry_price = market_no
        else:
            entry_side = ""
            entry_price = 0

        result = {
            "Market ID": row["Market ID"],
            "Market": market,
            "Ticker": ticker,
            "Current Price": round(current, 4),
            "Target": target,
            "Upper": upper,
            "Resolution Date": row["Resolution Date"],
            "Days": days,
            "Type": market_type,
            "Direction": direction,
            "Market Prob %": round(market_yes, 2),
            "No Prob %": round(market_no, 2),
            "EWMA Prob %": round(ewma, 2),
            "Historical Prob %": round(hist, 2),
            "Base Prob %": round(base, 2),
            "Momentum": momentum,
            "Momentum Adj %": round(mom_adj, 2),
            "Final Prob %": round(final, 2),
            "Calibrated Prob %": round(calibrated_prob, 2),
            "Calibration Status": calibration_status,
            "YES Edge %": round(yes_edge, 2),
            "NO Edge %": round(no_edge, 2),
            "Edge %": round(edge, 2),
            "Signal": signal,
            "Entry Side": entry_side,
            "Entry Price %": round(entry_price, 2),
            "Position Size $": size,
            "clobTokenIds": row["clobTokenIds"],
        }
        approved, reason, selected_prob = evaluate_execution_approval(result)
        result["Selected Model Prob %"] = round(selected_prob, 2)
        result["Execution Approved"] = approved
        result["Execution Decision"] = reason
        result["Forecast Confidence"] = forecast_confidence(
            result["EWMA Prob %"], result["Historical Prob %"], row.get("Liquidity", 0)
        )
        return result


# Reliable aliases are checked first. Unknown assets are resolved dynamically
# through Yahoo Finance only after the market has passed the binary-price filter.
ASSET_ALIASES = {
    # Crypto
    "bitcoin": "BTC-USD", "btc": "BTC-USD",
    "ethereum": "ETH-USD", "ether": "ETH-USD", "eth": "ETH-USD",
    "solana": "SOL-USD", "sol": "SOL-USD", "xrp": "XRP-USD",
    "dogecoin": "DOGE-USD", "doge": "DOGE-USD",
    "cardano": "ADA-USD", "ada": "ADA-USD",
    "avalanche": "AVAX-USD", "avax": "AVAX-USD",
    "chainlink": "LINK-USD", "link": "LINK-USD",
    "sui": "SUI20947-USD", "bnb": "BNB-USD",
    # Major equities
    "apple": "AAPL", "tesla": "TSLA", "nvidia": "NVDA",
    "microsoft": "MSFT", "amazon": "AMZN", "alphabet": "GOOGL",
    "google": "GOOGL", "meta": "META", "netflix": "NFLX",
    "amd": "AMD", "coinbase": "COIN", "palantir": "PLTR",
    "spotify": "SPOT", "uber": "UBER", "berkshire hathaway": "BRK-B",
    # Indices
    "s&p 500": "^GSPC", "s&p500": "^GSPC", "sp 500": "^GSPC",
    "nasdaq 100": "^NDX", "nasdaq": "^IXIC", "dow jones": "^DJI",
    "russell 2000": "^RUT",
    # Commodities
    "gold": "GC=F", "silver": "SI=F", "wti": "CL=F",
    "crude oil": "CL=F", "oil": "CL=F", "brent": "BZ=F",
    "natural gas": "NG=F", "copper": "HG=F",
    # FX
    "eur/usd": "EURUSD=X", "euro": "EURUSD=X",
    "gbp/usd": "GBPUSD=X", "pound": "GBPUSD=X",
    "usd/jpy": "JPY=X", "yen": "JPY=X",
}

PRICE_MARKET_PATTERNS = [
    r"\b(?:reach|hit|touch|dip to|fall to|rise to)\b.*?\$?\d",
    r"\b(?:above|below|over|under|greater than|less than)\b.*?\$?\d",
    r"\b(?:close|closes|finish|finishes|settle|settles)\b.*?\b(?:above|below|over|under)\b",
    r"\bbetween\b.*?\d.*?\band\b.*?\d",
]

NON_PRICE_EVENT_WORDS = [
    "earnings", "revenue", "eps", "market cap", "fully diluted", "fdv",
    "acquire", "acquisition", "merger", "ipo", "etf approval", "approve",
    "regulation", "lawsuit", "ceo", "president", "election", "nominee",
    "fed chair", "interest rate", "cpi", "inflation", "gdp", "unemployment",
    "tariff", "win", "wins", "champion", "world cup", "ufc", "nba",
    "nfl", "mlb", "tennis", "reserves", "inventory", "inventories",
    "production", "output", "transit", "transits", "shipments",
    "strait of hormuz", "temperature", "rainfall", "kills", "total rounds",
]


def classify_market(market):
    """Classify only binary financial price markets supported by the model."""
    text = str(market or "").lower().strip()

    if any(word in text for word in NON_PRICE_EVENT_WORDS):
        return "event"
    if "between" in text and len(extract_numbers(text)) >= 2:
        return "range"
    if re.search(r"\b(?:close|closes|finish|finishes|settle|settles)\b", text) and re.search(
        r"\b(?:above|below|over|under)\b", text
    ):
        return "daily_close"
    if re.search(r"\b(?:hit|reach|touch|dip to|fall to|rise to|trade as high as|trade as low as)\b", text) or re.search(r"\((?:high|low)\)", text):
        return "barrier"
    if any(re.search(pattern, text) for pattern in PRICE_MARKET_PATTERNS):
        return "price"
    return "event"


def infer_direction(market):
    text = str(market or "").lower()
    if re.search(r"\b(?:below|under|less than|dip|fall to|low)\b", text):
        return "below"
    return "above"


def _parse_number(value):
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def extract_price_bounds(market):
    """Extract contract price levels while ignoring dates and asset-name numbers."""
    text = str(market or "")
    number = r"(?:US\$|\$)?\s*(\d[\d,]*(?:\.\d+)?)"

    range_match = re.search(
        rf"\bbetween\s+{number}\s+(?:and|to|-)\s+{number}",
        text,
        flags=re.I,
    )
    if range_match:
        lower = _parse_number(range_match.group(1))
        upper = _parse_number(range_match.group(2))
        if lower is not None and upper is not None and 0 < lower < upper:
            return lower, upper

    trigger_patterns = [
        rf"\b(?:close|closes|closed|finish|finishes|settle|settles)[^?]*?\b(?:above|below|over|under)\s+{number}",
        rf"\b(?:above|below|over|under|greater than|less than)\s+{number}",
        rf"\b(?:reach|hit|touch|dip to|fall to|rise to|trade as high as|trade as low as)\s+(?:\((?:high|low)\)\s*)?{number}",
        rf"\((?:high|low)\)\s*{number}",
    ]
    for pattern in trigger_patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            value = _parse_number(match.group(1))
            if value is not None and value > 0:
                return value, None

    return None, None


def extract_numbers(market):
    lower, upper = extract_price_bounds(market)
    return [value for value in (lower, upper) if value is not None]


def extract_target(market):
    return extract_price_bounds(market)[0]


def extract_upper(market):
    return extract_price_bounds(market)[1]


def extract_asset_phrase(market):
    """Extract the likely underlying name from a binary price question."""
    text = str(market or "").strip()
    text = re.sub(r"^(will|can|could|does|is)\s+", "", text, flags=re.I)
    split_patterns = [
        r"\s+(?:reach|hit|touch|dip to|fall to|rise to)\s+",
        r"\s+(?:close|closes|finish|finishes|settle|settles)\s+",
        r"\s+(?:be|trade)\s+(?:above|below|over|under|between)\s+",
        r"\s+(?:above|below|over|under|greater than|less than)\s+",
    ]
    phrase = text
    for pattern in split_patterns:
        parts = re.split(pattern, phrase, maxsplit=1, flags=re.I)
        if len(parts) > 1:
            phrase = parts[0]
            break
    phrase = re.sub(r"\b(price|stock|shares|token|coin|index|futures?)\b", "", phrase, flags=re.I)
    phrase = re.sub(r"[^A-Za-z0-9&./\- ]", " ", phrase)
    return re.sub(r"\s+", " ", phrase).strip(" -?")


@st.cache_data(ttl=86400, show_spinner=False)
def yahoo_symbol_search(query):
    """Resolve a name to a Yahoo symbol. Returns None when confidence is weak."""
    query = str(query or "").strip()
    if not query:
        return None
    try:
        response = requests.get(
            "https://query1.finance.yahoo.com/v1/finance/search",
            params={"q": query, "quotesCount": 8, "newsCount": 0},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=8,
        )
        response.raise_for_status()
        quotes = response.json().get("quotes", [])
        allowed = {"EQUITY", "ETF", "INDEX", "CRYPTOCURRENCY", "FUTURE", "CURRENCY"}
        for quote in quotes:
            symbol = quote.get("symbol")
            quote_type = str(quote.get("quoteType", "")).upper()
            if symbol and quote_type in allowed:
                return symbol
    except Exception:
        return None
    return None


def find_ticker(market):
    raw_text = str(market or "")
    text = raw_text.lower()

    # Explicit SPY contracts reference the ETF price, not the index level.
    if re.search(r"(?<!\w)spy(?!\w)", text):
        return "SPY"

    # Longest aliases first to avoid matching "oil" before "crude oil".
    for key in sorted(ASSET_ALIASES, key=len, reverse=True):
        if re.search(rf"(?<!\w){re.escape(key)}(?!\w)", text):
            return ASSET_ALIASES[key]

    # Direct ticker notation such as $AAPL or (AAPL).
    direct = re.search(r"\$([A-Z]{1,6})\b|\(([A-Z]{1,6})\)", str(market or ""))
    if direct:
        return direct.group(1) or direct.group(2)

    return yahoo_symbol_search(extract_asset_phrase(market))


@st.cache_data(ttl=300)
def pull_markets():
    """Paginate the stable Gamma discovery feed and keep supported price markets.

    Only market retrieval is expanded here. The model, execution, journal, and
    approval logic remain unchanged.
    """
    url = "https://gamma-api.polymarket.com/markets"
    page_size = 100
    max_pages = 20  # up to 2,000 active markets per scan
    markets_raw = []
    seen_ids = set()

    for page_number in range(max_pages):
        offset = page_number * page_size
        params = {
            "closed": "false",
            "active": "true",
            "limit": page_size,
            "offset": offset,
            "order": "volume",
            "ascending": "false",
        }

        response = requests.get(url, params=params, timeout=25)
        response.raise_for_status()
        page = response.json()

        if not isinstance(page, list) or not page:
            break

        new_on_page = 0
        for market in page:
            market_id = str(market.get("id", "")).strip()
            dedupe_key = market_id or str(market.get("conditionId", "")).strip() or str(market.get("question", "")).strip()
            if not dedupe_key or dedupe_key in seen_ids:
                continue
            seen_ids.add(dedupe_key)
            markets_raw.append(market)
            new_on_page += 1

        # A short page means the catalogue is exhausted. Also stop if the API
        # returns a repeated page that adds nothing new.
        if len(page) < page_size or new_on_page == 0:
            break

    rows = []
    for m in markets_raw:
        # Defensive checks in case Gamma returns stale records despite the query.
        if bool(m.get("closed", False)) or m.get("active") is False:
            continue

        try:
            outcome_prices = json.loads(m.get("outcomePrices", "[]"))
        except Exception:
            outcome_prices = []

        # The current model and executor require a binary YES/NO market.
        if len(outcome_prices) != 2:
            continue

        try:
            yes_probability = float(outcome_prices[0]) * 100
            no_probability = float(outcome_prices[1]) * 100
        except (TypeError, ValueError):
            continue

        rows.append({
            "Market ID": m.get("id"),
            "Market": m.get("question"),
            "Resolution Date": m.get("endDate"),
            "Market Prob %": yes_probability,
            "No Prob %": no_probability,
            "Volume": pd.to_numeric(m.get("volumeNum"), errors="coerce"),
            "Liquidity": pd.to_numeric(m.get("liquidityNum"), errors="coerce"),
            "clobTokenIds": m.get("clobTokenIds"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.drop_duplicates(subset=["Market ID"], keep="first")
    df["Resolution Date"] = pd.to_datetime(df["Resolution Date"], errors="coerce", utc=True)
    df["Days"] = (df["Resolution Date"] - pd.Timestamp.now(tz="UTC")).dt.total_seconds() / 86400
    df["Market Type"] = df["Market"].apply(classify_market)

    financial_candidates = df[df["Market Type"].isin(["price", "barrier", "range", "daily_close"])].copy()
    financial_candidates["Asset Phrase"] = financial_candidates["Market"].apply(extract_asset_phrase)
    financial_candidates["Ticker"] = financial_candidates["Market"].apply(find_ticker)
    financial_candidates["Target"] = financial_candidates["Market"].apply(extract_target)
    financial_candidates["Upper"] = pd.Series(np.nan, index=financial_candidates.index, dtype="float64")
    range_mask = financial_candidates["Market Type"].eq("range")
    if range_mask.any():
        parsed_upper = pd.to_numeric(
            financial_candidates.loc[range_mask, "Market"].apply(extract_upper),
            errors="coerce",
        )
        financial_candidates.loc[range_mask, "Upper"] = parsed_upper.to_numpy()
    financial_candidates["Direction"] = financial_candidates["Market"].apply(infer_direction)

    def rejection_reason(row):
        if pd.isna(row["Resolution Date"]): return "Missing resolution date"
        if pd.isna(row["Ticker"]) or not str(row["Ticker"]).strip(): return "Asset could not be resolved"
        if pd.isna(row["Target"]): return "Target price could not be parsed"
        if row["Market Type"] == "range" and pd.isna(row["Upper"]): return "Upper range could not be parsed"
        if row["Days"] < 0: return "Market already expired"
        if row["Days"] > MAX_DAYS: return f"More than {MAX_DAYS} days to expiry"
        if pd.isna(row["Liquidity"]) or row["Liquidity"] < MIN_LIQUIDITY: return "Insufficient liquidity"
        return "Eligible"

    financial_candidates["Screen Result"] = financial_candidates.apply(rejection_reason, axis=1)
    eligible = financial_candidates[financial_candidates["Screen Result"] == "Eligible"].copy()

    eligible.attrs["scan_stats"] = {
        "all_binary_markets": len(df),
        "binary_price_markets": len(financial_candidates),
        "eligible_markets": len(eligible),
        "rejected_markets": len(financial_candidates) - len(eligible),
        "catalogue_markets_fetched": len(markets_raw),
    }
    eligible.attrs["rejections"] = financial_candidates[
        financial_candidates["Screen Result"] != "Eligible"
    ][["Market", "Market Type", "Asset Phrase", "Ticker", "Screen Result", "Liquidity", "Days"]]
    return eligible


def get_news(ticker, limit=5):
    query = ticker.replace("-", " ")

    url = (
        "https://news.google.com/rss/search?"
        f"q={query}+finance+stock+crypto&hl=en-US&gl=US&ceid=US:en"
    )

    try:
        r = requests.get(url, timeout=10)
        root = ET.fromstring(r.content)

        news = []

        for item in root.findall(".//item")[:limit]:
            news.append(
                {
                    "Title": item.find("title").text,
                    "Date": item.find("pubDate").text,
                    "Link": item.find("link").text,
                }
            )

        return pd.DataFrame(news)

    except Exception as e:
        return pd.DataFrame(
            [
                {
                    "Title": f"News fetch failed: {e}",
                    "Date": "",
                    "Link": "",
                }
            ]
        )


def fetch_market_by_id(market_id):
    try:
        url = f"https://gamma-api.polymarket.com/markets/{market_id}"
        r = requests.get(url, timeout=10)

        if r.status_code == 200:
            return r.json()

    except Exception:
        pass

    return None


def infer_winner_from_market(market):
    if not market:
        return None, False

    closed = market.get("closed", False)

    if not closed:
        return None, False

    winner = (
        market.get("winningOutcome")
        or market.get("winner")
        or market.get("resolution")
    )

    if winner:
        winner = str(winner).upper()

        if "YES" in winner:
            return "YES", True

        if "NO" in winner:
            return "NO", True

    try:
        prices = json.loads(market.get("outcomePrices", "[]"))

        if len(prices) >= 2:
            yes = float(prices[0])
            no = float(prices[1])

            if yes > 0.95:
                return "YES", True

            if no > 0.95:
                return "NO", True

    except Exception:
        pass

    return None, True


def calculate_pnl(entry_side, winner, entry_price_pct, position_size):
    if not entry_side or not winner or position_size <= 0 or entry_price_pct <= 0:
        return 0

    entry_price = entry_price_pct / 100
    shares = position_size / entry_price

    if entry_side == winner:
        payout = shares * 1
        return round(payout - position_size, 2)

    return round(-position_size, 2)


JOURNAL_COLUMNS = [
    "Market ID",
    "Market",
    "Ticker",
    "Current Price",
    "Target",
    "Upper",
    "Resolution Date",
    "Days",
    "Type",
    "Direction",
    "Market Prob %",
    "No Prob %",
    "EWMA Prob %",
    "Historical Prob %",
    "Base Prob %",
    "Momentum",
    "Momentum Adj %",
    "Final Prob %",
    "Calibrated Prob %",
    "Calibration Status",
    "YES Edge %",
    "NO Edge %",
    "Edge %",
    "Signal",
    "Entry Side",
    "Entry Price %",
    "Position Size $",
    "Forecast Confidence",
    "clobTokenIds",
    "Execution Token ID",
    "Execution Outcome",
    "Estimated Fill Price",
    "Executed Amount pUSD",
    "MCP Order ID",
    "CLOB Order ID",
    "Execution Status",
    "Execution Response",
    "Date Saved",
    "Status",
    "Result",
    "PnL",
]

NUMERIC_JOURNAL_COLUMNS = [
    "Current Price",
    "Target",
    "Upper",
    "Days",
    "Market Prob %",
    "No Prob %",
    "EWMA Prob %",
    "Historical Prob %",
    "Base Prob %",
    "Momentum",
    "Momentum Adj %",
    "Final Prob %",
    "Calibrated Prob %",
    "YES Edge %",
    "NO Edge %",
    "Edge %",
    "Entry Price %",
    "Position Size $",
    "Estimated Fill Price",
    "Executed Amount pUSD",
    "PnL",
]


def _clean_sheet_value(value):
    """Convert Python and pandas values to Google Sheets-safe values."""
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()

    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value)

    if isinstance(value, np.generic):
        return value.item()

    return value


@st.cache_resource
def get_journal_worksheet():
    """Connect to the permanent Google Sheets trading journal."""
    required_sections = {"google_service_account", "google_sheets"}
    missing_sections = required_sections.difference(st.secrets.keys())

    if missing_sections:
        missing = ", ".join(sorted(missing_sections))
        raise RuntimeError(
            f"Missing Streamlit Secrets section(s): {missing}. "
            "Add the Google service-account and sheet settings first."
        )

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]

    credentials_info = dict(st.secrets["google_service_account"])
    credentials = Credentials.from_service_account_info(
        credentials_info,
        scopes=scopes,
    )
    client = gspread.authorize(credentials)

    spreadsheet_name = st.secrets["google_sheets"].get(
        "spreadsheet_name",
        SPREADSHEET_NAME,
    )
    worksheet_name = st.secrets["google_sheets"].get(
        "worksheet_name",
        WORKSHEET_NAME,
    )

    spreadsheet = client.open(spreadsheet_name)

    try:
        worksheet = spreadsheet.worksheet(worksheet_name)
    except gspread.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(
            title=worksheet_name,
            rows=1000,
            cols=len(JOURNAL_COLUMNS),
        )

    current_headers = worksheet.row_values(1)

    if not current_headers:
        worksheet.append_row(JOURNAL_COLUMNS, value_input_option="RAW")
    elif current_headers != JOURNAL_COLUMNS:
        # Preserve existing data while making sure all required columns exist.
        merged_headers = current_headers.copy()
        for column in JOURNAL_COLUMNS:
            if column not in merged_headers:
                merged_headers.append(column)

        worksheet.resize(cols=max(len(merged_headers), worksheet.col_count))
        worksheet.update(
            range_name=f"A1:{gspread.utils.rowcol_to_a1(1, len(merged_headers))}",
            values=[merged_headers],
            value_input_option="RAW",
        )

    return worksheet


def ensure_worksheet_capacity(worksheet, required_row=None, required_cols=None, row_buffer=500):
    """Expand the worksheet before a write would exceed its grid limits."""
    target_row = int(required_row or worksheet.row_count)
    target_cols = int(required_cols or worksheet.col_count)

    new_rows = worksheet.row_count
    new_cols = worksheet.col_count

    if target_row > worksheet.row_count:
        new_rows = max(target_row, worksheet.row_count + row_buffer)
    if target_cols > worksheet.col_count:
        new_cols = target_cols

    if new_rows != worksheet.row_count or new_cols != worksheet.col_count:
        worksheet.resize(rows=new_rows, cols=new_cols)


def save_to_journal(row, update_existing=False):
    """Save a trade, updating its existing journal row when requested."""
    worksheet = get_journal_worksheet()
    journal_row = row.copy()
    headers = worksheet.row_values(1)

    existing_row_number = None
    existing_record = {}

    if update_existing:
        market_id = str(journal_row.get("Market ID", "")).strip()
        market_name = str(journal_row.get("Market", "")).strip()
        all_values = worksheet.get_all_values()

        # Search from the bottom so the most recent matching journal entry is updated.
        # Prefer an entry that has not already been executed.
        fallback = None
        for sheet_row_number in range(len(all_values), 1, -1):
            values = all_values[sheet_row_number - 1]
            record = {
                header: values[index] if index < len(values) else ""
                for index, header in enumerate(headers)
            }
            same_trade = (
                market_id and str(record.get("Market ID", "")).strip() == market_id
            ) or (
                not market_id
                and market_name
                and str(record.get("Market", "")).strip() == market_name
            )
            if not same_trade:
                continue

            if fallback is None:
                fallback = (sheet_row_number, record)

            if not str(record.get("MCP Order ID", "")).strip():
                existing_row_number, existing_record = sheet_row_number, record
                break

        if existing_row_number is None and fallback is not None:
            existing_row_number, existing_record = fallback

    if existing_row_number is not None:
        # Preserve journal-management fields and the original save date unless explicitly supplied.
        for field in ("Date Saved", "Status", "Result", "PnL"):
            if journal_row.get(field, "") in ("", None):
                journal_row[field] = existing_record.get(field, "")

        if not journal_row.get("Date Saved"):
            journal_row["Date Saved"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not journal_row.get("Status"):
            journal_row["Status"] = "Open"
        if journal_row.get("Result") is None:
            journal_row["Result"] = ""
        if journal_row.get("PnL") in ("", None):
            journal_row["PnL"] = 0.0

        # Keep any existing values for columns not present in the incoming model row.
        merged = existing_record.copy()
        merged.update({key: value for key, value in journal_row.items() if value is not None})
        values = [_clean_sheet_value(merged.get(column, "")) for column in headers]
        end_cell = gspread.utils.rowcol_to_a1(existing_row_number, len(headers))
        worksheet.update(
            range_name=f"A{existing_row_number}:{end_cell}",
            values=[values],
            value_input_option="USER_ENTERED",
        )
        return "updated", existing_row_number

    journal_row["Date Saved"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    journal_row["Status"] = journal_row.get("Status") or "Open"
    journal_row["Result"] = journal_row.get("Result", "")
    journal_row["PnL"] = journal_row.get("PnL", 0.0)

    values = [_clean_sheet_value(journal_row.get(column, "")) for column in headers]

    # Write to the exact next populated row rather than relying on append_row.
    # This lets us expand the grid first and return the real journal row number.
    next_row = len(worksheet.get_all_values()) + 1
    ensure_worksheet_capacity(
        worksheet,
        required_row=next_row,
        required_cols=len(headers),
    )
    end_cell = gspread.utils.rowcol_to_a1(next_row, len(headers))
    worksheet.update(
        range_name=f"A{next_row}:{end_cell}",
        values=[values],
        value_input_option="USER_ENTERED",
    )
    return "created", next_row


def verify_execution_fields(sheet_row_number, expected):
    """Write and verify MCP execution fields in a specific Google Sheets row."""
    worksheet = get_journal_worksheet()
    headers = worksheet.row_values(1)

    required = [
        "Execution Token ID",
        "Execution Outcome",
        "Estimated Fill Price",
        "Executed Amount pUSD",
        "MCP Order ID",
        "CLOB Order ID",
        "Execution Status",
        "Execution Response",
    ]

    missing = [column for column in required if column not in headers]
    if missing:
        raise RuntimeError(
            "Google Sheets is missing execution column(s): " + ", ".join(missing)
        )

    # Update each execution field by its actual header location. This remains
    # reliable even when older sheets have the new columns appended at the end.
    updates = []
    for column in required:
        col_number = headers.index(column) + 1
        cell = gspread.utils.rowcol_to_a1(sheet_row_number, col_number)
        updates.append({
            "range": cell,
            "values": [[_clean_sheet_value(expected.get(column, ""))]],
        })

    worksheet.batch_update(updates, value_input_option="USER_ENTERED")

    # Read the row back so the app does not claim success when IDs were not saved.
    saved_values = worksheet.row_values(sheet_row_number)
    saved = {
        header: saved_values[index] if index < len(saved_values) else ""
        for index, header in enumerate(headers)
    }

    for column in ("MCP Order ID", "CLOB Order ID", "Execution Status"):
        expected_value = str(_clean_sheet_value(expected.get(column, ""))).strip()
        saved_value = str(saved.get(column, "")).strip()
        if expected_value and saved_value != expected_value:
            raise RuntimeError(
                f"The trade executed, but Google Sheets did not retain {column}. "
                "Check the sheet permissions and column headers."
            )

    return saved


@st.cache_data(ttl=30)
def load_journal():
    """Load the permanent journal from Google Sheets."""
    try:
        worksheet = get_journal_worksheet()
        records = worksheet.get_all_records(
            expected_headers=worksheet.row_values(1),
            default_blank="",
        )
    except Exception as error:
        st.error(f"Could not load the trading journal: {error}")
        return pd.DataFrame(columns=JOURNAL_COLUMNS)

    if not records:
        return pd.DataFrame(columns=JOURNAL_COLUMNS)

    df = pd.DataFrame(records)

    for column in JOURNAL_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    for column in NUMERIC_JOURNAL_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "PnL" in df.columns:
        df["PnL"] = df["PnL"].fillna(0.0)

    return df


def _write_journal_dataframe(df):
    """Replace the worksheet contents after result updates."""
    worksheet = get_journal_worksheet()

    for column in JOURNAL_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    # Keep required columns first and retain any older extra columns afterward.
    ordered_columns = JOURNAL_COLUMNS + [
        column for column in df.columns if column not in JOURNAL_COLUMNS
    ]
    df = df[ordered_columns]

    values = [ordered_columns]
    values.extend(
        [
            [_clean_sheet_value(value) for value in row]
            for row in df.itertuples(index=False, name=None)
        ]
    )

    worksheet.clear()
    worksheet.resize(
        rows=max(len(values) + 100, 1000),
        cols=max(len(ordered_columns), worksheet.col_count),
    )
    end_cell = gspread.utils.rowcol_to_a1(len(values), len(ordered_columns))
    worksheet.update(
        range_name=f"A1:{end_cell}",
        values=values,
        value_input_option="USER_ENTERED",
    )
    load_journal.clear()


def update_results():
    """Check open Polymarket trades and save settled results to Google Sheets."""
    df = load_journal().copy()

    if df.empty:
        return df, 0

    if "Status" not in df.columns:
        df["Status"] = "Open"

    if "Result" not in df.columns:
        df["Result"] = ""

    if "PnL" not in df.columns:
        df["PnL"] = 0.0

    df["Status"] = df["Status"].astype("object")
    df["Result"] = df["Result"].astype("object")
    df["PnL"] = pd.to_numeric(df["PnL"], errors="coerce").fillna(0.0)

    updates = 0
    now_utc = pd.Timestamp.now(tz="UTC")

    for i, row in df.iterrows():
        resolution_date = pd.to_datetime(
            row.get("Resolution Date"),
            utc=True,
            errors="coerce",
        )

        if pd.notna(resolution_date) and now_utc < resolution_date:
            if str(row.get("Status", "")) == "Closed":
                df.loc[i, "Status"] = "Open"
                df.loc[i, "Result"] = ""
                df.loc[i, "PnL"] = 0.0
            continue

        if str(row.get("Status", "")) == "Closed":
            continue

        market_id = row.get("Market ID")

        if pd.isna(market_id) or str(market_id).strip() == "":
            continue

        market = fetch_market_by_id(str(market_id).strip())
        winner, is_closed = infer_winner_from_market(market)

        if is_closed and winner:
            pnl = calculate_pnl(
                entry_side=str(row.get("Entry Side", "")),
                winner=str(winner),
                entry_price_pct=float(row.get("Entry Price %", 0) or 0),
                position_size=float(row.get("Position Size $", 0) or 0),
            )

            df.loc[i, "Status"] = "Closed"
            df.loc[i, "Result"] = str(winner)
            df.loc[i, "PnL"] = float(pnl)
            updates += 1

    if updates > 0:
        _write_journal_dataframe(df)

    return df, updates

tab1, tab2, tab3 = st.tabs(["Dashboard", "Journal", "Analytics"])


with tab1:
    st.subheader("Run Market Screener")

    if st.button("Run MCP Screener", key="run_screener_button"):
        markets_df = pull_markets()
        st.session_state["markets_df"] = markets_df
        st.session_state["scan_stats"] = markets_df.attrs.get("scan_stats", {})
        st.session_state["rejections"] = markets_df.attrs.get("rejections", pd.DataFrame())

        engine = MCPQuantEngine()
        scored = []

        # Use only CLOSED trades with a resolved YES/NO outcome for calibration.
        # This uses the same definition of a resolved trade as Analytics.
        # Open trades never train the calibration layer.
        calibration_journal = load_journal()

        if (
            not calibration_journal.empty
            and "Status" in calibration_journal.columns
            and "Result" in calibration_journal.columns
        ):
            status = calibration_journal["Status"].astype(str).str.strip().str.upper()
            result = calibration_journal["Result"].astype(str).str.strip().str.upper()

            resolved_for_calibration = calibration_journal[
                status.eq("CLOSED") & result.isin(["YES", "NO"])
            ].copy()
        else:
            resolved_for_calibration = pd.DataFrame()

        # If the journal has closed rows but the calibration set is empty,
        # surface the actual count in the UI rather than silently reporting
        # "no resolved trades".
        st.session_state["calibration_resolved_count"] = len(resolved_for_calibration)

        for _, row in markets_df.iterrows():
            try:
                scored.append(engine.score_market(row, resolved_for_calibration))
            except Exception:
                pass

        results = pd.DataFrame(scored)
        st.session_state["results"] = results

        if len(results) > 0:
            results = results.sort_values("Edge %", ascending=False)
            st.session_state["results"] = results
            results.to_csv("mcp_dashboard_results.csv", index=False)

    if "markets_df" in st.session_state:
        markets_df = st.session_state["markets_df"]

        stats = st.session_state.get("scan_stats", {})
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Binary Markets Scanned", stats.get("all_binary_markets", len(markets_df)))
        m2.metric("Binary Price Markets", stats.get("binary_price_markets", len(markets_df)))
        m3.metric("Model Eligible", stats.get("eligible_markets", len(markets_df)))
        m4.metric("Rejected", stats.get("rejected_markets", 0))

        calibration_count = st.session_state.get("calibration_resolved_count", 0)
        if calibration_count < CALIBRATION_MIN_SAMPLES:
            st.caption(
                f"Calibration: {calibration_count} resolved trades "
                f"(activates at {CALIBRATION_MIN_SAMPLES})"
            )
        else:
            st.caption(
                f"Calibration: ACTIVE — {calibration_count} resolved trades"
            )

        st.subheader("Filtered Markets")

        st.dataframe(
            markets_df[
                [
                    "Market",
                    "Market Type",
                    "Ticker",
                    "Target",
                    "Upper",
                    "Direction",
                    "Market Prob %",
                    "No Prob %",
                    "Days",
                    "Liquidity",
                ]
            ],
            use_container_width=True,
        )

        rejected = st.session_state.get("rejections", pd.DataFrame())
        if isinstance(rejected, pd.DataFrame) and not rejected.empty:
            with st.expander("See rejected binary price markets and reasons"):
                st.dataframe(rejected, use_container_width=True)

    if "results" in st.session_state:
        results = st.session_state["results"]
        if not isinstance(results, pd.DataFrame):
            results = pd.DataFrame()

        st.subheader("Top Trade Candidates")
        st.dataframe(results, use_container_width=True)

        buys = results[results["Signal"].isin(["BUY YES", "BUY NO"]) & results.get("Execution Approved", False).fillna(False).astype(bool)]

        st.subheader("Actionable Trades")
        st.dataframe(buys, use_container_width=True)

        st.markdown("---")
        st.subheader("📰 News Validation")

        if len(buys) > 0:
            selected_news_trade = st.selectbox(
                "Select actionable trade for news",
                buys["Market"].tolist(),
                key="news_trade_selectbox",
            )

            news_row = buys[buys["Market"] == selected_news_trade].iloc[0]
            ticker_for_news = news_row["Ticker"]

            if st.button("Get News", key="get_news_button"):
                news_df = get_news(ticker_for_news)

                st.dataframe(news_df, use_container_width=True)

                st.info(
                    "Use news as validation only. News should confirm or reject "
                    "the model signal, not create a trade by itself."
                )

                verdict = st.radio(
                    "Manual News Verdict",
                    ["Positive", "Neutral", "Negative"],
                    key="manual_news_verdict",
                )

                st.write(f"News Verdict: **{verdict}**")
        else:
            st.info("No actionable trades. News check skipped.")

        st.markdown("---")
        st.subheader("🔍 Explain Model")

        market_options = results["Market"].dropna().astype(str).drop_duplicates().tolist() if not results.empty and "Market" in results.columns else []
        if not market_options:
            st.info("No scored trades are available to explain.")
            explain = None
        else:
            selected_trade = st.selectbox(
                "Select a trade to explain",
                market_options,
                key="explain_trade_selectbox",
            )
            matching = results[results["Market"].astype(str) == str(selected_trade)]
            explain = matching.iloc[0] if not matching.empty else None

        if explain is not None:
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
            st.write(f"**Raw Model YES Probability:** {explain['Final Prob %']}%")
            st.write(f"**Calibrated YES Probability:** {explain.get('Calibrated Prob %', explain['Final Prob %'])}%")
            st.write(f"**YES Edge:** {explain['YES Edge %']}%")
            st.write(f"**NO Edge:** {explain['NO Edge %']}%")
            st.write(f"**Signal:** {explain['Signal']}")
            st.write(f"**Entry Side:** {explain['Entry Side']}")
            st.write(f"**Entry Price:** {explain['Entry Price %']}%")
            st.write(f"**Suggested Position Size:** ${explain['Position Size $']}")


            st.markdown("### Superforecasting Review")
            historical_prob = _safe_float(explain.get("Historical Prob %"))
            ewma_prob = _safe_float(explain.get("EWMA Prob %"))
            market_prob = _safe_float(explain.get("Market Prob %"))
            final_prob = _safe_float(explain.get("Final Prob %"))
            calibrated_prob = _safe_float(explain.get("Calibrated Prob %", final_prob))
            confidence = str(explain.get("Forecast Confidence", "Low"))
            disagreement = abs(ewma_prob - historical_prob)

            sf1, sf2, sf3, sf4 = st.columns(4)
            sf1.metric("Outside view", f"{historical_prob:.2f}%")
            sf2.metric("Current-data view", f"{ewma_prob:.2f}%")
            sf3.metric("Polymarket benchmark", f"{market_prob:.2f}%")
            sf4.metric("Confidence", confidence)

            reasoning = []
            if calibrated_prob > market_prob:
                reasoning.append(
                    f"The calibrated model assigns YES {calibrated_prob - market_prob:.2f} percentage points more probability than the market."
                )
            elif calibrated_prob < market_prob:
                reasoning.append(
                    f"The calibrated model assigns YES {market_prob - calibrated_prob:.2f} percentage points less probability than the market."
                )
            else:
                reasoning.append("The model and market assign the same YES probability.")
            reasoning.append(
                f"The historical base rate is {historical_prob:.2f}% and the current-volatility estimate is {ewma_prob:.2f}%."
            )
            if disagreement > 30:
                reasoning.append("The outside and current-data views disagree substantially, so confidence is reduced.")
            elif disagreement > 15:
                reasoning.append("The two model views disagree moderately, so the forecast should be monitored closely.")
            else:
                reasoning.append("The two model views are broadly aligned.")
            reasoning.append(
                "Treat this as a dated probability estimate: update it when meaningful evidence changes and score it after resolution."
            )
            for item in reasoning:
                st.write(f"• {item}")
    
            if explain["Signal"] == "BUY YES":
                st.success("✅ YES is underpriced based on the model probability.")
            elif explain["Signal"] == "BUY NO":
                st.error("❌ NO is underpriced based on the model probability.")
            else:
                st.info("⚪ The model does not see enough edge to trade.")
    
        st.markdown("---")
        st.subheader("Execute Model Trade through MCP")
        st.caption(
            "Only actionable model signals can be executed. The app logs in, selects the matching YES/NO token, "
            "gets a fresh FOK price estimate, and requires confirmation before sending a live order."
        )

        if len(buys) > 0:
            selected_execute = st.selectbox(
                "Select model trade to execute",
                buys["Market"].tolist(),
                key="execute_trade_selectbox",
            )
            execute_row = buys[buys["Market"] == selected_execute].iloc[0].to_dict()

            try:
                preview_token_id, preview_outcome = token_for_signal(execute_row)
                model_amount = float(execute_row.get("Position Size $", 0) or 0)

                client = MCPTradingClient()
                capped_amount = min(model_amount, client.max_order_amount)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Model Signal", execute_row["Signal"])
                c2.metric("Outcome Token", preview_outcome)
                c3.metric("Model Size", f"${model_amount:.2f}")
                c4.metric("Live Amount", f"${capped_amount:.2f}")

                if capped_amount < model_amount:
                    st.warning(
                        f"The model suggested ${model_amount:.2f}, but the live safety cap limits this order to "
                        f"${capped_amount:.2f}. Change max_order_amount in Streamlit Secrets when appropriate."
                    )

                if capped_amount < 1:
                    st.error("The live amount must be at least 1 pUSD for the CLOB minimum notional.")
                else:
                    if st.button("Get Fresh MCP Price Estimate", key="mcp_estimate_button"):
                        token = client.login()
                        estimate = client.price_estimate(
                            token, preview_token_id, "buy", capped_amount
                        )
                        st.session_state["mcp_trade_preview"] = {
                            "market": selected_execute,
                            "row": execute_row,
                            "token_id": preview_token_id,
                            "outcome": preview_outcome,
                            "amount": capped_amount,
                            "estimate": estimate,
                        }

                    preview = st.session_state.get("mcp_trade_preview")
                    if preview and preview.get("market") == selected_execute:
                        estimate_price = float(preview["estimate"].get("price", 0) or 0)
                        st.success(f"Fresh estimated execution price: ${estimate_price:.4f} per share")
                        st.json(preview["estimate"])

                        confirm = st.checkbox(
                            f"I confirm this live BUY {preview_outcome} order for ${capped_amount:.2f} pUSD.",
                            key="confirm_live_mcp_trade",
                        )

                        if st.button(
                            "Execute Confirmed Live Trade",
                            disabled=not confirm,
                            key="execute_live_mcp_trade_button",
                        ):
                            # Re-login and re-estimate immediately before execution so stale session data is not used.
                            token = client.login()
                            fresh_estimate = client.price_estimate(
                                token, preview_token_id, "buy", capped_amount
                            )

                            # Order placement is the critical step. Once this succeeds, a later
                            # journal failure must never be reported as a failed trade.
                            order_response = client.place_market_order(
                                token, preview_token_id, "buy", capped_amount
                            )

                            executed_row = execute_row.copy()
                            executed_row["Execution Token ID"] = preview_token_id
                            executed_row["Execution Outcome"] = preview_outcome
                            executed_row["Estimated Fill Price"] = fresh_estimate.get("price", "")
                            executed_row["Executed Amount pUSD"] = capped_amount
                            executed_row["MCP Order ID"] = (
                                order_response.get("order_id")
                                or order_response.get("id")
                                or order_response.get("taker_order_id")
                                or ""
                            )
                            executed_row["CLOB Order ID"] = (
                                order_response.get("clob_order_id")
                                or order_response.get("taker_order_id")
                                or ""
                            )
                            executed_row["Execution Status"] = (
                                order_response.get("clob_status")
                                or order_response.get("status")
                                or "Submitted"
                            )
                            executed_row["Execution Response"] = order_response
                            executed_row["Entry Price %"] = (
                                float(fresh_estimate.get("price", 0) or 0) * 100
                            )
                            executed_row["Position Size $"] = capped_amount

                            st.success("✅ Trade order was accepted by MCP.")
                            r1, r2, r3, r4 = st.columns(4)
                            r1.metric("Outcome", preview_outcome)
                            r2.metric("Amount", f"${capped_amount:.2f}")
                            r3.metric(
                                "Estimated Price",
                                f"${float(fresh_estimate.get('price', 0) or 0):.4f}",
                            )
                            r4.metric("Status", executed_row["Execution Status"])
                            if executed_row["MCP Order ID"]:
                                st.caption(f"MCP Order ID: {executed_row['MCP Order ID']}")
                            if executed_row["CLOB Order ID"]:
                                st.caption(f"CLOB Order ID: {executed_row['CLOB Order ID']}")
                            st.json(order_response)

                            # Journal persistence is non-critical. Report it separately so a
                            # Sheets problem cannot create uncertainty or trigger a duplicate order.
                            try:
                                journal_action, journal_row_number = save_to_journal(
                                    executed_row,
                                    update_existing=True,
                                )
                                verify_execution_fields(
                                    journal_row_number,
                                    {
                                        "Execution Token ID": preview_token_id,
                                        "Execution Outcome": preview_outcome,
                                        "Estimated Fill Price": fresh_estimate.get("price", ""),
                                        "Executed Amount pUSD": capped_amount,
                                        "MCP Order ID": executed_row["MCP Order ID"],
                                        "CLOB Order ID": executed_row["CLOB Order ID"],
                                        "Execution Status": executed_row["Execution Status"],
                                        "Execution Response": order_response,
                                    },
                                )
                                load_journal.clear()
                                if journal_action == "updated":
                                    st.success("📒 Existing Google Sheets journal row updated.")
                                else:
                                    st.success("📒 New Google Sheets journal row created.")
                            except Exception as journal_error:
                                st.warning(
                                    "⚠️ The trade was submitted successfully, but the Google Sheets "
                                    f"journal could not be updated: {journal_error}"
                                )
                                st.info(
                                    "Do not submit this trade again. Verify it in MCP Trade History "
                                    "and repair the journal separately."
                                )

                            st.session_state.pop("mcp_trade_preview", None)
            except Exception as error:
                message = str(error)
                if "422" in message and "no match" in message.lower():
                    st.warning("Trade unavailable: no matchable FOK liquidity. No order was placed.")
                else:
                    st.error(f"MCP execution setup error: {error}")
        else:
            st.info("No actionable model signals are available for execution.")

        st.markdown("---")
        st.subheader("Save Trade to Journal")

        save_options = results["Market"].dropna().astype(str).drop_duplicates().tolist() if not results.empty and "Market" in results.columns else []
        if save_options:
            selected_save = st.selectbox(
                "Select trade to save",
                save_options,
                key="save_trade_selectbox",
            )
        else:
            selected_save = None
            st.info("No scored trades are available to save.")

        if selected_save and st.button("Save Selected Trade", key="save_trade_button"):
            matching = results[results["Market"].astype(str) == str(selected_save)]
            if matching.empty:
                st.warning("The selected trade is no longer available. Run the screener again.")
            else:
                row = matching.iloc[0].to_dict()

                try:
                    journal_action, _ = save_to_journal(row, update_existing=True)
                    load_journal.clear()
                    if journal_action == "updated":
                        st.success("The existing Google Sheets trade row was refreshed.")
                    else:
                        st.success("Trade saved permanently to Google Sheets.")
                except Exception as error:
                    st.error(f"Trade could not be saved: {error}")


with tab2:
    st.subheader("Trade Journal")

    if st.button("Update Results", key="update_results_button"):
        try:
            journal, updates = update_results()
            load_journal.clear()
            st.success(f"Updated {updates} closed trades.")
        except Exception as error:
            st.error(f"Results could not be updated: {error}")

    journal = load_journal()

    if len(journal) > 0:
        st.dataframe(journal, use_container_width=True)
    else:
        st.info("No trades saved yet.")


with tab3:
    st.subheader("Analytics")

    journal = load_journal()

    if len(journal) > 0:
        if "Status" in journal.columns:
            closed = journal[journal["Status"] == "Closed"]
            open_trades = journal[journal["Status"] != "Closed"]
        else:
            closed = pd.DataFrame()
            open_trades = journal

        total_pnl = closed["PnL"].sum() if len(closed) > 0 else 0
        bankroll = STARTING_BANKROLL + total_pnl

        buy_count = journal[journal["Signal"].isin(["BUY YES", "BUY NO"])].shape[0]
        avg_edge = journal["Edge %"].mean()

        wins = closed[closed["PnL"] > 0].shape[0] if len(closed) > 0 else 0
        win_rate = (wins / len(closed) * 100) if len(closed) > 0 else 0

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Bankroll", f"${round(bankroll, 2)}")
        c2.metric("Total PnL", f"${round(total_pnl, 2)}")
        c3.metric("Closed Trades", len(closed))
        c4.metric("Win Rate", f"{round(win_rate, 2)}%")

        c5, c6, c7 = st.columns(3)

        c5.metric("Open Trades", len(open_trades))
        c6.metric("Buy Signals", buy_count)
        c7.metric("Avg Edge", round(avg_edge, 2))

        st.subheader("Edge Distribution")
        st.bar_chart(journal["Edge %"])

        if len(closed) > 0:
            st.subheader("PnL by Trade")
            st.bar_chart(closed["PnL"])

            resolved = closed[closed["Result"].astype(str).str.upper().isin(["YES", "NO"])].copy()
            if not resolved.empty:
                resolved["Outcome YES"] = resolved["Result"].astype(str).str.upper().eq("YES")
                resolved["Raw Model Brier"] = resolved.apply(
                    lambda row: brier_score(row.get("Final Prob %", 0), row["Outcome YES"]), axis=1
                )
                resolved["Calibrated Model Brier"] = resolved.apply(
                    lambda row: brier_score(
                        row.get("Calibrated Prob %", row.get("Final Prob %", 0)),
                        row["Outcome YES"],
                    ),
                    axis=1,
                )
                resolved["Model Brier"] = resolved["Calibrated Model Brier"]
                resolved["Market Brier"] = resolved.apply(
                    lambda row: brier_score(row.get("Market Prob %", 0), row["Outcome YES"]), axis=1
                )

                st.subheader("Forecast Accuracy")
                b1, b2, b3 = st.columns(3)
                model_brier = resolved["Model Brier"].mean()
                market_brier = resolved["Market Brier"].mean()
                b1.metric("Model Brier Score", f"{model_brier:.4f}")
                b2.metric("Market Brier Score", f"{market_brier:.4f}")
                b3.metric("Model vs Market", "Better" if model_brier < market_brier else "Worse")
                st.caption("Lower is better: 0 is perfect and 1 is the worst possible binary Brier score.")

                resolved["Final Prob %"] = pd.to_numeric(resolved["Final Prob %"], errors="coerce")
                if "Calibrated Prob %" in resolved.columns:
                    resolved["Calibrated Prob %"] = pd.to_numeric(
                        resolved["Calibrated Prob %"], errors="coerce"
                    )
                else:
                    resolved["Calibrated Prob %"] = resolved["Final Prob %"]
                resolved = resolved[resolved["Final Prob %"].notna()].copy()
                resolved["Probability Band"] = pd.cut(
                    resolved["Final Prob %"],
                    bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                    include_lowest=True,
                )
                calibration = (
                    resolved.groupby("Probability Band", observed=False)
                    .agg(
                        Forecasts=("Market ID", "count"),
                        Average_Model_Probability=("Final Prob %", "mean"),
                        Actual_YES_Rate=("Outcome YES", "mean"),
                    )
                    .reset_index()
                )
                calibration["Actual_YES_Rate"] *= 100
                calibration = calibration[calibration["Forecasts"] > 0]
                st.subheader("Calibration by Probability Band")
                st.dataframe(calibration, use_container_width=True)
            else:
                st.info("Brier score and calibration will appear after resolved YES/NO trades are available.")
    else:
        st.info("No analytics available yet.")
