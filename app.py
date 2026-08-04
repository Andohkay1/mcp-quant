import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import json
import re
import xml.etree.ElementTree as ET
from scipy.stats import norm
from datetime import datetime

import gspread
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="MCP Quant Dashboard", layout="wide")
st.title("MCP Quant Dashboard")

EDGE_THRESHOLD = 5
MIN_LIQUIDITY = 250
MAX_DAYS = 10
MOMENTUM_WEIGHT = 1.0
EWMA_LAMBDA = 0.94
SPREADSHEET_NAME = "Polymarket Journal"
WORKSHEET_NAME = "Trades"
STARTING_BANKROLL = 100
BUY_THRESHOLD = 6

# Automatic execution filters. A model signal can still appear in the research
# table even when it is not approved for live execution.
MIN_ACTIONABLE_EDGE = 8.0
MIN_TRADABLE_ENTRY_PRICE_PCT = 15.0
MAX_TRADABLE_ENTRY_PRICE_PCT = 85.0
MIN_SELECTED_MODEL_PROB_PCT = 55.0
MIN_TRADING_DAYS_REMAINING = 0.01  # about four minutes of a 6.5-hour session

# Unattended auto-trading safeguards. These may be overridden in [mcp_trading]
# Streamlit Secrets. Auto-trading only runs while the Streamlit app has an active
# session; use the separate worker deployment for true 24/7 execution.
AUTO_SCAN_INTERVAL = "10m"
DEFAULT_AUTO_MAX_TRADES_PER_DAY = 3
DEFAULT_AUTO_MAX_TRADES_PER_CYCLE = 1
DEFAULT_AUTO_MAX_OPEN_TRADES = 8
DEFAULT_AUTO_MIN_BALANCE = 75.0


# MCP competition tracking settings
COMPETITION_DAYS = 60
REQUIRED_COMPLETED_TRADES = 80
MARK_TO_MARKET_FLOOR = 70.0


def _safe_float(value, default=0.0):
    try:
        number = float(value)
        return number if np.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def evaluate_execution_approval(row):
    """Return whether a model signal is eligible for live execution and why.

    This separates forecasting from execution. Large apparent edges in nearly
    resolved or extremely one-sided markets remain visible for research but are
    blocked from automatic/live execution.
    """
    signal = str(row.get("Signal", ""))
    edge = _safe_float(row.get("Edge %"), 0.0)
    entry_price = _safe_float(row.get("Entry Price %"), 0.0)
    days_remaining = _safe_float(row.get("Days"), 0.0)
    model_yes = _safe_float(row.get("Final Prob %"), 0.0)

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
        reasons.append(
            f"selected-outcome model probability below {MIN_SELECTED_MODEL_PROB_PCT:.0f}%"
        )
    if days_remaining < MIN_TRADING_DAYS_REMAINING:
        reasons.append("too little trading time remaining")

    approved = not reasons
    return approved, ("Approved" if approved else "; ".join(reasons)), selected_model_probability


def _asset_class_from_ticker(ticker):
    ticker = str(ticker or "").upper().strip()
    if ticker.endswith("-USD"):
        return "Crypto"
    if ticker.endswith("=F"):
        return "Commodity"
    if ticker.startswith("^"):
        return "Index"
    if ticker.endswith("=X"):
        return "FX"
    return "Stock / ETF"


def _trade_return(row):
    size = _safe_float(row.get("Executed Amount pUSD"), 0.0)
    if size <= 0:
        size = _safe_float(row.get("Position Size $"), 0.0)
    pnl = _safe_float(row.get("PnL"), 0.0)
    return pnl / size if size > 0 else np.nan


def _max_drawdown_from_pnl(closed):
    if closed.empty:
        return 0.0, pd.Series(dtype=float)
    pnl = pd.to_numeric(closed["PnL"], errors="coerce").fillna(0.0)
    equity = STARTING_BANKROLL + pnl.cumsum()
    running_peak = equity.cummax()
    drawdown = (equity - running_peak) / running_peak.replace(0, np.nan)
    return abs(float(drawdown.min())) if len(drawdown) else 0.0, equity


def calculate_competition_metrics(journal):
    """Calculate competition metrics from the permanent journal.

    Sharpe and Sortino use resolved trade-level returns. Calmar uses realized
    cumulative return divided by realized maximum drawdown. Open-position
    mark-to-market is not included because the journal does not contain live
    position values.
    """
    if journal is None or journal.empty:
        return {
            "executed": pd.DataFrame(), "closed": pd.DataFrame(), "open": pd.DataFrame(),
            "trade_count": 0, "closed_count": 0, "open_count": 0,
            "total_pnl": 0.0, "realized_bankroll": STARTING_BANKROLL,
            "win_rate": np.nan, "sharpe": np.nan, "sortino": np.nan,
            "calmar": np.nan, "max_drawdown": 0.0, "equity": pd.Series(dtype=float),
            "days_elapsed": 0, "days_remaining": COMPETITION_DAYS,
            "required_weekly_pace": REQUIRED_COMPLETED_TRADES / (COMPETITION_DAYS / 7),
        }

    df = journal.copy()
    order_ids = df.get("MCP Order ID", pd.Series(index=df.index, dtype=object)).astype(str).str.strip()
    statuses = df.get("Execution Status", pd.Series(index=df.index, dtype=object)).astype(str).str.lower()
    executed_mask = order_ids.ne("") & ~statuses.str.contains("error|failed|rejected|cancel", regex=True, na=False)
    executed = df.loc[executed_mask].copy()
    if "MCP Order ID" in executed.columns:
        executed = executed.drop_duplicates(subset=["MCP Order ID"], keep="last")

    closed = executed[executed.get("Status", "").astype(str).str.lower().eq("closed")].copy()
    open_trades = executed[~executed.index.isin(closed.index)].copy()
    closed["Trade Return"] = closed.apply(_trade_return, axis=1) if not closed.empty else pd.Series(dtype=float)
    returns = pd.to_numeric(closed.get("Trade Return"), errors="coerce").dropna()

    total_pnl = pd.to_numeric(closed.get("PnL"), errors="coerce").fillna(0.0).sum() if not closed.empty else 0.0
    wins = int((pd.to_numeric(closed.get("PnL"), errors="coerce").fillna(0.0) > 0).sum()) if not closed.empty else 0
    win_rate = wins / len(closed) if len(closed) else np.nan

    sharpe = np.nan
    sortino = np.nan
    if len(returns) >= 2 and returns.std(ddof=1) > 0:
        sharpe = float(returns.mean() / returns.std(ddof=1) * np.sqrt(len(returns)))
    downside = returns[returns < 0]
    if len(returns) >= 2 and len(downside) >= 2 and downside.std(ddof=1) > 0:
        sortino = float(returns.mean() / downside.std(ddof=1) * np.sqrt(len(returns)))

    max_drawdown, equity = _max_drawdown_from_pnl(closed)
    realized_return = total_pnl / STARTING_BANKROLL
    calmar = realized_return / max_drawdown if max_drawdown > 0 else np.nan

    saved_dates = pd.to_datetime(executed.get("Date Saved"), errors="coerce") if not executed.empty else pd.Series(dtype="datetime64[ns]")
    first_date = saved_dates.dropna().min() if not saved_dates.dropna().empty else pd.NaT
    today = pd.Timestamp.now().normalize()
    days_elapsed = max(1, int((today - first_date.normalize()).days) + 1) if pd.notna(first_date) else 0
    days_remaining = max(COMPETITION_DAYS - days_elapsed, 0)
    trades_remaining = max(REQUIRED_COMPLETED_TRADES - len(executed), 0)
    weeks_remaining = max(days_remaining / 7, 1 / 7)

    return {
        "executed": executed, "closed": closed, "open": open_trades,
        "trade_count": len(executed), "closed_count": len(closed), "open_count": len(open_trades),
        "total_pnl": float(total_pnl), "realized_bankroll": STARTING_BANKROLL + float(total_pnl),
        "win_rate": win_rate, "sharpe": sharpe, "sortino": sortino,
        "calmar": calmar, "max_drawdown": max_drawdown, "equity": equity,
        "days_elapsed": days_elapsed, "days_remaining": days_remaining,
        "trades_remaining": trades_remaining,
        "required_weekly_pace": trades_remaining / weeks_remaining if trades_remaining else 0.0,
    }


def _metric_text(value, percent=False):
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value * 100:.1f}%" if percent else f"{value:.2f}"


def _band_table(df, value_column, bins, labels):
    if df.empty or value_column not in df.columns:
        return pd.DataFrame()
    work = df.copy()
    work[value_column] = pd.to_numeric(work[value_column], errors="coerce")
    work["Band"] = pd.cut(work[value_column], bins=bins, labels=labels, include_lowest=True, right=False)
    work["Win"] = pd.to_numeric(work["PnL"], errors="coerce").fillna(0) > 0
    grouped = work.dropna(subset=["Band"]).groupby("Band", observed=False).agg(
        Trades=("Win", "size"),
        Win_Rate=("Win", "mean"),
        Average_PnL=("PnL", "mean"),
    ).reset_index()
    grouped["Win Rate"] = (grouped.pop("Win_Rate") * 100).round(1)
    grouped["Average PnL"] = pd.to_numeric(grouped.pop("Average_PnL"), errors="coerce").round(3)
    return grouped


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
        self.auto_trading_enabled = bool(cfg.get("auto_trading_enabled", False))
        self.auto_max_trades_per_day = int(cfg.get("auto_max_trades_per_day", DEFAULT_AUTO_MAX_TRADES_PER_DAY))
        self.auto_max_trades_per_cycle = int(cfg.get("auto_max_trades_per_cycle", DEFAULT_AUTO_MAX_TRADES_PER_CYCLE))
        self.auto_max_open_trades = int(cfg.get("auto_max_open_trades", DEFAULT_AUTO_MAX_OPEN_TRADES))
        self.auto_min_balance = float(cfg.get("auto_min_balance", DEFAULT_AUTO_MIN_BALANCE))
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

    def list_markets(self, token, *, limit=100, offset=0, active=True, closed=False,
                     order="endDate", ascending=True, tag=None, keyword=None):
        """Browse or search MCP markets using the enhanced discovery endpoint.

        MCP does not allow keyword to be combined with tag, offset, order, or
        ascending, so this helper builds one valid query mode at a time.
        """
        params = {
            "limit": int(limit),
            "active": str(bool(active)).lower(),
            "closed": str(bool(closed)).lower(),
        }
        if keyword:
            params["keyword"] = str(keyword)
        else:
            params.update({
                "offset": int(offset),
                "order": str(order),
                "ascending": str(bool(ascending)).lower(),
            })
            if tag:
                params["tag"] = str(tag)

        return self._request("GET", "/v1/markets", token=token, params=params)

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

    def get_ohlc(self, ticker, period="5y", interval="1d", start=None, end=None):
        kwargs = {
            "auto_adjust": True,
            "progress": False,
            "interval": interval,
        }
        if start is not None:
            kwargs["start"] = start
            if end is not None:
                kwargs["end"] = end
        else:
            kwargs["period"] = period

        data = yf.download(ticker, **kwargs)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data.dropna(how="all")

    def ewma_probability(self, ticker, target, days, direction):
        close = self.get_prices(ticker, "1y")
        current = close.iloc[-1]
        vol = self.ewma_volatility(close)

        # Preserve fractional trading days. A market with only a few hours left
        # must not be treated as though it has a full day to move.
        horizon_days = max(float(days), 1 / 390)
        sigma = vol * np.sqrt(horizon_days / 252)
        z = np.log(target / current) / sigma

        if direction == "above":
            return (1 - norm.cdf(z)) * 100

        return norm.cdf(z) * 100

    def historical_probability(self, ticker, target, days, direction, lookback=252):
        close = self.get_prices(ticker, "5y").tail(lookback)
        current = close.iloc[-1]

        required_return = target / current - 1
        # Daily historical data cannot be shifted by a fraction. Use the next
        # full trading day as the nearest empirical comparison while EWMA uses
        # the exact fractional horizon.
        historical_days = max(int(np.ceil(float(days))), 1)
        future_returns = (close.shift(-historical_days) / close - 1).dropna()

        if direction == "above":
            return (future_returns >= required_return).mean() * 100

        return (future_returns <= required_return).mean() * 100

    def ewma_barrier_probability(self, ticker, target, days, direction):
        """Approximate first-passage probability for touching a barrier before expiry."""
        close = self.get_prices(ticker, "1y")
        current = float(close.iloc[-1])

        # If the current price is already beyond the barrier, the touch condition
        # has necessarily been met at the present instant.
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
        """Empirical frequency of touching an equivalent barrier within the horizon."""
        ohlc = self.get_ohlc(ticker, period="5y", interval="1d").tail(lookback)
        required = {"Close", "High", "Low"}
        if ohlc.empty or not required.issubset(set(ohlc.columns)):
            return np.nan

        current = float(ohlc["Close"].dropna().iloc[-1])
        target_ratio = float(target) / current
        horizon = max(int(np.ceil(float(days))), 1)
        hits = []

        for i in range(0, len(ohlc) - horizon):
            start_price = float(ohlc["Close"].iloc[i])
            if not np.isfinite(start_price) or start_price <= 0:
                continue
            window = ohlc.iloc[i + 1 : i + horizon + 1]
            if window.empty:
                continue

            if direction == "above":
                equivalent_barrier = start_price * target_ratio
                hit = float(window["High"].max()) >= equivalent_barrier
            else:
                equivalent_barrier = start_price * target_ratio
                hit = float(window["Low"].min()) <= equivalent_barrier
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

        horizon_days = max(float(days), 1 / 390)
        sigma = vol * np.sqrt(horizon_days / 252)

        z_low = np.log(lower / current) / sigma
        z_high = np.log(upper / current) / sigma

        return (norm.cdf(z_high) - norm.cdf(z_low)) * 100

    def score_market(self, row):
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

        model_yes = final
        model_no = 100 - final

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
            "Market Start Date": row.get("Market Start Date", ""),
            "Days": round(days, 4),
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
            "YES Edge %": round(yes_edge, 2),
            "NO Edge %": round(no_edge, 2),
            "Edge %": round(edge, 2),
            "Signal": signal,
            "Entry Side": entry_side,
            "Entry Price %": round(entry_price, 2),
            "Position Size $": size,
            "Liquidity": row.get("Liquidity", 0),
            "Barrier Already Hit": bool(row.get("Barrier Already Hit", False)),
            "clobTokenIds": row["clobTokenIds"],
        }

        approved, approval_reason, selected_model_probability = evaluate_execution_approval(result)
        result["Selected Model Prob %"] = round(selected_model_probability, 2)
        result["Execution Approved"] = approved
        result["Execution Decision"] = approval_reason
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
    "earnings", "revenue", "eps", "market cap", "acquire", "acquisition",
    "merger", "ipo", "etf approval", "approve", "regulation", "lawsuit",
    "ceo", "president", "election", "nominee", "fed chair", "interest rate",
    "cpi", "inflation", "gdp", "unemployment", "tariff", "win", "wins",
    "champion", "world cup", "ufc", "nba", "nfl", "mlb", "tennis",
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
    # Touch/barrier contracts resolve YES if the level is reached at any point,
    # not only if the asset finishes beyond the level at expiry.
    if re.search(r"\b(?:hit|reach|touch|dip to|fall to|rise to|trade as high as|trade as low as)\b", text):
        return "barrier"
    if re.search(r"\((?:high|low)\)", text):
        return "barrier"
    if any(re.search(pattern, text) for pattern in PRICE_MARKET_PATTERNS):
        return "price"
    return "event"


def infer_direction(market):
    text = str(market or "").lower()
    if re.search(r"\b(?:below|under|less than|dip|fall to|low)\b", text):
        return "below"
    return "above"


def extract_numbers(market):
    text = str(market or "").replace(",", "")
    # Prefer explicit prices and large index levels; remove dates and percentages.
    raw = re.findall(r"(?<![%\w])\$?(\d+(?:\.\d+)?)", text)
    values = []
    for item in raw:
        value = float(item)
        if 1900 <= value <= 2100:  # likely a year
            continue
        values.append(value)
    return values


def extract_target(market):
    nums = extract_numbers(market)
    return nums[0] if nums else None


def extract_upper(market):
    nums = extract_numbers(market)
    return nums[1] if len(nums) > 1 else None


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
    text = str(market or "").lower()
    # Longest aliases first to avoid matching "oil" before "crude oil".
    for key in sorted(ASSET_ALIASES, key=len, reverse=True):
        if re.search(rf"(?<!\w){re.escape(key)}(?!\w)", text):
            return ASSET_ALIASES[key]

    # Direct ticker notation such as $AAPL or (AAPL).
    direct = re.search(r"\$([A-Z]{1,6})\b|\(([A-Z]{1,6})\)", str(market or ""))
    if direct:
        return direct.group(1) or direct.group(2)

    return yahoo_symbol_search(extract_asset_phrase(market))


@st.cache_data(ttl=300, show_spinner=False)
def barrier_already_hit(ticker, target, direction, start_date):
    """Check whether a touch barrier has already been crossed since market inception."""
    try:
        if pd.isna(start_date):
            return False
        start_ts = pd.Timestamp(start_date)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        else:
            start_ts = start_ts.tz_convert("UTC")

        now_utc = pd.Timestamp.now(tz="UTC")
        age_days = max((now_utc - start_ts).total_seconds() / 86400, 0)
        # Intraday data gives a more precise check for recent markets; daily
        # high/low data extends the check for longer-lived markets.
        if age_days <= 7:
            interval = "5m"
        elif age_days <= 60:
            interval = "1h"
        else:
            interval = "1d"

        data = yf.download(
            ticker,
            start=start_ts.tz_localize(None),
            end=(now_utc + pd.Timedelta(days=1)).tz_localize(None),
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        if data.empty:
            return False

        if direction == "above":
            return float(data["High"].max()) >= float(target)
        return float(data["Low"].min()) <= float(target)
    except Exception:
        # A failed historical check must not silently assert that a barrier was hit.
        return False


@st.cache_data(ttl=300)
def pull_markets():
    """Browse the complete MCP market catalogue with pagination.

    The enhanced MCP endpoint defaults to high-volume markets, so relying on a
    single page can hide lower-volume financial contracts. This function uses
    browse mode only (never keyword mode), paginates with offset, sorts by end
    date, and then applies the app's existing financial/model filters.
    """
    client = MCPTradingClient()
    token = client.login()

    cfg = st.secrets.get("mcp_trading", {})
    page_size = max(1, min(int(cfg.get("discovery_page_size", 100)), 500))
    max_pages = max(1, int(cfg.get("discovery_max_pages", 50)))
    discovery_tag = str(cfg.get("discovery_tag", "")).strip() or None
    discovery_order = str(cfg.get("discovery_order", "endDate")).strip() or "endDate"
    discovery_ascending = bool(cfg.get("discovery_ascending", True))

    markets_raw = []
    offset = 0
    pages_fetched = 0

    for _ in range(max_pages):
        payload = client.list_markets(
            token,
            limit=page_size,
            offset=offset,
            active=True,
            closed=False,
            order=discovery_order,
            ascending=discovery_ascending,
            tag=discovery_tag,
        )

        # The wrapper currently preserves the Gamma response format, but these
        # fallbacks make the scanner safe if the API later wraps the list.
        if isinstance(payload, list):
            page = payload
        elif isinstance(payload, dict):
            page = (
                payload.get("markets")
                or payload.get("results")
                or payload.get("data")
                or payload.get("items")
                or []
            )
        else:
            page = []

        if not isinstance(page, list):
            raise RuntimeError("MCP /v1/markets returned an unexpected response shape.")

        pages_fetched += 1
        if not page:
            break

        markets_raw.extend(page)
        if len(page) < page_size:
            break
        offset += page_size

    # Prevent duplicates if catalogue pages shift while pagination is running.
    unique_markets = []
    seen_keys = set()
    for m in markets_raw:
        if not isinstance(m, dict):
            continue
        key = (
            m.get("condition_id")
            or m.get("conditionId")
            or m.get("id")
            or m.get("slug")
            or m.get("question")
        )
        key = str(key)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_markets.append(m)
    markets_raw = unique_markets

    def _json_list(value):
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                return parsed if isinstance(parsed, list) else []
            except Exception:
                return []
        return []

    rows = []
    for m in markets_raw:
        outcome_prices = _json_list(
            m.get("outcomePrices", m.get("outcome_prices", []))
        )
        outcomes = _json_list(m.get("outcomes", []))
        token_ids = m.get("clobTokenIds", m.get("clob_token_ids", m.get("token_ids", [])))
        token_ids = _json_list(token_ids)

        # Some API shapes provide token objects instead of parallel arrays.
        tokens = m.get("tokens")
        if (not outcome_prices or not token_ids) and isinstance(tokens, list):
            outcome_prices = [t.get("price") for t in tokens if isinstance(t, dict)]
            token_ids = [t.get("token_id") or t.get("tokenId") for t in tokens if isinstance(t, dict)]
            if not outcomes:
                outcomes = [t.get("outcome") for t in tokens if isinstance(t, dict)]

        # The model and executor require a binary YES/NO contract.
        if len(outcome_prices) != 2 or len(token_ids) != 2:
            continue
        if outcomes and len(outcomes) == 2:
            normalized_outcomes = [str(x).strip().lower() for x in outcomes]
            if set(normalized_outcomes) != {"yes", "no"}:
                continue

        try:
            yes_probability = float(outcome_prices[0]) * 100
            no_probability = float(outcome_prices[1]) * 100
        except (TypeError, ValueError, IndexError):
            continue

        rows.append({
            "Market ID": m.get("id") or m.get("market_id") or m.get("condition_id") or m.get("conditionId"),
            "Condition ID": m.get("condition_id") or m.get("conditionId"),
            "Market": m.get("question") or m.get("title"),
            "Resolution Date": m.get("endDate") or m.get("end_date") or m.get("end_date_iso"),
            "Market Start Date": m.get("startDate") or m.get("start_date") or m.get("createdAt") or m.get("created_at"),
            "Market Prob %": yes_probability,
            "No Prob %": no_probability,
            "Volume": pd.to_numeric(
                m.get("volumeNum", m.get("volume24hr", m.get("volume"))),
                errors="coerce",
            ),
            "Liquidity": pd.to_numeric(
                m.get("liquidityNum", m.get("liquidity")),
                errors="coerce",
            ),
            "clobTokenIds": token_ids,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        df.attrs["scan_stats"] = {
            "catalog_markets_retrieved": len(markets_raw),
            "pages_fetched": pages_fetched,
            "all_binary_markets": 0,
            "binary_price_markets": 0,
            "eligible_markets": 0,
            "rejected_markets": 0,
        }
        return df

    df["Resolution Date"] = pd.to_datetime(df["Resolution Date"], errors="coerce", utc=True)
    df["Market Start Date"] = pd.to_datetime(df["Market Start Date"], errors="coerce", utc=True)
    now_utc = pd.Timestamp.now(tz="UTC")
    df["Hours Remaining"] = (
        df["Resolution Date"] - now_utc
    ).dt.total_seconds() / 3600
    df["Days"] = df["Hours Remaining"] / 24
    df["Market Type"] = df["Market"].apply(classify_market)

    same_day_close = (
        df["Market Type"].eq("daily_close")
        & df["Resolution Date"].dt.date.eq(now_utc.date())
        & df["Hours Remaining"].ge(0)
    )
    df.loc[same_day_close, "Days"] = df.loc[same_day_close, "Hours Remaining"] / 6.5

    financial_candidates = df[df["Market Type"].isin(["price", "barrier", "range", "daily_close"])].copy()
    financial_candidates["Asset Phrase"] = financial_candidates["Market"].apply(extract_asset_phrase)
    financial_candidates["Ticker"] = financial_candidates["Market"].apply(find_ticker)
    financial_candidates["Target"] = financial_candidates["Market"].apply(extract_target)
    financial_candidates["Upper"] = pd.Series(
        np.nan,
        index=financial_candidates.index,
        dtype="float64",
    )
    range_mask = financial_candidates["Market Type"].eq("range")
    if range_mask.any():
        parsed_upper = pd.to_numeric(
            financial_candidates.loc[range_mask, "Market"].apply(extract_upper),
            errors="coerce",
        ).astype("float64")
        financial_candidates.loc[range_mask, "Upper"] = parsed_upper.to_numpy()
    financial_candidates["Direction"] = financial_candidates["Market"].apply(infer_direction)
    financial_candidates["Barrier Already Hit"] = False
    barrier_mask = financial_candidates["Market Type"].eq("barrier")
    for idx, barrier_row in financial_candidates.loc[barrier_mask].iterrows():
        financial_candidates.at[idx, "Barrier Already Hit"] = barrier_already_hit(
            barrier_row["Ticker"],
            barrier_row["Target"],
            barrier_row["Direction"],
            barrier_row["Market Start Date"],
        ) if pd.notna(barrier_row["Ticker"]) and pd.notna(barrier_row["Target"]) else False

    def rejection_reason(row):
        if pd.isna(row["Resolution Date"]): return "Missing resolution date"
        if pd.isna(row["Ticker"]) or not str(row["Ticker"]).strip(): return "Asset could not be resolved"
        if pd.isna(row["Target"]): return "Target price could not be parsed"
        if row["Market Type"] == "range" and pd.isna(row["Upper"]): return "Upper range could not be parsed"
        if row.get("Market Type") == "barrier" and bool(row.get("Barrier Already Hit", False)):
            return "Barrier was already hit before screening"
        if row["Days"] < 0: return "Market already expired"
        if row["Days"] > MAX_DAYS: return f"More than {MAX_DAYS} days to expiry"
        if pd.isna(row["Liquidity"]) or row["Liquidity"] < MIN_LIQUIDITY: return "Insufficient liquidity"
        return "Eligible"

    financial_candidates["Screen Result"] = financial_candidates.apply(rejection_reason, axis=1)
    eligible = financial_candidates[financial_candidates["Screen Result"] == "Eligible"].copy()

    eligible.attrs["scan_stats"] = {
        "catalog_markets_retrieved": len(markets_raw),
        "pages_fetched": pages_fetched,
        "all_binary_markets": len(df),
        "binary_price_markets": len(financial_candidates),
        "eligible_markets": len(eligible),
        "rejected_markets": len(financial_candidates) - len(eligible),
    }
    eligible.attrs["rejections"] = financial_candidates[
        financial_candidates["Screen Result"] != "Eligible"
    ][["Market", "Market Type", "Asset Phrase", "Ticker", "Screen Result", "Barrier Already Hit", "Liquidity", "Hours Remaining", "Days"]]
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
    "YES Edge %",
    "NO Edge %",
    "Edge %",
    "Signal",
    "Entry Side",
    "Entry Price %",
    "Position Size $",
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
    worksheet.append_row(values, value_input_option="USER_ENTERED")
    return "created", worksheet.row_count


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


def run_quant_scan():
    """Run the full market scan and return scored results."""
    markets_df = pull_markets()
    engine = MCPQuantEngine()
    scored = []
    for _, row in markets_df.iterrows():
        try:
            scored.append(engine.score_market(row))
        except Exception:
            continue
    results = pd.DataFrame(scored)
    if not results.empty:
        results = results.sort_values("Edge %", ascending=False)
    return markets_df, results


def _executed_market_ids(journal):
    if journal is None or journal.empty or "Market ID" not in journal.columns:
        return set()
    ids = journal["Market ID"].astype(str).str.strip()
    if "MCP Order ID" in journal.columns:
        order_ids = journal["MCP Order ID"].astype(str).str.strip()
        ids = ids[order_ids.ne("")]
    return set(ids[ids.ne("")])


def _executed_today_count(journal):
    if journal is None or journal.empty or "Date Saved" not in journal.columns:
        return 0
    work = journal.copy()
    dates = pd.to_datetime(work["Date Saved"], errors="coerce")
    today = pd.Timestamp.now().date()
    mask = dates.dt.date.eq(today)
    if "MCP Order ID" in work.columns:
        mask &= work["MCP Order ID"].astype(str).str.strip().ne("")
    return int(mask.sum())


def auto_execute_approved_trades(results):
    """Execute new approved signals with strict portfolio and duplicate controls."""
    client = MCPTradingClient()
    if not client.auto_trading_enabled:
        return []
    if results is None or results.empty:
        return [{"status": "skipped", "reason": "No scored markets"}]

    approved = results[
        results.get("Execution Approved", False).fillna(False).astype(bool)
        & results["Signal"].isin(["BUY YES", "BUY NO"])
    ].copy()
    if approved.empty:
        return [{"status": "skipped", "reason": "No approved trades"}]

    journal = load_journal()
    existing_market_ids = _executed_market_ids(journal)
    metrics = calculate_competition_metrics(journal)
    open_count = int(metrics.get("open_count", 0))
    today_count = _executed_today_count(journal)

    if today_count >= client.auto_max_trades_per_day:
        return [{"status": "skipped", "reason": "Daily auto-trade limit reached"}]
    if open_count >= client.auto_max_open_trades:
        return [{"status": "skipped", "reason": "Maximum open-trade limit reached"}]

    token = client.login()
    balance_payload = client.balance(token)
    balance = _safe_float(balance_payload.get("balance"), 0.0)
    if balance <= client.auto_min_balance:
        return [{"status": "skipped", "reason": f"Balance ${balance:.2f} is at/below auto floor"}]

    outcomes = []
    remaining_daily = client.auto_max_trades_per_day - today_count
    cycle_limit = min(client.auto_max_trades_per_cycle, remaining_daily)

    for _, candidate in approved.iterrows():
        if len([x for x in outcomes if x.get("status") == "matched"]) >= cycle_limit:
            break
        row = candidate.to_dict()
        market_id = str(row.get("Market ID", "")).strip()
        if not market_id or market_id in existing_market_ids:
            continue

        amount = min(_safe_float(row.get("Position Size $"), 0.0), client.max_order_amount)
        if amount < 1:
            outcomes.append({"status": "skipped", "market": row.get("Market"), "reason": "Amount below 1 pUSD"})
            continue
        # Reserve fee/headroom and preserve the configured competition floor.
        if balance - amount * 1.10 < client.auto_min_balance:
            outcomes.append({"status": "skipped", "market": row.get("Market"), "reason": "Insufficient floor buffer"})
            continue

        try:
            token_id, outcome = token_for_signal(row)
            estimate = client.price_estimate(token, token_id, "buy", amount)
            order_response = client.place_market_order(token, token_id, "buy", amount)
            status = str(order_response.get("clob_status") or order_response.get("status") or "submitted")

            executed_row = row.copy()
            executed_row.update({
                "Execution Token ID": token_id,
                "Execution Outcome": outcome,
                "Estimated Fill Price": estimate.get("price", ""),
                "Executed Amount pUSD": amount,
                "MCP Order ID": order_response.get("order_id") or order_response.get("id") or "",
                "CLOB Order ID": order_response.get("clob_order_id", ""),
                "Execution Status": status,
                "Execution Response": order_response,
                "Entry Price %": _safe_float(estimate.get("price"), 0.0) * 100,
                "Position Size $": amount,
            })
            action, row_number = save_to_journal(executed_row, update_existing=True)
            verify_execution_fields(row_number, {
                "Execution Token ID": token_id,
                "Execution Outcome": outcome,
                "Estimated Fill Price": estimate.get("price", ""),
                "Executed Amount pUSD": amount,
                "MCP Order ID": executed_row["MCP Order ID"],
                "CLOB Order ID": executed_row["CLOB Order ID"],
                "Execution Status": status,
                "Execution Response": order_response,
            })
            load_journal.clear()
            existing_market_ids.add(market_id)
            balance -= amount * 1.10
            outcomes.append({
                "status": "matched" if "match" in status.lower() else status,
                "market": row.get("Market"),
                "outcome": outcome,
                "amount": amount,
                "order_id": executed_row["MCP Order ID"],
            })
        except RuntimeError as error:
            message = str(error)
            if "422" in message and "no match" in message.lower():
                outcomes.append({"status": "skipped", "market": row.get("Market"), "reason": "No matchable FOK liquidity"})
                continue
            outcomes.append({"status": "error", "market": row.get("Market"), "reason": message})
        except Exception as error:
            outcomes.append({"status": "error", "market": row.get("Market"), "reason": str(error)})

    return outcomes or [{"status": "skipped", "reason": "All approved markets were duplicates or unavailable"}]


@st.fragment(run_every=AUTO_SCAN_INTERVAL)
def auto_trading_monitor():
    """Auto-scan and execute while this Streamlit session remains active."""
    try:
        client = MCPTradingClient()
    except Exception as error:
        st.error(f"Auto-trading configuration error: {error}")
        return

    if not client.auto_trading_enabled:
        st.info("Auto-trading is OFF. Set auto_trading_enabled = true in Streamlit Secrets to enable it.")
        return

    st.warning(
        "AUTO-TRADING IS ON. The app scans every 10 minutes while this page/session is active. "
        "Duplicate markets, daily limits, open-trade limits, price filters and the balance floor remain enforced."
    )
    try:
        markets_df, results = run_quant_scan()
        st.session_state["markets_df"] = markets_df
        st.session_state["results"] = results
        outcomes = auto_execute_approved_trades(results)
        st.write(f"Last automatic scan: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.dataframe(pd.DataFrame(outcomes), use_container_width=True)
    except Exception as error:
        st.error(f"Automatic scan failed safely; no further orders were attempted: {error}")


tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Journal", "Competition Tracker", "Research Analytics"])


with tab1:
    st.subheader("Automatic Trading Monitor")
    auto_trading_monitor()
    st.markdown("---")
    st.subheader("Run Market Screener")

    if st.button("Run MCP Screener", key="run_screener_button"):
        markets_df = pull_markets()
        st.session_state["markets_df"] = markets_df
        st.session_state["scan_stats"] = markets_df.attrs.get("scan_stats", {})
        st.session_state["rejections"] = markets_df.attrs.get("rejections", pd.DataFrame())

        engine = MCPQuantEngine()
        scored = []

        for _, row in markets_df.iterrows():
            try:
                scored.append(engine.score_market(row))
            except Exception:
                pass

        results = pd.DataFrame(scored)

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

        st.subheader("Top Trade Candidates")
        st.dataframe(results, use_container_width=True)

        model_signals = results[results["Signal"].isin(["BUY YES", "BUY NO"])].copy()
        buys = model_signals[model_signals["Execution Approved"] == True].copy()
        watchlist = model_signals[model_signals["Execution Approved"] != True].copy()

        st.subheader("Approved Actionable Trades")
        st.caption(
            "Only these trades can be sent to MCP. Approval requires sufficient edge, "
            "a 15%-85% entry price, at least 55% model probability for the selected outcome, "
            "and enough time remaining."
        )
        st.dataframe(buys, use_container_width=True)

        if not watchlist.empty:
            with st.expander("Model signals not approved for execution"):
                st.dataframe(
                    watchlist[[
                        "Market", "Signal", "Edge %", "Entry Price %",
                        "Selected Model Prob %", "Days", "Execution Decision"
                    ]],
                    use_container_width=True,
                )

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

        selected_trade = st.selectbox(
            "Select a trade to explain",
            results["Market"].tolist(),
            key="explain_trade_selectbox",
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
        st.write(f"**Model Horizon (trading days):** {explain['Days']}")
        st.write(f"**Momentum Score:** {explain['Momentum']}")
        st.write(f"**Momentum Adjustment:** {explain['Momentum Adj %']}%")
        st.write(f"**Model YES Probability:** {explain['Final Prob %']}%")
        st.write(f"**YES Edge:** {explain['YES Edge %']}%")
        st.write(f"**NO Edge:** {explain['NO Edge %']}%")
        st.write(f"**Signal:** {explain['Signal']}")
        st.write(f"**Entry Side:** {explain['Entry Side']}")
        st.write(f"**Entry Price:** {explain['Entry Price %']}%")
        st.write(f"**Suggested Position Size:** ${explain['Position Size $']}")

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
                                or ""
                            )
                            executed_row["CLOB Order ID"] = order_response.get("clob_order_id", "")
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
                            st.session_state.pop("mcp_trade_preview", None)
                            if journal_action == "updated":
                                st.success(
                                    "Model trade executed through MCP and the existing Google Sheets row was updated."
                                )
                            else:
                                st.success(
                                    "Model trade executed through MCP and a new Google Sheets row was created."
                                )
                            st.json(order_response)
            except Exception as error:
                message = str(error)
                if "422" in message and "no match" in message.lower():
                    st.warning(
                        "Trade unavailable: MCP found no matchable liquidity for the requested "
                        "FOK amount. No order was placed. Refresh the screener or try again later."
                    )
                else:
                    st.error(f"MCP execution setup error: {error}")
        else:
            st.info("No actionable model signals are available for execution.")

        st.markdown("---")
        st.subheader("Save Trade to Journal")

        selected_save = st.selectbox(
            "Select trade to save",
            results["Market"].tolist(),
            key="save_trade_selectbox",
        )

        if st.button("Save Selected Trade", key="save_trade_button"):
            row = results[results["Market"] == selected_save].iloc[0].to_dict()

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
    st.subheader("MCP Competition Performance Tracker")
    st.caption(
        "Counts unique MCP orders. Sharpe, Sortino and Calmar are based on resolved trade results in the journal. "
        "Open-position mark-to-market is not included in these calculated ratios."
    )

    journal = load_journal()
    metrics = calculate_competition_metrics(journal)

    live_cash = np.nan
    try:
        client = MCPTradingClient()
        token = client.login()
        live_cash = _safe_float(client.balance(token).get("balance"), np.nan)
    except Exception as error:
        st.warning(f"Live MCP cash balance could not be loaded: {error}")

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Completed Trades", f"{metrics['trade_count']} / {REQUIRED_COMPLETED_TRADES}")
    p2.metric("Days", f"{metrics['days_elapsed']} / {COMPETITION_DAYS}")
    p3.metric("Trades Remaining", metrics.get("trades_remaining", REQUIRED_COMPLETED_TRADES))
    p4.metric("Required Weekly Pace", f"{metrics['required_weekly_pace']:.1f}")

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("MCP Cash Balance", "N/A" if not np.isfinite(live_cash) else f"${live_cash:.2f}")
    b2.metric("Realized Bankroll", f"${metrics['realized_bankroll']:.2f}")
    b3.metric("Realized P&L", f"${metrics['total_pnl']:.2f}")
    floor_buffer = live_cash - MARK_TO_MARKET_FLOOR if np.isfinite(live_cash) else np.nan
    b4.metric("Cash Buffer Above $70", "N/A" if not np.isfinite(floor_buffer) else f"${floor_buffer:.2f}")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Win Rate", _metric_text(metrics['win_rate'], percent=True))
    r2.metric("Sharpe", _metric_text(metrics['sharpe']))
    r3.metric("Sortino", _metric_text(metrics['sortino']))
    r4.metric("Calmar", _metric_text(metrics['calmar']))

    d1, d2, d3 = st.columns(3)
    d1.metric("Realized Max Drawdown", _metric_text(metrics['max_drawdown'], percent=True))
    d2.metric("Resolved Trades", metrics['closed_count'])
    d3.metric("Open Executed Trades", metrics['open_count'])

    if np.isfinite(live_cash):
        if live_cash <= MARK_TO_MARKET_FLOOR:
            st.error("The MCP cash balance is at or below the $70 risk floor. Stop new orders and review exposure.")
        elif live_cash < 80:
            st.warning("The cash balance is within $10 of the $70 floor. Keep sizing conservative.")
        else:
            st.success("The current cash balance remains above the competition floor.")

    progress = min(metrics['trade_count'] / REQUIRED_COMPLETED_TRADES, 1.0)
    st.progress(progress, text=f"Trade requirement progress: {metrics['trade_count']} of {REQUIRED_COMPLETED_TRADES}")

    if len(metrics['equity']) > 0:
        equity_chart = pd.DataFrame({"Realized Equity": metrics['equity'].values})
        equity_chart.index = range(1, len(equity_chart) + 1)
        equity_chart.index.name = "Resolved Trade"
        st.subheader("Realized Equity Curve")
        st.line_chart(equity_chart)

    executed = metrics['executed']
    if not executed.empty:
        display_cols = [c for c in [
            "Date Saved", "Market", "Ticker", "Signal", "Edge %", "Final Prob %",
            "Executed Amount pUSD", "Execution Status", "Status", "Result", "PnL", "MCP Order ID"
        ] if c in executed.columns]
        st.subheader("Competition Trade Ledger")
        st.dataframe(executed[display_cols], use_container_width=True)
    else:
        st.info("No unique MCP executions have been recorded yet.")


with tab4:
    st.subheader("Research Analytics")
    st.caption("Use these tables after enough trades resolve. Very small samples can be misleading.")

    journal = load_journal()
    metrics = calculate_competition_metrics(journal)
    closed = metrics['closed'].copy()

    if not closed.empty:
        closed["Asset Class"] = closed.get("Ticker", "").apply(_asset_class_from_ticker)
        closed["Win"] = pd.to_numeric(closed["PnL"], errors="coerce").fillna(0) > 0
        closed["Trade Return"] = closed.apply(_trade_return, axis=1)

        st.subheader("Probability Calibration")
        calibration = _band_table(
            closed, "Final Prob %",
            bins=[0, 60, 70, 80, 90, 95, 101],
            labels=["<60", "60–69", "70–79", "80–89", "90–94", "95–100"],
        )
        if not calibration.empty:
            st.dataframe(calibration, use_container_width=True)

        st.subheader("Edge Validation")
        edge_table = _band_table(
            closed, "Edge %",
            bins=[0, 5, 10, 15, 20, 1000],
            labels=["<5", "5–9.9", "10–14.9", "15–19.9", "20+"],
        )
        if not edge_table.empty:
            st.dataframe(edge_table, use_container_width=True)

        st.subheader("Performance by Asset Class")
        by_asset = closed.groupby("Asset Class").agg(
            Trades=("Win", "size"),
            Win_Rate=("Win", "mean"),
            Total_PnL=("PnL", "sum"),
            Average_Return=("Trade Return", "mean"),
        ).reset_index()
        by_asset["Win Rate"] = (by_asset.pop("Win_Rate") * 100).round(1)
        by_asset["Total PnL"] = by_asset.pop("Total_PnL").round(3)
        by_asset["Average Return %"] = (by_asset.pop("Average_Return") * 100).round(1)
        st.dataframe(by_asset, use_container_width=True)

        type_col = "Type" if "Type" in closed.columns else "Market Type"
        if type_col in closed.columns:
            st.subheader("Performance by Market Type")
            by_type = closed.groupby(type_col).agg(
                Trades=("Win", "size"),
                Win_Rate=("Win", "mean"),
                Total_PnL=("PnL", "sum"),
            ).reset_index()
            by_type["Win Rate"] = (by_type.pop("Win_Rate") * 100).round(1)
            by_type["Total PnL"] = by_type.pop("Total_PnL").round(3)
            st.dataframe(by_type, use_container_width=True)

        st.subheader("Position Sizing Review")
        sizing_cols = [c for c in ["Market", "Edge %", "Final Prob %", "Executed Amount pUSD", "PnL", "Trade Return"] if c in closed.columns]
        st.dataframe(closed[sizing_cols], use_container_width=True)
    else:
        st.info("Research analytics will appear after executed trades resolve and the results are updated.")
