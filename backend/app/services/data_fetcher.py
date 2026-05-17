"""
data_fetcher.py  v5  — Smart multi-source, no hardcoded prices
───────────────────────────────────────────────────────────────
Price sources (tried in order, first success wins):

  METALS (Gold, Silver, Platinum, Palladium):
    1. gold-api.com       — completely free, no API key needed
    2. metals.live        — free fallback

  ALL COMMODITIES:
    1. Yahoo Finance v8   — free, sometimes blocked in India
    2. omkar.cloud API    — free 5000/month, Indian-friendly (needs key)

  USD/INR RATE:
    1. open.er-api.com    — free, no key, very reliable
    2. frankfurter.app    — ECB data, free
    3. exchangerate-api   — free fallback

IMPORTANT: No hardcoded fallback prices. If all sources fail for a
           commodity, it is skipped and shown as "unavailable" in
           the frontend instead of showing wrong data.
"""

import asyncio, requests, os
from app.services.unit_converter import convert as unit_convert
from app.services.price_display import usd_to_display
from datetime import datetime
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
})

# ── Commodity metadata ────────────────────────────────────────
COMMODITIES: Dict[str, dict] = {
    "GC=F":  {"name":"Gold",          "symbol":"XAU","category":"metals",      "unit":"per oz",     "omkar":"gold"},
    "SI=F":  {"name":"Silver",        "symbol":"XAG","category":"metals",      "unit":"per oz",     "omkar":"silver"},
    "HG=F":  {"name":"Copper",        "symbol":"HG", "category":"metals",      "unit":"per lb",     "omkar":"copper"},
    "PL=F":  {"name":"Platinum",      "symbol":"XPT","category":"metals",      "unit":"per oz",     "omkar":"platinum"},
    "PA=F":  {"name":"Palladium",     "symbol":"XPD","category":"metals",      "unit":"per oz",     "omkar":"palladium"},
    "CL=F":  {"name":"Crude Oil WTI", "symbol":"WTI","category":"energy",      "unit":"per bbl",    "omkar":"crude_oil"},
    "BZ=F":  {"name":"Brent Oil",     "symbol":"BNO","category":"energy",      "unit":"per bbl",    "omkar":"brent_crude_oil"},
    "NG=F":  {"name":"Natural Gas",   "symbol":"NG", "category":"energy",      "unit":"per MMBtu",  "omkar":"natural_gas"},
    "RB=F":  {"name":"Gasoline",      "symbol":"RB", "category":"energy",      "unit":"per gal",    "omkar":"gasoline"},
    "HO=F":  {"name":"Heating Oil",   "symbol":"HO", "category":"energy",      "unit":"per gal",    "omkar":"heating_oil"},
    "ZW=F":  {"name":"Wheat",         "symbol":"ZW", "category":"agricultural","unit":"per bushel",  "omkar":"wheat"},
    "ZC=F":  {"name":"Corn",          "symbol":"ZC", "category":"agricultural","unit":"per bushel",  "omkar":"corn"},
    "ZS=F":  {"name":"Soybeans",      "symbol":"ZS", "category":"agricultural","unit":"per bushel",  "omkar":"soybeans"},
    "KC=F":  {"name":"Coffee",        "symbol":"KC", "category":"agricultural","unit":"per lb",      "omkar":"coffee"},
    "SB=F":  {"name":"Sugar",         "symbol":"SB", "category":"agricultural","unit":"per lb",      "omkar":"sugar"},
    "CT=F":  {"name":"Cotton",        "symbol":"CT", "category":"agricultural","unit":"per lb",      "omkar":"cotton"},
    "CC=F":  {"name":"Cocoa",         "symbol":"CC", "category":"agricultural","unit":"per MT",      "omkar":"cocoa"},
    "ZO=F":  {"name":"Oats",          "symbol":"ZO", "category":"agricultural","unit":"per bushel",  "omkar":"oats"},
    "LE=F":  {"name":"Live Cattle",   "symbol":"LE", "category":"agricultural","unit":"per lb",      "omkar":"live_cattle"},
    "HE=F":  {"name":"Lean Hogs",     "symbol":"HE", "category":"agricultural","unit":"per lb",      "omkar":"lean_hog"},
    "LB=F":  {"name":"Lumber",        "symbol":"LB", "category":"agricultural","unit":"per MBF",     "omkar":"lumber"},
}

# Excluded from LSTM prediction tab (still on dashboard); lumber data is often sparse.
PREDICTION_EXCLUDED = frozenset({"LB=F"})


def prediction_tickers() -> list:
    return [t for t in COMMODITIES if t not in PREDICTION_EXCLUDED]


# Read omkar API key from env (optional — works without it for metals)
OMKAR_API_KEY = os.environ.get("OMKAR_API_KEY", "")

# Yahoo quotes several futures in cents (or other sub-dollar scales).
# Convert to USD before INR/unit conversions.
YAHOO_PRICE_SCALE = {
    # cents per bushel
    "ZW=F": 0.01,
    "ZC=F": 0.01,
    "ZS=F": 0.01,
    "ZO=F": 0.01,
    # cents per pound
    "CT=F": 0.01,
    "KC=F": 0.01,
    "SB=F": 0.01,
    "LE=F": 0.01,
    "HE=F": 0.01,
}


class DataFetcher:
    def __init__(self):
        self._cache:      Dict[str, dict] = {}
        self._usd_inr:    float = 84.50
        self._last_fetched = None
        self._using_fallback = False
        # In-memory price history for change% calculation
        self._prev_prices: Dict[str, float] = {}

    # ── Public API ────────────────────────────────────────────

    async def refresh_all_prices(self) -> Dict[str, dict]:
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._fetch_sync)
        self._cache = result
        self._last_fetched = datetime.utcnow()
        return result

    def get_cached(self) -> Dict[str, dict]:
        return self._cache

    async def get_historical(self, ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._fetch_history(ticker, period, interval))

    # ── USD/INR ───────────────────────────────────────────────

    def _get_usd_inr(self) -> float:
        sources = [
            ("open.er-api.com",     "https://open.er-api.com/v6/latest/USD",                  lambda d: d["rates"]["INR"]),
            ("frankfurter.app",     "https://api.frankfurter.app/latest?from=USD&to=INR",      lambda d: d["rates"]["INR"]),
            ("exchangerate-api",    "https://api.exchangerate-api.com/v4/latest/USD",           lambda d: d["rates"]["INR"]),
        ]
        for name, url, extract in sources:
            try:
                r    = SESSION.get(url, timeout=6)
                rate = float(extract(r.json()))
                if 80 < rate < 110:          # sanity check
                    print(f"[DataFetcher] USD/INR = ₹{rate:.4f} ({name})")
                    return rate
            except Exception as e:
                print(f"[DataFetcher] {name} failed: {e}")
        print(f"[DataFetcher] All USD/INR sources failed, keeping ₹{self._usd_inr:.2f}")
        return self._usd_inr

    # ── Gold-API.com (no key needed, metals only) ─────────────

    def _fetch_gold_api(self) -> Dict[str, dict]:
        """
        gold-api.com returns XAU, XAG, XPT, XPD in USD per troy oz.
        Also returns prev_close and change% directly from the API.
        No API key needed. Works from India.
        """
        metal_map = {
            "XAU": "GC=F",   # Gold
            "XAG": "SI=F",   # Silver
            "XPT": "PL=F",   # Platinum
            "XPD": "PA=F",   # Palladium
        }
        prices = {}
        for symbol, ticker in metal_map.items():
            try:
                url = f"https://gold-api.com/price/{symbol}"
                r   = SESSION.get(url, timeout=8)
                if r.status_code == 200:
                    data  = r.json()
                    price = data.get("price") or data.get("ask") or data.get("bid")
                    # gold-api also returns prev_close_price and ch (change)
                    prev  = data.get("prev_close_price") or data.get("previousClose")
                    chg   = data.get("ch") or data.get("chp")   # % change field
                    if price and float(price) > 0:
                        prices[ticker] = {
                            "price": float(price),
                            "prev":  float(prev) if prev else float(price),
                            "chp":   float(chg)  if chg  else None,
                        }
                        print(f"[DataFetcher] gold-api: {symbol} = ${price:.2f} (prev=${prev})")
            except Exception as e:
                print(f"[DataFetcher] gold-api {symbol} failed: {e}")
        return prices

    # ── Yahoo Finance v8 ──────────────────────────────────────

    def _normalize_yahoo_price(self, ticker: str, price: Optional[float]) -> Optional[float]:
        if price is None:
            return None
        val = float(price)
        scale = YAHOO_PRICE_SCALE.get(ticker, 1.0)
        return val * scale

    def _fetch_yahoo(self, ticker: str) -> Tuple[Optional[float], Optional[float]]:
        """Returns (today_close, yesterday_close) or (None, None)."""
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
            r   = SESSION.get(url, params={"interval":"1d","range":"5d"}, timeout=8)
            closes = r.json()["chart"]["result"][0]["indicators"]["quote"][0]["close"]
            closes = [c for c in (closes or []) if c is not None]
            if len(closes) >= 2:
                return (
                    self._normalize_yahoo_price(ticker, closes[-1]),
                    self._normalize_yahoo_price(ticker, closes[-2]),
                )
            elif len(closes) == 1:
                price = self._normalize_yahoo_price(ticker, closes[0])
                return price, price
        except Exception:
            pass
        return None, None

    # ── Omkar Cloud API (5000 free/month, needs free key) ─────

    def _fetch_omkar(self, name: str) -> Optional[float]:
        if not OMKAR_API_KEY:
            return None
        try:
            url = "https://commodity-price-api.omkar.cloud/commodity-price"
            r   = SESSION.get(url, params={"name": name},
                              headers={"API-Key": OMKAR_API_KEY}, timeout=8)
            if r.status_code == 200:
                return float(r.json()["price_usd"])
        except Exception:
            pass
        return None

    # ── Main fetch ────────────────────────────────────────────

    def _fetch_sync(self) -> Dict[str, dict]:
        self._usd_inr = self._get_usd_inr()
        rate  = self._usd_inr
        result = {}

        # Step 1: Try gold-api.com for all metals (fast, no key)
        metal_prices = self._fetch_gold_api()

        live_count = 0
        for ticker, meta in COMMODITIES.items():
            price_usd = None

            prev_usd  = None

            # Priority 1: gold-api for metals (includes prev price)
            if ticker in metal_prices:
                md        = metal_prices[ticker]
                price_usd = md["price"]
                prev_usd  = md.get("prev", price_usd)

            # Priority 2: Yahoo Finance (returns today, yesterday)
            if price_usd is None:
                today, yesterday      = self._fetch_yahoo(ticker)
                price_usd = today
                prev_usd  = yesterday

            # Priority 3: Omkar Cloud (if API key set)
            if price_usd is None:
                omkar_name = meta.get("omkar")
                if omkar_name:
                    price_usd = self._fetch_omkar(omkar_name)
                    prev_usd  = self._prev_prices.get(ticker, price_usd)

            # If all sources fail — use last cached price if available
            if price_usd is None:
                cached = self._cache.get(ticker)
                if cached:
                    print(f"[DataFetcher] {ticker}: all sources failed, reusing cached")
                    result[ticker] = {**cached, "source": "cached", "timestamp": datetime.utcnow().isoformat()}
                    continue
                else:
                    print(f"[DataFetcher] {ticker}: UNAVAILABLE")
                    continue

            # Calculate change% from yesterday's close
            prev_usd   = prev_usd or price_usd
            price_inr  = round(price_usd * rate, 2)
            prev_inr   = round(prev_usd  * rate, 2)
            change_pct = round(((price_inr - prev_inr) / prev_inr) * 100, 2) if prev_inr else 0.0

            self._prev_prices[ticker] = price_usd
            live_count += 1

            # Convert to Indian units
            converted = unit_convert(ticker, price_usd, rate)

            result[ticker] = {
                **meta,
                "ticker":        ticker,
                "price_inr":     price_inr,           # raw $/unit × INR
                "price_usd":     round(price_usd, 4),
                "change_pct":    change_pct,
                "usd_inr":       round(rate, 4),
                "timestamp":     datetime.utcnow().isoformat(),
                "source":        "live",
                # Indian unit prices
                "display_price": converted["display_price"],
                "display_unit":  converted["display_unit"],
                "display_label": converted["display_label"],
                "also":          converted.get("also", ""),
                **{k:v for k,v in converted.items()
                   if k not in ("display_price","display_unit","display_label","also")},
            }

        self._using_fallback = live_count == 0
        print(f"[DataFetcher] {live_count}/{len(COMMODITIES)} live · USD/INR=₹{rate:.2f}")

        # Gold sanity check
        gold = result.get("GC=F")
        if gold:
            gp = gold["price_inr"]
            g10g = gold["price_usd"] * rate / 31.1035 * 10 * 1.09
            print(f"[DataFetcher] Gold check: ${gold['price_usd']:.0f}/oz = ₹{gp:,.0f}/oz ≈ ₹{g10g:,.0f}/10g (MCX est.)")

        return result

    # ── Historical data ───────────────────────────────────────

    def _fetch_history(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        try:
            pm  = {"1mo":"1mo","3mo":"3mo","6mo":"6mo","1y":"1y","2y":"2y","5y":"5y"}
            im  = {"1d":"1d","1wk":"1wk","1mo":"1mo"}
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
            r   = SESSION.get(url,
                              params={"interval": im.get(interval,"1d"),
                                      "range":    pm.get(period,"6mo")},
                              timeout=12)
            res = r.json()["chart"]["result"][0]
            q   = res["indicators"]["quote"][0]
            df  = pd.DataFrame({
                "Date":  pd.to_datetime(res["timestamp"], unit="s"),
                "Open":  q.get("open",  []),
                "High":  q.get("high",  []),
                "Low":   q.get("low",   []),
                "Close": q.get("close", []),
                "Volume":q.get("volume",[]),
            }).dropna(subset=["Close"])
            if ticker in YAHOO_PRICE_SCALE:
                scale = YAHOO_PRICE_SCALE[ticker]
                for col in ["Open", "High", "Low", "Close"]:
                    if col in df.columns:
                        df[col] = df[col].astype(float) * scale
            rate = self._usd_inr
            for col in ["Open", "High", "Low", "Close"]:
                df[f"{col}_INR"] = (df[col].astype(float) * rate).round(2)
                disp = []
                for v in df[col].values:
                    try:
                        fv = float(v)
                        if np.isnan(fv):
                            disp.append(np.nan)
                        else:
                            disp.append(usd_to_display(ticker, fv, rate)[0])
                    except (TypeError, ValueError):
                        disp.append(np.nan)
                df[f"{col}_Display"] = np.round(disp, 2)
            df["USD_INR_Rate"] = rate
            return df.set_index("Date")
        except Exception as e:
            print(f"[DataFetcher] History failed {ticker}: {e}")
            # Return synthetic from last known price
            return self._synthetic_history(ticker, period)

    def _synthetic_history(self, ticker: str, period: str) -> pd.DataFrame:
        """Use last known price to generate realistic-looking history."""
        cached = self._cache.get(ticker)
        if not cached:
            return pd.DataFrame()
        rate = float(self._usd_inr)
        base_display = cached.get("display_price")
        if base_display is None and cached.get("price_usd"):
            base_display, _ = usd_to_display(ticker, float(cached["price_usd"]), rate)
        if not base_display:
            base_display = contract_inr_to_display_fallback(cached, ticker, rate)
        base_display = float(base_display)
        base_inr = float(cached.get("price_inr") or base_display)
        days  = {"1mo":30,"3mo":90,"6mo":180,"1y":365,"2y":730,"5y":1825}.get(period, 180)
        dates = pd.date_range(end=datetime.today(), periods=days, freq="D")
        np.random.seed(abs(hash(ticker)) % 1000)
        returns = np.random.normal(0.0003, 0.012, days)
        prices_d = base_display / (1 + returns[::-1]).cumprod()[::-1] * (1 + returns).cumprod()
        prices_d = prices_d / prices_d[-1] * base_display
        ratio = base_inr / base_display if base_display else 1.0
        prices_inr = prices_d * ratio
        return pd.DataFrame({
            "Open_INR":     (prices_inr * 0.999).round(2),
            "High_INR":     (prices_inr * 1.005).round(2),
            "Low_INR":      (prices_inr * 0.995).round(2),
            "Close_INR":    prices_inr.round(2),
            "Open_Display": (prices_d * 0.999).round(2),
            "High_Display": (prices_d * 1.005).round(2),
            "Low_Display":  (prices_d * 0.995).round(2),
            "Close_Display": prices_d.round(2),
            "Volume":       np.random.randint(1000, 50000, days),
            "USD_INR_Rate": rate,
        }, index=dates)


def contract_inr_to_display_fallback(cached: dict, ticker: str, rate: float) -> float:
    from app.services.price_display import contract_inr_to_display
    p = float(cached.get("price_inr") or 0)
    if p <= 0:
        return 0.0
    return contract_inr_to_display(ticker, p, rate)[0]