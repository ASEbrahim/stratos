"""
STRAT_OS - Market Data Fetcher
Fetches market data from Yahoo Finance (stocks/metals) and Binance (crypto).
"""

import yfinance as yf
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

# ── Binance integration for crypto tickers ──
BINANCE_API = "https://api.binance.com/api/v3"

# Map Yahoo-style symbols to Binance pairs
CRYPTO_SYMBOL_MAP = {
    "BTC-USD":  "BTCUSDT",
    "ETH-USD":  "ETHUSDT",
    "SOL-USD":  "SOLUSDT",
    "XRP-USD":  "XRPUSDT",
    "ADA-USD":  "ADAUSDT",
    "DOGE-USD": "DOGEUSDT",
    "DOT-USD":  "DOTUSDT",
    "AVAX-USD": "AVAXUSDT",
    "MATIC-USD":"MATICUSDT",
    "LINK-USD": "LINKUSDT",
    "BNB-USD":  "BNBUSDT",
}

# Map our interval keys to Binance kline intervals + limits
BINANCE_INTERVAL_MAP = {
    "1m":      {"interval": "1m",  "limit": 1000},   # ~16.7 hrs — "Today" view
    "5m":      {"interval": "5m",  "limit": 1000},   # ~3.5 days — "5D" view
    "1d_1mo":  {"interval": "1d",  "limit": 365},    # 1 year daily
    "1d_1y":   {"interval": "1d",  "limit": 1000},   # ~2.7 years daily
    "1wk":     {"interval": "1w",  "limit": 520},    # 10 years weekly
}


class MarketFetcher:
    """Fetches market data for configured tickers."""
    
    # Yahoo Finance-matching intervals with scroll-back history.
    # Key insight: intraday (1m/5m) needs HIGH max_points with SHORT periods
    # so candles stay real (not aggregated into multi-hour fake bars).
    # Gold trades 23hr/day = 1,380 1m-candles/day or 276 5m-candles/day.
    # Stocks trade 6.5hr/day = 390 1m-candles/day or 78 5m-candles/day.
    DEFAULT_INTERVALS = {
        "1m":      {"period": "3d",   "max_points": 5000},                        # 1D btn — 3 days, effectively no downsampling
        "5m":      {"period": "15d",  "max_points": 5000},                        # 5D btn — 15 days, effectively no downsampling
        "1d_1mo":  {"period": "1y",   "max_points": 365, "yf_interval": "1d"},    # 1M btn
        "1d_1y":   {"period": "5y",   "max_points": 1260, "yf_interval": "1d"},   # 1Y btn
        "1wk":     {"period": "10y",  "max_points": 520},                         # 5Y btn
    }

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with market configuration.
        
        Args:
            config: Market config section from config.yaml
        """
        self.tickers = {t["symbol"]: t for t in config.get("tickers", [])}
        # Only use DEFAULT_INTERVALS — ignore legacy config intervals
        # (old configs had 1m/5m which waste API calls for no frontend use)
        self.intervals = dict(self.DEFAULT_INTERVALS)
        self.alert_threshold = config.get("alert_threshold_percent", 5.0)
        self._cache: Dict[str, Any] = {}
        self._cache_time: Optional[datetime] = None
    
    def fetch_all(self, cache_ttl_seconds: int = 60) -> tuple[Dict[str, Any], List[Dict]]:
        """
        Fetch data for all tickers with caching.
        
        Args:
            cache_ttl_seconds: How long to cache results
            
        Returns:
            Dict with market data for all tickers (FRS schema)
        """
        # Check cache
        if self._cache and self._cache_time:
            age = (datetime.now() - self._cache_time).total_seconds()
            if age < cache_ttl_seconds:
                logger.debug(f"Using cached market data (age: {age:.0f}s)")
                return self._cache
        
        logger.info(f"Fetching market data for {len(self.tickers)} tickers...")
        result = {}
        alerts = []
        
        for symbol, config in self.tickers.items():
            try:
                if self._is_crypto(symbol):
                    ticker_data, ticker_alerts = self._fetch_binance(symbol, config["name"])
                else:
                    ticker_data, ticker_alerts = self._fetch_ticker(symbol, config["name"])
                if ticker_data:
                    result[symbol] = ticker_data
                    alerts.extend(ticker_alerts)
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                result[symbol] = {
                    "name": config["name"],
                    "error": str(e),
                    "data": {}
                }
        
        # Update cache (store as tuple to match return type)
        self._cache = (result, alerts)
        self._cache_time = datetime.now()

        return result, alerts
    
    def _fetch_ticker(self, symbol: str, name: str) -> tuple[Dict[str, Any], List[Dict]]:
        """
        Fetch data for a single ticker at all intervals.
        
        Args:
            symbol: Ticker symbol (e.g., 'NVDA', 'BTC-USD')
            name: Human-readable name
            
        Returns:
            Tuple of (ticker data dict, list of alerts)
        """
        ticker = yf.Ticker(symbol)
        alerts = []
        
        result = {
            "name": name,
            "data": {}
        }
        
        for interval_key, settings in self.intervals.items():
            try:
                # interval_key maps to yfinance unless overridden by yf_interval
                yf_interval = settings.get("yf_interval", interval_key)
                
                hist = ticker.history(
                    period=settings["period"],
                    interval=yf_interval
                )
                
                if hist.empty:
                    logger.warning(f"No data for {symbol} at {interval_key}")
                    continue
                
                # Get current price and change
                current_price = float(hist["Close"].iloc[-1])
                
                # Calculate change from first available point
                if len(hist) > 1:
                    first_price = float(hist["Close"].iloc[0])
                    change_pct = ((current_price - first_price) / first_price) * 100
                else:
                    change_pct = 0.0
                
                # Check for alert condition (>5% move)
                if abs(change_pct) >= self.alert_threshold and interval_key == "1m":
                    direction = "up" if change_pct > 0 else "down"
                    alerts.append({
                        "symbol": symbol,
                        "name": name,
                        "change": change_pct,
                        "direction": direction,
                        "price": current_price
                    })
                
                # Downsample with proper OHLCV aggregation (not naive picking)
                max_points = settings.get("max_points", 60)
                raw_len = len(hist)
                
                if raw_len <= max_points:
                    # No downsampling needed — send everything
                    history = hist["Close"].tolist()
                    opens = hist["Open"].tolist()
                    highs = hist["High"].tolist()
                    lows = hist["Low"].tolist()
                    volumes = hist["Volume"].tolist() if "Volume" in hist.columns else [0] * raw_len
                    timestamps_raw = hist.index.tolist()
                else:
                    # Aggregate groups of candles into one proper OHLCV bar
                    group_size = raw_len / max_points
                    history, opens, highs, lows, volumes, timestamps_raw = [], [], [], [], [], []
                    
                    for i in range(max_points):
                        start = int(i * group_size)
                        end = int((i + 1) * group_size)
                        if i == max_points - 1:
                            end = raw_len  # Include last point
                        
                        chunk = hist.iloc[start:end]
                        if chunk.empty:
                            continue
                        
                        opens.append(float(chunk["Open"].iloc[0]))
                        highs.append(float(chunk["High"].max()))
                        lows.append(float(chunk["Low"].min()))
                        history.append(float(chunk["Close"].iloc[-1]))
                        vol = float(chunk["Volume"].sum()) if "Volume" in chunk.columns else 0
                        volumes.append(vol)
                        timestamps_raw.append(chunk.index[0])
                
                timestamps_str = []
                for ts in timestamps_raw:
                    try:
                        timestamps_str.append(ts.strftime('%Y-%m-%dT%H:%M'))
                    except Exception:
                        timestamps_str.append(str(ts))
                
                # Summary stats from full (non-downsampled) data
                open_price = round(float(hist["Open"].iloc[0]), 2) if not hist.empty else 0
                high_price = round(float(hist["High"].max()), 2) if not hist.empty else 0
                low_price = round(float(hist["Low"].min()), 2) if not hist.empty else 0
                volume = int(hist["Volume"].sum()) if "Volume" in hist.columns and not hist.empty else 0
                prev_close_price = round(float(hist["Close"].iloc[0]), 2) if len(hist) > 1 else round(current_price, 2)
                
                result["data"][interval_key] = {
                    "price": round(current_price, 2),
                    "change": round(change_pct, 2),
                    "history": [round(p, 2) for p in history],
                    "timestamps": timestamps_str,
                    "opens": [round(p, 2) for p in opens],
                    "highs": [round(p, 2) for p in highs],
                    "lows": [round(p, 2) for p in lows],
                    "volumes": [int(v) for v in volumes],
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "volume": volume,
                    "prev_close": prev_close_price,
                }
                
            except Exception as e:
                logger.error(f"Failed to fetch {symbol} at {interval_key}: {e}")
                result["data"][interval_key] = {
                    "price": 0,
                    "change": 0,
                    "history": [],
                    "error": str(e)
                }
        
        return result, alerts
    
    @staticmethod
    def _is_crypto(symbol: str) -> bool:
        """Check if a symbol is a crypto ticker that should use Binance."""
        return symbol in CRYPTO_SYMBOL_MAP

    def _fetch_binance(self, symbol: str, name: str) -> tuple[Dict[str, Any], List[Dict]]:
        """
        Fetch crypto data from Binance public API.
        Uses /api/v3/klines for OHLCV and /api/v3/ticker/24hr for 24h stats.
        """
        binance_symbol = CRYPTO_SYMBOL_MAP[symbol]
        alerts = []
        result = {"name": name, "data": {}}

        # Fetch 24hr stats once for the ticker
        stats_24h = {}
        try:
            r = requests.get(f"{BINANCE_API}/ticker/24hr",
                             params={"symbol": binance_symbol}, timeout=10)
            if r.status_code == 200:
                stats_24h = r.json()
        except Exception as e:
            logger.warning(f"Binance 24hr stats failed for {symbol}: {e}")

        for interval_key, settings in self.intervals.items():
            try:
                bconf = BINANCE_INTERVAL_MAP.get(interval_key)
                if not bconf:
                    continue

                # Fetch klines
                params = {
                    "symbol": binance_symbol,
                    "interval": bconf["interval"],
                    "limit": bconf["limit"],
                }
                r = requests.get(f"{BINANCE_API}/klines", params=params, timeout=15)
                if r.status_code != 200:
                    logger.warning(f"Binance klines {symbol} {interval_key}: HTTP {r.status_code}")
                    continue

                klines = r.json()
                if not klines:
                    logger.warning(f"No Binance data for {symbol} at {interval_key}")
                    continue

                # Parse klines: [open_time, open, high, low, close, volume, ...]
                timestamps_str = []
                opens, highs, lows, history, volumes = [], [], [], [], []
                for k in klines:
                    ts_ms = int(k[0])
                    dt = datetime.fromtimestamp(ts_ms / 1000, tz=None)
                    timestamps_str.append(dt.strftime('%Y-%m-%dT%H:%M'))
                    opens.append(float(k[1]))
                    highs.append(float(k[2]))
                    lows.append(float(k[3]))
                    history.append(float(k[4]))  # close
                    volumes.append(float(k[5]))

                # Downsample if exceeding max_points
                max_points = settings.get("max_points", 5000)
                raw_len = len(history)
                if raw_len > max_points:
                    group_size = raw_len / max_points
                    ds_ts, ds_o, ds_h, ds_l, ds_c, ds_v = [], [], [], [], [], []
                    for i in range(max_points):
                        start = int(i * group_size)
                        end = int((i + 1) * group_size) if i < max_points - 1 else raw_len
                        ds_o.append(opens[start])
                        ds_h.append(max(highs[start:end]))
                        ds_l.append(min(lows[start:end]))
                        ds_c.append(history[end - 1])
                        ds_v.append(sum(volumes[start:end]))
                        ds_ts.append(timestamps_str[start])
                    timestamps_str, opens, highs, lows, history, volumes = ds_ts, ds_o, ds_h, ds_l, ds_c, ds_v

                current_price = history[-1] if history else 0
                first_price = history[0] if history else 0
                change_pct = ((current_price - first_price) / first_price * 100) if first_price else 0

                # Alert check
                if abs(change_pct) >= self.alert_threshold and interval_key == "1m":
                    alerts.append({
                        "symbol": symbol, "name": name,
                        "change": change_pct,
                        "direction": "up" if change_pct > 0 else "down",
                        "price": current_price,
                    })

                # Round prices based on magnitude
                def _rp(v):
                    return round(v, 2) if v >= 1 else round(v, 6)

                # Summary stats from full data
                all_highs = [float(k[2]) for k in klines]
                all_lows = [float(k[3]) for k in klines]
                all_vols = [float(k[5]) for k in klines]

                result["data"][interval_key] = {
                    "price": _rp(current_price),
                    "change": round(change_pct, 2),
                    "history": [_rp(p) for p in history],
                    "timestamps": timestamps_str,
                    "opens": [_rp(p) for p in opens],
                    "highs": [_rp(p) for p in highs],
                    "lows": [_rp(p) for p in lows],
                    "volumes": [int(v) for v in volumes],
                    "open": _rp(float(klines[0][1])),
                    "high": _rp(max(all_highs)),
                    "low": _rp(min(all_lows)),
                    "volume": int(sum(all_vols)),
                    "prev_close": _rp(float(klines[0][4])),
                    "source": "binance",
                }

                # Inject 24h stats into the 1m interval for header display
                if interval_key == "1m" and stats_24h:
                    result["data"][interval_key]["stats_24h"] = {
                        "high": float(stats_24h.get("highPrice", 0)),
                        "low": float(stats_24h.get("lowPrice", 0)),
                        "volume": float(stats_24h.get("volume", 0)),
                        "quote_volume": float(stats_24h.get("quoteVolume", 0)),
                        "change_pct": float(stats_24h.get("priceChangePercent", 0)),
                        "change_abs": float(stats_24h.get("priceChange", 0)),
                    }

            except Exception as e:
                logger.error(f"Binance fetch failed {symbol} at {interval_key}: {e}")
                result["data"][interval_key] = {
                    "price": 0, "change": 0, "history": [], "error": str(e)
                }

        return result, alerts

    def get_trend_description(self, symbol: str, data: Dict) -> str:
        """
        Generate a human-readable trend description.
        
        Args:
            symbol: Ticker symbol
            data: Ticker data with intervals
            
        Returns:
            Trend description string
        """
        if "1m" not in data.get("data", {}):
            return "No recent data available."
        
        d = data["data"]["1m"]
        change = d.get("change", 0)
        price = d.get("price", 0)
        name = data.get("name", symbol)
        
        if abs(change) < 0.5:
            trend = "stable"
        elif change > 0:
            trend = "up"
        else:
            trend = "down"
        
        if abs(change) >= self.alert_threshold:
            intensity = "significantly "
        elif abs(change) >= 2:
            intensity = "notably "
        else:
            intensity = ""
        
        if trend == "stable":
            return f"{name} is trading flat at ${price:.2f}."
        else:
            direction = "gained" if trend == "up" else "declined"
            return f"{name} has {intensity}{direction} {abs(change):.1f}% to ${price:.2f}."


def fetch_market_data(config: Dict[str, Any]) -> tuple[Dict[str, Any], List[Dict]]:
    """
    Convenience function to fetch market data from config.
    
    Args:
        config: Full configuration dict
        
    Returns:
        Tuple of (market data dict, alerts list)
    """
    market_config = config.get("market", {})
    cache_ttl = config.get("cache", {}).get("market_ttl_seconds", 60)
    
    fetcher = MarketFetcher(market_config)
    return fetcher.fetch_all(cache_ttl_seconds=cache_ttl)


if __name__ == "__main__":
    # Quick test
    import yaml
    
    logging.basicConfig(level=logging.INFO)
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    data, alerts = fetch_market_data(config)
    
    print("\n=== Market Data ===")
    for symbol, info in data.items():
        print(f"\n{symbol} ({info['name']}):")
        for interval, interval_data in info.get("data", {}).items():
            if "error" not in interval_data:
                print(f"  {interval}: ${interval_data['price']} ({interval_data['change']:+.2f}%)")
    
    if alerts:
        print("\n=== Alerts ===")
        for alert in alerts:
            print(f"  {alert['name']}: {alert['change']:+.2f}%")
