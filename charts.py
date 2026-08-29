import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
import pytz
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import feedparser
from urllib.parse import quote
from html import escape as html_escape

from config import (FUTURES_GROUPS, THEMES, SYMBOL_NAMES, FONTS, clean_symbol)

logger = logging.getLogger(__name__)

CHART_CONFIGS = [
    ('Day (15m)', '15m', '5-Session High/Low', 'session'),
    ('Weekly (1H)', '1h', 'Week High/Low', 'week'),
    ('Monthly (Daily)', '1d', 'Month High/Low', 'month'),
    ('Year (Weekly)', '1wk', 'Year High/Low', 'year'),
]

STATUS_LABELS = {
    'above_high': '▲ BREAKOUT',  'above_mid': '● BULL',
    'below_mid':  '● BEAR',      'below_low': '▼ BREAKDOWN',
}

def get_theme():
    name = st.session_state.get('theme', 'Dark')
    return THEMES.get(name, THEMES['Dark'])

def zone_colors():
    t = get_theme()
    return {'above_high': t['zone_hi'], 'above_mid': t['zone_amid'], 'below_mid': t['zone_bmid'], 'below_low': t['zone_lo']}

# =============================================================================
# HELPERS
# =============================================================================

def get_dynamic_period(boundary_type):
    now = pd.Timestamp.now()
    if boundary_type == 'session': return '10d'
    elif boundary_type == 'week': return f'{int(now.weekday() + 1 + 14 + 3)}d'
    elif boundary_type == 'month': return f'{int(now.day + 65 + 5)}d'
    elif boundary_type == 'year': return '3y'
    return '90d'


# =============================================================================
# TIMEZONE HELPERS
# =============================================================================

def _tz_now(hist_index):
    """Return a tz-aware 'now' aligned with the data's timezone."""
    tz = hist_index.tz
    if tz is not None:
        return pd.Timestamp.now(tz=tz)
    return pd.Timestamp.now()


def _to_date(ts):
    """Extract .date() safely from a Timestamp index entry."""
    return ts.date()


def _slice_period(hist, period_type, now=None):
    """Return (prev_period, current_bars) DataFrames for a given period boundary.

    Returns (prev_period_df, current_bars_df) or (None, None) if insufficient data.
    """
    if now is None:
        now = _tz_now(hist.index)

    dates = hist.index.map(_to_date)

    if period_type == 'session':
        gaps = hist.index.to_series().diff()
        median_gap = gaps.median()
        today = now.date()
        if median_gap < pd.Timedelta(hours=4):
            # Intraday data: use last 5 sessions for a stable zone
            prev_data = hist[dates < today]
            if prev_data.empty:
                return None, None
            unique_dates = sorted(set(prev_data.index.map(_to_date)))
            n_sessions = min(5, len(unique_dates))
            cutoff_dates = set(unique_dates[-n_sessions:])
            prev_period = prev_data[prev_data.index.map(_to_date).isin(cutoff_dates)]
            current_bars = hist[dates >= today]
        else:
            # Daily data: use last 5 trading days
            prev_data = hist[dates < today]
            if prev_data.empty:
                return None, None
            prev_period = prev_data.iloc[-5:]
            current_bars = hist[dates >= today]

    elif period_type == 'week':
        wsd = (now - pd.Timedelta(days=now.weekday())).date()
        prev_data = hist[dates < wsd]
        if prev_data.empty:
            return None, None
        pwsd = wsd - pd.Timedelta(days=7)
        prev_period = prev_data[prev_data.index.map(_to_date) >= pwsd]
        if prev_period.empty:
            prev_period = prev_data.tail(5)
        current_bars = hist[dates >= wsd]

    elif period_type == 'month':
        msd = now.replace(day=1).date()
        prev_data = hist[dates < msd]
        if prev_data.empty:
            return None, None
        pm = (now.month - 2) % 12 + 1
        py = now.year if now.month > 1 else now.year - 1
        prev_period = prev_data[(prev_data.index.month == pm) & (prev_data.index.year == py)]
        current_bars = hist[dates >= msd]

    elif period_type == 'year':
        ysd = now.replace(month=1, day=1).date()
        prev_data = hist[dates < ysd]
        if prev_data.empty:
            return None, None
        prev_period = prev_data[prev_data.index.year == now.year - 1]
        current_bars = hist[dates >= ysd]

    else:
        return None, None

    if prev_period.empty:
        return None, None
    return prev_period, current_bars

# =============================================================================
# PERIOD BOUNDARIES
# =============================================================================

@dataclass
class PeriodBoundary:
    idx: int
    date: pd.Timestamp
    prev_high: float
    prev_low: float
    prev_close: float

class PeriodBoundaryCalculator:
    @staticmethod
    def get_boundaries(df, boundary_type, symbol=''):
        if df is None or len(df) == 0: return []
        boundaries = []
        
        # Session: use midnight date change for all markets
        if boundary_type == 'session' and len(df) >= 2:
            is_break = lambda i: df.index[i].date() != df.index[i-1].date()
        else:
            is_break = {
                'year': lambda i: df.index[i].year != df.index[i-1].year,
                'month': lambda i: (df.index[i].month != df.index[i-1].month or df.index[i].year != df.index[i-1].year),
                'week': lambda i: (df.index[i].isocalendar()[1] != df.index[i-1].isocalendar()[1] or df.index[i].year != df.index[i-1].year),
            }.get(boundary_type, lambda i: False)

        prev_start = 0
        for i in range(1, len(df)):
            if is_break(i):
                prev_data = df.iloc[prev_start:i]
                if len(prev_data) > 0:
                    boundaries.append(PeriodBoundary(
                        idx=i, date=df.index[i],
                        prev_high=prev_data['High'].max(),
                        prev_low=prev_data['Low'].min(),
                        prev_close=prev_data['Close'].iloc[-1]))
                prev_start = i

        # Session boundaries: make the last boundary's H/L span the prior 5 sessions
        # so the zone is a stable 5-day range instead of a single volatile session
        if boundary_type == 'session' and len(boundaries) >= 2:
            n_lookback = min(5, len(boundaries))
            lookback = boundaries[-n_lookback:]
            combined_high = max(b.prev_high for b in lookback)
            combined_low = min(b.prev_low for b in lookback)
            last = boundaries[-1]
            boundaries[-1] = PeriodBoundary(
                idx=last.idx, date=last.date,
                prev_high=combined_high,
                prev_low=combined_low,
                prev_close=last.prev_close)

        return boundaries

def calculate_rsi(closes, period=14):
    if len(closes) < period + 1: return np.nan
    delta = closes.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not rsi.empty else np.nan

# =============================================================================
# DATA FETCHING
# =============================================================================

@dataclass
class FuturesMetrics:
    symbol: str
    price: float
    change_day: float
    change_wtd: float
    change_mtd: float
    change_ytd: float
    timestamp: datetime
    lag_minutes: float
    decimals: int
    hist_vol: float = np.nan
    day_sharpe: float = np.nan
    wtd_sharpe: float = np.nan
    mtd_sharpe: float = np.nan
    ytd_sharpe: float = np.nan
    day_status: str = ''
    week_status: str = ''
    month_status: str = ''
    year_status: str = ''
    current_dd: float = np.nan
    day_reversal: str = ''
    week_reversal: str = ''
    month_reversal: str = ''
    year_reversal: str = ''

class FuturesDataFetcher:
    def __init__(self, symbol):
        self.symbol = symbol
        self.ticker = yf.Ticker(symbol)
        self.est = pytz.timezone('US/Eastern')
        self.decimals = 4 if symbol.endswith('=X') else 2
        self._hist_yearly = None
        self._hist_intraday = None

    def fetch(self):
        """Fetch data from yfinance directly (used by chart tab)."""
        try:
            hist_yearly = self.ticker.history(period='1y')
            if hist_yearly.empty: return None
            hist_intraday = self.ticker.history(period='1d', interval='1m')
        except Exception as e:
            logger.warning(f"[{self.symbol}] fetch API error: {e}")
            return None
        return self._compute_metrics(hist_yearly, hist_intraday)

    def fetch_from_cache(self):
        """Compute metrics from pre-loaded data (used by batch scanner)."""
        hist_yearly = self._hist_yearly
        hist_intraday = self._hist_intraday if self._hist_intraday is not None else pd.DataFrame()
        if hist_yearly is None or hist_yearly.empty:
            return None
        return self._compute_metrics(hist_yearly, hist_intraday)

    def _compute_metrics(self, hist_yearly, hist_intraday):
        try:
            if hist_yearly.empty: return None
            if not hist_intraday.empty:
                current_price = hist_intraday['Close'].iloc[-1]
                daily_open = hist_intraday['Open'].iloc[0]
                daily_change = ((current_price - daily_open) / daily_open) * 100
                last_timestamp = hist_intraday.index[-1]
            else:
                current_price = hist_yearly['Close'].iloc[-1]
                daily_open = hist_yearly['Open'].iloc[0]
                daily_change = ((current_price - daily_open) / daily_open) * 100
                last_timestamp = hist_yearly.index[-1]

            now_est = datetime.now(self.est)
            try:
                lag_minutes = (now_est - last_timestamp.tz_convert(self.est)).total_seconds() / 60
            except Exception:
                lag_minutes = 0
            wtd, mtd, ytd = self._calculate_period_returns(hist_yearly, current_price)
            day_status = self._calculate_period_status(hist_yearly, current_price, 'session')
            week_status = self._calculate_period_status(hist_yearly, current_price, 'week')
            month_status = self._calculate_period_status(hist_yearly, current_price, 'month')
            year_status = self._calculate_period_status(hist_yearly, current_price, 'year')
            hist_vol = self._calculate_hist_vol(hist_yearly)
            current_dd = self._calculate_current_dd(hist_yearly, current_price)
            ytd_sharpe = self._calculate_ytd_sharpe(hist_yearly, current_price)
            mtd_sharpe = self._calculate_period_sharpe(hist_yearly, 'mtd')
            wtd_sharpe = self._calculate_period_sharpe(hist_yearly, 'wtd')
            day_sharpe = self._calculate_intraday_sharpe(hist_intraday) if not hist_intraday.empty else np.nan
            day_rev = self._check_reversal(hist_yearly, 'session')
            week_rev = self._check_reversal(hist_yearly, 'week')
            month_rev = self._check_reversal(hist_yearly, 'month')
            year_rev = self._check_reversal(hist_yearly, 'year')
            return FuturesMetrics(
                symbol=self.symbol, price=round(current_price, self.decimals),
                change_day=round(daily_change, 2),
                change_wtd=round(wtd, 2) if not np.isnan(wtd) else np.nan,
                change_mtd=round(mtd, 2) if not np.isnan(mtd) else np.nan,
                change_ytd=round(ytd, 2) if not np.isnan(ytd) else np.nan,
                timestamp=last_timestamp, lag_minutes=round(lag_minutes, 0),
                decimals=self.decimals,
                hist_vol=round(hist_vol, 1) if not np.isnan(hist_vol) else np.nan,
                day_sharpe=round(day_sharpe, 2) if not np.isnan(day_sharpe) else np.nan,
                wtd_sharpe=round(wtd_sharpe, 2) if not np.isnan(wtd_sharpe) else np.nan,
                mtd_sharpe=round(mtd_sharpe, 2) if not np.isnan(mtd_sharpe) else np.nan,
                ytd_sharpe=round(ytd_sharpe, 2) if not np.isnan(ytd_sharpe) else np.nan,
                day_status=day_status, week_status=week_status,
                month_status=month_status, year_status=year_status,
                current_dd=round(current_dd, 2) if not np.isnan(current_dd) else np.nan,
                day_reversal=day_rev, week_reversal=week_rev,
                month_reversal=month_rev, year_reversal=year_rev)
        except Exception as e:
            logger.warning(f"[{self.symbol}] fetch failed: {e}")
            return None

    def _calculate_hist_vol(self, hist):
        try:
            if len(hist) < 20: return np.nan
            dr = hist['Close'].pct_change().dropna()
            return dr.std() * np.sqrt(252) * 100 if len(dr) >= 20 else np.nan
        except Exception as e:
            logger.debug(f"[{self.symbol}] hist_vol error: {e}")
            return np.nan

    def _calculate_current_dd(self, hist, current_price):
        try:
            if len(hist) < 2: return np.nan
            peak = hist['High'].max()
            return ((current_price - peak) / peak) * 100 if peak != 0 else np.nan
        except Exception as e:
            logger.debug(f"[{self.symbol}] current_dd error: {e}")
            return np.nan

    def _calculate_ytd_sharpe(self, hist, current_price):
        try:
            now = _tz_now(hist.index)
            ytd_start = now.replace(month=1, day=1).date()
            ytd_hist = hist[hist.index.map(_to_date) >= ytd_start]
            if len(ytd_hist) < 10: return np.nan
            dr = ytd_hist['Close'].pct_change().dropna()
            if len(dr) < 5 or dr.std() == 0: return np.nan
            return (dr.mean() / dr.std()) * np.sqrt(252)
        except Exception as e:
            logger.debug(f"[{self.symbol}] ytd_sharpe error: {e}")
            return np.nan

    def _calculate_period_sharpe(self, hist, period):
        try:
            now = _tz_now(hist.index)
            if period == 'wtd':
                start_date = (now - pd.Timedelta(days=now.weekday())).date()
                min_bars = 2
            elif period == 'mtd':
                start_date = now.replace(day=1).date()
                min_bars = 3
            else:
                return np.nan
            ph = hist[hist.index.map(_to_date) >= start_date]
            if len(ph) < min_bars: return np.nan
            dr = ph['Close'].pct_change().dropna()
            if len(dr) < 2 or dr.std() == 0: return np.nan
            return (dr.mean() / dr.std()) * np.sqrt(252)
        except Exception as e:
            logger.debug(f"[{self.symbol}] period_sharpe ({period}) error: {e}")
            return np.nan

    def _calculate_intraday_sharpe(self, hist_intraday):
        try:
            if len(hist_intraday) < 30: return np.nan
            r = hist_intraday['Close'].pct_change().dropna()
            if len(r) < 20 or r.std() == 0: return np.nan
            return (r.mean() / r.std()) * np.sqrt(252 * 26)  # 26 bars/day at 15m
        except Exception as e:
            logger.debug(f"[{self.symbol}] intraday_sharpe error: {e}")
            return np.nan

    def _calculate_period_status(self, hist, current_price, period_type):
        try:
            prev_period, _ = _slice_period(hist, period_type)
            if prev_period is None:
                return ''
            ph, pl = prev_period['High'].max(), prev_period['Low'].min()
            pm = (ph + pl) / 2
            if current_price > ph: return 'above_high'
            elif current_price < pl: return 'below_low'
            elif current_price > pm: return 'above_mid'
            else: return 'below_mid'
        except Exception as e:
            logger.debug(f"[{self.symbol}] period_status ({period_type}) error: {e}")
            return ''

    def _check_reversal(self, hist, period_type):
        """Return 'buy' if bounced off low, 'sell' if rejected at high, '' otherwise."""
        try:
            if len(hist) < 3: return ''
            prev_period, current_bars = _slice_period(hist, period_type)
            if prev_period is None or current_bars is None or current_bars.empty:
                return ''
            ph, pl = prev_period['High'].max(), prev_period['Low'].min()
            current_close = current_bars['Close'].iloc[-1]
            period_high = current_bars['High'].max()
            period_low = current_bars['Low'].min()
            # Rejected at high → sell signal (amber)
            if period_high > ph and current_close <= ph: return 'sell'
            # Bounced off low → buy signal (green)
            if period_low < pl and current_close >= pl: return 'buy'
            return ''
        except Exception as e:
            logger.debug(f"[{self.symbol}] check_reversal ({period_type}) error: {e}")
            return ''

    def _calculate_period_returns(self, hist, current_price):
        try:
            now = _tz_now(hist.index)
            periods = {
                'wtd': (now - pd.Timedelta(days=now.weekday())).date(),
                'mtd': now.replace(day=1).date(),
                'ytd': now.replace(month=1, day=1).date()
            }
            returns = []
            for pn, sd in periods.items():
                ph = hist[hist.index.map(_to_date) >= sd]
                if not ph.empty:
                    sp = ph['Open'].iloc[0]
                    returns.append(((current_price - sp) / sp) * 100)
                else:
                    returns.append(np.nan)
            return tuple(returns)
        except Exception as e:
            logger.debug(f"[{self.symbol}] period_returns error: {e}")
            return (np.nan, np.nan, np.nan)


# =============================================================================
# CACHED DATA FETCHING
# =============================================================================

@st.cache_resource(ttl=900, show_spinner=False)
def fetch_sector_data(sector_name, symbols_override=None):
    """Fetch metrics for all symbols in a sector.

    Uses yf.download for a single batch request to get 1y daily data,
    then individual 1-minute calls only for intraday price/lag.
    Pass symbols_override to fetch arbitrary symbols (e.g. Custom mode).
    """
    if symbols_override is not None:
        symbols = symbols_override
    else:
        symbols = FUTURES_GROUPS.get(sector_name, [])
    if not symbols:
        return []

    # Batch download 1y daily data for the whole sector at once
    try:
        batch_daily = yf.download(symbols, period='1y', group_by='ticker',
                                   threads=True, progress=False)
    except Exception as e:
        logger.warning(f"Batch download failed for {sector_name}: {e}")
        batch_daily = pd.DataFrame()

    metrics = []
    for symbol in symbols:
        try:
            # Extract this symbol's daily data from batch result
            hist_yearly = pd.DataFrame()
            if not batch_daily.empty:
                if isinstance(batch_daily.columns, pd.MultiIndex):
                    if symbol in batch_daily.columns.get_level_values(0):
                        hist_yearly = batch_daily[symbol].dropna(subset=['Close'])
                else:
                    # Single-symbol download returns flat columns
                    hist_yearly = batch_daily.copy()

            # Fallback: individual fetch if batch missed this symbol
            if hist_yearly.empty:
                try:
                    ticker_fb = yf.Ticker(symbol)
                    hist_yearly = ticker_fb.history(period='1y')
                except Exception as e:
                    logger.warning(f"[{symbol}] individual fallback failed: {e}")
                    continue

            if hist_yearly.empty:
                continue

            # Individual intraday call (can't batch 1m intervals)
            ticker = yf.Ticker(symbol)
            hist_intraday = ticker.history(period='1d', interval='1m')

            fetcher = FuturesDataFetcher(symbol)
            fetcher._hist_yearly = hist_yearly
            fetcher._hist_intraday = hist_intraday

            result = fetcher.fetch_from_cache()
            if result:
                metrics.append(result)
        except Exception as e:
            logger.warning(f"[{symbol}] sector fetch error: {e}")
    return metrics

@st.cache_data(ttl=900, show_spinner=False)
def fetch_chart_data(symbol, period, interval):
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period, interval=interval)
    is_crypto = '-USD' in symbol
    if not is_crypto and not hist.empty:
        hist = hist[hist.index.dayofweek < 5]
    return hist

@st.cache_data(ttl=900, show_spinner=False)
def fetch_news(symbol):
    NEWS_TERMS = {
        'ES=F': 'S&P 500', 'NQ=F': 'Nasdaq 100', 'YM=F': 'Dow Jones', 'RTY=F': 'Russell 2000',
        'NKD=F': 'Nikkei 225', 'ZB=F': 'US treasury bonds', 'ZN=F': '10 year treasury yield',
        'ZF=F': '5 year treasury', 'ZT=F': '2 year treasury', '6E=F': 'EUR USD euro',
        '6J=F': 'USD JPY yen', '6B=F': 'GBP USD pound', '6A=F': 'AUD USD australian',
        'USDSGD=X': 'USD SGD Singapore dollar', 'CL=F': 'crude oil', 'NG=F': 'natural gas',
        'GC=F': 'gold price', 'SI=F': 'silver price', 'PL=F': 'platinum', 'HG=F': 'copper price',
        'ZS=F': 'soybean', 'ZC=F': 'corn grain', 'ZW=F': 'wheat', 'ZM=F': 'soybean meal',
        'SB=F': 'sugar commodity', 'KC=F': 'coffee arabica', 'CC=F': 'cocoa', 'CT=F': 'cotton commodity',
        'BTC-USD': 'bitcoin', 'ETH-USD': 'ethereum', 'SOL-USD': 'solana crypto', 'XRP-USD': 'XRP ripple',
        'BTC=F': 'bitcoin futures CME', 'ETH=F': 'ethereum futures CME',
    }
    st_term = NEWS_TERMS.get(symbol)
    if not st_term:
        fn = SYMBOL_NAMES.get(symbol, '')
        st_term = fn if fn else clean_symbol(symbol)
    results = []; seen = set()
    for when in ['1d', '3d']:
        if when == '3d' and len(results) >= 2: break
        try:
            url = f"https://news.google.com/rss/search?q={quote(st_term)}+when:{when}&hl=en&gl=US&ceid=US:en"
            feed = feedparser.parse(url)
            for entry in feed.entries[:12]:
                title = entry.get('title', '').strip()
                if not title or title in seen: continue
                link = entry.get('link', ''); provider = ''
                if ' - ' in title:
                    parts = title.rsplit(' - ', 1)
                    if len(parts) == 2 and len(parts[1]) < 40:
                        title, provider = parts[0].strip(), parts[1].strip()
                date_str = ''
                pub = entry.get('published', '') or entry.get('updated', '')
                if pub:
                    try:
                        dt = pd.Timestamp(pub)
                        now_ts = pd.Timestamp.now(tz=dt.tzinfo) if dt.tzinfo else pd.Timestamp.now()
                        dh = (now_ts - dt).total_seconds() / 3600
                        date_str = f"{int(dh*60)}m ago" if dh < 1 else f"{int(dh)}h ago" if dh < 24 else dt.strftime('%d %b %H:%M')
                    except Exception as e:
                        logger.debug(f"News date parse error: {e}")
                seen.add(title)
                # HTML-escape user-facing text to prevent injection
                results.append({
                    'title': html_escape(title),
                    'url': link,
                    'provider': html_escape(provider),
                    'date': date_str,
                })
        except Exception as e:
            logger.debug(f"News fetch error for {symbol} ({when}): {e}")
    # Score + rank by source tier and recency before returning
    def _tier(src):
        s = src.lower()
        t1 = ['bloomberg','reuters','financial times','wsj','wall street journal',"barron's",'barrons','ft.com','marketwatch']
        t3 = ['yahoo finance','motley fool','benzinga','msn','aol','fortune','usa today','thestreet']
        if any(k in s for k in t1): return 1
        if any(k in s for k in t3): return 3
        return 2
    def _rec(d):
        import re as _re
        m = _re.match(r'(\d+)([mh])', (d or '').lower())
        if not m: return 1
        mins = int(m.group(1)) if m.group(2)=='m' else int(m.group(1))*60
        return 3 if mins<=120 else 2 if mins<=360 else 1
    results.sort(key=lambda x: (4-_tier(x['provider']))*10 + _rec(x['date']), reverse=True)
    # Deduplicate by title prefix
    seen, deduped = set(), []
    for r in results:
        key = r['title'][:50].lower()
        if key not in seen:
            seen.add(key); deduped.append(r)
    return deduped[:8]


# =============================================================================
# SCANNER TABLE
# =============================================================================

def render_return_bars(metrics, sort_by='Default'):
    """Butterfly bar chart — symbol dead center, bars extend left/right.
    Height auto-matches scanner table: 40px header + n*28px rows + 6px padding."""
    t = get_theme(); pos_c = t['pos']; neg_c = t['neg']
    field_map = {
        'Default': ('change_day', 'DAY %', True), 'Day %': ('change_day', 'DAY %', True),
        'WTD %': ('change_wtd', 'WTD %', True), 'MTD %': ('change_mtd', 'MTD %', True),
        'YTD %': ('change_ytd', 'YTD %', True),
        'HV': ('hist_vol', 'HV %', False), 'DD': ('current_dd', 'DD %', False),
        'Sharpe Day': ('day_sharpe', 'SHARPE DAY', False),
        'Sharpe WTD': ('wtd_sharpe', 'SHARPE WTD', False),
        'Sharpe MTD': ('mtd_sharpe', 'SHARPE MTD', False),
        'Sharpe YTD': ('ytd_sharpe', 'SHARPE YTD', False),
    }
    attr, label, is_change = field_map.get(sort_by, ('change_day', 'DAY %', True))
    vals = [(clean_symbol(m.symbol), getattr(m, attr, 0) if not pd.isna(getattr(m, attr, 0)) else 0) for m in metrics]
    if not vals: return
    if sort_by in ('HV', 'DD'):
        vals.sort(key=lambda x: x[1])
    else:
        vals.sort(key=lambda x: x[1], reverse=True)
    max_abs = max(abs(v) for _, v in vals) or 1

    n = len(vals)
    # Match scanner: 2-row thead ~52px + n rows ~26px each + 2px border
    scanner_h = 52 + n * 26 + 2
    row_h = max((scanner_h - 46) // n, 18) if n > 0 else 22

    rows = ""
    for sym, v in vals:
        bar_pct = max(abs(v) / max_abs * 95, 3)
        if is_change:
            c = pos_c if v >= 0 else neg_c
        elif sort_by in ('HV', 'DD'):
            c = neg_c
        else:
            c = pos_c if v >= 0 else neg_c
        sign = '+' if v > 0 and is_change else ''
        fmt = f"{sign}{v:.1f}" if abs(v) < 100 else f"{sign}{v:.0f}"

        if v >= 0 or sort_by in ('HV',):
            left_content = ""
            right_content = (
                f"<div style='height:15px;width:{bar_pct}%;background:linear-gradient(90deg,{c}15,{c}55);border-radius:0 3px 3px 0'></div>"
                f"<span style='color:{c};font-size:8px;font-weight:700;margin-left:3px;font-family:{FONTS};white-space:nowrap;font-variant-numeric:tabular-nums'>{fmt}</span>"
            )
        else:
            left_content = (
                f"<span style='color:{c};font-size:8px;font-weight:700;margin-right:3px;font-family:{FONTS};white-space:nowrap;font-variant-numeric:tabular-nums'>{fmt}</span>"
                f"<div style='height:15px;width:{bar_pct}%;background:linear-gradient(270deg,{c}15,{c}55);border-radius:3px 0 0 3px'></div>"
            )
            right_content = ""

        rows += f"""<div style='display:flex;align-items:center;height:{row_h}px'>
            <div style='flex:1;display:flex;align-items:center;justify-content:flex-end'>{left_content}</div>
            <span style='width:36px;text-align:center;color:{t.get("text2","#9d9d9d")};font-size:9px;font-weight:600;font-family:{FONTS};flex-shrink:0'>{sym}</span>
            <div style='flex:1;display:flex;align-items:center'>{right_content}</div>
        </div>"""

    _bg0 = t.get('bg3', '#0f1522'); _bdr0 = t.get('border', '#1e293b')
    html = f"""<div style='background:{_bg0};border:1px solid {_bdr0};border-radius:6px;padding:0 6px 0 6px;overflow:hidden;height:{scanner_h}px'>
        <div style='display:flex;align-items:flex-end;height:46px;padding:0 2px'>
            <span style='color:#f8fafc;font-size:9px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;font-family:{FONTS}'>{label}</span>
            <div style='flex:1;height:1px;background:{_bdr0};margin-left:8px'></div>
        </div>
        {rows}
    </div>"""
    st.markdown(html, unsafe_allow_html=True)


def render_scanner_table(metrics, selected_symbol):
    if not metrics:
        _mut0 = get_theme().get('muted', '#475569')
        st.markdown(f"<div style='padding:10px;color:{_mut0};font-size:11px;'>No data — markets may be closed</div>", unsafe_allow_html=True)
        return

    t = get_theme(); zc = zone_colors()
    pos_c, neg_c = t['pos'], t['neg']

    _mut = t.get('muted', '#475569')

    def _fv(val):
        if pd.isna(val): return f"<span style='color:{_mut}'>—</span>"
        c = pos_c if val >= 0 else neg_c
        return f"<span style='color:{c};font-weight:600'>{'+' if val >= 0 else ''}{val:.2f}%</span>"

    def _dot(status, reversal=''):
        ico = ""
        c = zc.get(status, _mut)
        if status == 'above_high': ico = f"<span style='color:{c};font-weight:700;font-size:8px'>▲</span>"
        elif status == 'below_low': ico = f"<span style='color:{c};font-weight:700;font-size:8px'>▼</span>"
        if reversal == 'buy': ico += f"<span style='color:{pos_c};font-size:8px'>●</span>"
        elif reversal == 'sell': ico += f"<span style='color:{neg_c};font-size:8px'>●</span>"
        return f"<span style='display:inline-block;width:20px;text-align:left;vertical-align:middle;margin-left:2px'>{ico}</span>"

    def _chg(val, status, reversal=''):
        return f"<span style='display:inline-block;width:56px;text-align:right;font-variant-numeric:tabular-nums'>{_fv(val)}</span>{_dot(status, reversal)}"

    def _sharpe(val):
        if pd.isna(val): return f"<span style='color:{_mut}'>—</span>"
        c = pos_c if val >= 0 else neg_c
        return f"<span style='color:{c};font-weight:600'>{val:+.2f}</span>"

    def _trend(m):
        statuses = [m.year_status, m.month_status, m.week_status, m.day_status]
        bullish = sum(1 for s in statuses if s in ('above_high', 'above_mid')); bearish = 4 - bullish
        if bullish >= 3: conf = f"<span style='color:{pos_c};font-weight:700;font-size:10px'>{bullish}/4▲</span>"
        elif bearish >= 3: conf = f"<span style='color:{neg_c};font-weight:700;font-size:10px'>{bearish}/4▼</span>"
        else: conf = "<span style='color:#6b7280;font-weight:700;font-size:10px'>2/4─</span>"
        hb = all(s in ('above_high','above_mid') for s in statuses[:2])
        hr = all(s in ('below_mid','below_low') for s in statuses[:2])
        lb = any(s in ('above_high','above_mid') for s in statuses[2:])
        lr = any(s in ('below_mid','below_low') for s in statuses[2:])
        if bullish == 4: sig, sc = 'STR▲', t['str_up']
        elif bearish == 4: sig, sc = 'STR▼', t['str_dn']
        elif hb and lr: sig, sc = 'PULL', t['pull']
        elif hr and lb: sig, sc = 'BNCE', t['bnce']
        else: sig, sc = 'MIX', '#6b7280'
        return f"{conf} <span style='color:{sc};font-size:9px;font-weight:600'>{sig}</span>"

    _bdr = t.get('border', '#1e293b'); _bg3 = t.get('bg3', '#0f172a'); _mut = t.get('muted', '#475569')
    _row_alt = t.get('bg3', '#2a2a2a')
    th = f"padding:5px 8px;border-bottom:1px solid {_bdr};color:#f8fafc;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.06em;text-align:center;"
    td = "padding:4px 8px;border:none;"

    # Compute shared height for scanner/bar chart alignment
    _n_rows = len(metrics)
    _scanner_h = 52 + _n_rows * 26 + 2

    def _zone_dots(m):
        """Render 4 colored dots for D/W/M/Y zone status."""
        labels = ['D', 'W', 'M', 'Y']
        statuses = [m.day_status, m.week_status, m.month_status, m.year_status]
        dots = ''
        for lbl, s in zip(labels, statuses):
            c = zc.get(s, _mut)
            if s == 'above_high': icon = '▲'
            elif s == 'above_mid': icon = '●'
            elif s == 'below_mid': icon = '●'
            elif s == 'below_low': icon = '▼'
            else: icon = '·'
            status_lbl = STATUS_LABELS.get(s, '—')
            dots += f"<span title='{lbl}: {status_lbl}' style='color:{c};font-size:9px;font-weight:700'>{icon}</span>"
        return dots

    html = f"""<div style='overflow-x:auto;-webkit-overflow-scrolling:touch;border:1px solid {_bdr};border-radius:6px;height:{_scanner_h}px;overflow-y:hidden'><table style='border-collapse:collapse;font-family:{FONTS};font-size:11px;width:100%;line-height:1.3'>
        <thead style='background:{_bg3}'><tr>
            <th style='{th}text-align:left' rowspan='2'></th><th style='{th}' rowspan='2'>PRICE</th>
            <th style='{th}border-bottom:none' colspan='4'>CHANGE</th>
            <th style='{th}' rowspan='2'>ZONES</th>
            <th style='{th}' rowspan='2'>TREND</th>
            <th style='{th}' rowspan='2'>HV</th><th style='{th}' rowspan='2'>DD</th>
            <th style='{th}border-bottom:none' colspan='4'>SHARPE</th>
        </tr><tr>
            <th style='{th}'>DAY</th><th style='{th}'>WTD</th>
            <th style='{th}'>MTD</th><th style='{th}'>YTD</th>
            <th style='{th}'>DAY</th><th style='{th}'>WTD</th>
            <th style='{th}'>MTD</th><th style='{th}'>YTD</th>
        </tr></thead><tbody>"""

    _txt1 = t.get('text', '#e2e8f0'); _txt2 = t.get('text2', '#94a3b8')

    for idx, m in enumerate(metrics):
        pf = f"{m.price:,.{m.decimals}f}"
        ss = clean_symbol(m.symbol)
        if m.symbol == selected_symbol:
            bg = f'linear-gradient(90deg,{pos_c}08,{t.get("bg3","#1a2744")},{pos_c}08)'
        else:
            bg = _row_alt if idx % 2 == 1 else 'transparent'
        hv = f"<span style='color:{_txt2}'>{m.hist_vol:.1f}%</span>" if not pd.isna(m.hist_vol) else f"<span style='color:{_mut}'>—</span>"
        dd = f"<span style='color:{neg_c};font-weight:600'>{m.current_dd:.1f}%</span>" if not pd.isna(m.current_dd) else f"<span style='color:{_mut}'>—</span>"
        html += f"""<tr style='background:{bg}'>
            <td style='{td}color:{_txt1};font-weight:600;text-align:left;white-space:nowrap'>{ss}</td>
            <td style='{td}color:#f8fafc;font-weight:700;text-align:center'>{pf}</td>
            <td style='{td}text-align:center;white-space:nowrap'>{_chg(m.change_day, m.day_status, m.day_reversal)}</td>
            <td style='{td}text-align:center;white-space:nowrap'>{_chg(m.change_wtd, m.week_status, m.week_reversal)}</td>
            <td style='{td}text-align:center;white-space:nowrap'>{_chg(m.change_mtd, m.month_status, m.month_reversal)}</td>
            <td style='{td}text-align:center;white-space:nowrap'>{_chg(m.change_ytd, m.year_status, m.year_reversal)}</td>
            <td style='{td}text-align:center;white-space:nowrap;letter-spacing:2px'>{_zone_dots(m)}</td>
            <td style='{td}text-align:center;white-space:nowrap'>{_trend(m)}</td>
            <td style='{td}text-align:center'>{hv}</td>
            <td style='{td}text-align:center'>{dd}</td>
            <td style='{td}text-align:center'>{_sharpe(m.day_sharpe)}</td>
            <td style='{td}text-align:center'>{_sharpe(m.wtd_sharpe)}</td>
            <td style='{td}text-align:center'>{_sharpe(m.mtd_sharpe)}</td>
            <td style='{td}text-align:center'>{_sharpe(m.ytd_sharpe)}</td>
        </tr>"""
    html += "</tbody></table></div>"
    st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# 4-CHART GRID
# =============================================================================

def create_4_chart_grid(symbol, chart_type='line', mobile=False):
    zc = zone_colors(); t = get_theme()

    live_price = None
    try:
        hist_lag = fetch_chart_data(symbol, '1d', '5m')
        if not hist_lag.empty:
            live_price = float(hist_lag['Close'].iloc[-1])
    except Exception as e:
        logger.debug(f"[{symbol}] live price fetch error: {e}")

    if mobile:
        fig = make_subplots(rows=4, cols=1,
            subplot_titles=[tf[0].upper() for tf in CHART_CONFIGS],
            vertical_spacing=0.06)
        positions = [(1,1),(2,1),(3,1),(4,1)]
    else:
        fig = make_subplots(rows=2, cols=2,
            subplot_titles=[tf[0].upper() for tf in CHART_CONFIGS],
            vertical_spacing=0.08, horizontal_spacing=0.06)
        positions = [(1,1),(1,2),(2,1),(2,2)]
    chart_statuses = {}; chart_rsis = {}; computed_levels = {}

    for chart_idx, (label, interval, zone_desc, boundary_type) in enumerate(CHART_CONFIGS):
        row, col = positions[chart_idx]
        period = get_dynamic_period(boundary_type)
        hist = fetch_chart_data(symbol, period, interval)
        if hist.empty: continue

        boundaries = PeriodBoundaryCalculator.get_boundaries(hist, boundary_type, symbol)
        current_price = hist['Close'].iloc[-1]
        if live_price is not None: current_price = live_price

        x_vals = list(range(len(hist)))

        if boundary_type == 'session':
            tick_indices = [i for i, dt in enumerate(hist.index) if dt.minute == 0 and dt.hour % 4 == 0]
            tick_labels = [hist.index[i].strftime('%d %b') if hist.index[i].hour == 0 else hist.index[i].strftime('%H:%M') for i in tick_indices]
        elif boundary_type == 'week':
            n = 8; tick_indices = list(range(0, len(hist), max(1, len(hist)//n)))
            tick_labels = [hist.index[i].strftime('%a %d') for i in tick_indices]
        elif boundary_type == 'month':
            n = 8; tick_indices = list(range(0, len(hist), max(1, len(hist)//n)))
            tick_labels = [hist.index[i].strftime('%d %b') for i in tick_indices]
        else:
            n = 8; tick_indices = list(range(0, len(hist), max(1, len(hist)//n)))
            tick_labels = [hist.index[i].strftime("%b '%y") for i in tick_indices]

        line_color = '#6b7280'

        def get_zone(price, high, low, mid):
            if price > high: return 'above_high'
            elif price < low: return 'below_low'
            elif price > mid: return 'above_mid'
            else: return 'below_mid'

        def plot_line(x_data, closes, start_offset=0, datetimes=None, color='rgba(255,255,255,0.85)', width=1.8, zone_color=False, high=None, low=None, mid=None):
            """Draw a line. If zone_color=True, color segments by zone (green above mid, amber below)."""
            all_x = x_data if isinstance(x_data, list) else list(range(start_offset, start_offset+len(closes)))
            all_y = list(closes)
            all_dt = [dt.strftime('%d %b %H:%M') if boundary_type == 'session' else dt.strftime('%d %b %Y') for dt in datetimes] if datetimes is not None else None
            hover = '%{customdata}<br>%{y:.2f}<extra></extra>' if all_dt else '%{y:.2f}<extra></extra>'

            if not zone_color or mid is None:
                fig.add_trace(go.Scatter(x=all_x, y=all_y, mode='lines',
                    line=dict(color=color, width=width, shape='linear'),
                    showlegend=False, customdata=all_dt, hovertemplate=hover), row=row, col=col)
                return

            # Zone-colored: split into segments at zone transitions
            zones = []
            for c in closes:
                if c >= mid: zones.append('up')
                else: zones.append('dn')

            i = 0
            while i < len(zones):
                zone = zones[i]; start_i = i
                while i < len(zones) and zones[i] == zone: i += 1
                # Include 1 extra point for overlap (no gaps)
                end_i = min(i + 1, len(all_x))
                seg_x = all_x[start_i:end_i]
                seg_y = all_y[start_i:end_i]
                seg_dt = all_dt[start_i:end_i] if all_dt else None
                seg_color = t['pos'] if zone == 'up' else t['neg']
                fig.add_trace(go.Scatter(x=seg_x, y=seg_y, mode='lines',
                    line=dict(color=seg_color, width=width, shape='linear'),
                    showlegend=False, customdata=seg_dt, hovertemplate=hover), row=row, col=col)

        if boundaries:
            last_b = boundaries[-1]
            mid = (last_b.prev_high + last_b.prev_low) / 2
            zone_status = get_zone(current_price, last_b.prev_high, last_b.prev_low, mid)
            # Line color matches theme: green above mid, amber below
            line_color = t['pos'] if zone_status in ('above_high', 'above_mid') else t['neg']
            boundary_idx = last_b.idx

            if chart_type == 'bars':
                fig.add_trace(go.Candlestick(x=x_vals, open=hist['Open'].values, high=hist['High'].values,
                    low=hist['Low'].values, close=hist['Close'].values,
                    increasing_line_color=t['pos'], decreasing_line_color=t['neg'],
                    increasing_fillcolor=t['pos'], decreasing_fillcolor=t['neg'],
                    showlegend=False, line=dict(width=1)), row=row, col=col)

            if len(boundaries) >= 2:
                prev_b = boundaries[-2]; prev_mid = (prev_b.prev_high + prev_b.prev_low) / 2
                ps, pe = prev_b.idx, boundary_idx
                if pe > ps and chart_type == 'line':
                    plot_line(x_vals[ps:pe], hist['Close'].values[ps:pe], ps, hist.index[ps:pe], width=1.5, zone_color=True, mid=prev_mid)

            first_tracked = boundaries[-2].idx if len(boundaries) >= 2 else boundary_idx
            if first_tracked > 0 and chart_type == 'line':
                plot_line(x_vals[:first_tracked], hist['Close'].values[:first_tracked], 0, hist.index[:first_tracked], color='rgba(255,255,255,0.25)', width=1.0)

            if boundary_idx < len(hist) and chart_type == 'line':
                plot_line(x_vals[boundary_idx:], hist['Close'].values[boundary_idx:], boundary_idx, hist.index[boundary_idx:], width=2.0, zone_color=True, mid=mid)

        elif not boundaries:
            if chart_type == 'bars':
                fig.add_trace(go.Candlestick(x=x_vals, open=hist['Open'].values, high=hist['High'].values,
                    low=hist['Low'].values, close=hist['Close'].values,
                    increasing_line_color=t['pos'], decreasing_line_color=t['neg'],
                    increasing_fillcolor=t['pos'], decreasing_fillcolor=t['neg'],
                    showlegend=False, line=dict(width=1)), row=row, col=col)
            else:
                plot_line(x_vals, hist['Close'].values, 0, hist.index, color='rgba(255,255,255,0.5)', width=1.5)

        if boundaries:
            zone_status = get_zone(current_price, last_b.prev_high, last_b.prev_low, mid)
            chart_statuses[chart_idx] = STATUS_LABELS[zone_status]
            # Compute RB/RS from current period rolling high/low
            _rb, _rs = None, None
            _cp = hist.iloc[last_b.idx:]
            if len(_cp) > 1:
                _rh = _cp['High'].expanding().max().iloc[-1]
                _rl = _cp['Low'].expanding().min().iloc[-1]
                _rb = (_rh + last_b.prev_low) / 2
                _rs = (_rl + last_b.prev_high) / 2
            computed_levels[boundary_type] = {'high': last_b.prev_high, 'low': last_b.prev_low, 'mid': mid, 'price': current_price, 'status': zone_status, 'label': label, 'rb': _rb, 'rs': _rs}

        rsi_value = calculate_rsi(hist['Close']); chart_rsis[chart_idx] = rsi_value
        if boundary_type in computed_levels:
            computed_levels[boundary_type]['rsi'] = rsi_value

        # X-range: last data point at ~60% of visible chart width
        xref = f'xaxis{chart_idx+1}' if chart_idx > 0 else 'xaxis'
        last_bar = len(hist) - 1
        if boundary_type == 'session':
            # Show ~last 24h: for 15m bars that's ~96 bars
            bars_24h = min(96, last_bar)
            x_left = max(0, last_bar - bars_24h) - 2
        else:
            x_left = -2
        x_right = x_left + int((last_bar - x_left) / 0.6)
        fig.update_layout(**{xref: dict(range=[x_left, x_right])})

        # Y-axis range — fit to visible data only
        visible_start = max(0, x_left)
        visible_hist = hist.iloc[visible_start:]
        y_low_vals = visible_hist['Low'].dropna(); y_high_vals = visible_hist['High'].dropna()
        if len(y_low_vals) > 10:
            y_min = y_low_vals.quantile(0.005); y_max = y_high_vals.quantile(0.995)
        else:
            y_min = y_low_vals.min(); y_max = y_high_vals.max()
        pad = (y_max - y_min) * 0.08
        yref = f'yaxis{chart_idx+1}' if chart_idx > 0 else 'yaxis'
        fig.update_layout(**{yref: dict(range=[y_min-pad, y_max+pad], side='right', tickfont=dict(size=9, color='#94a3b8'))})

        # MAs on weekly chart
        if boundary_type == 'year':
            ma_20 = hist['Close'].rolling(window=20).mean(); ma_40 = hist['Close'].rolling(window=40).mean()
            if ma_20.notna().any():
                fig.add_trace(go.Scatter(x=x_vals, y=ma_20.values, mode='lines', line=dict(color='rgba(255,255,255,0.3)', width=0.7), showlegend=False, hovertemplate='MA20: %{y:.2f}<extra></extra>'), row=row, col=col)
                _ma20_last = ma_20.dropna().iloc[-1]
                _ma20_y = float(np.clip(_ma20_last, y_min + (y_max-y_min)*0.05, y_max - (y_max-y_min)*0.05))
                fig.add_annotation(x=1.02, y=_ma20_y,
                    xref=f'x{chart_idx+1} domain' if chart_idx > 0 else 'x domain',
                    yref=f'y{chart_idx+1}' if chart_idx > 0 else 'y',
                    text='MA20', showarrow=False,
                    font=dict(color='rgba(255,255,255,0.6)', size=8),
                    bgcolor='rgba(0,0,0,0.4)', bordercolor='rgba(255,255,255,0.2)', borderwidth=1, borderpad=2, xanchor='left')
            if ma_40.notna().any():
                fig.add_trace(go.Scatter(x=x_vals, y=ma_40.values, mode='lines', line=dict(color='rgba(168,85,247,0.5)', width=0.7), showlegend=False, hovertemplate='MA40: %{y:.2f}<extra></extra>'), row=row, col=col)
                _ma40_last = ma_40.dropna().iloc[-1]
                _ma40_y = float(np.clip(_ma40_last, y_min + (y_max-y_min)*0.05, y_max - (y_max-y_min)*0.05))
                fig.add_annotation(x=1.02, y=_ma40_y,
                    xref=f'x{chart_idx+1} domain' if chart_idx > 0 else 'x domain',
                    yref=f'y{chart_idx+1}' if chart_idx > 0 else 'y',
                    text='MA40', showarrow=False,
                    font=dict(color='rgba(168,85,247,0.8)', size=8),
                    bgcolor='rgba(0,0,0,0.4)', bordercolor='rgba(168,85,247,0.3)', borderwidth=1, borderpad=2, xanchor='left')

        # Boundary lines
        num_boundaries = min(2, len(boundaries))
        for j in range(num_boundaries):
            b = boundaries[-(j+1)]; px = b.idx; ex = len(hist)-1 if j == 0 else boundaries[-1].idx
            fig.add_vline(x=px, line=dict(color='rgba(255,255,255,0.25)', width=0.8, dash='dot'), row=row, col=col)
            ml = (b.prev_high + b.prev_low) / 2
            fig.add_trace(go.Scatter(x=[px,ex], y=[b.prev_high]*2, mode='lines', line=dict(color=zc['above_high'], width=0.9), showlegend=False, hovertemplate=f'High: {b.prev_high:.2f}<extra></extra>'), row=row, col=col)
            fig.add_trace(go.Scatter(x=[px,ex], y=[b.prev_low]*2, mode='lines', line=dict(color=zc['below_low'], width=0.9), showlegend=False, hovertemplate=f'Low: {b.prev_low:.2f}<extra></extra>'), row=row, col=col)
            fig.add_trace(go.Scatter(x=[px,ex], y=[b.prev_close]*2, mode='lines', line=dict(color='#475569', width=0.6, dash='dot'), showlegend=False, hovertemplate=f'Close: {b.prev_close:.2f}<extra></extra>'), row=row, col=col)
            fig.add_trace(go.Scatter(x=[px,ex], y=[ml]*2, mode='lines', line=dict(color='#94a3b8', width=0.6, dash='dot'), showlegend=False, hovertemplate=f'50%: {ml:.2f}<extra></extra>'), row=row, col=col)

        # Dynamic retrace lines — all timeframes
        # Buy retrace:  (Rolling Period High + Prev Period Low)  / 2 → green dashed
        # Sell retrace: (Rolling Period Low  + Prev Period High) / 2 → red dashed
        # Line traces bar-by-bar as the period high/low evolves
        if boundaries:
            last_b = boundaries[-1]
            current_period = hist.iloc[last_b.idx:]
            if len(current_period) > 1:
                prev_high = last_b.prev_high
                prev_low  = last_b.prev_low
                rolling_high = current_period['High'].expanding().max()
                rolling_low  = current_period['Low'].expanding().min()
                buy_y_vals  = ((rolling_high + prev_low)  / 2).values
                sell_y_vals = ((rolling_low  + prev_high) / 2).values
                rx_vals = list(range(last_b.idx, last_b.idx + len(current_period)))
                fig.add_trace(go.Scatter(
                    x=rx_vals, y=buy_y_vals, mode='lines',
                    line=dict(color='#22c55e', width=1.2, dash='dot'), showlegend=False,
                    hovertemplate='Buy retrace: %{y:.2f}<extra></extra>'), row=row, col=col)
                fig.add_trace(go.Scatter(
                    x=rx_vals, y=sell_y_vals, mode='lines',
                    line=dict(color='#ef4444', width=1.2, dash='dot'), showlegend=False,
                    hovertemplate='Sell retrace: %{y:.2f}<extra></extra>'), row=row, col=col)

        # Reversal dots — close-to-close failed breakout:
        # Closed above high then back below → sell (amber)
        # Closed below low then back above → buy (green)
        if boundaries:
            buy_x, buy_y, sell_x, sell_y = [], [], [], []
            bi = last_b.idx; ph, pl = last_b.prev_high, last_b.prev_low
            for j in range(bi, len(hist) - 1):
                c0 = hist['Close'].iloc[j]; c1 = hist['Close'].iloc[j + 1]
                if c0 > ph and c1 <= ph: sell_x.append(j + 1); sell_y.append(c1)
                elif c0 < pl and c1 >= pl: buy_x.append(j + 1); buy_y.append(c1)
            if len(boundaries) >= 2:
                pb = boundaries[-2]; end_i = last_b.idx
                for j in range(pb.idx, end_i - 1):
                    c0 = hist['Close'].iloc[j]; c1 = hist['Close'].iloc[j + 1]
                    if c0 > pb.prev_high and c1 <= pb.prev_high: sell_x.append(j + 1); sell_y.append(c1)
                    elif c0 < pb.prev_low and c1 >= pb.prev_low: buy_x.append(j + 1); buy_y.append(c1)
            if buy_x:
                fig.add_trace(go.Scatter(x=buy_x, y=buy_y, mode='markers',
                    marker=dict(color=t['pos'], size=7, symbol='circle', line=dict(color='rgba(0,0,0,0.3)', width=0.5)),
                    showlegend=False, hovertemplate='Buy reversal: %{y:.2f}<extra></extra>'), row=row, col=col)
            if sell_x:
                fig.add_trace(go.Scatter(x=sell_x, y=sell_y, mode='markers',
                    marker=dict(color=t['neg'], size=7, symbol='circle', line=dict(color='rgba(0,0,0,0.3)', width=0.5)),
                    showlegend=False, hovertemplate='Sell reversal: %{y:.2f}<extra></extra>'), row=row, col=col)

        # Axis formatting
        if tick_indices:
            axis_name = f'xaxis{chart_idx+1}' if chart_idx > 0 else 'xaxis'
            fig.update_layout(**{axis_name: dict(tickmode='array', tickvals=tick_indices, ticktext=tick_labels, tickfont=dict(color='#e2e8f0', size=9))})

        pd_dec = 4 if '=X' in symbol else 2

        # Collect all right-axis labels, then de-conflict positions
        # Each entry: (price, color, text, font_size, bold)
        _axis_labels = []
        _axis_labels.append((current_price, line_color, f'{current_price:.{pd_dec}f} C', 11, True))

        if boundaries:
            _mid = (last_b.prev_high + last_b.prev_low) / 2
            _axis_labels.append((last_b.prev_high, zc['above_high'], f'{last_b.prev_high:.{pd_dec}f} H', 9, True))
            _axis_labels.append((_mid,             '#94a3b8',         f'{_mid:.{pd_dec}f} M',             9, True))
            _axis_labels.append((last_b.prev_low,  zc['below_low'],   f'{last_b.prev_low:.{pd_dec}f} L',  9, True))

            current_period = hist.iloc[last_b.idx:]
            if len(current_period) > 1:
                _roll_high = current_period['High'].expanding().max().iloc[-1]
                _roll_low  = current_period['Low'].expanding().min().iloc[-1]
                _axis_labels.append(((_roll_high + last_b.prev_low)  / 2, '#22c55e', f'{(_roll_high + last_b.prev_low)/2:.{pd_dec}f} RB', 8, False))
                _axis_labels.append(((_roll_low  + last_b.prev_high) / 2, '#ef4444', f'{(_roll_low  + last_b.prev_high)/2:.{pd_dec}f} RS', 8, False))

        # De-collision: sort desc, enforce min gap, iterate to convergence
        if _axis_labels:
            _y_range = max(y_max - y_min, 1)
            _min_gap = _y_range * 0.028
            _lbls = sorted(_axis_labels, key=lambda x: x[0], reverse=True)
            _pos  = [l[0] for l in _lbls]
            for _ in range(30):
                _changed = False
                for _i in range(1, len(_pos)):
                    if _pos[_i-1] - _pos[_i] < _min_gap:
                        _mid2 = (_pos[_i-1] + _pos[_i]) / 2
                        _pos[_i-1] = _mid2 + _min_gap / 2
                        _pos[_i]   = _mid2 - _min_gap / 2
                        _changed = True
                if not _changed:
                    break

            _xref = f'x{chart_idx+1} domain' if chart_idx > 0 else 'x domain'
            _yref = f'y{chart_idx+1}' if chart_idx > 0 else 'y'
            for (_price, _col, _lbl, _fs, _bold), _y in zip(_lbls, _pos):
                _txt = f'<b>{_lbl}</b>' if _bold else _lbl
                _bp  = 3 if _fs >= 11 else 2
                fig.add_annotation(
                    x=1.02, y=_y, xref=_xref, yref=_yref,
                    text=_txt, showarrow=False,
                    font=dict(color=_col, size=_fs),
                    bgcolor='rgba(0,0,0,0.45)',
                    bordercolor=_col, borderwidth=1, borderpad=_bp,
                    xanchor='left')

    # Update subplot titles — clean: symbol + timeframe + RSI + status only
    stc = {'▲ ABOVE HIGH': zc['above_high'], '● ABOVE MID': zc['above_mid'],
           '● BELOW MID': zc['below_mid'], '▼ BELOW LOW': zc['below_low']}
    title_labels = [tf[0].upper() for tf in CHART_CONFIGS]
    _clean_sym = clean_symbol(symbol)

    for idx, ann in enumerate(fig['layout']['annotations']):
        txt = str(ann.text) if hasattr(ann, 'text') else ''
        if txt in title_labels:
            status = chart_statuses.get(idx, ''); rsi = chart_rsis.get(idx, np.nan)
            parts = [f"<b>{_clean_sym} {txt}</b>"]
            if not np.isnan(rsi):
                rc = zc['above_mid'] if rsi > 50 else zc['below_low']
                parts.append(f"<span style='color:{rc};font-size:9px'>RSI {rsi:.0f}</span>")
            if status:
                c = stc.get(status, '#64748b')
                parts.append(f"<span style='color:{c};font-size:9px'>{status}</span>")
            ann['text'] = '  '.join(parts)
            ann['font'] = dict(color='#f8fafc', size=10)

    # Single universal legend — top centre of figure
    _leg = (
        f"<span style='font-size:13px;color:{zc['above_high']}'>■</span>"
        f"<span style='font-size:11px;color:#94a3b8'> H  </span>"
        f"<span style='font-size:13px;color:#94a3b8'>■</span>"
        f"<span style='font-size:11px;color:#94a3b8'> M  </span>"
        f"<span style='font-size:13px;color:{zc['below_low']}'>■</span>"
        f"<span style='font-size:11px;color:#94a3b8'> L  </span>"
        f"<span style='font-size:13px;color:#22c55e'>■</span>"
        f"<span style='font-size:11px;color:#94a3b8'> RB  </span>"
        f"<span style='font-size:13px;color:#ef4444'>■</span>"
        f"<span style='font-size:11px;color:#94a3b8'> RS</span>"
    )
    fig.add_annotation(
        x=0.5, y=1.01, xref='paper', yref='paper',
        text=_leg, showarrow=False,
        font=dict(size=11, color='#94a3b8'),
        xanchor='center', yanchor='bottom'
    )

    _t = get_theme()
    _pbg = _t.get('plot_bg', '#121212'); _grd = _t.get('grid', '#1f1f1f')
    _axl = _t.get('axis_line', '#2a2a2a')
    _tpl = 'plotly_white' if _t.get('mode') == 'light' else 'plotly_dark'

    fig.update_layout(template=_tpl, height=1100 if mobile else 700, margin=dict(l=40,r=80,t=50,b=20),
        showlegend=False, plot_bgcolor=_pbg, paper_bgcolor=_pbg,
        dragmode='pan', hovermode='closest', autosize=True)
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.03)', linecolor='rgba(0,0,0,0)', tickfont=dict(color='#e2e8f0', size=9),
        showgrid=True, showticklabels=True, tickangle=0, rangeslider=dict(visible=False),
        fixedrange=False, showspikes=True, spikecolor='#475569', spikethickness=0.5, spikedash='dot', spikemode='across')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.03)', linecolor='rgba(0,0,0,0)', showgrid=True, side='right', tickfont=dict(color='#94a3b8', size=9),
        fixedrange=False, showspikes=True, spikecolor='#475569', spikethickness=0.5, spikedash='dot', spikemode='across')

    return fig, computed_levels




def render_key_levels(symbol, levels, target_height=None):
    zc = zone_colors(); t = get_theme(); pos_c = t['pos']
    ds = clean_symbol(symbol); fn = SYMBOL_NAMES.get(symbol, symbol)
    if not levels: return

    tfo = ['session','week','month','year']
    statuses = [levels.get(tf,{}).get('status','') for tf in tfo]
    bull = sum(1 for s in statuses if s in ('above_high','above_mid')); bear = 4 - bull
    hb = all(s in ('above_high','above_mid') for s in statuses[2:])
    hr = all(s in ('below_mid','below_low') for s in statuses[2:])
    lb = any(s in ('above_high','above_mid') for s in statuses[:2])
    lr = any(s in ('below_mid','below_low') for s in statuses[:2])
    if bull == 4: sig, sc = 'STRONG ▲', t['str_up']
    elif bear == 4: sig, sc = 'STRONG ▼', t['str_dn']
    elif hb and lr: sig, sc = 'PULLBACK ↻', '#fbbf24'
    elif hr and lb: sig, sc = 'BOUNCE ↻', '#a855f7'
    else: sig, sc = 'MIXED', '#6b7280'

    dec = 2; price = None
    for tf in tfo:
        if tf in levels: price = levels[tf]['price']; dec = 2 if price > 10 else 4; break

    _t = get_theme(); _il = _t.get('mode') == 'light'
    _hdr_bg = _t.get('bg3', '#1a2744'); _body_bg = _t.get('bg', '#1e1e1e')
    _bdr_ln = _t.get('border', '#2a2a2a')
    _txt1 = _t.get('text', '#e2e8f0'); _txt2 = _t.get('text2', '#b0b0b0')
    _mut = _t.get('muted', '#6d6d6d')
    _row_alt = _t.get('bg3', '#131b2e')

    th = f"padding:4px 10px;font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:0.07em;border-bottom:1px solid {_bdr_ln};text-align:right"
    td = f"padding:4px 10px;font-size:10px;font-variant-numeric:tabular-nums;text-align:right;border-bottom:1px solid {_bdr_ln}20"

    html = f"""<div style='padding:6px 10px;background:{_hdr_bg};border-left:2px solid {pos_c};display:flex;justify-content:space-between;align-items:center;font-family:{FONTS};border-radius:4px 4px 0 0'>
        <span><span style='color:#f8fafc;font-size:11px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase'>{ds} LEVELS</span>
        <span style='color:{_mut};font-size:9px;margin-left:6px'>{fn}</span></span>
        <span style='color:{sc};font-size:10px;font-weight:700'>{sig}</span></div>"""

    _levels_h = target_height or (44 + 52 + 9 * 26)
    html += f"<div style='background:{_body_bg};border:1px solid {_bdr_ln};border-top:none;border-radius:0 0 4px 4px;overflow-x:auto;height:{_levels_h - 44}px;overflow-y:auto'>"
    html += f"<table style='border-collapse:collapse;font-family:{FONTS};width:100%;line-height:1.2'><thead><tr>"
    html += f"<th style='{th};text-align:left;color:{_mut}'>LEVEL</th>"
    for tf in tfo:
        html += f"<th style='{th};color:#f8fafc'>{tf.upper()}</th>"
    html += "</tr></thead><tbody>"

    # Price row
    if price is not None:
        html += f"<tr style='background:{_row_alt}'>"
        html += f"<td style='{td};text-align:left;color:{_txt1};font-weight:700'>C</td>"
        for tf in tfo:
            p = levels.get(tf,{}).get('price')
            html += f"<td style='{td};color:#ffffff;font-weight:700'>{f'{p:,.{dec}f}' if p else '—'}</td>"
        html += "</tr>"

    _zc_hi = zc['above_high']; _zc_lo = zc['below_low']

    # H row
    html += f"<tr><td style='{td};text-align:left;color:{_zc_hi};font-weight:700'>H</td>"
    for tf in tfo:
        v = levels.get(tf,{}).get('high')
        html += f"<td style='{td};color:{_zc_hi}'>{f'{v:,.{dec}f}' if v else '—'}</td>"
    html += "</tr>"

    # RB row
    html += f"<tr style='background:{_row_alt}'><td style='{td};text-align:left;color:#22c55e;font-weight:700'>RB</td>"
    for tf in tfo:
        v = levels.get(tf,{}).get('rb')
        html += f"<td style='{td};color:#22c55e'>{f'{v:,.{dec}f}' if v else '—'}</td>"
    html += "</tr>"

    # M row
    html += f"<tr><td style='{td};text-align:left;color:{_mut};font-weight:700'>M</td>"
    for tf in tfo:
        v = levels.get(tf,{}).get('mid')
        html += f"<td style='{td};color:{_mut}'>{f'{v:,.{dec}f}' if v else '—'}</td>"
    html += "</tr>"

    # RS row
    html += f"<tr style='background:{_row_alt}'><td style='{td};text-align:left;color:#ef4444;font-weight:700'>RS</td>"
    for tf in tfo:
        v = levels.get(tf,{}).get('rs')
        html += f"<td style='{td};color:#ef4444'>{f'{v:,.{dec}f}' if v else '—'}</td>"
    html += "</tr>"

    # L row
    html += f"<tr><td style='{td};text-align:left;color:{_zc_lo};font-weight:700'>L</td>"
    for tf in tfo:
        v = levels.get(tf,{}).get('low')
        html += f"<td style='{td};color:{_zc_lo}'>{f'{v:,.{dec}f}' if v else '—'}</td>"
    html += "</tr>"

    # Status row
    html += f"<tr style='background:{_row_alt}'><td style='{td};text-align:left;color:{_mut};font-weight:700'>STATUS</td>"
    for tf in tfo:
        st_val = levels.get(tf,{}).get('status','')
        sco = zc.get(st_val, _mut)
        stx = STATUS_LABELS.get(st_val,'—')
        html += f"<td style='{td};color:{sco};font-size:9px;font-weight:700;text-align:right'>{stx}</td>"
    html += "</tr>"

    # DIST row — % distance from current price to nearest level (per TF)
    html += f"<tr><td style='{td};text-align:left;color:{_mut};font-weight:700'>DIST</td>"
    for tf in tfo:
        lv = levels.get(tf, {})
        cp = lv.get('price')
        candidates = {k: lv.get(k) for k in ('high','rb','mid','rs','low') if lv.get(k)}
        if cp and candidates:
            nearest_k = min(candidates, key=lambda k: abs(candidates[k] - cp))
            nearest_v = candidates[nearest_k]
            pct = (cp - nearest_v) / nearest_v * 100
            dc = zc['above_mid'] if pct >= 0 else zc['below_mid']
            sign = '+' if pct >= 0 else ''
            html += f"<td style='{td};color:{dc};font-size:9px'>{sign}{pct:.1f}%<span style='color:{_mut};font-size:8px'> {nearest_k.upper()}</span></td>"
        else:
            html += f"<td style='{td};color:{_mut}'>—</td>"
    html += "</tr>"

    # RSI row
    def _rsi_color(v):
        if np.isnan(v): return _mut
        if v >= 70: return '#f59e0b'   # overbought — amber
        if v >= 50: return zc['above_mid']
        if v >= 30: return zc['below_mid']
        return '#22c55e'               # oversold — green

    html += f"<tr><td style='{td};text-align:left;color:{_mut};font-weight:700'>RSI</td>"
    for tf in tfo:
        v = levels.get(tf, {}).get('rsi', np.nan)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            html += f"<td style='{td};color:{_mut}'>—</td>"
        else:
            rc = _rsi_color(float(v))
            html += f"<td style='{td};color:{rc};font-weight:700'>{float(v):.0f}</td>"
    html += "</tr>"

    html += "</tbody></table></div>"
    st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# NEWS PANEL
# =============================================================================

def render_news_panel(symbol, target_height=None):
    ds = clean_symbol(symbol); fn = SYMBOL_NAMES.get(symbol, symbol)
    t = get_theme(); pos_c = t['pos']
    _il = t.get('mode') == 'light'
    _hdr_bg = t.get('bg3', '#1a2744'); _body_bg = t.get('bg', '#1e1e1e')
    _bdr_ln = t.get('border', '#2a2a2a')
    _txt1 = t.get('text', '#e2e8f0'); _mut = t.get('muted', '#6d6d6d')
    _link_c = '#334155' if _il else '#c9d1d9'
    news = fetch_news(symbol)

    html = f"""<div style='padding:8px 12px;background:{_hdr_bg};border-left:2px solid {pos_c};font-family:{FONTS};margin-top:8px;border-radius:4px 4px 0 0'>
        <span style='color:#f8fafc;font-size:11px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase'>{ds} NEWS</span>
        <span style='color:{_mut};font-size:10px;margin-left:6px;font-weight:400'>{fn}</span></div>"""

    if not news:
        html += f"<div style='padding:12px;background-color:{_body_bg};border:1px solid {_bdr_ln};border-top:none;border-radius:0 0 4px 4px;color:{_mut};font-size:11px;font-family:{FONTS}'>No news available</div>"
    else:
        _row_alt = t.get('bg3', '#131b2e')
        _news_h = (target_height or 330) - 44
        html += f"<div style='background-color:{_body_bg};border:1px solid {_bdr_ln};border-top:none;border-radius:0 0 4px 4px;height:{_news_h}px;overflow-y:auto'>"
        for i, item in enumerate(news):
            t_text = item['title']; u = item['url']; p = item['provider']; d = item['date']
            row_bg = _body_bg if i % 2 == 0 else _row_alt
            title_el = f"<a href='{u}' target='_blank' style='color:{_link_c};text-decoration:none;font-size:9.5px;font-weight:500;overflow:hidden;text-overflow:ellipsis'>{t_text}</a>" if u else f"<span style='color:{_link_c};font-size:9.5px'>{t_text}</span>"
            src_html = f"<span style='color:{pos_c};font-weight:600;font-size:8.5px'>{p}</span>" if p else ""
            date_html = f"<span style='color:{_mut};font-size:8.5px'>{d}</span>" if d else ""
            html += (
                f"<div style='padding:1px 8px;border-bottom:1px solid {_bdr_ln}10;font-family:{FONTS};background:{row_bg};"
                f"display:flex;align-items:baseline;gap:0;white-space:nowrap;overflow:hidden'>"
                f"<span style='flex-shrink:0;width:90px;text-align:left'>{src_html}</span>"
                f"<span style='flex-shrink:0;width:55px;text-align:left'>{date_html}</span>"
                f"<span style='overflow:hidden;text-overflow:ellipsis'>{title_el}</span>"
                f"</div>"
            )
        html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

def render_charts_tab(is_mobile, est):
    """Chart tab content — sector scanner, asset charts, levels, news."""
    t = get_theme()
    pos_c = t['pos']

    # Determine if we're in Custom mode
    _is_custom = (st.session_state.sector == 'Custom')

    if not _is_custom:
        symbols = FUTURES_GROUPS[st.session_state.sector]
        sym_labels = [clean_symbol(s) for s in symbols]
        if st.session_state.symbol not in symbols:
            st.session_state.symbol = symbols[0]
        current_idx = symbols.index(st.session_state.symbol)
    else:
        symbols = []
        sym_labels = []
        current_idx = 0

    # Sector change callback
    def _on_sector_change():
        new_sector = st.session_state.sel_sector
        st.session_state.sector = new_sector
        if new_sector != 'Custom':
            st.session_state.symbol = FUTURES_GROUPS[new_sector][0]
        if 'sel_asset' in st.session_state:
            del st.session_state.sel_asset

    # Controls row
    _lbl = f"color:#f8fafc;font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;font-family:{FONTS}"

    if _is_custom:
        # Custom mode: SECTOR + TICKER input + CHART
        if is_mobile:
            col_sec, col_ticker, col_ct = st.columns([3, 4, 1])
        else:
            col_sec, col_ticker, col_ct = st.columns([3, 5, 1])

        with col_sec:
            st.markdown(f"<div style='{_lbl}'>SECTOR</div>", unsafe_allow_html=True)
            sector_names = list(FUTURES_GROUPS.keys()) + ['Custom']
            sector_idx = sector_names.index('Custom')
            st.selectbox("Sector", sector_names, index=sector_idx,
                key='sel_sector', label_visibility='collapsed',
                on_change=_on_sector_change)

        with col_ticker:
            st.markdown(f"<div style='{_lbl}'>TICKER SYMBOL</div>", unsafe_allow_html=True)
            _prev = st.session_state.get('custom_ticker', '')
            custom_input = st.text_input("Ticker", value=_prev,
                placeholder="e.g. AAPL, SPY, BTC-USD, NVDA",
                key='custom_ticker_input', label_visibility='collapsed')
            custom_input = custom_input.strip().upper()
            if custom_input and custom_input != _prev:
                st.session_state['custom_ticker'] = custom_input
                st.session_state.symbol = custom_input
                st.rerun()
            elif custom_input:
                st.session_state.symbol = custom_input

        with col_ct:
            st.markdown(f"<div style='{_lbl}'>CHART</div>", unsafe_allow_html=True)
            chart_options = ['Line', 'Bars']
            ct_idx = 0 if st.session_state.chart_type == 'line' else 1
            ct = st.selectbox("Chart", chart_options, index=ct_idx,
                key='chart_select', label_visibility='collapsed')
            st.session_state.chart_type = 'line' if ct == 'Line' else 'bars'

        # No ticker entered yet
        if not st.session_state.get('custom_ticker'):
            st.info("Enter a ticker symbol above to load charts, levels and news.")
            return

        # Fetch and render scanner for the single custom symbol
        with st.spinner('Loading market data...'):
            metrics = fetch_sector_data('Custom', symbols_override=[st.session_state.symbol])
        if metrics:
            col_scan, col_bars = st.columns([55, 45])
            with col_scan:
                render_scanner_table(metrics, st.session_state.symbol)
            with col_bars:
                render_return_bars(metrics, 'Default')

    else:
        # Normal mode: SECTOR + ASSET + SORT + CHART
        if is_mobile:
            col_sec, col_ast, col_sort, col_ct = st.columns([3, 2, 2, 1])
        else:
            col_sec, col_ast, col_sort, col_ct = st.columns([3, 3, 2, 1])

        with col_sec:
            st.markdown(f"<div style='{_lbl}'>SECTOR</div>", unsafe_allow_html=True)
            sector_names = list(FUTURES_GROUPS.keys()) + ['Custom']
            sel_idx = sector_names.index(st.session_state.sector) if st.session_state.sector in sector_names else 0
            if st.session_state.get('sel_sector') != st.session_state.sector:
                st.session_state.sel_sector = st.session_state.sector
            st.selectbox("Sector", sector_names, index=sel_idx,
                key='sel_sector', label_visibility='collapsed',
                on_change=_on_sector_change)

        with col_ast:
            st.markdown(f"<div style='{_lbl}'>ASSET</div>", unsafe_allow_html=True)
            selected_label = st.selectbox("Asset", sym_labels, index=current_idx,
                key='sel_asset', label_visibility='collapsed')
            selected_sym = symbols[sym_labels.index(selected_label)]
            if selected_sym != st.session_state.symbol:
                st.session_state.symbol = selected_sym
                st.rerun()

        with col_sort:
            st.markdown(f"<div style='{_lbl}'>SORT</div>", unsafe_allow_html=True)
            sort_options = ['Default', 'Day %', 'WTD %', 'MTD %', 'YTD %', 'HV', 'DD',
                            'Sharpe Day', 'Sharpe WTD', 'Sharpe MTD', 'Sharpe YTD']
            sort_by = st.selectbox("Sort", sort_options, index=0,
                key='scanner_sort', label_visibility='collapsed')

        with col_ct:
            st.markdown(f"<div style='{_lbl}'>CHART</div>", unsafe_allow_html=True)
            chart_options = ['Line', 'Bars']
            ct_idx = 0 if st.session_state.chart_type == 'line' else 1
            ct = st.selectbox("Chart", chart_options, index=ct_idx,
                key='chart_select', label_visibility='collapsed')
            st.session_state.chart_type = 'line' if ct == 'Line' else 'bars'

        # Fetch and render scanner
        with st.spinner('Loading market data...'):
            metrics = fetch_sector_data(st.session_state.sector)

        if metrics and sort_by != 'Default':
            sort_map = {
                'Day %': 'change_day', 'WTD %': 'change_wtd', 'MTD %': 'change_mtd', 'YTD %': 'change_ytd',
                'HV': 'hist_vol', 'DD': 'current_dd',
                'Sharpe Day': 'day_sharpe', 'Sharpe WTD': 'wtd_sharpe', 'Sharpe MTD': 'mtd_sharpe', 'Sharpe YTD': 'ytd_sharpe',
            }
            attr = sort_map.get(sort_by)
            if attr:
                reverse = sort_by not in ('HV', 'DD')
                metrics = sorted(metrics, key=lambda m: getattr(m, attr, 0) if not pd.isna(getattr(m, attr, None)) else -999,
                               reverse=reverse)

        if metrics:
            col_scan, col_bars = st.columns([55, 45])
            with col_scan:
                render_scanner_table(metrics, st.session_state.symbol)
            with col_bars:
                render_return_bars(metrics, sort_by)

    # Spacer between scanner and charts
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # Chart header markup (reused below)
    _sym = st.session_state.symbol
    _ds = clean_symbol(_sym); _fn = SYMBOL_NAMES.get(_sym, _sym)
    _hdr_bg = t.get('bg3', '#1a2744'); _bdr = t.get('border', '#1e293b')
    _hdr_mut = t.get('muted', '#475569')
    _chart_hdr = (
        f"<div style='padding:8px 12px;background:{_hdr_bg};"
        f"border-left:2px solid {pos_c};font-family:{FONTS};border-radius:4px 4px 0 0'>"
        f"<span style='color:#f8fafc;font-size:11px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase'>{_ds} CHARTS</span>"
        f"<span style='color:{_hdr_mut};font-size:10px;margin-left:6px;font-weight:400'>{_fn}</span></div>")

    # Charts + Levels + News
    if is_mobile:
        st.markdown(_chart_hdr, unsafe_allow_html=True)
        with st.spinner('Loading charts...'):
            try:
                fig, levels = create_4_chart_grid(st.session_state.symbol, st.session_state.chart_type, mobile=True)
                st.plotly_chart(fig, width='stretch', config={'scrollZoom': True, 'displayModeBar': False, 'responsive': True})
            except Exception as e:
                st.error(f"Chart error: {str(e)}"); levels = {}
        render_key_levels(st.session_state.symbol, levels)
        render_news_panel(st.session_state.symbol)
    else:
        # ── Fetch chart data once ──
        with st.spinner('Loading charts...'):
            try:
                fig, levels = create_4_chart_grid(st.session_state.symbol, st.session_state.chart_type, mobile=False)
                chart_ok = True
            except Exception as e:
                st.error(f"Chart error: {str(e)}")
                fig, levels, chart_ok = None, {}, False

        # ── Levels + News side by side (equal columns, same height) ──
        col_lvl, col_news = st.columns([1, 1])
        _panel_h = 44 + 52 + 9 * 26   # 330px — levels table height
        with col_lvl:
            render_key_levels(st.session_state.symbol, levels, target_height=_panel_h)
        with col_news:
            render_news_panel(st.session_state.symbol, target_height=_panel_h)

        # ── 2x2 Chart grid below, full width ──
        st.markdown(_chart_hdr, unsafe_allow_html=True)
        if chart_ok:
            st.plotly_chart(fig, width='stretch', config={
                'scrollZoom': True, 'displayModeBar': False,
                'responsive': True})

    # Auto-refresh handled globally in app.py


# =============================================================================
# SCANNER CHART TAB — one timeframe, all assets in sector, 2-col grid
# =============================================================================

# Maps user-facing label → (interval, boundary_type, period_label)
SCANNER_TF_OPTIONS = {
    'Intraday (15m)':   ('15m',  'session', 'Day High/Low'),
    'Short-Term (4H)':  ('1h',   'week',    'Week High/Low'),
    'Medium (Daily)':   ('1d',   'month',   'Month High/Low'),
    'Long-Term (Wkly)': ('1wk',  'year',    'Year High/Low'),
}


def create_single_asset_chart(symbol, chart_type, interval, boundary_type, mobile=False):
    """One-panel chart for a single symbol at a fixed timeframe.
    Returns (fig, zone_status_str) — stripped-down version of create_4_chart_grid."""
    zc = zone_colors(); t = get_theme()
    display_symbol = clean_symbol(symbol)

    live_price = None
    try:
        hist_lag = fetch_chart_data(symbol, '1d', '5m')
        if not hist_lag.empty:
            live_price = float(hist_lag['Close'].iloc[-1])
    except Exception:
        pass

    fig = make_subplots(rows=1, cols=1)

    period = get_dynamic_period(boundary_type)
    hist = fetch_chart_data(symbol, period, interval)
    if hist.empty:
        return fig, ''

    boundaries = PeriodBoundaryCalculator.get_boundaries(hist, boundary_type, symbol)
    current_price = hist['Close'].iloc[-1]
    if live_price is not None:
        current_price = live_price

    x_vals = list(range(len(hist)))

    if boundary_type == 'session':
        tick_indices = [i for i, dt in enumerate(hist.index) if dt.minute == 0 and dt.hour % 4 == 0]
        tick_labels = [hist.index[i].strftime('%d %b') if hist.index[i].hour == 0 else hist.index[i].strftime('%H:%M') for i in tick_indices]
    elif boundary_type == 'week':
        n = 8; tick_indices = list(range(0, len(hist), max(1, len(hist)//n)))
        tick_labels = [hist.index[i].strftime('%a %d') for i in tick_indices]
    elif boundary_type == 'month':
        n = 8; tick_indices = list(range(0, len(hist), max(1, len(hist)//n)))
        tick_labels = [hist.index[i].strftime('%d %b') for i in tick_indices]
    else:
        n = 8; tick_indices = list(range(0, len(hist), max(1, len(hist)//n)))
        tick_labels = [hist.index[i].strftime("%b '%y") for i in tick_indices]

    line_color = '#6b7280'
    zone_status = ''

    def get_zone(price, high, low, mid):
        if price > high: return 'above_high'
        elif price < low: return 'below_low'
        elif price > mid: return 'above_mid'
        else: return 'below_mid'

    def plot_line(x_data, closes, datetimes=None, color='rgba(255,255,255,0.85)', width=1.8, zone_color=False, mid=None):
        all_x = x_data; all_y = list(closes)
        all_dt = [dt.strftime('%d %b %H:%M') if boundary_type == 'session' else dt.strftime('%d %b %Y') for dt in datetimes] if datetimes is not None else None
        hover = '%{customdata}<br>%{y:.2f}<extra></extra>' if all_dt else '%{y:.2f}<extra></extra>'
        if not zone_color or mid is None:
            fig.add_trace(go.Scatter(x=all_x, y=all_y, mode='lines',
                line=dict(color=color, width=width, shape='linear'),
                showlegend=False, customdata=all_dt, hovertemplate=hover))
            return
        zones = ['up' if c >= mid else 'dn' for c in closes]
        i = 0
        while i < len(zones):
            zone = zones[i]; start_i = i
            while i < len(zones) and zones[i] == zone: i += 1
            end_i = min(i + 1, len(all_x))
            seg_x = all_x[start_i:end_i]; seg_y = all_y[start_i:end_i]
            seg_dt = all_dt[start_i:end_i] if all_dt else None
            seg_color = t['pos'] if zone == 'up' else t['neg']
            fig.add_trace(go.Scatter(x=seg_x, y=seg_y, mode='lines',
                line=dict(color=seg_color, width=width, shape='linear'),
                showlegend=False, customdata=seg_dt, hovertemplate=hover))

    if boundaries:
        last_b = boundaries[-1]
        mid = (last_b.prev_high + last_b.prev_low) / 2
        zone_status = get_zone(current_price, last_b.prev_high, last_b.prev_low, mid)
        line_color = t['pos'] if zone_status in ('above_high', 'above_mid') else t['neg']
        boundary_idx = last_b.idx

        if chart_type == 'bars':
            fig.add_trace(go.Candlestick(
                x=x_vals, open=hist['Open'].values, high=hist['High'].values,
                low=hist['Low'].values, close=hist['Close'].values,
                increasing_line_color=t['pos'], decreasing_line_color=t['neg'],
                increasing_fillcolor=t['pos'], decreasing_fillcolor=t['neg'],
                showlegend=False, line=dict(width=1)))

        if len(boundaries) >= 2 and chart_type == 'line':
            prev_b = boundaries[-2]; prev_mid = (prev_b.prev_high + prev_b.prev_low) / 2
            ps, pe = prev_b.idx, boundary_idx
            if pe > ps:
                plot_line(x_vals[ps:pe], hist['Close'].values[ps:pe], hist.index[ps:pe], width=1.5, zone_color=True, mid=prev_mid)

        if len(boundaries) >= 2 and chart_type == 'line':
            first_tracked = boundaries[-2].idx
            if first_tracked > 0:
                plot_line(x_vals[:first_tracked], hist['Close'].values[:first_tracked], hist.index[:first_tracked], color='rgba(255,255,255,0.25)', width=1.0)

        if boundary_idx < len(hist) and chart_type == 'line':
            plot_line(x_vals[boundary_idx:], hist['Close'].values[boundary_idx:], hist.index[boundary_idx:], width=2.0, zone_color=True, mid=mid)

    elif not boundaries:
        if chart_type == 'bars':
            fig.add_trace(go.Candlestick(
                x=x_vals, open=hist['Open'].values, high=hist['High'].values,
                low=hist['Low'].values, close=hist['Close'].values,
                increasing_line_color=t['pos'], decreasing_line_color=t['neg'],
                increasing_fillcolor=t['pos'], decreasing_fillcolor=t['neg'],
                showlegend=False, line=dict(width=1)))
        else:
            plot_line(x_vals, hist['Close'].values, hist.index, color='rgba(255,255,255,0.5)', width=1.5)

    # Boundary lines + retrace
    if boundaries:
        num_boundaries = min(2, len(boundaries))
        for j in range(num_boundaries):
            b = boundaries[-(j+1)]; px = b.idx; ex = len(hist) - 1 if j == 0 else boundaries[-1].idx
            fig.add_vline(x=px, line=dict(color='rgba(255,255,255,0.2)', width=0.8, dash='dot'))
            ml = (b.prev_high + b.prev_low) / 2
            fig.add_trace(go.Scatter(x=[px, ex], y=[b.prev_high]*2, mode='lines', line=dict(color=zc['above_high'], width=0.9), showlegend=False, hovertemplate=f'High: {b.prev_high:.2f}<extra></extra>'))
            fig.add_trace(go.Scatter(x=[px, ex], y=[b.prev_low]*2,  mode='lines', line=dict(color=zc['below_low'],  width=0.9), showlegend=False, hovertemplate=f'Low: {b.prev_low:.2f}<extra></extra>'))
            fig.add_trace(go.Scatter(x=[px, ex], y=[ml]*2,          mode='lines', line=dict(color='#94a3b8', width=0.6, dash='dot'), showlegend=False, hovertemplate=f'50%: {ml:.2f}<extra></extra>'))

        # Rolling retrace lines
        last_b = boundaries[-1]
        current_period = hist.iloc[last_b.idx:]
        if len(current_period) > 1:
            rolling_high = current_period['High'].expanding().max()
            rolling_low  = current_period['Low'].expanding().min()
            rx_vals = list(range(last_b.idx, last_b.idx + len(current_period)))
            buy_y  = ((rolling_high + last_b.prev_low)  / 2).values
            sell_y = ((rolling_low  + last_b.prev_high) / 2).values
            fig.add_trace(go.Scatter(x=rx_vals, y=buy_y,  mode='lines', line=dict(color='#22c55e', width=1.0, dash='dot'), showlegend=False, hovertemplate='RB: %{y:.2f}<extra></extra>'))
            fig.add_trace(go.Scatter(x=rx_vals, y=sell_y, mode='lines', line=dict(color='#ef4444', width=1.0, dash='dot'), showlegend=False, hovertemplate='RS: %{y:.2f}<extra></extra>'))

    # Axis ticks
    if tick_indices:
        fig.update_layout(xaxis=dict(tickmode='array', tickvals=tick_indices, ticktext=tick_labels, tickfont=dict(color='#e2e8f0', size=8)))

    # X range
    last_bar = len(hist) - 1
    if boundary_type == 'session':
        bars_24h = min(96, last_bar)
        x_left = max(0, last_bar - bars_24h) - 2
    else:
        x_left = -2
    x_right = x_left + int((last_bar - x_left) / 0.6)
    fig.update_layout(xaxis=dict(range=[x_left, x_right]))

    # Y range
    visible_hist = hist.iloc[max(0, x_left):]
    y_lows = visible_hist['Low'].dropna(); y_highs = visible_hist['High'].dropna()
    if len(y_lows) > 10:
        y_min = y_lows.quantile(0.005); y_max = y_highs.quantile(0.995)
    else:
        y_min = y_lows.min(); y_max = y_highs.max()
    pad = (y_max - y_min) * 0.08
    fig.update_layout(yaxis=dict(range=[y_min - pad, y_max + pad], side='right', tickfont=dict(size=8, color='#94a3b8')))

    pd_dec = 4 if '=X' in symbol else 2

    # Price label
    fig.add_annotation(x=1.02, y=current_price, xref='x domain', yref='y',
        text=f'<b>{current_price:.{pd_dec}f} C</b>', showarrow=False,
        font=dict(color='#ffffff', size=10), bgcolor='rgba(0,0,0,0.5)',
        bordercolor=line_color, borderwidth=1, borderpad=2, xanchor='left')

    # H/M/L labels
    if boundaries:
        _mid = (last_b.prev_high + last_b.prev_low) / 2
        for _lvl, _col, _lbl in [
            (last_b.prev_high, zc['above_high'], f'{last_b.prev_high:.{pd_dec}f} H'),
            (_mid,             '#94a3b8',         f'{_mid:.{pd_dec}f} M'),
            (last_b.prev_low,  zc['below_low'],   f'{last_b.prev_low:.{pd_dec}f} L'),
        ]:
            fig.add_annotation(x=1.02, y=_lvl, xref='x domain', yref='y',
                text=f'<b>{_lbl}</b>', showarrow=False,
                font=dict(color=_col, size=8), bgcolor='rgba(0,0,0,0.35)',
                bordercolor=_col, borderwidth=1, borderpad=2, xanchor='left')

    # Chart title — symbol + zone status + RSI
    rsi_val = calculate_rsi(hist['Close'])
    stc = {'above_high': zc['above_high'], 'above_mid': zc['above_mid'],
           'below_mid': zc['below_mid'], 'below_low': zc['below_low']}
    title_parts = [f"<b>{display_symbol}</b>"]
    if not np.isnan(rsi_val):
        rc = zc['above_mid'] if rsi_val > 50 else zc['below_low']
        title_parts.append(f"<span style='color:{rc};font-size:9px'>RSI {rsi_val:.0f}</span>")
    if zone_status:
        sc = stc.get(zone_status, '#64748b')
        title_parts.append(f"<span style='color:{sc};font-size:9px'>{STATUS_LABELS[zone_status]}</span>")
    fig.update_layout(title=dict(text='  '.join(title_parts), font=dict(color='#f8fafc', size=10), x=0.02, xanchor='left', y=0.97, yanchor='top'))

    _t = get_theme()
    _pbg = _t.get('plot_bg', '#121212')
    _tpl = 'plotly_white' if _t.get('mode') == 'light' else 'plotly_dark'
    fig.update_layout(
        template=_tpl, height=260, margin=dict(l=10, r=80, t=36, b=20),
        showlegend=False, plot_bgcolor=_pbg, paper_bgcolor=_pbg,
        dragmode='pan', hovermode='closest', autosize=True)
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.03)', linecolor='rgba(0,0,0,0)', tickfont=dict(color='#e2e8f0', size=8),
        showgrid=True, tickangle=0, rangeslider=dict(visible=False), fixedrange=False,
        showspikes=True, spikecolor='#475569', spikethickness=0.5, spikedash='dot', spikemode='across')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.03)', linecolor='rgba(0,0,0,0)', showgrid=True, side='right',
        tickfont=dict(color='#94a3b8', size=8), fixedrange=False,
        showspikes=True, spikecolor='#475569', spikethickness=0.5, spikedash='dot', spikemode='across')

    return fig, zone_status


def _get_levels_for_symbol(symbol, interval, boundary_type):
    """Fetch H/RB/M/RS/L/Status/Price levels for one symbol at one timeframe.
    Returns a dict with keys: price, high, rb, mid, rs, low, status."""
    try:
        period = get_dynamic_period(boundary_type)
        hist = fetch_chart_data(symbol, period, interval)
        if hist.empty:
            return {}

        # Live price
        live_price = None
        try:
            hist_lag = fetch_chart_data(symbol, '1d', '5m')
            if not hist_lag.empty:
                live_price = float(hist_lag['Close'].iloc[-1])
        except Exception:
            pass

        boundaries = PeriodBoundaryCalculator.get_boundaries(hist, boundary_type, symbol)
        if not boundaries:
            return {}

        current_price = live_price if live_price is not None else float(hist['Close'].iloc[-1])
        last_b = boundaries[-1]
        mid = (last_b.prev_high + last_b.prev_low) / 2

        # Zone status
        if current_price > last_b.prev_high:   status = 'above_high'
        elif current_price < last_b.prev_low:  status = 'below_low'
        elif current_price > mid:               status = 'above_mid'
        else:                                   status = 'below_mid'

        # Rolling retrace levels
        rb = rs = None
        current_period = hist.iloc[last_b.idx:]
        if len(current_period) > 1:
            rh = current_period['High'].expanding().max().iloc[-1]
            rl = current_period['Low'].expanding().min().iloc[-1]
            rb = (rh + last_b.prev_low)  / 2
            rs = (rl + last_b.prev_high) / 2

        rsi_value = calculate_rsi(hist['Close'])

        return {
            'price':  current_price,
            'high':   last_b.prev_high,
            'rb':     rb,
            'mid':    mid,
            'rs':     rs,
            'low':    last_b.prev_low,
            'status': status,
            'rsi':    rsi_value,
        }
    except Exception:
        return {}


def render_scanner_levels_table(symbols, interval, boundary_type, selected_sector, tf_choice, zone_desc):
    """Wide levels table: rows = level labels, columns = each symbol in sector."""
    zc  = zone_colors()
    t   = get_theme()
    pos_c = t['pos']
    _hdr_bg   = t.get('bg3',    '#1a2744')
    _body_bg  = t.get('bg',     '#1e1e1e')
    _bdr_ln   = t.get('border', '#2a2a2a')
    _mut      = t.get('muted',  '#6d6d6d')
    _row_alt  = t.get('bg3',    '#131b2e')

    # Fetch levels for every symbol
    all_levels = {}
    for sym in symbols:
        all_levels[sym] = _get_levels_for_symbol(sym, interval, boundary_type)

    # Determine signal for each symbol (for header colouring)
    def _sig(lvl):
        s = lvl.get('status', '')
        if s == 'above_high': return zc['above_high']
        if s == 'above_mid':  return zc['above_mid']
        if s == 'below_mid':  return zc['below_mid']
        if s == 'below_low':  return zc['below_low']
        return _mut

    th = (f"padding:4px 8px;font-size:9px;font-weight:700;text-transform:uppercase;"
          f"letter-spacing:0.07em;border-bottom:1px solid {_bdr_ln};text-align:center;white-space:nowrap")
    td = (f"padding:4px 8px;font-size:9px;font-variant-numeric:tabular-nums;"
          f"text-align:center;border-bottom:1px solid {_bdr_ln}20;white-space:nowrap")

    # ── Header ──
    html = (
        f"<div style='padding:7px 12px;background:{_hdr_bg};border-left:2px solid {pos_c};"
        f"font-family:{FONTS};border-radius:4px 4px 0 0;display:flex;align-items:center;gap:10px'>"
        f"<span style='color:#f8fafc;font-size:11px;font-weight:700;letter-spacing:0.08em;"
        f"text-transform:uppercase'>{selected_sector} LEVELS</span>"
        f"<span style='color:{_mut};font-size:10px'>{tf_choice} · {zone_desc}</span>"
        f"</div>"
    )

    html += (
        f"<div style='background:{_body_bg};border:1px solid {_bdr_ln};border-top:none;"
        f"border-radius:0 0 4px 4px;overflow-x:auto'>"
        f"<table style='border-collapse:collapse;font-family:{FONTS};width:100%;line-height:1.2'>"
        f"<thead><tr>"
        f"<th style='{th};text-align:left;color:{_mut};min-width:48px'>LEVEL</th>"
    )

    # Symbol header cols — coloured by zone status
    for sym in symbols:
        ds   = clean_symbol(sym)
        lvl  = all_levels.get(sym, {})
        sc   = _sig(lvl)
        html += f"<th style='{th};color:{sc}'>{ds}</th>"
    html += "</tr></thead><tbody>"

    # Helper to fmt a price value
    def _fmt(v, dec):
        return f"{v:,.{dec}f}" if v is not None and not (isinstance(v, float) and np.isnan(v)) else '—'

    def _rsi_color(v):
        if v is None or (isinstance(v, float) and np.isnan(v)): return _mut
        v = float(v)
        if v >= 70: return '#f59e0b'       # overbought — amber
        if v >= 50: return zc['above_mid'] # bullish
        if v >= 30: return zc['below_mid'] # bearish
        return '#22c55e'                   # oversold — green

    # Row definitions: (label, key, colour, alt_bg)
    rows = [
        ('C',      'price',  '#ffffff',       True),
        ('H',      'high',   zc['above_high'],False),
        ('RB',     'rb',     '#22c55e',       True),
        ('M',      'mid',    _mut,            False),
        ('RS',     'rs',     '#ef4444',       True),
        ('L',      'low',    zc['below_low'], False),
        ('STATUS', 'status', None,            True),
    ]

    for lbl, key, colour, use_alt in rows:
        bg = f"background:{_row_alt}" if use_alt else ''
        html += f"<tr style='{bg}'>"
        html += f"<td style='{td};text-align:left;color:{colour or _mut};font-weight:700'>{lbl}</td>"
        for sym in symbols:
            lvl = all_levels.get(sym, {})
            val = lvl.get(key)
            dec = 4 if '=X' in sym else 2
            if key == 'status':
                sc   = zc.get(val, _mut) if val else _mut
                text = STATUS_LABELS.get(val, '—') if val else '—'
                html += f"<td style='{td};color:{sc};font-size:8px;font-weight:700'>{text}</td>"
            else:
                cell_c = colour if colour else _mut
                # Price cell: colour by zone
                if key == 'price' and val:
                    cell_c = _sig(lvl)
                html += f"<td style='{td};color:{cell_c}'>{_fmt(val, dec)}</td>"
        html += "</tr>"

    # RSI row
    html += f"<tr style='background:{_row_alt}'>"
    html += f"<td style='{td};text-align:left;color:{_mut};font-weight:700'>RSI</td>"
    for sym in symbols:
        lvl = all_levels.get(sym, {})
        v = lvl.get('rsi')
        if v is None or (isinstance(v, float) and np.isnan(v)):
            html += f"<td style='{td};color:{_mut}'>—</td>"
        else:
            rc = _rsi_color(v)
            html += f"<td style='{td};color:{rc};font-weight:700'>{float(v):.0f}</td>"
    html += "</tr>"

    # DIST row — % to nearest level
    html += "<tr>"
    html += f"<td style='{td};text-align:left;color:{_mut};font-weight:700'>DIST</td>"
    for sym in symbols:
        lv = all_levels.get(sym, {})
        cp = lv.get('price')
        candidates = {k: lv.get(k) for k in ('high','rb','mid','rs','low') if lv.get(k)}
        if cp and candidates:
            nk = min(candidates, key=lambda k: abs(candidates[k] - cp))
            nv = candidates[nk]
            pct = (cp - nv) / nv * 100
            dc = zc['above_mid'] if pct >= 0 else zc['below_mid']
            sign = '+' if pct >= 0 else ''
            html += f"<td style='{td};color:{dc};font-size:8px'>{sign}{pct:.1f}%<span style='color:{_mut};font-size:7px'>{nk.upper()}</span></td>"
        else:
            html += f"<td style='{td};color:{_mut}'>—</td>"
    html += "</tr>"

    html += "</tbody></table></div>"
    st.markdown(html, unsafe_allow_html=True)


def render_scanner_charts_tab(is_mobile, est):
    """Scanner Charts tab — lock one timeframe, levels table + all assets in 2-col chart grid."""
    t = get_theme()
    pos_c = t['pos']
    _lbl = f"color:#f8fafc;font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;font-family:{FONTS}"

    # ── Controls: SECTOR | TIMEFRAME | CHART TYPE ──
    def _on_sc_sector_change():
        new_sector = st.session_state.sc_sector
        st.session_state.sector = new_sector
        st.session_state.symbol = FUTURES_GROUPS[new_sector][0]

    sector_names = list(FUTURES_GROUPS.keys())
    if st.session_state.get('sc_sector') != st.session_state.sector:
        st.session_state.sc_sector = st.session_state.sector

    col_sec, col_tf, col_ct = st.columns([3, 3, 2])

    with col_sec:
        st.markdown(f"<div style='{_lbl}'>SECTOR</div>", unsafe_allow_html=True)
        st.selectbox("Sector", sector_names, key='sc_sector',
            label_visibility='collapsed', on_change=_on_sc_sector_change)
        selected_sector = st.session_state.sc_sector

    with col_tf:
        st.markdown(f"<div style='{_lbl}'>TIMEFRAME</div>", unsafe_allow_html=True)
        tf_options = list(SCANNER_TF_OPTIONS.keys())
        tf_choice = st.selectbox("Timeframe", tf_options, key='sc_timeframe',
            label_visibility='collapsed')
        interval, boundary_type, zone_desc = SCANNER_TF_OPTIONS[tf_choice]

    with col_ct:
        st.markdown(f"<div style='{_lbl}'>CHART</div>", unsafe_allow_html=True)
        ct = st.selectbox("Chart", ['Line', 'Bars'],
            index=0 if st.session_state.chart_type == 'line' else 1,
            key='sc_chart_type', label_visibility='collapsed')
        chart_type = 'line' if ct == 'Line' else 'bars'

    symbols = FUTURES_GROUPS[selected_sector]
    n = len(symbols)

    # ── Wide levels table (full width, single panel) ──
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    with st.spinner('Loading levels…'):
        render_scanner_levels_table(symbols, interval, boundary_type, selected_sector, tf_choice, zone_desc)

    # ── Charts header ──
    _hdr_bg  = t.get('bg3', '#1a2744')
    _hdr_mut = t.get('muted', '#475569')
    st.markdown(
        f"<div style='padding:8px 12px;background:{_hdr_bg};border-left:2px solid {pos_c};"
        f"font-family:{FONTS};border-radius:4px 4px 0 0;margin-top:14px'>"
        f"<span style='color:#f8fafc;font-size:11px;font-weight:700;letter-spacing:0.08em;"
        f"text-transform:uppercase'>{selected_sector} CHARTS</span>"
        f"<span style='color:{_hdr_mut};font-size:10px;margin-left:8px;font-weight:400'>"
        f"{tf_choice} · {zone_desc}</span></div>",
        unsafe_allow_html=True)
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ── 2-col chart grid ──
    with st.spinner(f'Loading {n} charts…'):
        pairs = [(symbols[i], symbols[i+1] if i+1 < n else None) for i in range(0, n, 2)]
        for sym_a, sym_b in pairs:
            cols = st.columns(2)
            for col, sym in zip(cols, [sym_a, sym_b]):
                if sym is None:
                    continue
                with col:
                    try:
                        fig, zone_status = create_single_asset_chart(
                            sym, chart_type, interval, boundary_type, mobile=is_mobile)
                        st.plotly_chart(fig, width='stretch',
                            config={'scrollZoom': True, 'displayModeBar': False, 'responsive': True})
                    except Exception as e:
                        st.error(f"{clean_symbol(sym)}: {e}")

    # Auto-refresh handled globally in app.py


