import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import combinations
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

from collections import OrderedDict

from config import (FUTURES_GROUPS, SYMBOL_NAMES, FONTS, clean_symbol,
                    basket_category, sort_val)

logger = logging.getLogger(__name__)

# =============================================================================
# SPREAD STATISTICS
# =============================================================================

def _spread_sharpe(returns, ann_factor=252):
    if returns.std() == 0 or len(returns) < 5: return 0.0
    return float((returns.mean() / returns.std()) * np.sqrt(ann_factor))

def _spread_sortino(returns, ann_factor=252):
    """Downside deviation is the RMS of min(r, 0) over EVERY bar.

    Averaging the squares over the losing bars alone divides by the count of
    losers instead of the count of bars, inflating the denominator by
    sqrt(n / n_losers) and understating Sortino by about 44% on these series.
    The zeros are part of the measure: a spread that loses rarely is supposed to
    score better for it.
    """
    if returns.std() == 0 or len(returns) < 5: return 0.0
    ds = float(np.sqrt(np.mean(np.minimum(returns, 0) ** 2)) * np.sqrt(ann_factor))
    return float(returns.mean() * ann_factor / ds) if ds else 0.0

def _spread_curve(returns):
    """Equity curve anchored at its true origin.

    (1 + r).cumprod() starts at 1 + r[0], i.e. one bar in, because r came from a
    pct_change().dropna() that already ate the first observation. That makes the
    first bar its own running maximum, so an opening loss can never register: a
    series that gapped 5% down on bar one and drifted up thereafter reported a
    maximum drawdown of 0.00%. Prepending 1.0 restores bar zero.
    """
    return np.r_[1.0, (1 + np.asarray(returns, dtype=float)).cumprod()]

def _spread_drawdowns(returns):
    cum = pd.Series(_spread_curve(returns))
    peak = cum.cummax()
    dd = (cum - peak) / peak
    mdd = float(dd.min() * 100)
    add = float(dd[dd < 0].mean() * 100) if (dd < 0).any() else 0.0
    return mdd, add

def _observed(index):
    """Mask of steps that happened during hours the window actually watched.

    An intraday window is a few hours a day stitched together, so the overnight
    gap between one session's close and the next session's open contributes its
    whole move to ER's numerator while costing a single step of the denominator.
    That is not efficiency, it is a gap being counted as a trend. Daily and
    weekly bars are left alone: a weekend is not an unobserved gap in a daily
    series, and masking there would throw away every Monday.
    """
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return None
    step = pd.Series(index[1:] - index[:-1])
    modal = step.mode()
    if modal.empty or modal[0] >= pd.Timedelta(days=1):
        return None
    keep = np.ones(len(index), bool)
    keep[1:] = (index[1:] - index[:-1]) <= modal[0] * 2
    return keep

def _spread_er(returns):
    """Kaufman efficiency ratio: |net move| / path length, signed.

    1.0 is a straight line, 0.0 is chop that goes nowhere, negative is a clean
    downtrend. Scale-free and purely descriptive, so it stays meaningful on
    short windows where Sharpe -- an estimate of a forward parameter -- does not.

    It IS bar-size dependent: a coarser bar traces a shorter path over the same
    net move, so never compare raw ER across intervals, only within one.
    """
    if len(returns) < 5: return 0.0
    d = np.diff(_spread_curve(returns))
    obs = _observed(getattr(returns, 'index', None))
    if obs is not None:
        d = d[obs]
    path = np.abs(d).sum()
    if path == 0: return 0.0
    net = d.sum()
    er = abs(net) / path
    return float(er if net >= 0 else -er)

# A drawdown of nothing is not a denominator. Floor both ratios and cap the
# result, or a spread that barely moved posts a MAR in the hundreds.
MIN_DD_PCT = 0.05
MAX_RATIO = 99.0

def _spread_roa(returns, mdd):
    """Total return over the worst hole, the way the desk sizes a trade.

    Refuses to score a spread whose worst drawdown is smaller than one typical
    bar of its own movement: it did not avoid a drawdown, it was measured too
    coarsely to have had one.
    """
    if len(returns) < 5: return 0.0
    r = np.asarray(returns, dtype=float)
    floor = max(MIN_DD_PCT, float(np.median(np.abs(r))) * 100)
    if abs(mdd) < floor: return 0.0
    total = float((np.prod(1 + r) - 1) * 100)
    return float(np.clip(total / abs(mdd), -MAX_RATIO, MAX_RATIO))

def _spread_r2(returns):
    if len(returns) < 5: return 0.0
    cum = (1 + returns).cumprod().values
    x = np.arange(len(cum))
    xm, ym = x.mean(), cum.mean()
    ss_xy = np.sum(x * cum) - len(cum) * xm * ym
    ss_xx = np.sum(x * x) - len(cum) * xm * xm
    ss_yy = np.sum(cum * cum) - len(cum) * ym * ym
    slope = ss_xy / ss_xx if ss_xx else 0
    r2 = (ss_xy ** 2) / (ss_xx * ss_yy) if (ss_xx * ss_yy) else 0
    r2 = float(np.clip(r2, 0, 1))
    return r2 if slope > 0 else -r2


def annualization_factor(index, fallback=252.0):
    """Bars per year, measured off the index instead of assumed.

    A flat 252 only describes daily equity bars. Crypto prints 365 days a year,
    and one intraday session is 26 15m bars for US equities but ~77 for futures
    and ~95 for FX, so assuming the equity session understated Sharpe, Sortino
    and MAR on those groups by up to 1.9x.
    """
    n = len(index)
    if n < 3:
        return float(fallback)
    span_days = (index[-1] - index[0]).total_seconds() / 86400.0
    if span_days <= 0:
        return float(fallback)
    # Clamped: a window of a few bars over a few minutes would otherwise
    # annualise into the millions.
    return float(np.clip(n / (span_days / 365.25), 12, 8760))

# =============================================================================
# ALIGNMENT
# =============================================================================

MIN_BARS, MIN_SYMBOLS = 20, 4
SESSION_SHARE = 0.5     # of symbols that must really print for a date to count


def align_frames(frames, intraday, min_bars=MIN_BARS, min_symbols=MIN_SYMBOLS):
    """Align {symbol: close Series} by shedding COLUMNS, not rows.

    Dropping rows -- ffill then dropna -- lets one sparsely listed symbol govern
    the whole matrix: ffill cannot backfill a late listing, so dropna deletes
    every row before it, and on intraday the intersection collapses to whichever
    market trades the fewest hours. One 2024 IPO in a basket of thirty can cut a
    two-year window to a few months.

    Daily is the exception, and only for dates nothing traded on. Crypto prints
    every day of the year, so a Saturday enters the union index with two real
    prices and the rest forward-filled -- a bar on which most of the board could
    not have been traded, contributing a zero return for every closed market and
    a live one for BTC. Those are dropped before the ffill, so a genuine single
    market holiday still fills from its own last price.

    Returns (frame, dropped_symbols).
    """
    syms = [k for k, v in frames.items() if v is not None and len(v) > 0]
    if len(syms) < 2:
        return None, []
    union = frames[syms[0]].index
    for s in syms[1:]:
        union = union.union(frames[s].index)
    cov = {k: float(frames[k].reindex(union).notna().mean()) for k in syms}
    order = sorted(syms, key=lambda k: cov[k])   # thinnest coverage dropped first
    keep, dropped = list(syms), []
    df = None
    while True:
        df = pd.DataFrame({k: frames[k] for k in keep})
        if intraday:
            df = df.dropna()
        else:
            df = df[df.notna().mean(axis=1) >= SESSION_SHARE].ffill().dropna()
        if len(df) >= min_bars or len(keep) <= min_symbols:
            break
        victim = next((k for k in order if k in keep), None)
        if victim is None:
            break
        keep.remove(victim)
        dropped.append(victim)
    if df is None or len(df) < 5 or len(df.columns) < 2:
        return None, dropped
    return df, dropped

# =============================================================================
# DATA FETCHING
# =============================================================================

LOOKBACK_OPTIONS = {
    'YTD': 0,
    '30 Days': 30,
    '60 Days': 60,
    '120 Days': 120,
    '240 Days': 240,
    '520 Days': 520,
}

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_sector_spread_data(sector, lookback_days=0):
    symbols = FUTURES_GROUPS.get(sector, [])
    if not symbols: return None
    if lookback_days == 0:
        start = datetime.now().replace(month=1, day=1).strftime('%Y-%m-%d')
    else:
        start = (datetime.now() - pd.Timedelta(days=int(lookback_days * 1.5))).strftime('%Y-%m-%d')
    data = pd.DataFrame()
    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            # auto_adjust pinned: yfinance has flipped this default between
            # versions, and on a fund like MSTY -- 40 distributions totalling
            # $12.56 on a $16 share this year -- unadjusted prices turn a
            # Sharpe of 0.81 into 5.09. A short seller pays those payouts.
            hist = ticker.history(start=start, auto_adjust=True)
            if not hist.empty:
                closes = hist['Close'].copy()
                closes.index = closes.index.tz_localize(None) if closes.index.tz else closes.index
                closes.index = closes.index.normalize()
                closes = closes.groupby(closes.index).last()
                data[sym] = closes
        except Exception as e:
            logger.debug(f"[{sym}] spread data fetch error: {e}")
    if data.empty or len(data.columns) < 2: return None
    data = data.ffill().dropna()
    if lookback_days > 0 and len(data) > lookback_days:
        data = data.iloc[-lookback_days:]
    if len(data) < 2: return None
    data = 100 * (data / data.iloc[0])
    return data

# =============================================================================
# SPREAD COMPUTATION
# =============================================================================

# What Composite averages the ranks of. Sharpe is the risk-adjusted edge,
# Sortino the same edge counting only the downside, ROA the return against the
# worst hole, ER whether the curve got there in a straight line. MAR and R2 are
# shown but deliberately not ranked on: MAR pins to its cap on short windows and
# contributes nothing but arbitrary tie-breaking, and R2 measures almost the
# same thing ER does, so including both double-counts straightness.
COMPOSITE_METRICS = ['Sharpe', 'Sortino', 'ROA', 'ER']


# Where a metric sits when it carries no information. Shrinking pulls an
# under-sampled estimate toward THIS point, not toward zero: 55% of bars up over
# three months is evidence of very little, and what "very little" looks like for
# a win rate is 50%, not 0%. Everything else here -- Sharpe, Sortino, ROA, ER,
# MAR, R2, a total return -- is already centred on zero. The lowercase names are
# PORTFOLIO's spelling of the same two metrics.
NEUTRAL = {'Win%': 50.0, 'win_rate': 0.5}

# Scores that are ALREADY a rank over shrunk values. Shrinking one of these a
# second time would discount the short windows twice over.
PRE_ADJUSTED = {'_score'}


def _confidence(items):
    """How much of the longest window in the set each item actually saw, as
    sqrt(span / longest span).

    Three months of history does not earn the same credit as three years at the
    same Sharpe: the standard error of every one of these estimates falls with
    the square root of the sample, so a short series is simply a noisier draw.

    Span is measured in calendar days, not bars. Bars would penalise a crypto
    group for printing 365 of them a year against an equity group's 252 over the
    SAME window -- a denser sample, not a longer one. Days falls back to Bars
    only if a caller never recorded it.

    Within one basket every pair shares an index, so the factor is 1 across the
    board and nothing moves. It bites where the windows genuinely differ -- the
    scan ranking a group of 2024 listings against one with ten years of history.
    """
    span = [float(p.get('Days') or p.get('Bars') or 0) for p in items]
    longest = max(span) if span else 0.0
    if longest <= 0:
        return [1.0] * len(items)
    return [min(1.0, float(np.sqrt(sp / longest))) for sp in span]


def length_adjusted(items, key, neutral=None):
    """Every item's KEY shrunk toward its neutral point by _confidence.

    This is what makes two rows measured over different windows comparable at
    all, and it applies to every metric the tables rank on, not just the four
    inside Composite: an eighteen-month basket posting the best Sharpe, MAR or
    win rate in the set is usually the smallest sample in the set.

    Values are NOT written back -- the table still shows the real Sharpe, and
    only the order changes.
    """
    if key in PRE_ADJUSTED:
        return [sort_val(p.get(key)) for p in items]
    if neutral is None:
        neutral = NEUTRAL.get(key, 0.0)
    return [neutral + (sort_val(p.get(key)) - neutral) * c
            for p, c in zip(items, _confidence(items))]


def rank_by_length_adjusted(items, key, reverse=True):
    """ITEMS ordered on KEY after the sample-size shrink."""
    adj = length_adjusted(items, key)
    order = sorted(range(len(items)), key=lambda i: adj[i], reverse=reverse)
    return [items[i] for i in order]


def composite_ranks(items, metrics=COMPOSITE_METRICS, score_key='_score'):
    """Average rank across METRICS, each value shrunk for sample size first.

    Mutates and returns ITEMS; lower score is better, 1.0 being the best an item
    can score. See _confidence for what the shrink is and why.
    """
    n = len(items)
    if n == 0:
        return items
    if n == 1:
        items[0][score_key] = 1.0
        return items
    for metric in metrics:
        vals = length_adjusted(items, metric)
        order = sorted(range(n), key=lambda i: -vals[i])
        for rank, idx in enumerate(order):
            items[idx][f'_{metric}_rank'] = rank + 1
    for p in items:
        p[score_key] = float(np.mean([p[f'_{m}_rank'] for m in metrics]))
    return items


def compute_sector_spreads(data, ann_factor=252):
    # The same metrics exist a second time, vectorised, as
    # spreads_portfolio._window_stats -- it scores thousands of columns per
    # rebalance, which this per-pair loop is far too slow for. They are checked
    # to agree to three decimals on Sharpe, ROA and ER; change one and change
    # the other.
    if data is None or len(data.columns) < 2: return []

    asset_sharpes = {}
    for sym in data.columns:
        ret = data[sym].pct_change().dropna()
        asset_sharpes[sym] = _spread_sharpe(ret, ann_factor)
    best_long_sym = max(asset_sharpes, key=asset_sharpes.get)
    best_long_sharpe = asset_sharpes[best_long_sym]

    # Calendar span of the window, the sample-size measure composite_ranks
    # discounts on. Constant across a basket; it separates one group from another.
    try:
        span_days = max(int((data.index[-1] - data.index[0]).days), 1)
    except Exception:
        span_days = len(data)

    pairs = []
    for s1, s2 in combinations(data.columns.tolist(), 2):
        r1 = data[s1].pct_change().dropna()
        r2 = data[s2].pct_change().dropna()
        spread_ret = (r1 - r2).dropna()

        sh = _spread_sharpe(spread_ret, ann_factor)

        # A spread with negative Sharpe is the same trade the other way round, so
        # swap the legs. Sharpe can simply be negated -- std(-r) == std(r) -- but
        # Sortino cannot: its denominator looks only at losing bars, and flipping
        # turns the old winners into the losers, so it has to be recomputed on the
        # flipped series. Negating it was understating 72% of flipped pairs.
        if sh < 0:
            spread_ret = -spread_ret
            sh = -sh
            s1, s2 = s2, s1
        so = _spread_sortino(spread_ret, ann_factor)

        mdd, add = _spread_drawdowns(spread_ret)
        cum_spread = (1 + spread_ret).cumprod()
        total = float((cum_spread.iloc[-1] - 1) * 100)
        ann = float(spread_ret.mean() * ann_factor * 100)
        vol = float(spread_ret.std() * np.sqrt(ann_factor) * 100)
        mar = float(np.clip(ann / max(abs(add), MIN_DD_PCT), -MAX_RATIO, MAX_RATIO))
        roa = _spread_roa(spread_ret, mdd)
        er = _spread_er(spread_ret)
        r2_val = _spread_r2(spread_ret)
        corr = float(r1.corr(r2))
        win_rate = float((spread_ret > 0).sum() / len(spread_ret) * 100) if len(spread_ret) > 0 else 50.0

        cum1 = data[s1]
        cum2 = data[s2]
        cum_sp = pd.Series(100.0, index=data.index[:1])
        cum_sp = pd.concat([cum_sp, 100 * (1 + spread_ret).cumprod()])
        cum_sp = cum_sp[~cum_sp.index.duplicated(keep='last')]

        pairs.append({
            'long': s1, 'short': s2,
            'Sharpe': sh, 'Sortino': so, 'MAR': mar, 'ROA': roa, 'ER': er, 'R²': r2_val,
            'Tot%': total, 'Ann%': ann, 'Vol%': vol, 'MDD%': mdd, 'ADD%': add,
            'Corr': corr, 'Win%': win_rate,
            'Bars': len(spread_ret), 'Days': span_days,
            'beats_long': sh > best_long_sharpe,
            'cum_long': cum1, 'cum_short': cum2, 'cum_spread': cum_sp,
        })

    if not pairs: return []
    composite_ranks(pairs)
    pairs.sort(key=lambda x: -x['Sharpe'])

    for p in pairs:
        p['best_long_sym'] = best_long_sym
        p['best_long_sharpe'] = best_long_sharpe

    return pairs

def compute_basket_singles(data, ann_factor=252, direction='Long only'):
    """Score every symbol on its own, held long or held short.

    Same metric set and the same dict shape as compute_sector_spreads, so the
    table and the ranker do not care which one produced the rows. A spread can
    be flipped to face the right way; an outright cannot, so the direction is
    the user's choice and a name that fell simply scores badly when held long.

    One leg is left blank: the table prints an em dash for the side that is not
    traded.
    """
    if data is None or len(data.columns) < 1:
        return []
    short = direction == 'Short only'
    sign = -1.0 if short else 1.0

    try:
        span_days = max(int((data.index[-1] - data.index[0]).days), 1)
    except Exception:
        span_days = len(data)

    out = []
    for sym in data.columns:
        r = (data[sym].pct_change().dropna()) * sign
        if len(r) < 5:
            continue
        sh = _spread_sharpe(r, ann_factor)
        so = _spread_sortino(r, ann_factor)
        mdd, add = _spread_drawdowns(r)
        cum = (1 + r).cumprod()
        total = float((cum.iloc[-1] - 1) * 100)
        ann = float(r.mean() * ann_factor * 100)
        vol = float(r.std() * np.sqrt(ann_factor) * 100)
        mar = float(np.clip(ann / max(abs(add), MIN_DD_PCT), -MAX_RATIO, MAX_RATIO))
        curve = pd.Series(100.0, index=data.index[:1])
        curve = pd.concat([curve, 100 * (1 + r).cumprod()])
        curve = curve[~curve.index.duplicated(keep='last')]
        out.append({
            'long': '' if short else sym, 'short': sym if short else '',
            'Sharpe': sh, 'Sortino': so, 'MAR': mar,
            'ROA': _spread_roa(r, mdd), 'ER': _spread_er(r), 'R²': _spread_r2(r),
            'Tot%': total, 'Ann%': ann, 'Vol%': vol, 'MDD%': mdd, 'ADD%': add,
            'Corr': float('nan'), 'Win%': float((r > 0).sum() / len(r) * 100),
            'Bars': len(r), 'Days': span_days,
            # Nothing to beat: the row IS the leg.
            'beats_long': False, 'best_long_sym': '', 'best_long_sharpe': 0.0,
            # The price goes on the side it is actually traded, so the
            # chart colours it long or short without being told.
            'cum_long': None if short else data[sym],
            'cum_short': data[sym] if short else None,
            'cum_spread': curve,
        })
    if not out:
        return []
    composite_ranks(out)
    out.sort(key=lambda x: -x['Sharpe'])
    return out


# =============================================================================
# INTERVAL FETCH
# =============================================================================
# Kept when the Sector tab was folded in. Measured across Futures, Crypto and
# US Sectors, rankings at 15m/1h/4h correlate +0.90 to +1.00 with daily and
# pick the same top spread -- so interval is for looking at the shape of a
# move, not for deciding which spread is strongest.

# yf interval, resample target, bars per trading day, max calendar days yfinance allows.
# bars_per_day here describes a 6.5h US equity session and is only a fallback --
# the real rate is measured off the fetched index, because futures run ~77 15m
# bars a day and FX ~95.
INTERVAL_CONFIG = {
    '15m': {'yf': '15m', 'resample': None, 'bars_per_day': 26,  'max_cal_days': 59},
    '1h':  {'yf': '1h',  'resample': None, 'bars_per_day': 7,   'max_cal_days': 729},
    '4h':  {'yf': '1h',  'resample': '4h', 'bars_per_day': 2,   'max_cal_days': 729},
    '1d':  {'yf': '1d',  'resample': None, 'bars_per_day': 1,   'max_cal_days': None},
    '1wk': {'yf': '1wk', 'resample': None, 'bars_per_day': 0.2, 'max_cal_days': None},
}

# Fallback only, for when the window is too short to measure the real bar rate.
ANN_FACTORS = {
    '15m': 26 * 252,
    '1h':  7 * 252,
    '4h':  2 * 252,
    '1d':  252,
    '1wk': 52,
}

@st.cache_data(ttl=900, show_spinner=False)
def fetch_interval_data(symbols, interval_key, lookback_days):
    """Returns (normalised prices, annualisation factor), or (None, fallback)."""
    cfg = INTERVAL_CONFIG[interval_key]
    fallback_af = float(ANN_FACTORS[interval_key])
    symbols = list(symbols or ())
    if len(symbols) < 2:
        return None, fallback_af

    if lookback_days == 0:  # YTD
        start = datetime.now().replace(month=1, day=1).strftime('%Y-%m-%d')
    else:
        cal_days = int(lookback_days * 1.6)
        if cfg['max_cal_days']:
            cal_days = min(cal_days, cfg['max_cal_days'])
        start = (datetime.now() - pd.Timedelta(days=max(cal_days, 2))).strftime('%Y-%m-%d')

    frames = {}
    for sym in symbols:
        try:
            hist = yf.Ticker(sym).history(start=start, interval=cfg['yf'],
                                          auto_adjust=True)
            if hist.empty:
                continue
            closes = hist['Close'].copy()
            if closes.index.tz is not None:
                closes.index = closes.index.tz_convert('UTC').tz_localize(None)
            if cfg.get('resample'):
                closes = closes.resample(cfg['resample']).last().dropna()
            if interval_key in ('1d', '1wk'):
                closes.index = closes.index.normalize()
                closes = closes.groupby(closes.index).last()
            frames[sym] = closes
        except Exception as e:
            logger.debug(f"[{sym}] fetch error ({interval_key}): {e}")

    # Shed thin columns rather than rows: one late listing used to delete every
    # bar before it, and on intraday the intersection collapsed to whichever
    # market trades the fewest hours.
    data, _thin = align_frames(frames, intraday=interval_key in ('15m', '1h', '4h'))
    if data is None or len(data.columns) < 2:
        return None, fallback_af

    # Measure the real bar rate off the full fetch, before slicing. The config
    # constants assume a 6.5h equity session, which made 'Lookback 30 Days' mean
    # about ten days on futures and FX.
    bars_per_day = cfg['bars_per_day']
    if interval_key in ('15m', '1h', '4h') and len(data) > 1:
        sessions = max(data.index.normalize().nunique(), 1)
        bars_per_day = max(len(data) / sessions, 0.1)
    ann_factor = annualization_factor(data.index, fallback_af)

    if lookback_days > 0:
        bars = max(int(lookback_days * bars_per_day), 5)
        if len(data) > bars:
            data = data.iloc[-bars:]

    if len(data) < 5:
        return None, ann_factor
    return 100 * (data / data.iloc[0]), ann_factor


# =============================================================================
# BASKET PICKER — shared by the Scan and Portfolio sub-tabs
# =============================================================================

def basket_picker(prefix, is_mobile, theme, label='Baskets in play'):
    """Checkbox picker over every basket, one column per theme. Returns the ticked names.

    PREFIX namespaces the widget keys, so two tabs can each carry their own
    selection without stepping on one another.

    Two Streamlit traps are designed around here, both found the hard way:

    1. The boxes are seeded once and then their own keys ARE the state. Passing
       value= and key= together makes the default fight the stored value on
       every rerun.
    2. EVERY toggle renders before ANY checkbox. Writing a checkbox's key only
       takes effect while that widget does not yet exist in the run, and widgets
       are created in script order -- with a theme button inside its own column,
       the earlier columns' boxes already existed and the write was dropped: the
       count read 18/18 while every box stayed clear.
    """
    _mut = theme.get('muted', '#475569')
    accent = theme.get('accent', '#4ade80')
    names = list(FUTURES_GROUPS.keys())
    k = lambda n: f'{prefix}_bk_{n}'

    if f'{prefix}_seeded' not in st.session_state:
        for n in names:
            st.session_state.setdefault(k(n), False)
        st.session_state[f'{prefix}_seeded'] = True

    by_cat = OrderedDict()
    for n in names:
        by_cat.setdefault(basket_category(n), []).append(n)
    cat_names = list(by_cat.keys())

    # Checkboxes appear nowhere else in the app, so this can be blunt. Streamlit
    # gives each one a block with the global gap on top; at 58 boxes that is
    # half a screen of air.
    st.markdown("""<style>
        [data-testid="stCheckbox"] { margin-bottom: -10px !important; }
        [data-testid="stCheckbox"] label { font-size: 11px !important; }
        [data-testid="stCheckbox"] label > div:first-child { transform: scale(0.85); }
    </style>""", unsafe_allow_html=True)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    if is_mobile:
        c_lbl, c_all, c_none = st.columns([2, 1, 1])
    else:
        c_lbl, c_all, c_none, _sp = st.columns([2, 1, 1, 6])
    with c_lbl:
        st.markdown(f"<div style='font-size:10px;font-weight:600;letter-spacing:0.08em;"
                    f"text-transform:uppercase;color:#cbd5e1;font-family:{FONTS};"
                    f"padding:11px 0 0 2px'>{label}</div>", unsafe_allow_html=True)
    with c_all:
        tick_all = st.button('Select all', key=f'{prefix}_all', use_container_width=True,
                             help=f'Tick all {len(names)} baskets.')
    with c_none:
        tick_none = st.button('Clear', key=f'{prefix}_none', use_container_width=True,
                              help='Untick everything.')
    if tick_all or tick_none:
        for n in names:
            st.session_state[k(n)] = bool(tick_all)

    per_row = 2 if is_mobile else len(cat_names)
    for start in range(0, len(cat_names), per_row):
        chunk = cat_names[start:start + per_row]
        # strict=False on purpose: the final row can hold fewer themes than
        # columns, and the spare columns are meant to stay empty.
        for col, cat in zip(st.columns(per_row), chunk, strict=False):
            members = by_cat[cat]
            full = all(st.session_state.get(k(n)) for n in members)
            with col:
                # Label is the theme alone: a count baked into it is one
                # interaction stale, because the label is emitted before the
                # click that changes it is processed.
                if st.button(cat, key=f'{prefix}_cat_{cat}', use_container_width=True,
                             help=f'{"Untick" if full else "Tick"} all '
                                  f'{len(members)} {cat} baskets.'):
                    for n in members:
                        st.session_state[k(n)] = not full
        for col, cat in zip(st.columns(per_row), chunk, strict=False):
            members = by_cat[cat]
            on_now = sum(1 for n in members if st.session_state.get(k(n)))
            with col:
                # The checkbox rule below pulls each box up by 10px, which ate
                # into this line; the margin puts the air back under it.
                st.markdown(
                    f"<div style='font-size:9px;letter-spacing:0.06em;color:"
                    f"{accent if on_now else _mut};font-family:{FONTS};"
                    f"line-height:1.6;margin:2px 0 10px 2px'>"
                    f"{on_now}/{len(members)} ticked</div>",
                    unsafe_allow_html=True)
                for name in members:
                    st.session_state.setdefault(k(name), False)
                    st.checkbox(f"{name} ({len(FUTURES_GROUPS[name])})", key=k(name))

    st.markdown(f"<div style='height:1px;background:{theme.get('border', '#1e293b')};"
                f"margin:14px 0 2px 0'></div>", unsafe_allow_html=True)
    return [n for n in names if st.session_state.get(k(n))]


# =============================================================================
# SORTING
# =============================================================================

SORT_KEYS = {
    'Composite': '_score', 'Sharpe': 'Sharpe', 'Sortino': 'Sortino',
    'ROA': 'ROA', 'ER': 'ER', 'MAR': 'MAR', 'R²': 'R²',
    'Total': 'Tot%', 'Win Rate': 'Win%'
}
SORT_OPTIONS = list(SORT_KEYS.keys())

def sort_spread_pairs(pairs, sort_key='Composite', ascending=False):
    """Rank on the length-adjusted value, whichever metric is chosen.

    Within one basket every pair shares a window and this is an ordinary sort.
    Across baskets it is not: a pair out of a basket with eighteen months of
    history is shrunk back toward the uninformative value before it is compared
    with one that has ten years.
    """
    key = SORT_KEYS.get(sort_key, sort_key)
    default_reverse = (key != '_score')
    reverse = not default_reverse if ascending else default_reverse
    return rank_by_length_adjusted(pairs, key, reverse)

# =============================================================================
# SHARED TABLE RENDERER
# =============================================================================

def render_spread_table(pairs, theme, top_n=10):
    show = pairs[:top_n]
    pos_c = theme['pos']; neg_c = theme['neg']; short_c = theme['short']
    _bg3 = theme.get('bg3', '#0f172a'); _bdr = theme.get('border', '#1e293b')
    _txt = theme.get('text', '#e2e8f0'); _txt2 = theme.get('text2', '#94a3b8'); _mut = theme.get('muted', '#475569')
    th = f"padding:4px 8px;border-bottom:1px solid {_bdr};color:#f8fafc;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.06em;"
    td = f"padding:5px 8px;border-bottom:1px solid {_bdr}22;"

    html = f"""<div style='overflow-x:auto;border:1px solid {_bdr};border-radius:6px'><table style='border-collapse:collapse;font-family:{FONTS};font-size:11px;width:100%;line-height:1.3'>
        <thead style='background:{_bg3}'><tr>
            <th style='{th}text-align:left'>RANK</th>
            <th style='{th}text-align:left'>LONG</th>
            <th style='{th}text-align:left'>SHORT</th>
            <th style='{th}text-align:right'>COMPOSITE</th>
            <th style='{th}text-align:right'>SHARPE</th>
            <th style='{th}text-align:right'>SORTINO</th>
            <th style='{th}text-align:right'>ROA</th>
            <th style='{th}text-align:right'>ER</th>
            <th style='{th}text-align:right'>MAR</th>
            <th style='{th}text-align:right'>R²</th>
            <th style='{th}text-align:right'>WIN%</th>
            <th style='{th}text-align:right'>TOT%</th>
            <th style='{th}text-align:right'>VOL%</th>
            <th style='{th}text-align:right'>MDD%</th>
            <th style='{th}text-align:right'>CORR</th>
            <th style='{th}text-align:center'>vs LONG</th>
        </tr></thead><tbody>"""

    for rank, p in enumerate(show, 1):
        ln = SYMBOL_NAMES.get(p['long'], clean_symbol(p['long']))
        sn = SYMBOL_NAMES.get(p['short'], clean_symbol(p['short']))
        sh_c = pos_c if p['Sharpe'] >= 0 else neg_c
        tot_c = pos_c if p['Tot%'] >= 0 else neg_c
        tot_s = '+' if p['Tot%'] >= 0 else ''
        win_c = pos_c if p['Win%'] >= 55 else (neg_c if p['Win%'] < 45 else _txt2)
        # ER above 0.3 is a genuinely directional curve; under 0.1 is chop.
        _er = p.get('ER', 0)
        er_c = pos_c if _er >= 0.30 else (_mut if _er < 0.10 else _txt2)
        _roa = p.get('ROA', 0)
        roa_c = pos_c if _roa >= 3 else (_mut if _roa <= 0 else _txt2)
        vs = f"<span style='color:{pos_c};font-weight:700'>▲</span>" if p['beats_long'] else f"<span style='color:{_mut}'>—</span>"
        bg = f'linear-gradient(90deg,{pos_c}08,{_bg3},{pos_c}08)' if p['beats_long'] else 'transparent'
        score = p.get('_score', 0)
        sc_c = pos_c if score <= 3 else (_txt2 if score <= 6 else _mut)
        html += f"""<tr style='background:{bg}'>
            <td style='{td}color:{_mut};text-align:left'>{rank}</td>
            <td style='{td}color:{pos_c};font-weight:600;text-align:left'>{ln}</td>
            <td style='{td}color:{short_c};font-weight:600;text-align:left'>{sn}</td>
            <td style='{td}text-align:right;color:{sc_c};font-weight:600'>{score:.1f}</td>
            <td style='{td}text-align:right'><span style='color:{sh_c};font-weight:700'>{p["Sharpe"]:.2f}</span></td>
            <td style='{td}text-align:right;color:{_txt2}'>{p["Sortino"]:.2f}</td>
            <td style='{td}text-align:right;color:{roa_c}'>{p.get("ROA", 0):.1f}</td>
            <td style='{td}text-align:right;color:{er_c}'>{p.get("ER", 0):.2f}</td>
            <td style='{td}text-align:right;color:{_txt2}'>{p["MAR"]:.2f}</td>
            <td style='{td}text-align:right;color:{_txt2}'>{p["R²"]:.3f}</td>
            <td style='{td}text-align:right'><span style='color:{win_c};font-weight:600'>{p["Win%"]:.0f}%</span></td>
            <td style='{td}text-align:right'><span style='color:{tot_c};font-weight:600'>{tot_s}{p["Tot%"]:.1f}%</span></td>
            <td style='{td}text-align:right;color:{_txt2}'>{p["Vol%"]:.1f}%</td>
            <td style='{td}text-align:right;color:{neg_c}'>{p["MDD%"]:.1f}%</td>
            <td style='{td}text-align:right;color:{_txt2}'>{p["Corr"]:.2f}</td>
            <td style='{td}text-align:center'>{vs}</td>
        </tr>"""
    html += "</tbody></table></div>"
    st.markdown(html, unsafe_allow_html=True)

# =============================================================================
# SHARED CHART RENDERER
# =============================================================================

def render_spread_charts(pairs, data, theme, mobile=False):
    top_n = min(6, len(pairs))
    if top_n == 0: return

    _pbg = theme.get('plot_bg', '#121212'); _grd = theme.get('grid', '#1f1f1f')
    _axl = theme.get('axis_line', '#2a2a2a'); _tk = theme.get('tick', '#888888')
    _mut = theme.get('muted', '#475569')

    n_cols = 1 if mobile else min(3, top_n)
    n_rows = (top_n + n_cols - 1) // n_cols

    subtitles = []
    for i in range(top_n):
        ln = SYMBOL_NAMES.get(pairs[i]['long'], clean_symbol(pairs[i]['long']))
        sn = SYMBOL_NAMES.get(pairs[i]['short'], clean_symbol(pairs[i]['short']))
        lc = theme['long']; sc = theme['short']
        subtitles.append(f"<span style='color:{lc}'>■</span> {ln}  <span style='color:{sc}'>■</span> {sn}  <span style='color:#ffffff'>■</span> Spread")
    while len(subtitles) < n_rows * n_cols: subtitles.append("")

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=subtitles,
        horizontal_spacing=0.06, vertical_spacing=0.18 if not mobile else 0.08)

    for i in range(top_n):
        p = pairs[i]; row = i // n_cols + 1; col = i % n_cols + 1
        fig.add_trace(go.Scatter(x=list(range(len(p['cum_long']))), y=p['cum_long'].values,
            mode='lines', line=dict(color=theme['long'], width=1.3, shape='spline', smoothing=1.0),
            showlegend=False, hovertemplate='Long: %{y:.1f}<extra></extra>'), row=row, col=col)
        fig.add_trace(go.Scatter(x=list(range(len(p['cum_short']))), y=p['cum_short'].values,
            mode='lines', line=dict(color=theme['short'], width=1.3, shape='spline', smoothing=1.0),
            showlegend=False, hovertemplate='Short: %{y:.1f}<extra></extra>'), row=row, col=col)
        fig.add_trace(go.Scatter(x=list(range(len(p['cum_spread']))), y=p['cum_spread'].values,
            mode='lines', line=dict(color='#ffffff', width=1.5, dash='dot', shape='spline', smoothing=1.0),
            showlegend=False, hovertemplate='Spread: %{y:.1f}<extra></extra>'), row=row, col=col)
        fig.add_hline(y=100, line=dict(color=_grd, width=0.8, dash='dot'), row=row, col=col)

        axis_idx = (row - 1) * n_cols + col
        fig.add_annotation(
            text=f"<b>{i+1}</b>", x=0.02, y=0.95,
            xref=f"x{'' if axis_idx == 1 else axis_idx} domain",
            yref=f"y{'' if axis_idx == 1 else axis_idx} domain",
            showarrow=False, font=dict(size=12, color=_mut, family=FONTS),
            xanchor='left', yanchor='top')

        n_ticks = 4; idx_step = max(1, len(data) // n_ticks)
        tick_vals = list(range(0, len(data), idx_step))
        if (len(data) - 1) not in tick_vals: tick_vals.append(len(data) - 1)
        tick_text = [data.index[j].strftime('%d %b') for j in tick_vals if j < len(data)]
        tick_vals = tick_vals[:len(tick_text)]
        axis_key = 'xaxis' if axis_idx == 1 else f'xaxis{axis_idx}'
        fig.update_layout(**{axis_key: dict(tickmode='array', tickvals=tick_vals, ticktext=tick_text)})

    for ann in fig['layout']['annotations']:
        xref_str = str(ann['xref']) if ann['xref'] else ''
        if 'domain' not in xref_str:
            ann['font'] = dict(size=10, family=FONTS)

    chart_h = 350 * n_rows if mobile else 220 * n_rows
    fig.update_layout(
        template='plotly_dark', height=chart_h,
        margin=dict(l=40, r=40, t=45, b=30),
        plot_bgcolor=_pbg, paper_bgcolor=_pbg,
        showlegend=False, hovermode='x unified', font=dict(family=FONTS))
    fig.update_xaxes(gridcolor=_grd, linecolor=_axl,
        tickfont=dict(color=_tk, size=8, family=FONTS), showgrid=False, tickangle=0)
    fig.update_yaxes(gridcolor=_grd, linecolor=_axl,
        tickfont=dict(color=_tk, size=8, family=FONTS), side='right')

    st.plotly_chart(fig, width='stretch', config={
        'scrollZoom': True, 'displayModeBar': False, 'responsive': True})

# =============================================================================
# MAIN RENDER — sub-tabs
# =============================================================================

def render_spreads_tab(is_mobile):
    from spreads_scan import render_scan_tab
    from spreads_portfolio import render_spread_portfolio_tab

    # Green underline on nested sub-tabs only (inside a tab panel)
    st.markdown(f"""<style>
        .stTabs .stTabs [data-baseweb="tab-list"] button {{
            font-family: {FONTS};
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #64748b;
            padding: 8px 20px;
        }}
        .stTabs .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
            color: #f8fafc;
            border-bottom: 2px solid #4ade80 !important;
        }}
        .stTabs .stTabs [data-baseweb="tab-highlight"] {{
            background-color: #4ade80 !important;
        }}
        /* Native Streamlit labels, restyled to the SANPO scale. Hand-rolled
           markdown labels get clipped to a sliver by the tab container, which
           left these fields effectively unlabelled. Same fix as PORTFOLIO. */
        .stSelectbox label p, .stTextInput label p {{
            font-size: 10px !important; font-weight: 600 !important; text-transform: uppercase;
            letter-spacing: 0.08em; color: #cbd5e1 !important; font-family: {FONTS} !important;
        }}
        .stSelectbox label, .stTextInput label {{ margin-bottom: 1px !important; }}
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] div[aria-selected] {{
            font-family: {FONTS} !important; font-size: 13px !important; letter-spacing: 0.01em !important;
        }}
        .stTextInput input {{ font-family: {FONTS} !important; font-size: 13px !important; letter-spacing: 0.01em !important; }}
    </style>""", unsafe_allow_html=True)

    # Two sub-tabs, not three. 'Sector' and 'All' were the same computation
    # over different scopes, and the scope is now a checkbox list with a RANK
    # control: tick one basket and rank its pairs, and that IS the old Sector
    # view. The old tab's Interval control went with it -- measured across
    # Futures, Crypto and US Sectors, rankings at 15m/1h/4h correlate +0.90 to
    # +1.00 with daily and pick the same top spread, for ~90x the bars.
    tab_sector, tab_port = st.tabs(['Sector', 'Portfolio'])

    with tab_sector:
        render_scan_tab(is_mobile)

    with tab_port:
        render_spread_portfolio_tab(is_mobile)
