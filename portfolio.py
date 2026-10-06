import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from collections import OrderedDict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

from config import THEMES, SYMBOL_NAMES, FONTS, drop_listing_artifacts
# The statistics are SPREADS' definitions, imported rather than re-derived so a
# number means the same thing on both tabs. Each carries a guard that matters:
# ER masks unobserved intraday gaps, ROA refuses a drawdown too small to be
# real, Sortino divides by every bar rather than the losing ones, and the
# anchored curve lets an opening loss register as a drawdown. See spreads.py.
from spreads import (_spread_er, _spread_roa, _spread_sortino, _spread_curve,
                     MIN_DD_PCT, MAX_RATIO,
                     composite_ranks as _spread_composite_ranks,
                     length_adjusted as _spread_length_adjusted,
                     rank_by_length_adjusted as _spread_rank_by_length_adjusted)

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

# Series colours (data-carrying) stay as they are; surfaces and text follow the
# neutral charcoal ramp in config.THEMES. C_MUTE was #475569, near-invisible on
# a #212121 page.
C_POS = '#60a5fa'; C_NEG = '#fb7185'; C_TXT = '#e2e8f0'; C_TXT2 = '#9fb2ca'
C_MUTE = '#9fb2ca'; C_BG = 'rgba(4,8,16,0.46)'; C_HDR = 'rgba(18,33,60,0.40)'
C_BORDER = '#1e2e4c'
C_GOLD = '#fbbf24'; C_EW = '#9fb2ca'
# Benchmarks are drawn in order from this ramp; MAX_BENCHMARKS caps the list.
BENCH_COLORS = ['#c084fc', '#f472b6', '#38bdf8', '#fb923c']  # kept clear of the strategy green
C_BENCH = BENCH_COLORS[0]; MAX_BENCHMARKS = len(BENCH_COLORS)
TH = "padding:4px 8px;border-bottom:1px solid #1e2e4c;color:#f8fafc;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.06em;"
TD = "padding:5px 8px;border-bottom:1px solid #1e2e4c22;"

def _short(sym):
    return SYMBOL_NAMES.get(sym, sym.replace('=F','').replace('=X','').replace('.SI',''))

PORTFOLIO_APPROACHES = OrderedDict([
    ('3mo',              {'windows': OrderedDict([('3mo', 63)]),
                          'blend': {'3mo': 1.0}, 'min_days': 63}),
    ('6mo',              {'windows': OrderedDict([('6mo', 126)]),
                          'blend': {'6mo': 1.0}, 'min_days': 126}),
    ('12mo',             {'windows': OrderedDict([('12mo', 252)]),
                          'blend': {'12mo': 1.0}, 'min_days': 252}),
    ('24mo',             {'windows': OrderedDict([('24mo', 504)]),
                          'blend': {'24mo': 1.0}, 'min_days': 504}),
    # The fastest blend on the board, and the only one that still has something
    # to say on a short shared window: 63 days of warm-up against 252 for the
    # 12mo blends and 504 for the 24mo ones. Weighted the way every Recency
    # approach is -- front-loaded onto the shortest window, decaying back.
    ('1mo Recency',      {'windows': OrderedDict([('1mo', 21), ('2mo', 42), ('3mo', 63)]),
                          'blend': {'1mo': 0.50, '2mo': 0.30, '3mo': 0.20}, 'min_days': 63}),
    ('12mo Avg',         {'windows': OrderedDict([('3mo', 63), ('6mo', 126), ('9mo', 189), ('12mo', 252)]),
                          'blend': {'3mo': 0.25, '6mo': 0.25, '9mo': 0.25, '12mo': 0.25}, 'min_days': 252}),
    ('12mo Recency',     {'windows': OrderedDict([('3mo', 63), ('6mo', 126), ('9mo', 189), ('12mo', 252)]),
                          'blend': {'3mo': 0.40, '6mo': 0.30, '9mo': 0.20, '12mo': 0.10}, 'min_days': 252}),
    ('12mo Inv Recency', {'windows': OrderedDict([('3mo', 63), ('6mo', 126), ('9mo', 189), ('12mo', 252)]),
                          'blend': {'3mo': 0.10, '6mo': 0.20, '9mo': 0.30, '12mo': 0.40}, 'min_days': 252}),
    ('24mo Avg',         {'windows': OrderedDict([('3mo', 63), ('6mo', 126), ('12mo', 252), ('24mo', 504)]),
                          'blend': {'3mo': 0.25, '6mo': 0.25, '12mo': 0.25, '24mo': 0.25}, 'min_days': 504}),
    ('24mo Recency',     {'windows': OrderedDict([('3mo', 63), ('6mo', 126), ('12mo', 252), ('24mo', 504)]),
                          'blend': {'3mo': 0.40, '6mo': 0.30, '12mo': 0.20, '24mo': 0.10}, 'min_days': 504}),
    ('24mo Inv Recency', {'windows': OrderedDict([('3mo', 63), ('6mo', 126), ('12mo', 252), ('24mo', 504)]),
                          'blend': {'3mo': 0.10, '6mo': 0.20, '12mo': 0.30, '24mo': 0.40}, 'min_days': 504}),
    # Anchored: the window does not roll, it grows. Every rebalance scores
    # EVERYTHING from the start of the data up to that date, so the estimate
    # keeps getting steadier instead of forgetting at a fixed rate. It is the
    # opposite end of the scale from 1mo Recency -- the slowest thing on the
    # board, and the one that cannot be whipsawed by a single quarter. 252 bars
    # is the floor before it will trade at all.
    ('Anchored',         {'windows': OrderedDict([('all', 252)]),
                          'blend': {'all': 1.0}, 'min_days': 252, 'anchored': True}),
])

# How the chosen weights are sized once the optimiser has picked the names.
#   Optimized    -- the weights the search actually produced.
#   Equal weight -- the search is used only to SELECT, then every holding gets
#     1/N. Optimised weights are the noisiest thing an optimiser produces; the
#     selection is usually the part that carries signal. Pair it with Max Pos
#     and this reads as "hold the best 20 names, equally".
# The MODE picks the engine; the WEIGHTING below picks how its answer is sized.
# The third mode used to be called "Equal Weight", which read as the same thing
# as Weighting -> Equal weight and is not: this one skips the search entirely and
# holds every symbol in the universe, so there is no ranking to take a Max Pos
# from. Naming it after what it holds stops that confusion at the dropdown.
MODE_WF = 'Monte Carlo (Walk-Forward)'
MODE_FS = 'Monte Carlo (Full Sample)'
MODE_EW = 'Equal Weight (ranked)'
MODES = [MODE_WF, MODE_FS, MODE_EW]

WEIGHTING_OPTIMIZED = 'Optimized'
WEIGHTING_EQUAL = 'Equal weight'
WEIGHTINGS = [WEIGHTING_OPTIMIZED, WEIGHTING_EQUAL]

REBAL_OPTIONS = OrderedDict([
    ('No Rebalance', -1), ('Weekly', 0), ('Monthly', 1), ('Quarterly', 3), ('Semi-Annual', 6), ('Annual', 12),
])

PERIOD_OPTIONS = OrderedDict([
    ('1 Year', 365), ('2 Years', 730), ('3 Years', 1095), ('5 Years', 1825),
    ('10 Years', 3650), ('Max', 9999),
])

# The objective list and its order are SPREADS' SORT_OPTIONS, so the same nine
# names mean the same nine things on both tabs. Composite leads because it is
# the only one that does not let a single statistic decide on its own.
OBJECTIVES = ['Composite', 'Sharpe', 'Sortino', 'ROA', 'ER', 'MAR', 'R\u00b2',
              'Total Return', 'Win Rate']

SCORE_TO_RANK = {
    'Composite': '_score', 'Sharpe': 'sharpe', 'Sortino': 'sortino',
    'ROA': 'roa', 'ER': 'er', 'MAR': 'mar', 'R\u00b2': 'r2',
    'Total Return': 'total_ret', 'Win Rate': 'win_rate',
}

# What Composite averages the ranks of, in PORTFOLIO's metric names. Same four
# SPREADS ranks on, and for the same reasons: MAR pins to its cap on short
# windows and R2 measures almost what ER does, so ranking on either would be
# arbitrary tie-breaking or double-counted straightness.
COMPOSITE_METRICS = ('sharpe', 'sortino', 'roa', 'er')

# How the universe is assembled over time.
#   Shared window -- every symbol must have a price on every day of the test, so
#     one 2024 listing drags three hundred names down to its own listing date.
#   As listed     -- a symbol is absent until it has enough history to be scored,
#     then joins at the next rebalance. Nothing is discarded for being late and
#     nothing governs the window, which is what you actually want from a basket
#     that keeps gaining members.
UNIVERSE_SHARED = 'Shared window'
UNIVERSE_ASLISTED = 'As listed'
UNIVERSES = [UNIVERSE_ASLISTED, UNIVERSE_SHARED]

# Metrics a smaller number wins on. Composite scores a mean RANK, so 1.0 is the
# best possible and sorting it the usual way would put the worst row on top.
LOWER_IS_BETTER = {'_score', 'ann_vol', 'max_dd', 'avg_dd'}


def _with_span(items):
    """SPREADS reads a window length off a 'Days' key; PORTFOLIO records it as
    span_days. One place to bridge the two names."""
    for m in items:
        m.setdefault('Days', m.get('span_days') or m.get('n_days') or 0)
    return items


def rank_rows(items, key, reverse=True):
    """ITEMS ordered on KEY after SPREADS' sample-size shrink.

    Every metric the tables rank on goes through this, not only the four inside
    Composite. Rows that scored different-length windows are not otherwise
    comparable: the walk-forward approaches burn different warm-ups, and two
    baskets can differ by years because one of them is full of 2024 listings.
    The shrink only reorders -- the columns still show the real numbers.
    """
    return _spread_rank_by_length_adjusted(_with_span(items), key, reverse)


def composite_ranks(items, metrics=COMPOSITE_METRICS, score_key='_score'):
    """SPREADS' composite, computed on PORTFOLIO's metric names.

    One implementation for both tabs: the average rank across Sharpe, Sortino,
    ROA and ER, each value first shrunk by sqrt(span / longest span in the set)
    so a series that only has a few months of history cannot top the board on a
    noisy draw. It is the discount that makes these rows comparable at all --
    walk-forward approaches burn different warm-ups, so a 24mo approach scores a
    visibly shorter window than a 3mo one on the same basket, and two baskets
    can differ by years because one of them is full of 2024 listings.

    The span comes off a 'Days' key, which is what SPREADS' implementation reads
    and what the setdefault below supplies from _calc_oos_metrics' calendar span.
    Mutates and returns ITEMS; lower score is better.
    """
    return _spread_composite_ranks(_with_span(items), metrics=list(metrics),
                                   score_key=score_key)


def best_approach(results, rank_metric):
    """The winning approach name under RANK_METRIC.

    Length-adjusted and direction-aware, so it agrees with the order the ranking
    table draws rather than second-guessing it.
    """
    names = list(results)
    if not names:
        return None
    adj = _spread_length_adjusted(_with_span([results[k]['metrics'] for k in names]),
                                  rank_metric)
    sign = -1.0 if rank_metric in LOWER_IS_BETTER else 1.0
    return names[max(range(len(names)), key=lambda i: sign * adj[i])]

# =============================================================================
# DATA FETCHING
# =============================================================================

# Why the last fetch came back the way it did, keyed the same as the cache so a
# cache hit still matches. Lets the UI name the symbol that truncated the window
# instead of shrugging with "not enough history".
_FETCH_NOTES = {}


def fetch_notes(symbols, days, min_history_days=0):
    return _FETCH_NOTES.get((tuple(symbols), days, min_history_days))


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_symbol_history(symbols_tuple, days=1800, min_history_days=0, align='common'):
    symbols = list(symbols_tuple)
    if not symbols: return None, []
    note = {'n_requested': len(symbols), 'no_data': [], 'n_ok': 0,
            'union_rows': 0, 'common_rows': 0, 'start': None, 'end': None, 'limiters': [],
            'too_new': [], 'cutoff': None, 'trimmed': [], 'firsts': [], 'index': None}
    _FETCH_NOTES[(tuple(symbols_tuple), days, min_history_days)] = note
    start = (datetime.now() - pd.Timedelta(days=days)).strftime('%Y-%m-%d')
    cols = {}; valid = []
    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            hist = ticker.history(start=start, auto_adjust=True)
            if not hist.empty and len(hist) >= 50:
                closes = hist['Close'].copy()
                closes.index = closes.index.tz_localize(None) if closes.index.tz else closes.index
                closes.index = closes.index.normalize()
                closes = closes.groupby(closes.index).last()
                # Before anything measures this series: a pre-listing placeholder
                # price makes one impossible bar, and the optimiser will chase it.
                closes, cut = drop_listing_artifacts(closes)
                if cut is not None:
                    note['trimmed'].append((sym, cut))
                if len(closes) < 50:
                    note['no_data'].append(sym); continue
                cols[sym] = closes; valid.append(sym)
            else:
                note['no_data'].append(sym)
        except Exception as e:
            note['no_data'].append(sym)
            logger.warning(f"[{sym}] portfolio fetch error: {e}")
    # Drop the newly listed before the inner join: one 2026 IPO in a basket of
    # 150 otherwise drags everyone down to its own listing date.
    if min_history_days and valid:
        cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=min_history_days)
        note['cutoff'] = cutoff
        kept = []
        for sym in valid:
            (kept if cols[sym].index[0] <= cutoff else note['too_new']).append(sym)
        valid = kept
    note['n_ok'] = len(valid)
    if len(valid) < 2: return None, valid
    data = pd.concat({s: cols[s] for s in valid}, axis=1)[valid].ffill()
    note['union_rows'] = len(data)
    # Whoever listed last sets the common start date -- name them, they are the
    # reason a 5-year request can come back as 2 years of overlap.
    firsts = [(sym, data[sym].first_valid_index()) for sym in valid]
    firsts = [(sym, d) for sym, d in firsts if d is not None]
    note['firsts'] = firsts; note['index'] = data.index
    common = data.dropna()
    note['common_rows'] = len(common)
    if len(common):
        note['start'] = common.index[0]; note['end'] = common.index[-1]
        note['limiters'] = sorted(firsts, key=lambda kv: kv[1], reverse=True)[:3]
    if align == 'union':
        # Every bar any symbol printed, forward-filled, NaN before a symbol's
        # first print. Nothing is thrown away for being late: a 2024 listing is
        # simply absent until 2024 and joins the book at the first rebalance that
        # can score it. This is the frame the As-listed universe walks over.
        if len(data) < 50: return None, valid
        return data, valid
    if len(common) < 50: return None, valid
    return common, valid

# Windows worth asking for, in trading days: one, two, three and five years.
WINDOW_TARGETS = (252, 504, 756, 1260)

# Below a year of shared history the 12mo family of approaches cannot run at all,
# and most of the board lives there. It is the bar Auto tries to clear.
AUTO_TARGET_DAYS = 252


def min_hist_frontier(notes, fetch_days, targets=WINDOW_TARGETS):
    """The cheapest way to buy a given window, priced in symbols.

    A years cutoff is the wrong unit for this question. What governs the shared
    window is not an age but a handful of specific listings, and the honest
    answer is "drop these three and you get two years" -- so that is what this
    returns. Dropping the k newest listings leaves the (k+1)-th newest governing
    the window, which makes the whole frontier one pass down the sorted listing
    dates: exact, and no grid to fall between.

    Returns [(n_dropped, n_kept, n_total, n_days, cutoff_date)], cheapest first.
    """
    # notes can be None: _FETCH_NOTES is only written while the fetch body runs,
    # and @st.cache_data skips the body on a hit, so a cached universe has no
    # note to read. Nothing to say about a window we cannot see.
    if not notes:
        return []
    firsts = sorted((d for _s, d in (notes.get('firsts') or [])), reverse=True)
    idx = notes.get('index')
    total = len(firsts)
    if total < 2 or idx is None or not len(idx):
        return []
    rows, k = [], 0
    for target in targets:
        # Asking for more history than Period even fetched is not an option.
        if target > len(idx):
            break
        while k <= total - 2 and int((idx >= firsts[k]).sum()) < target:
            k += 1
        if k > total - 2:
            break
        cutoff = firsts[k]
        days = int((idx >= cutoff).sum())
        # Count by DATE, not by k: several symbols can share a listing date, and
        # a cutoff keeps all of them or none.
        kept = sum(1 for d in firsts if d <= cutoff)
        row = (total - kept, kept, total, days, cutoff)
        if rows and row[0] == rows[-1][0]:
            rows[-1] = row if row[3] > rows[-1][3] else rows[-1]
            continue
        rows.append(row)
    return rows


def min_hist_auto(notes, fetch_days, target_days=AUTO_TARGET_DAYS):
    """The cheapest exclusion that buys a workable window, or None to leave it be.

    Prefers the LEAST exclusion that clears the target: dropping symbols is a
    real cost, and the job is to stop two 2026 listings governing three hundred
    names, not to prune the universe for its own sake. If nothing clears the
    bar, take whatever buys the longest window, since the alternative is a run
    that cannot happen at all.

    Returns a frontier row, or None when there is nothing to gain.
    """
    rows = min_hist_frontier(notes, fetch_days)
    if not rows or not notes:
        return None
    current = notes.get('common_rows') or 0
    usable = [r for r in rows if r[3] >= target_days]
    best = usable[0] if usable else max(rows, key=lambda r: r[3])
    return best if best[3] > current else None


def min_hist_days_for(cutoff):
    """The Min Hist Y cutoff date, as the number of days the fetch filter wants."""
    return max(int((pd.Timestamp.now().normalize() - cutoff).days), 0)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_benchmark_history(symbol, days=1800):
    """Close series for a single benchmark ticker. Separate from
    fetch_symbol_history, which needs >= 2 symbols to build a portfolio."""
    if not symbol: return None
    start = (datetime.now() - pd.Timedelta(days=days)).strftime('%Y-%m-%d')
    try:
        hist = yf.Ticker(symbol).history(start=start, auto_adjust=True)
    except Exception as e:
        logger.warning(f"[{symbol}] benchmark fetch error: {e}")
        return None
    if hist.empty or len(hist) < 50: return None
    closes = hist['Close'].copy()
    closes.index = closes.index.tz_localize(None) if closes.index.tz else closes.index
    closes.index = closes.index.normalize()
    closes = closes.groupby(closes.index).last()
    # A benchmark can be a token too, and a comparison line drawn off a
    # placeholder price flatters or damns the portfolio for nothing.
    closes, _cut = drop_listing_artifacts(closes)
    return closes if closes is not None and len(closes) >= 50 else None


def benchmark_series(symbols, fetch_days, price_index):
    """Daily returns for each benchmark ticker, aligned to the portfolio's *price*
    dates so they land on the same index as data.pct_change(). Returns
    ([(symbol, returns), ...], [symbols with no usable history])."""
    out = []; failed = []
    for sym in _bench_list(symbols):
        prices = fetch_benchmark_history(sym, days=fetch_days)
        ret = None
        if prices is not None:
            px = prices.reindex(prices.index.union(price_index)).ffill().reindex(price_index).dropna()
            ret = px.pct_change().dropna()
        if ret is None or len(ret) < 20: failed.append(sym)
        else: out.append((sym, ret))
    return out, failed


def _bench_list(symbols):
    """Accept 'SPY, XLV' or ['SPY', 'XLV'] -> ['SPY', 'XLV'], deduped and capped."""
    if not symbols: return []
    if isinstance(symbols, str):
        symbols = symbols.replace(';', ',').split(',')
    out = []
    for sym in symbols:
        sym = (sym or '').strip().upper()
        if sym and sym not in out: out.append(sym)
    return out[:MAX_BENCHMARKS]


def _bench_metrics(series, start=None):
    """[(symbol, returns)] -> [(symbol, returns, metrics)] over the window from
    `start`, dropping any benchmark too short to measure there."""
    out = []
    for sym, ret in series or []:
        r = ret.loc[ret.index >= start] if start is not None else ret
        if len(r) < 5: continue
        m = _calc_oos_metrics(r)
        if m: out.append((sym, r, m))
    return out


# =============================================================================
# MC OPTIMIZATION ENGINE
# =============================================================================

def _score_rows(port_returns, score_type, max_vol=None, min_ann_ret=None):
    """Score every ROW of a (n, bars) return matrix on SCORE_TYPE.

    The optimiser feeds it candidate portfolios; the screen feeds it individual
    symbols. Identical formulas either way, which is the point -- "best 20 names
    by ROA" and "best weights by ROA" have to mean the same ROA or comparing
    them is meaningless. Returns (scores, penalty).
    """
    if port_returns.ndim == 1:
        port_returns = port_returns[None, :]
    n_rows = port_returns.shape[0]
    ann_rets = np.mean(port_returns, axis=1) * 252
    ann_vols = np.std(port_returns, axis=1, ddof=1) * np.sqrt(252)

    penalty = np.zeros(n_rows)
    if max_vol is not None and max_vol > 0:
        penalty += np.maximum(ann_vols - max_vol, 0) * 50
    if min_ann_ret is not None:
        penalty += np.maximum(min_ann_ret - ann_rets, 0) * 50

    # The row-wise twins of spreads._spread_* , one row per candidate portfolio.
    # They are the same formulas; change one and change the other.
    #
    # Everything below is (n_portfolios x n_bars) -- 20MB at 5,000 sims over two
    # years -- so the cost here is memory bandwidth, not arithmetic. The curve
    # and the drawdown are each built ONCE per call and handed to whoever needs
    # them; ROA and ER were previously building their own, which was a second
    # full cumprod for nothing.
    _shared = {}

    def _curve():
        """Equity anchored at 1.0, as in spreads._spread_curve. port_returns came
        from a pct_change that already ate bar zero, so without the anchor the
        first bar is its own running maximum and an opening loss cannot
        register."""
        if 'curve' not in _shared:
            _shared['curve'] = np.hstack([np.ones((port_returns.shape[0], 1)),
                                          np.cumprod(1 + port_returns, axis=1)])
        return _shared['curve']

    def _drawdown():
        """Running drawdown off the anchored curve."""
        if 'dd' not in _shared:
            curve = _curve()
            peak = np.maximum.accumulate(curve, axis=1)
            _shared['dd'] = (curve - peak) / peak
        return _shared['dd']

    def _median_abs_bar():
        """Median |bar| per candidate, the scale ROA's floor is measured in.

        np.median copies the array and partitions the copy, so at this size it
        is two more 20MB allocations per call. abs() writes into one buffer and
        the partition then runs in place on it, which is the same number down to
        the last bit for about 60% of the time."""
        if 'med' not in _shared:
            buf = np.abs(port_returns)
            n_bars = buf.shape[1]
            k = n_bars // 2
            if n_bars % 2:
                buf.partition(k, axis=1)
                _shared['med'] = buf[:, k].copy()
            else:
                buf.partition([k - 1, k], axis=1)
                _shared['med'] = (buf[:, k - 1] + buf[:, k]) / 2.0
        return _shared['med']

    def _vec_downside_vol(pr):
        """RMS of min(r, 0) over EVERY bar, as in spreads._spread_sortino.
        Averaging the squares over the losing bars alone divides by the count of
        losers, which understated Sortino by about 44% on these series."""
        neg = np.minimum(pr, 0)
        return np.sqrt(np.mean(neg**2, axis=1)) * np.sqrt(252)

    def _vec_avg_dd(pr):
        dd = _drawdown()
        neg_dd = np.where(dd < 0, dd, 0)
        n_neg = np.maximum(np.sum(dd < 0, axis=1).astype(float), 1)
        return np.sum(neg_dd, axis=1) / n_neg

    def _vec_roa(pr):
        """Total return over the worst hole. Refuses to score a candidate whose
        worst drawdown is under one typical bar of its own movement -- that is
        not a portfolio avoiding a hole, it is too few bars to have had one."""
        curve = _curve()
        mdd = _drawdown().min(axis=1) * 100.0
        total = (curve[:, -1] - 1.0) * 100.0
        floor = np.maximum(MIN_DD_PCT, _median_abs_bar() * 100.0)
        denom = np.where(np.abs(mdd) > 0, np.abs(mdd), 1.0)
        return np.where(np.abs(mdd) < floor, 0.0,
                        np.clip(total / denom, -MAX_RATIO, MAX_RATIO))

    def _vec_er(pr):
        """Kaufman efficiency ratio, signed: |net move| / path length. 1.0 is a
        straight line, 0.0 is chop that goes nowhere. No gap mask here -- these
        are daily bars, where a weekend is not an unobserved gap."""
        d = np.diff(_curve(), axis=1)
        path = np.abs(d).sum(axis=1)
        net = d.sum(axis=1)
        er = np.where(path > 0, np.abs(net) / np.where(path > 0, path, 1.0), 0.0)
        return np.where(net >= 0, er, -er)

    if score_type == 'Win Rate':
        scores = np.mean(port_returns > 0, axis=1)
    elif score_type == 'Total Return':
        cum_final = np.prod(1 + port_returns, axis=1)
        scores = cum_final - 1
    elif score_type == 'Sortino':
        dv = _vec_downside_vol(port_returns)
        scores = np.where(dv > 0, ann_rets / dv, 0)
    elif score_type == 'MAR':
        avg_dd = _vec_avg_dd(port_returns)
        scores = np.where(avg_dd < 0, ann_rets / np.abs(avg_dd), 0)
    elif score_type == 'ROA':
        scores = _vec_roa(port_returns)
    elif score_type == 'ER':
        scores = _vec_er(port_returns)
    elif score_type == 'R²':
        # Its own cumprod on purpose. Slicing the shared anchored curve would
        # give the right numbers off a non-contiguous view, and the regression
        # below then reads it column-wise -- measured 10% SLOWER than just
        # building the contiguous array it wants.
        cum = np.cumprod(1 + port_returns, axis=1)
        n = cum.shape[1]; x = np.arange(n, dtype=float); xm = x.mean()
        ss_xx = np.sum(x * x) - n * xm * xm
        ym = cum.mean(axis=1, keepdims=True)
        ss_xy = (cum @ x) - n * xm * ym.ravel()
        ss_yy = np.sum(cum * cum, axis=1) - n * (ym.ravel() ** 2)
        denom = ss_xx * ss_yy
        scores = np.where(denom > 0, (ss_xy ** 2) / denom, 0)
        slope = np.where(ss_xx > 0, ss_xy / ss_xx, 0)
        scores = np.where(slope > 0, scores, -scores)
    elif score_type == 'Composite':
        # The same four COMPOSITE_METRICS averages, equally weighted, as a rank
        # percentile instead of a rank so that larger is better here. No
        # sample-size discount: every candidate is scored on the SAME window, so
        # the factor composite_ranks applies across baskets would be 1.0 for all
        # of them.
        sharpes = np.where(ann_vols > 0, ann_rets / ann_vols, 0)
        dv = _vec_downside_vol(port_returns)
        sortinos = np.where(dv > 0, ann_rets / dv, 0)
        def _rank_pct(a):
            r = a.argsort().argsort().astype(float)
            return r / max(len(r) - 1, 1)
        scores = np.mean([_rank_pct(v) for v in (sharpes, sortinos,
                                                 _vec_roa(port_returns),
                                                 _vec_er(port_returns))], axis=0)
    else:  # Sharpe
        scores = np.where(ann_vols > 0, ann_rets / ann_vols, 0)


    return scores, penalty


def _optimize_window_vectorized(returns_array, n_portfolios, n_assets, max_weight,
                                score_type='Win Rate', min_weight=0.0, allow_short=False,
                                max_vol=None, min_ann_ret=None):
    if allow_short:
        weights = np.random.randn(n_portfolios, n_assets)
        weights = weights / weights.sum(axis=1, keepdims=True)
        for _ in range(30):
            violated = (weights > max_weight) | (weights < -max_weight)
            if not np.any(violated): break
            weights = np.clip(weights, -max_weight, max_weight)
            weights = weights / weights.sum(axis=1, keepdims=True)
    elif n_assets == 2:
        lo = max(1 - max_weight, min_weight); hi = min(max_weight, 1 - min_weight)
        if lo >= hi: lo, hi = 0.3, 0.7
        w1 = np.random.uniform(lo, hi, n_portfolios)
        weights = np.column_stack([w1, 1 - w1])
    else:
        n_half = n_portfolios // 2
        w_conc = np.random.dirichlet(np.ones(n_assets) * 0.5, n_half)
        w_div = np.random.dirichlet(np.ones(n_assets) * 1.0, n_portfolios - n_half)
        weights = np.vstack([w_conc, w_div])
        for _ in range(30):
            violated = (weights > max_weight) | (weights < min_weight)
            if not np.any(violated): break
            weights = np.maximum(weights, min_weight)
            weights = np.minimum(weights, max_weight)
            weights = weights / weights.sum(axis=1, keepdims=True)

    port_returns = weights @ returns_array.T
    scores, penalty = _score_rows(port_returns, score_type, max_vol, min_ann_ret)

    # Normalize scores to 0-1 range, then apply constraint penalty
    s_min, s_max = scores.min(), scores.max()
    if s_max > s_min:
        norm_scores = (scores - s_min) / (s_max - s_min)
    else:
        norm_scores = np.ones_like(scores) * 0.5
    # Penalty is 0 when constraints satisfied, large when violated
    final_scores = norm_scores - penalty
    return weights[np.argmax(final_scores)]

# =============================================================================
# WALK-FORWARD ENGINE
# =============================================================================

def _cap_to_max_weight(w, max_weight):
    """Pull anything over the ceiling back to it and push the excess onto the
    names that still have room.

    The obvious loop -- clip, then rescale back to 100% -- barely converges when
    a holding is already sitting ON the ceiling: the rescale lifts it straight
    back over, so each pass claws back a few basis points and five passes are
    nowhere near enough. Concentrating 12 names into 5 left a book of 12.

    Moving the excess sideways instead keeps the sum at 100% by construction, so
    there is nothing to rescale and it lands in a pass or two. Returns None when
    the ceiling genuinely cannot hold the whole portfolio -- every name pinned
    and still short of 100% -- which is the caller's cue to leave it alone.
    """
    out = np.asarray(w, dtype=float).copy()
    for _ in range(50):
        over = np.abs(out) > max_weight + 1e-12
        if not over.any():
            return out
        excess = float(np.sum(out[over] - np.sign(out[over]) * max_weight))
        out[over] = np.sign(out[over]) * max_weight
        # Only onto names that are ALREADY held. Both callers get here by sending
        # weights to exactly zero -- dust in one case, everything outside the top
        # N in the other -- and a zero has the most headroom of anything in the
        # vector, so ignoring this handed the excess straight back to the names
        # that were just dropped: Max Pos 5 returned a book of 7.
        room = (~over) & (out != 0.0)
        headroom = np.maximum(max_weight - np.abs(out), 0.0) * room
        total_room = float(headroom.sum())
        if total_room <= 1e-12:
            return None
        out = out + excess * headroom / total_room
    return out if not np.any(np.abs(out) > max_weight + 1e-9) else None


def _apply_min_pos(w, min_pos, max_weight=None):
    """Drop positions smaller than min_pos and rescale the survivors back to 100%.
    Unlike min_weight (a floor that forces every asset in), this removes dust:
    a 0.4% sliver becomes 0 and its weight goes to the positions worth trading."""
    if not min_pos or min_pos <= 0: return w
    keep = np.abs(w) >= min_pos
    if not keep.any(): return w  # everything is dust — leave the optimizer's answer alone
    out = np.where(keep, w, 0.0)
    total = out.sum()
    if total <= 0: return w
    out = out / total
    # Rescaling the survivors can lift one past the ceiling; spread the excess
    # over the names that still have room rather than rescale a second time.
    if max_weight:
        capped = _cap_to_max_weight(out, max_weight)
        if capped is None: return w
        out = capped
    return out


def _apply_max_pos(w, max_pos, max_weight=None):
    """Keep only the MAX_POS largest holdings and rescale them back to 100%.

    Min Pos % cuts by size, which is the wrong tool once the universe is 350
    names pooled out of fourteen baskets: whatever threshold you pick, the
    number of positions that survives is whatever it happens to be. This caps
    the count directly, so "twenty names" means twenty names.

    Ranked on |weight|, so it does the right thing in Long/Short: a conviction
    short is a position worth keeping, not a small number to discard. Ties break
    by position -- arbitrary, but stable, so the same weights always give the
    same book.
    """
    if not max_pos or max_pos <= 0: return w
    if int(np.sum(np.abs(w) > 0)) <= max_pos: return w
    keep = np.argsort(-np.abs(w), kind='stable')[:int(max_pos)]
    out = np.zeros_like(w)
    out[keep] = w[keep]
    total = out.sum()
    if total <= 0: return w
    out = out / total
    # Concentrating into fewer names pushes the survivors up, which can carry one
    # past the ceiling. None back means no book of this many names both sums to
    # 100% and fits under Max Wt -- hand the uncapped one back rather than a
    # levered book nobody asked for. _validate rejects the settings that cause
    # it, so in practice this is the Long/Short case where the kept names cancel.
    if max_weight:
        capped = _cap_to_max_weight(out, max_weight)
        if capped is None: return w
        out = capped
    return out


def _equalize(w):
    """Flatten the held names to 1/N, keeping the selection and discarding the
    sizing. Signs are preserved so a Long/Short book stays the shape the search
    chose; if the legs cancel so completely that the result cannot be scaled to
    100%, the optimiser's own weights are handed back rather than a levered one.
    """
    held = np.flatnonzero(np.abs(w) > 0)
    if len(held) == 0:
        return w
    out = np.zeros_like(w)
    out[held] = np.sign(w[held]) / len(held)
    total = out.sum()
    if total <= 1e-12:
        return w
    return out / total


def _round_weights(w, step):
    """Snap weights to a step (0.01 = whole percents) using largest-remainder, so
    the rounded weights still add to exactly 100% instead of 99.9 or 100.1.
    Anything under half a step rounds away to 0 — a 0.4% sliver at step 1% goes."""
    if not step or step <= 0: return w
    units = int(round(1.0 / step))
    if units <= 0: return w
    raw = w * units
    base = np.floor(raw)
    remainder = raw - base
    short = int(round(units - base.sum()))
    if short > 0:
        # Hand the leftover units to the weights that lost the most to flooring
        for i in np.argsort(-remainder)[:short]: base[i] += 1
    elif short < 0:
        for i in np.argsort(remainder):
            if short == 0: break
            if base[i] > 0: base[i] -= 1; short += 1
    return base / units


def _optimize_at_rebalance(returns_df, approach, score_type, n_portfolios, mw, mnw=0.0, allow_short=False,
                           max_vol=None, min_ann_ret=None, window_cache=None, min_pos=0.0, round_step=0.0,
                           max_pos=0, weighting=WEIGHTING_OPTIMIZED):
    """Blended weights at one rebalance, over whatever is tradeable by then.

    RETURNS_DF may carry NaN before a symbol's first print. Each window keeps the
    symbols that have at least WDAYS observations, which is exactly the set whose
    last WDAYS rows are dense -- the gaps are all at the top. A symbol old enough
    for the 3mo window but not the 12mo one therefore earns weight from the 3mo
    leg only, which is the honest answer rather than either excluding it outright
    or pretending it has a year of history.
    """
    n_assets = returns_df.shape[1]; data_len = len(returns_df)
    counts = returns_df.notna().sum().values
    anchored = bool(approach.get('anchored'))
    window_weights_list = []; blend_wts = []
    for wname, wdays in approach['windows'].items():
        if data_len < wdays:
            continue
        live = np.flatnonzero(counts >= wdays)
        if len(live) < 2:
            continue
        cache_key = (wname, data_len) if window_cache is not None else None
        if cache_key and cache_key in window_cache:
            best_w = window_cache[cache_key]
        else:
            # Anchored takes everything up to here; the rest take the last WDAYS.
            # Either way the slice is dense for the columns selected above, since
            # an eligible column's gaps are all at the top -- except on anchored,
            # where a late listing is NaN at the start of the full history and is
            # caught by the finite check below.
            w_ret = (returns_df.iloc[:, live].values if anchored
                     else returns_df.iloc[-wdays:, live].values)
            if not np.isfinite(w_ret).all():
                # A hole in the middle rather than at the top -- a trading halt,
                # or a symbol that stopped printing. Those columns cannot be
                # scored on this window.
                ok = np.isfinite(w_ret).all(axis=0)
                if ok.sum() < 2:
                    continue
                live = live[ok]; w_ret = w_ret[:, ok]
            sub_w = _optimize_window_vectorized(w_ret, n_portfolios, len(live), mw, score_type,
                                                mnw, allow_short, max_vol=max_vol,
                                                min_ann_ret=min_ann_ret)
            best_w = np.zeros(n_assets)
            best_w[live] = sub_w
            if cache_key and window_cache is not None:
                window_cache[cache_key] = best_w
        window_weights_list.append(best_w); blend_wts.append(approach['blend'][wname])
    if not window_weights_list: return None
    blend_wts = np.array(blend_wts); blend_wts /= blend_wts.sum()
    all_w = np.array(window_weights_list)
    opt_w = np.average(all_w, axis=0, weights=blend_wts)
    total = opt_w.sum()
    if total == 0: return None
    opt_w /= total
    # Dust first, then the count cap, then the rounding step: Min Pos drops the
    # slivers that would otherwise occupy slots in the top N, and rounding has to
    # come last or it cannot guarantee the weights still add to 100%.
    opt_w = _apply_max_pos(_apply_min_pos(opt_w, min_pos, mw), max_pos, mw)
    # Equal weight LAST of the three, so it flattens exactly the names that
    # survived the dust cut and the count cap -- the book, not the candidates.
    if weighting == WEIGHTING_EQUAL:
        opt_w = _equalize(opt_w)
    return _round_weights(opt_w, round_step)


def _screen_at_rebalance(returns_df, approach, score_type, max_pos, mw,
                         max_vol=None, min_ann_ret=None, min_pos=0.0, round_step=0.0):
    """Rank the symbols themselves, take the best MAX_POS, hold them 1/N.

    No Monte Carlo anywhere. Where the optimiser searches thousands of weight
    combinations and asks which PORTFOLIO scored best, this asks which SYMBOLS
    scored best and then refuses to express an opinion about sizing. That is a
    narrower claim and a much cheaper one -- it is also blind to how the names
    move together, which the search is not, so it is a genuinely different
    strategy rather than a faster approximation of the same one.

    Scores are blended across the approach's windows on its own blend weights,
    so '12mo Recency' means the same thing here as it does there: the recent
    window counts for more. Ranking happens once, on the blend.
    """
    n_assets = returns_df.shape[1]; data_len = len(returns_df)
    counts = returns_df.notna().sum().values
    anchored = bool(approach.get('anchored'))
    blended = np.zeros(n_assets); seen = np.zeros(n_assets)
    for wname, wdays in approach['windows'].items():
        if data_len < wdays:
            continue
        live = np.flatnonzero(counts >= wdays)
        if len(live) < 2:
            continue
        w_ret = (returns_df.iloc[:, live].values if anchored
                 else returns_df.iloc[-wdays:, live].values)
        ok = np.isfinite(w_ret).all(axis=0)
        if ok.sum() < 2:
            continue
        live = live[ok]; w_ret = w_ret[:, ok]
        # One row per SYMBOL -- the transpose is the whole trick.
        sc, pen = _score_rows(w_ret.T, score_type, max_vol, min_ann_ret)
        sc = sc - pen
        # Rank percentile, not the raw score: ROA and Sharpe live on different
        # scales, and a blend across windows has to add comparable things.
        rank = sc.argsort().argsort().astype(float) / max(len(sc) - 1, 1)
        blended[live] += rank * approach['blend'][wname]
        seen[live] += approach['blend'][wname]
    eligible = np.flatnonzero(seen > 0)
    if len(eligible) < 2:
        return None
    score = np.full(n_assets, -np.inf)
    score[eligible] = blended[eligible] / seen[eligible]

    keep = eligible[np.argsort(-score[eligible])]
    if max_pos and max_pos > 0:
        keep = keep[:int(max_pos)]
    if len(keep) < 1:
        return None
    w = np.zeros(n_assets)
    w[keep] = 1.0 / len(keep)
    # Min Pos and Round still apply; they just have far less to do once every
    # holding is already the same size.
    w = _apply_min_pos(w, min_pos, mw)
    capped = _cap_to_max_weight(w, mw) if mw else w
    if capped is not None:
        w = capped
    return _round_weights(w, round_step)


def _walk_forward_single(returns_df, approach, score_type, rebal_months,
                         n_portfolios=10000, max_weight=0.50, min_weight=0.0,
                         txn_cost=0.001, allow_short=False,
                         max_vol=None, min_ann_ret=None, window_cache=None, min_pos=0.0, round_step=0.0,
                         max_pos=0, weighting=WEIGHTING_OPTIMIZED, screen=False):
    n_assets = returns_df.shape[1]; mw = max_weight; mnw = min_weight
    # The SHORTEST window decides when trading can start. On a shared window
    # every symbol is present from bar zero so this is the same as the longest;
    # on a growing one it is what lets the test begin before the whole universe
    # has a year of history behind it.
    min_is_days = min(approach['windows'].values()); dates = returns_df.index

    if rebal_months == -1:
        candidate_dates = [dates[min_is_days]] if len(dates) > min_is_days else []
    elif rebal_months == 0:
        # Weekly: rebalance on first trading day of each week
        candidate_dates = []; seen_weeks = set()
        for d in dates:
            yw = (d.year, d.isocalendar()[1])
            if yw not in seen_weeks:
                seen_weeks.add(yw); candidate_dates.append(d)
    else:
        if rebal_months == 1: rebal_month_set = set(range(1, 13))
        elif rebal_months == 3: rebal_month_set = {1, 4, 7, 10}
        elif rebal_months == 6: rebal_month_set = {1, 7}
        else: rebal_month_set = {1}

        candidate_dates = []; seen_months = set()
        for d in dates:
            ym = (d.year, d.month)
            if ym not in seen_months and d.month in rebal_month_set:
                seen_months.add(ym); candidate_dates.append(d)

    rebal_dates = []
    for d in candidate_dates:
        idx = dates.get_loc(d)
        if idx >= min_is_days: rebal_dates.append((idx, d))
    # No Rebalance means exactly one: optimise at the end of the warm-up and hold
    # to the end. That is a real walk-forward -- one split, trained before the
    # period it is scored on -- and demanding two of them rejected buy-and-hold
    # outright, which is the one strategy everything else should be beating.
    need = 1 if rebal_months == -1 else 2
    if len(rebal_dates) < need: return None

    oos_segments = []; weight_history = []
    prev_weights = np.ones(n_assets) / n_assets

    for i, (ri, rd) in enumerate(rebal_dates):
        is_data = returns_df.iloc[:ri + 1]
        if screen:
            opt_w = _screen_at_rebalance(is_data, approach, score_type, max_pos, mw,
                                         max_vol=max_vol, min_ann_ret=min_ann_ret,
                                         min_pos=min_pos, round_step=round_step)
        else:
            opt_w = _optimize_at_rebalance(is_data, approach, score_type, n_portfolios, mw, mnw, allow_short,
                                        max_vol=max_vol, min_ann_ret=min_ann_ret, window_cache=window_cache,
                                        min_pos=min_pos, round_step=round_step, max_pos=max_pos,
                                        weighting=weighting)
        if opt_w is None: continue
        oos_start = ri + 1
        oos_end = rebal_dates[i + 1][0] if i + 1 < len(rebal_dates) else len(dates)
        if oos_start >= oos_end: continue
        oos_data = returns_df.iloc[oos_start:oos_end]
        # Only the names the optimiser could actually score carry weight, so the
        # NaNs belonging to symbols not yet listed never reach the dot product.
        held = np.flatnonzero(np.abs(opt_w) > 0)
        port_oos = np.nan_to_num(oos_data.values[:, held], nan=0.0) @ opt_w[held]
        turnover = np.sum(np.abs(opt_w - prev_weights)) / 2.0
        if txn_cost > 0 and turnover > 0: port_oos[0] -= turnover * txn_cost
        prev_weights = opt_w.copy()
        oos_segments.append(pd.Series(port_oos, index=oos_data.index))
        weight_history.append({'date': rd, 'weights': opt_w.copy(),
            'oos_start': dates[oos_start], 'oos_end': dates[min(oos_end - 1, len(dates) - 1)],
            'oos_days': len(oos_data)})

    if not oos_segments or len(weight_history) < need: return None
    if screen:
        current_w = _screen_at_rebalance(returns_df, approach, score_type, max_pos, mw,
                                         max_vol=max_vol, min_ann_ret=min_ann_ret,
                                         min_pos=min_pos, round_step=round_step)
    else:
        current_w = _optimize_at_rebalance(returns_df, approach, score_type, n_portfolios, mw, mnw, allow_short,
                                        max_vol=max_vol, min_ann_ret=min_ann_ret, window_cache=window_cache,
                                        min_pos=min_pos, round_step=round_step, max_pos=max_pos,
                                        weighting=weighting)
    if current_w is None: current_w = weight_history[-1]['weights']
    full_oos = pd.concat(oos_segments)
    return {'oos_returns': full_oos, 'weight_history': weight_history,
            'current_weights': current_w, 'last_rebalance': weight_history[-1]['date']}


def _calc_oos_metrics(returns_series):
    r = returns_series.values; n = len(r)
    if n < 5: return None
    cum = np.cumprod(1 + r); total = float(cum[-1] - 1)
    ann_ret = float(np.mean(r) * 252); ann_vol = float(np.std(r, ddof=1) * np.sqrt(252))
    # Drawdowns off the anchored curve, as in spreads._spread_drawdowns: cum
    # starts one bar in, so unanchored the first bar is its own running maximum
    # and a portfolio that gapped down on day one reported a 0.00% drawdown.
    anchored = _spread_curve(r)
    a_peak = np.maximum.accumulate(anchored); dd = (anchored - a_peak) / a_peak
    max_dd = float(np.min(dd)); avg_dd = float(np.mean(dd[dd < 0])) if np.any(dd < 0) else 0.0
    win_rate = float(np.sum(r > 0) / n)
    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else 0.0
    # The four Composite ranks on, all SPREADS' definitions. Sortino's
    # denominator is the RMS of min(r, 0) over EVERY bar; dividing by the count
    # of losing bars instead inflated it and understated Sortino by about 44%.
    # MAR takes the same floor and cap, or a portfolio that barely moved posts
    # one in the hundreds. _spread_roa wants max_dd in percent.
    sortino = _spread_sortino(returns_series, 252)
    mar = float(np.clip(ann_ret * 100 / max(abs(avg_dd * 100), MIN_DD_PCT),
                        -MAX_RATIO, MAX_RATIO))
    er = _spread_er(returns_series)
    roa = _spread_roa(returns_series, max_dd * 100)
    if n > 2:
        x = np.arange(n, dtype=float); xm, ym = x.mean(), cum.mean()
        ss_xy = np.sum(x * cum) - n * xm * ym
        ss_xx = np.sum(x * x) - n * xm * xm
        ss_yy = np.sum(cum * cum) - n * ym * ym
        slope = ss_xy / ss_xx if ss_xx else 0
        r2 = float(np.clip((ss_xy**2) / (ss_xx * ss_yy), 0, 1)) if (ss_xx * ss_yy) else 0.0
        r2 = r2 if slope > 0 else -r2
    else: r2 = 0.0
    now = pd.Timestamp.now(); idx = returns_series.index
    ytd_mask = idx >= pd.Timestamp(now.year, 1, 1)
    mtd_mask = idx >= pd.Timestamp(now.year, now.month, 1)
    ytd = float(np.prod(1 + r[ytd_mask]) - 1) if ytd_mask.any() else 0.0
    mtd = float(np.prod(1 + r[mtd_mask]) - 1) if mtd_mask.any() else 0.0
    # Calendar span, not bars: it is what composite_ranks discounts on, and bars
    # would penalise a crypto basket for printing 365 of them a year against an
    # equity basket's 252 over the SAME window -- a denser sample, not a longer.
    span_days = max(int((idx[-1] - idx[0]).days), 1)
    return {'total_ret': total, 'ann_ret': ann_ret, 'ann_vol': ann_vol,
            'max_dd': max_dd, 'avg_dd': avg_dd, 'win_rate': win_rate,
            'sharpe': sharpe, 'sortino': sortino, 'mar': mar, 'r2': r2,
            'roa': roa, 'er': er,
            'ytd': ytd, 'mtd': mtd, 'n_days': n, 'span_days': span_days,
            'oos_years': round(span_days / 365.25, 1)}

# =============================================================================
# GRID SEARCH
# =============================================================================

def run_walkforward_grid(symbols, score_type='Win Rate', rebal_months=3, n_portfolios=10000,
                         fetch_days=1800, max_weight=0.50, min_weight=0.0,
                         txn_cost=0.001, allow_short=False, progress_bar=None,
                         max_vol=None, min_ann_ret=None, benchmarks=None, min_pos=0.0, round_step=0.0,
                         min_history_days=0, max_pos=0, universe='common',
                         weighting=WEIGHTING_OPTIMIZED, screen=False):
    data, valid = fetch_symbol_history(tuple(symbols), days=fetch_days,
                                       min_history_days=min_history_days,
                                       align='union' if universe == UNIVERSE_ASLISTED else 'common')
    if data is None or len(valid) < 2: return None
    # dropna() on the union frame would undo the whole point of fetching it, so
    # only the first row -- the one pct_change cannot produce -- is dropped.
    returns = data.pct_change()
    returns = returns.iloc[1:] if universe == UNIVERSE_ASLISTED else returns.dropna()
    n_assets = len(valid)

    # Equal weight benchmark with drift + transaction costs
    eq_w = np.ones(n_assets) / n_assets
    if rebal_months == -1:
        dates = returns.index; eq_rebal_set = set()
    elif rebal_months == 0:
        # Weekly
        dates = returns.index; eq_rebal_set = set(); seen_weeks = set()
        for d in dates:
            yw = (d.year, d.isocalendar()[1])
            if yw not in seen_weeks: seen_weeks.add(yw); eq_rebal_set.add(d)
    else:
        if rebal_months == 1: rebal_month_set = set(range(1, 13))
        elif rebal_months == 3: rebal_month_set = {1, 4, 7, 10}
        elif rebal_months == 6: rebal_month_set = {1, 7}
        else: rebal_month_set = {1}
        dates = returns.index; eq_rebal_set = set(); seen = set()
        for d in dates:
            ym = (d.year, d.month)
            if ym not in seen and d.month in rebal_month_set: seen.add(ym); eq_rebal_set.add(d)
    ret_arr = returns.values; n_days = len(ret_arr); eq_daily = np.zeros(n_days)
    live_arr = np.isfinite(ret_arr)
    curr_w = eq_w.copy()
    for t in range(n_days):
        row = ret_arr[t]
        if universe == UNIVERSE_ASLISTED:
            # 1/N over whatever is listed on the day, so the benchmark the
            # strategy is measured against grows the same way the strategy does.
            # It re-equalises at each rebalance rather than drifting across a
            # changing membership, which has no well-defined drift.
            live = live_arr[t]
            n_live = int(live.sum())
            if n_live < 1:
                continue
            w = np.zeros(n_assets); w[live] = 1.0 / n_live
            eq_daily[t] = float(np.nansum(w * row))
            if t + 1 < n_days and dates[t + 1] in eq_rebal_set:
                eq_daily[t] -= (1.0 / max(n_live, 1)) * txn_cost
            continue
        eq_daily[t] = curr_w @ row
        grown = curr_w * (1 + row); g_sum = grown.sum()
        curr_w = grown / g_sum if g_sum != 0 else eq_w.copy()
        if t + 1 < n_days and dates[t + 1] in eq_rebal_set:
            turnover = np.sum(np.abs(eq_w - curr_w)) / 2.0
            eq_daily[t] -= turnover * txn_cost; curr_w = eq_w.copy()
    eq_ret = pd.Series(eq_daily, index=returns.index)
    bench, bench_failed = benchmark_series(benchmarks, fetch_days, data.index)

    results = OrderedDict()
    window_cache = {}  # shared cache: (window_name, data_len) -> best_weights
    approach_list = list(PORTFOLIO_APPROACHES.items())
    for i, (name, approach) in enumerate(approach_list):
        if progress_bar: progress_bar.progress((i + 1) / len(approach_list), text=f'Walk-forward: {name}')
        try:
            wf = _walk_forward_single(returns, approach, score_type, rebal_months,
                                      n_portfolios, max_weight, min_weight, txn_cost, allow_short,
                                      max_vol=max_vol, min_ann_ret=min_ann_ret,
                                      window_cache=window_cache, min_pos=min_pos, round_step=round_step,
                                      max_pos=max_pos, weighting=weighting, screen=screen)
            if wf is not None:
                metrics = _calc_oos_metrics(wf['oos_returns'])
                if metrics is not None:
                    metrics['n_rebalances'] = len(wf['weight_history'])
                    results[name] = {'wf': wf, 'metrics': metrics}
        except Exception as e:
            logger.warning(f"Walk-forward failed for {name}: {e}")

    if not results: return None
    logger.info(f"Window cache: {len(window_cache)} unique optimizations (vs ~{sum(len(a['windows']) * 30 for _, a in approach_list)} without cache)")
    # Store per-approach EW metrics (aligned to each approach's OOS start)
    for name, r in results.items():
        oos_start = r['wf']['oos_returns'].index[0]
        eq_aligned = eq_ret.loc[eq_ret.index >= oos_start]
        r['eq_returns'] = eq_aligned
        r['eq_metrics'] = _calc_oos_metrics(eq_aligned)
        r['eq_n_rebals'] = sum(1 for d in eq_rebal_set if d >= oos_start)
        r['bench'] = _bench_metrics(bench, oos_start)
    # Every approach burns a different warm-up, so they do NOT share a window:
    # 24mo Avg scores two years less than 3mo on the same basket. The discount
    # inside composite_ranks is what keeps that comparison honest.
    composite_ranks([r['metrics'] for r in results.values()])
    return {'results': results, 'symbols': valid, 'returns': returns,
            'eq_returns': eq_ret, 'eq_rebal_set': eq_rebal_set,
            'bench_symbols': [b[0] for b in bench], 'bench_failed': bench_failed,
            'score_type': score_type, 'rebal_months': rebal_months, 'txn_cost': txn_cost}

# =============================================================================
# FULL SAMPLE (IN-SAMPLE) OPTIMIZATION
# =============================================================================

def run_fullsample(symbols, score_type='Win Rate', n_portfolios=10000,
                   fetch_days=1800, max_weight=0.50, min_weight=0.0,
                   txn_cost=0.001, allow_short=False, progress_bar=None,
                   max_vol=None, min_ann_ret=None, rebal_months=3, benchmarks=None, min_pos=0.0,
                   round_step=0.0, min_history_days=0, max_pos=0, universe='common',
                   weighting=WEIGHTING_OPTIMIZED):
    """Run MC optimization on full dataset — no walk-forward split.
    Tests all PORTFOLIO_APPROACHES lookback windows that fit in the data,
    returns weights + in-sample backtest for each."""
    # Always the shared window, whatever the caller asked for. Full sample fits
    # ONE weight vector to the whole history and scores it on that same history;
    # a symbol that did not exist for half of it cannot carry a constant weight
    # through the half it was missing. As listed is a walk-forward idea -- the
    # universe grows because the test moves forward in time -- and there is no
    # honest way to express it here. The UI says so rather than offering it.
    data, valid = fetch_symbol_history(tuple(symbols), days=fetch_days,
                                       min_history_days=min_history_days)
    if data is None or len(valid) < 2: return None
    returns = data.pct_change().dropna(); n_assets = len(valid)

    # Equal weight benchmark
    eq_w = np.ones(n_assets) / n_assets
    ret_arr = returns.values; n_days = len(ret_arr)
    eq_daily = np.zeros(n_days); curr_w = eq_w.copy()
    dates = returns.index

    # Build rebalance set for EW
    if rebal_months == -1:
        eq_rebal_set = set()
    elif rebal_months == 0:
        eq_rebal_set = set(); seen_weeks = set()
        for d in dates:
            yw = (d.year, d.isocalendar()[1])
            if yw not in seen_weeks: seen_weeks.add(yw); eq_rebal_set.add(d)
    else:
        if rebal_months == 1: rebal_month_set = set(range(1, 13))
        elif rebal_months == 3: rebal_month_set = {1, 4, 7, 10}
        elif rebal_months == 6: rebal_month_set = {1, 7}
        else: rebal_month_set = {1}
        eq_rebal_set = set(); seen = set()
        for d in dates:
            ym = (d.year, d.month)
            if ym not in seen and d.month in rebal_month_set: seen.add(ym); eq_rebal_set.add(d)

    for t in range(n_days):
        eq_daily[t] = curr_w @ ret_arr[t]
        grown = curr_w * (1 + ret_arr[t]); g_sum = grown.sum()
        curr_w = grown / g_sum if g_sum != 0 else eq_w.copy()
        if t + 1 < n_days and dates[t + 1] in eq_rebal_set:
            turnover = np.sum(np.abs(eq_w - curr_w)) / 2.0
            eq_daily[t] -= turnover * txn_cost; curr_w = eq_w.copy()
    eq_ret = pd.Series(eq_daily, index=returns.index)
    eq_metrics = _calc_oos_metrics(eq_ret)
    bench, bench_failed = benchmark_series(benchmarks, fetch_days, data.index)
    bench_full = _bench_metrics(bench)

    results = OrderedDict()
    window_cache = {}
    approach_list = list(PORTFOLIO_APPROACHES.items())

    for i, (name, approach) in enumerate(approach_list):
        if progress_bar:
            progress_bar.progress((i + 1) / len(approach_list), text=f'Full sample: {name}')
        try:
            min_data_needed = max(approach['windows'].values())
            if len(returns) < min_data_needed:
                continue

            # Optimize on ALL data
            opt_w = _optimize_at_rebalance(returns, approach, score_type, n_portfolios,
                                           max_weight, min_weight, allow_short,
                                           max_vol=max_vol, min_ann_ret=min_ann_ret,
                                           window_cache=window_cache, min_pos=min_pos,
                                           round_step=round_step, max_pos=max_pos,
                                           weighting=weighting)
            if opt_w is None:
                continue

            # In-sample backtest with these fixed weights (rebalanced per schedule)
            is_daily = np.zeros(n_days); curr_opt = opt_w.copy()
            for t in range(n_days):
                is_daily[t] = curr_opt @ ret_arr[t]
                grown = curr_opt * (1 + ret_arr[t]); g_sum = grown.sum()
                curr_opt = grown / g_sum if g_sum != 0 else opt_w.copy()
                if t + 1 < n_days and dates[t + 1] in eq_rebal_set:
                    turnover = np.sum(np.abs(opt_w - curr_opt)) / 2.0
                    is_daily[t] -= turnover * txn_cost; curr_opt = opt_w.copy()

            is_ret = pd.Series(is_daily, index=returns.index)
            metrics = _calc_oos_metrics(is_ret)
            if metrics is None:
                continue
            metrics['n_rebalances'] = sum(1 for d in eq_rebal_set if d in set(dates))

            results[name] = {
                'wf': {
                    'oos_returns': is_ret,
                    'current_weights': opt_w,
                    'weight_history': [{'date': dates[0], 'weights': opt_w.copy(),
                                        'oos_start': dates[0], 'oos_end': dates[-1],
                                        'oos_days': n_days}],
                    'last_rebalance': dates[0],
                },
                'metrics': metrics,
                'eq_returns': eq_ret,
                'eq_metrics': eq_metrics,
                'eq_n_rebals': len(eq_rebal_set),
                'bench': bench_full,
            }
        except Exception as e:
            logger.warning(f"Full sample failed for {name}: {e}")

    if not results: return None
    # Full sample scores every approach on the whole series, so the spans match
    # and the discount is 1.0 throughout -- this is here for the _score key.
    composite_ranks([r['metrics'] for r in results.values()])
    return {'results': results, 'symbols': valid, 'returns': returns,
            'eq_returns': eq_ret, 'eq_rebal_set': eq_rebal_set,
            'bench_symbols': [b[0] for b in bench], 'bench_failed': bench_failed,
            'score_type': score_type, 'rebal_months': rebal_months, 'txn_cost': txn_cost}

def sweep_configs(symbols, objectives, rebalances, period_days, n_sims, max_wt, min_wt,
                   txn_cost, allow_short, max_pos, min_hist_days,
                   progress=None, engine=None, universe='common',
                   min_pos=0.0, round_step=0.0, weighting=WEIGHTING_OPTIMIZED):
    """One walk-forward grid per (objective, rebalance). Each call sweeps the
    eleven lookbacks itself, so the third dimension comes free with the second.

    Rows carry the span they were scored over, because they genuinely differ:
    a weekly rebalance starts trading sooner than an annual one, so it banks a
    longer out-of-sample record on identical data.
    """
    engine = engine or run_walkforward_grid
    rows = []
    combos = [(o, r) for o in objectives for r in rebalances]
    for i, (obj, rebal_label) in enumerate(combos):
        if progress:
            progress.progress((i + 1) / len(combos), text=f'{obj} · {rebal_label}')
        try:
            grid = engine(
                symbols, score_type=obj, rebal_months=REBAL_OPTIONS[rebal_label],
                fetch_days=period_days, n_portfolios=n_sims,
                max_weight=max_wt, min_weight=min_wt, txn_cost=txn_cost,
                allow_short=allow_short, max_pos=max_pos,
                min_history_days=min_hist_days, universe=universe,
                min_pos=min_pos, round_step=round_step, weighting=weighting)
            if not grid or not grid['results']:
                continue
            # Each cell is judged by the objective it was optimised for -- that is
            # the question being asked. Ranking every cell on one metric would
            # just rediscover which objective most resembles that metric.
            name = best_approach(grid['results'], SCORE_TO_RANK.get(obj, 'win_rate'))
            m = dict(grid['results'][name]['metrics'])
            m['objective'] = obj
            m['rebal'] = rebal_label
            m['lookback'] = name
            m['weights'] = grid['results'][name]['wf']['current_weights']
            m['symbols'] = grid['symbols']
            rows.append(m)
        except Exception as e:
            logger.warning(f'sweep {obj}/{rebal_label}: {e}')
    if progress:
        progress.empty()
    return rows



# =============================================================================
# DISPLAY: RANKING TABLE
# =============================================================================

def _fc(v, fmt='f2', neg_is_bad=True):
    if fmt == 'pct': s = f"{v*100:.1f}%"
    elif fmt == 'f3': s = f"{v:.3f}"
    else: s = f"{v:.2f}"
    if neg_is_bad: c = C_POS if v > 0 else (C_NEG if v < 0 else C_TXT)
    else: c = C_NEG
    return f"<span style='color:{c}'>{s}</span>"


# The 12 metric columns between 'Approach' and 'OOS': (key, format, neg_is_bad).
# ROA and ER sit beside Sortino in the same order SPREADS prints them.
_RANK_COLS = [('win_rate','pct',True), ('sharpe','f2',True), ('sortino','f2',True),
              ('roa','f2',True), ('er','f2',True),
              ('mar','f2',True), ('r2','f3',True), ('total_ret','pct',True),
              ('ann_ret','pct',True), ('ann_vol','pct',False), ('max_dd','pct',False),
              ('ytd','pct',True)]


def _tint(hex_color, alpha):
    """'#c084fc' -> 'rgba(192,132,252,0.07)' for row backgrounds."""
    h = hex_color.lstrip('#')
    return f"rgba({int(h[:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{alpha})"


def _compare_row(label, m, rebals, bg, label_color=C_TXT, rule=C_EW):
    """A non-ranked comparison row (equal weight, benchmark) under the ranking."""
    h = f"<tr><td colspan='17' style='border-bottom:1px solid {rule};padding:0;height:0'></td></tr>"
    h += f"<tr style='background:{bg}'>"
    h += f"<td style='{TD}color:{C_MUTE}'>&mdash;</td>"
    h += f"<td style='{TD}color:{label_color};font-weight:700'>{label}</td>"
    # Equal weight and the benchmarks are comparisons, not candidates: they were
    # never in the ranked set, so they have no composite to show.
    h += f"<td style='{TD}text-align:right;color:{C_MUTE}'>&mdash;</td>"
    for key, fmt, nib in _RANK_COLS:
        fw = 'font-weight:700;' if key == 'win_rate' else ('font-weight:600;' if key == 'total_ret' else '')
        h += f"<td style='{TD}text-align:right;{fw}'>{_fc(m[key], fmt, nib)}</td>"
    h += f"<td style='{TD}text-align:right;color:{C_TXT2}'>{m['oos_years']}y</td>"
    h += f"<td style='{TD}text-align:right;color:{C_TXT2}'>{rebals}</td></tr>"
    return h


def _delta_row(label, best_m, other_m, rule=C_EW):
    """Best approach minus a comparison row, green when the approach wins."""
    lower_is_better = {'ann_vol', 'max_dd'}
    h = f"<tr><td colspan='17' style='border-bottom:1px solid {rule};padding:0;height:0'></td></tr>"
    h += "<tr style='background:rgba(251,191,36,0.06)'>"
    h += f"<td style='{TD}color:{C_GOLD}'>&Delta;</td>"
    h += f"<td style='{TD}color:{C_GOLD};font-weight:600'>{label}</td>"
    h += f"<td style='{TD}text-align:right;color:{C_MUTE}'>&mdash;</td>"
    for key, fmt, _nib in _RANK_COLS:
        bv = best_m[key]; ev = other_m[key]; d = bv - ev
        good = abs(bv) < abs(ev) if key in lower_is_better else d > 0
        c = '#4ade80' if good else '#fb7185'; sign = '+' if d > 0 else ''
        if fmt == 'pct': ds = f"{sign}{d*100:.1f}%"
        elif fmt == 'f3': ds = f"{sign}{d:.3f}"
        else: ds = f"{sign}{d:.2f}"
        if abs(d) < 1e-6: ds = "&mdash;"; c = C_MUTE
        h += f"<td style='{TD}text-align:right;color:{c};font-weight:600'>{ds}</td>"
    h += f"<td style='{TD}text-align:right;color:{C_MUTE}'>&mdash;</td>"
    h += f"<td style='{TD}text-align:right;color:{C_MUTE}'>&mdash;</td></tr>"
    return h


def render_ranking_table(grid, rank_by='win_rate'):
    results = grid['results']
    items = [(name, r['metrics'], r) for name, r in results.items()]
    # The approaches do not share a window -- 24mo Avg burns two more years of
    # warm-up than 3mo -- so the order is taken on the length-adjusted value of
    # whichever metric was chosen, not on the raw one the columns print.
    adj = _spread_length_adjusted(_with_span([m for _n, m, _r in items]), rank_by)
    sign = -1.0 if rank_by in LOWER_IS_BETTER else 1.0
    items = [items[i] for i in sorted(range(len(items)), key=lambda i: -sign * adj[i])]
    best_name = items[0][0] if items else None

    html = f"<div style='overflow-x:auto;border:1px solid {C_BORDER};border-radius:6px'><table style='border-collapse:collapse;font-family:{FONTS};font-size:11px;width:100%;line-height:1.3'>"
    html += "<thead><tr>"
    for label, align in [('#','left'),('Approach','left'),('Score','right'),('Win%','right'),('Sharpe','right'),
                          ('Sortino','right'),('ROA','right'),('ER','right'),
                          ('MAR','right'),('R²','right'),('Total','right'),
                          ('Ann Ret','right'),('Vol','right'),('MaxDD','right'),('YTD','right'),
                          ('OOS','right'),('Rebals','right')]:
        html += f"<th style='{TH}text-align:{align}'>{label}</th>"
    html += "</tr></thead><tbody>"

    for rank, (name, m, _r) in enumerate(items, 1):
        is_best = name == best_name
        bg = 'rgba(96,165,250,0.08)' if is_best else 'transparent'
        badge = f" <span style='color:{C_GOLD};font-size:9px'>★</span>" if is_best else ""
        nc = C_GOLD if is_best else C_TXT; fw = '700' if is_best else '500'
        best_border = f'border-top:2px solid {C_GOLD};border-bottom:2px solid {C_GOLD};' if is_best else ''
        html += f"<tr style='background:{bg};{best_border}'>"
        html += f"<td style='{TD}color:{C_MUTE}'>{rank}</td>"
        html += f"<td style='{TD}color:{nc};font-weight:{fw}'>{name}{badge}</td>"
        # Composite: the mean rank across Sharpe, Sortino, ROA and ER after the
        # length discount, so 1.0 is the best an approach can score and the
        # column reads the opposite way to every other one.
        sc = m.get('_score')
        sc_c = C_MUTE if sc is None else (C_POS if sc <= 3 else (C_TXT2 if sc <= 6 else C_MUTE))
        sc_s = '&mdash;' if sc is None else f'{sc:.2f}'
        html += f"<td style='{TD}text-align:right;color:{sc_c};font-weight:600'>{sc_s}</td>"
        html += f"<td style='{TD}text-align:right;font-weight:700'>{_fc(m['win_rate'],'pct')}</td>"
        html += f"<td style='{TD}text-align:right'>{_fc(m['sharpe'])}</td>"
        html += f"<td style='{TD}text-align:right'>{_fc(m['sortino'])}</td>"
        html += f"<td style='{TD}text-align:right'>{_fc(m['roa'])}</td>"
        html += f"<td style='{TD}text-align:right'>{_fc(m['er'])}</td>"
        html += f"<td style='{TD}text-align:right'>{_fc(m['mar'])}</td>"
        html += f"<td style='{TD}text-align:right'>{_fc(m['r2'],'f3')}</td>"
        html += f"<td style='{TD}text-align:right;font-weight:600'>{_fc(m['total_ret'],'pct')}</td>"
        html += f"<td style='{TD}text-align:right'>{_fc(m['ann_ret'],'pct')}</td>"
        html += f"<td style='{TD}text-align:right'>{_fc(m['ann_vol'],'pct',False)}</td>"
        html += f"<td style='{TD}text-align:right'>{_fc(m['max_dd'],'pct',False)}</td>"
        html += f"<td style='{TD}text-align:right'>{_fc(m['ytd'],'pct')}</td>"
        html += f"<td style='{TD}text-align:right;color:{C_TXT2}'>{m['oos_years']}y</td>"
        html += f"<td style='{TD}text-align:right;color:{C_TXT2}'>{m['n_rebalances']}</td>"
        html += "</tr>"

    # Comparison rows — aligned to the best approach's OOS period
    if best_name and items:
        best_r = items[0][2]; best_m = items[0][1]
        eq = best_r.get('eq_metrics')
        if eq:
            html += _compare_row('◆ Equal Weight (1/N)', eq, best_r.get('eq_n_rebals', '—'),
                                 'rgba(100,116,139,0.06)')
            html += _delta_row('★ vs Equal Weight', best_m, eq)
        for i, (bsym, _br, bm) in enumerate(best_r.get('bench') or []):
            bc = BENCH_COLORS[i % len(BENCH_COLORS)]
            html += _compare_row(f"◇ {bsym} <span style='font-size:9px;color:{C_MUTE}'>benchmark</span>",
                                 bm, '—', _tint(bc, 0.07), label_color=bc, rule=bc)
            html += _delta_row(f'★ vs {bsym}', best_m, bm, rule=bc)

    html += "</tbody></table></div>"
    st.markdown(html, unsafe_allow_html=True)
    sorted_names = [name for name, _, _ in items]
    return best_name, sorted_names

# =============================================================================
# DISPLAY: WEIGHTS TABLE
# =============================================================================

def render_weights_table(grid, approach_name):
    wf = grid['results'][approach_name]['wf']
    syms = grid['symbols']; w = wf['current_weights']
    n_assets = len(syms); eq_w = 1.0 / n_assets
    # Held names only. Max Pos, Min Pos % and Round % all work by sending weights
    # to exactly 0, so a 323-symbol universe capped at 20 names used to print the
    # book followed by 303 rows of "0.0%" -- the part you trade buried in the
    # part you do not. Ranked on |weight| so a short sorts by size, not sign.
    held = [i for i in np.argsort(-np.abs(w)) if abs(w[i]) > 5e-5]
    n_dropped = n_assets - len(held)
    sorted_idx = held if held else list(np.argsort(-w))

    html = f"<div style='overflow-x:auto;border:1px solid {C_BORDER};border-radius:6px'><table style='border-collapse:collapse;font-family:{FONTS};font-size:11px;width:100%;line-height:1.3'>"
    html += f"<thead><tr><th style='{TH}text-align:left'>Asset</th><th style='{TH}text-align:left;width:60px'>Ticker</th>"
    html += f"<th style='{TH}text-align:right'>Weight</th><th style='{TH}text-align:right'>vs EW</th>"
    html += f"<th style='{TH}text-align:left;width:140px'>Allocation</th></tr></thead><tbody>"

    for i in sorted_idx:
        sym = syms[i]; sn = _short(sym); wi = w[i]; delta = wi - eq_w
        if wi < 0: wc = C_NEG
        elif wi > 0.20: wc = C_POS
        elif wi > 0.05: wc = C_TXT
        else: wc = C_MUTE
        dc = '#4ade80' if delta > 0.01 else ('#fb7185' if delta < -0.01 else C_MUTE)
        ds = '+' if delta > 0 else ''
        bar_pct = min(abs(wi) / 0.50 * 100, 100)
        bar_color = C_NEG if wi < 0 else C_POS
        bar = (f"<div style='background:{C_BORDER};border-radius:2px;height:10px;width:100%'>"
               f"<div style='background:{bar_color};border-radius:2px;height:10px;width:{bar_pct:.0f}%'></div></div>")
        html += f"<tr><td style='{TD}color:{wc};font-weight:600'>{sn}</td>"
        html += f"<td style='{TD}color:{C_MUTE};font-size:10px'>{sym}</td>"
        html += f"<td style='{TD}text-align:right;color:{wc};font-weight:700;font-size:12px'>{wi*100:.1f}%</td>"
        html += f"<td style='{TD}text-align:right;color:{dc};font-size:10px'>{ds}{delta*100:.1f}%</td>"
        html += f"<td style='{TD}'>{bar}</td></tr>"

    html += f"<tr style='border-top:2px solid {C_BORDER}'>"
    html += f"<td style='{TD}color:{C_TXT};font-weight:700'>TOTAL</td><td style='{TD}'></td>"
    html += f"<td style='{TD}text-align:right;color:{C_TXT};font-weight:700'>{np.sum(w)*100:.1f}%</td>"
    tail = (f"{len(held)} of {n_assets} names held · {n_dropped} at 0% not shown · "
            if n_dropped else '')
    html += f"<td colspan='2' style='{TD}color:{C_MUTE};font-size:10px'>{tail}Optimized on all data through today</td></tr>"
    html += "</tbody></table></div>"
    st.markdown(html, unsafe_allow_html=True)

# =============================================================================
# DISPLAY: OOS EQUITY CHART
# =============================================================================

def render_oos_chart(grid, approach_name):
    # Read theme fresh from session state every time
    theme_name = st.session_state.get('theme', 'Dark')
    theme = THEMES.get(theme_name, THEMES['Dark'])
    pos_c = theme['pos']; neg_c = theme['neg']

    wf = grid['results'][approach_name]['wf']
    oos = wf['oos_returns']; m = grid['results'][approach_name]['metrics']
    # Use per-approach aligned EW
    r_entry = grid['results'][approach_name]
    eq_aligned = r_entry['eq_returns'].loc[r_entry['eq_returns'].index >= oos.index[0]]
    eq_m = _calc_oos_metrics(eq_aligned) or r_entry['eq_metrics']
    opt_cum = np.cumprod(1 + oos.values); eq_cum = np.cumprod(1 + eq_aligned.values)
    opt_peak = np.maximum.accumulate(opt_cum); opt_dd = (opt_cum - opt_peak) / opt_peak
    eq_peak = np.maximum.accumulate(eq_cum); eq_dd = (eq_cum - eq_peak) / eq_peak
    opt_pct = (opt_cum[-1] - 1) * 100; eq_pct = (eq_cum[-1] - 1) * 100

    # Optional benchmark tickers — same OOS window as the approach
    benches = []
    for i, (b_sym, b_ret, b_m) in enumerate(r_entry.get('bench') or []):
        b_cum = np.cumprod(1 + b_ret.values)
        b_peak = np.maximum.accumulate(b_cum)
        benches.append({'sym': b_sym, 'idx': b_ret.index, 'cum': b_cum,
                        'dd': (b_cum - b_peak) / b_peak,
                        'color': BENCH_COLORS[i % len(BENCH_COLORS)],
                        'label': (f'{b_sym} ({(b_cum[-1]-1)*100:+.1f}%)  '
                                  f'Sharpe {b_m["sharpe"]:.2f} · Win {b_m["win_rate"]*100:.0f}%')})

    # Concise legend — just Sharpe + Win%
    opt_lbl = f'{approach_name} ({opt_pct:+.1f}%)  Sharpe {m["sharpe"]:.2f} · Win {m["win_rate"]*100:.0f}%'
    eq_lbl = f'Equal Weight ({eq_pct:+.1f}%)  Sharpe {eq_m["sharpe"]:.2f} · Win {eq_m["win_rate"]*100:.0f}%'

    fig = make_subplots(rows=2, cols=1, row_heights=[0.75, 0.25], shared_xaxes=True, vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=oos.index, y=opt_cum, mode='lines',
        line=dict(color=pos_c, width=2.2),
        name=opt_lbl, hovertemplate='WF: $%{y:.3f}<extra></extra>'), row=1, col=1)
    fig.add_trace(go.Scatter(x=eq_aligned.index, y=eq_cum, mode='lines',
        line=dict(color=_tint(C_EW, 0.80), width=1.2), name=eq_lbl,
        hovertemplate='EW: $%{y:.3f}<extra></extra>'), row=1, col=1)
    # Benchmarks sit behind the strategy: thin, translucent, no dashes
    for b in benches:
        fig.add_trace(go.Scatter(x=b['idx'], y=b['cum'], mode='lines',
            line=dict(color=_tint(b['color'], 0.78), width=1.2),
            name=b['label'],
            hovertemplate=f"{b['sym']}: $%{{y:.3f}}<extra></extra>"), row=1, col=1)
    for wh in wf['weight_history']:
        if wh['date'] >= oos.index[0]:
            fig.add_vline(x=wh['date'], line=dict(color=C_GOLD, width=0.5, dash='dot'), opacity=0.22, row=1, col=1)
    fig.add_hline(y=1.0, line=dict(color='#1f1f1f', width=0.8, dash='dash'), row=1, col=1)

    # End value annotations
    fig.add_annotation(x=oos.index[-1], y=opt_cum[-1], text=f'${opt_cum[-1]:.2f}',
        showarrow=False, xanchor='left', xshift=5,
        font=dict(size=11, color=pos_c, family=FONTS), row=1, col=1)
    fig.add_annotation(x=eq_aligned.index[-1], y=eq_cum[-1], text=f'${eq_cum[-1]:.2f}',
        showarrow=False, xanchor='left', xshift=5,
        font=dict(size=11, color=C_EW, family=FONTS), row=1, col=1)
    for i, b in enumerate(benches):
        # Nudge each label off the last one so close finishes stay readable
        fig.add_annotation(x=b['idx'][-1], y=b['cum'][-1], text=f"{b['sym']} ${b['cum'][-1]:.2f}",
            showarrow=False, xanchor='left', xshift=6, yshift=(-1) ** i * 8 * i,
            font=dict(size=9, color=_tint(b['color'], 0.9), family=FONTS), row=1, col=1)

    nr = neg_c.lstrip('#'); rv, gv, bv = int(nr[:2], 16), int(nr[2:4], 16), int(nr[4:6], 16)
    fig.add_trace(go.Scatter(x=oos.index, y=opt_dd * 100, mode='lines', fill='tozeroy',
        line=dict(color=neg_c, width=1), fillcolor=f'rgba({rv},{gv},{bv},0.2)',
        name='Drawdown', showlegend=False, hovertemplate='DD: %{y:.1f}%<extra></extra>'), row=2, col=1)
    fig.add_trace(go.Scatter(x=eq_aligned.index, y=eq_dd * 100, mode='lines',
        line=dict(color=_tint(C_EW, 0.45), width=0.9),
        name='EW Drawdown', showlegend=False, hovertemplate='EW DD: %{y:.1f}%<extra></extra>'), row=2, col=1)
    for b in benches:
        fig.add_trace(go.Scatter(x=b['idx'], y=b['dd'] * 100, mode='lines',
            line=dict(color=_tint(b['color'], 0.40), width=0.9),
            name=f"{b['sym']} Drawdown", showlegend=False,
            hovertemplate=f"{b['sym']} DD: %{{y:.1f}}%<extra></extra>"), row=2, col=1)

    # Title — use preset name instead of symbols
    params = st.session_state.get('port_params', st.session_state.get('port_fs_params', {}))
    title_name = params.get('preset_name', 'Portfolio').upper()
    is_fullsample = 'port_fs_result' in st.session_state and st.session_state.get('port_mode') == 'Monte Carlo (Full Sample)'
    mode_tag = 'FULL SAMPLE (IN-SAMPLE)' if is_fullsample else 'OOS WALK-FORWARD'
    tag_color = '#60a5fa' if is_fullsample else C_GOLD
    fig.update_layout(template='plotly_dark', height=400, margin=dict(l=55, r=55, t=35, b=25),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', showlegend=True,
        legend=dict(x=0.01, y=0.90, bgcolor='rgba(10,14,22,0.35)', tracegroupgap=2,
                    font=dict(size=11, color='#e2e8f0', family=FONTS), borderwidth=0),
        hovermode='x unified', font=dict(family=FONTS))
    fig.add_annotation(text=f"<b>{title_name}</b>  <span style='font-size:10px;color:{tag_color}'>{mode_tag}</span>",
        x=0.01, y=0.99, xref='paper', yref='paper', showarrow=False,
        font=dict(size=14, color='#ffffff', family=FONTS), xanchor='left', yanchor='top')
    # White axes
    fig.update_xaxes(gridcolor='#1f1f1f', linecolor='#2a2a2a', tickfont=dict(size=9, color='#94a3b8', family=FONTS))
    fig.update_yaxes(gridcolor='#1f1f1f', linecolor='#2a2a2a', tickfont=dict(size=9, color='#94a3b8', family=FONTS), side='right')
    fig.update_yaxes(tickprefix='$', tickformat='.2f', row=1, col=1)
    fig.update_yaxes(ticksuffix='%', row=2, col=1)
    st.plotly_chart(fig, width='stretch', config={'scrollZoom': True, 'displayModeBar': False, 'responsive': True})

# =============================================================================
# DISPLAY: MONTHLY RETURNS
# =============================================================================

def render_monthly_table(oos_returns):
    monthly = oos_returns.groupby([oos_returns.index.year, oos_returns.index.month]).apply(
        lambda x: float((1 + x).prod() - 1))
    years = sorted(monthly.index.get_level_values(0).unique())
    mlbl = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    html = f"<div style='overflow-x:auto;border:1px solid {C_BORDER};border-radius:6px'><table style='border-collapse:collapse;font-family:{FONTS};font-size:11px;width:100%;line-height:1.3'>"
    html += f"<thead><tr><th style='{TH}text-align:left'>Year</th>"
    for m in mlbl: html += f"<th style='{TH}text-align:right'>{m}</th>"
    html += f"<th style='{TH}text-align:right;font-weight:700'>YTD</th></tr></thead><tbody>"
    for yr in years:
        html += f"<tr><td style='{TD}color:{C_TXT};font-weight:700'>{yr}</td>"
        ytd = 1.0
        for mo in range(1, 13):
            if (yr, mo) in monthly.index:
                v = monthly[(yr, mo)]; ytd *= (1 + v)
                c = C_POS if v >= 0 else C_NEG
                html += f"<td style='{TD}text-align:right;color:{c}'>{v*100:.1f}%</td>"
            else: html += f"<td style='{TD}text-align:right;color:{C_MUTE}'>-</td>"
        yv = ytd - 1; yc = C_POS if yv >= 0 else C_NEG
        html += f"<td style='{TD}text-align:right;color:{yc};font-weight:700'>{yv*100:.1f}%</td></tr>"
    html += "</tbody></table></div>"
    st.markdown(html, unsafe_allow_html=True)

# =============================================================================
# SECTION HEADER HELPER
# =============================================================================

def _section(title, subtitle=''):
    sub = f"<span style='color:{C_MUTE};font-size:10px;margin-left:8px'>{subtitle}</span>" if subtitle else ""
    st.markdown(f"""<div style='margin-top:12px;padding:8px 12px;background:linear-gradient(90deg,{C_EW}12,{C_HDR});
        border-left:2px solid {C_EW};font-family:{FONTS};border-radius:4px'>
        <span style='color:#f8fafc;font-size:11px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase'>{title}</span>{sub}
    </div>""", unsafe_allow_html=True)

# =============================================================================
# MAIN RENDER — sub-tabs
# =============================================================================

def render_portfolio_tab(is_mobile):
    from portfolio_single import render_single_tab
    from portfolio_all import render_all_tab
    from portfolio_sweep import render_sweep_tab

    # Green underline on nested sub-tabs only
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
    </style>""", unsafe_allow_html=True)

    tab_single, tab_all, tab_sweep = st.tabs(['Single', 'All', 'Optimal'])

    with tab_single:
        render_single_tab(is_mobile)

    with tab_all:
        render_all_tab(is_mobile)

    with tab_sweep:
        render_sweep_tab(is_mobile)
