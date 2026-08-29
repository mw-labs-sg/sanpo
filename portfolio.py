import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from collections import OrderedDict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

from config import THEMES, SYMBOL_NAMES, FONTS

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

# Series colours (data-carrying) stay as they are; surfaces and text follow the
# neutral charcoal ramp in config.THEMES. C_MUTE was #475569, near-invisible on
# a #212121 page.
C_POS = '#60a5fa'; C_NEG = '#fb7185'; C_TXT = '#e2e8f0'; C_TXT2 = '#9fb2ca'
C_MUTE = '#6b7f9c'; C_BG = 'rgba(9,15,29,0.58)'; C_HDR = 'rgba(28,45,76,0.52)'
C_BORDER = '#28395a'
C_GOLD = '#fbbf24'; C_EW = '#6b7f9c'
TH = "padding:4px 8px;border-bottom:1px solid #28395a;color:#f8fafc;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.06em;"
TD = "padding:5px 8px;border-bottom:1px solid #28395a22;"

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
])

REBAL_OPTIONS = OrderedDict([
    ('No Rebalance', -1), ('Weekly', 0), ('Monthly', 1), ('Quarterly', 3), ('Semi-Annual', 6), ('Annual', 12),
])

PERIOD_OPTIONS = OrderedDict([
    ('1 Year', 365), ('2 Years', 730), ('5 Years', 1825),
    ('10 Years', 3650), ('Max', 9999),
])

SCORE_TO_RANK = {
    'Win Rate': 'win_rate', 'Sharpe': 'sharpe', 'Sortino': 'sortino',
    'MAR': 'mar', 'R²': 'r2', 'Composite': 'sharpe', 'Total Return': 'total_ret',
}

# =============================================================================
# DATA FETCHING
# =============================================================================

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_symbol_history(symbols_tuple, days=1800):
    symbols = list(symbols_tuple)
    if not symbols: return None, []
    start = (datetime.now() - pd.Timedelta(days=days)).strftime('%Y-%m-%d')
    data = pd.DataFrame(); valid = []
    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            hist = ticker.history(start=start)
            if not hist.empty and len(hist) >= 50:
                closes = hist['Close'].copy()
                closes.index = closes.index.tz_localize(None) if closes.index.tz else closes.index
                closes.index = closes.index.normalize()
                closes = closes.groupby(closes.index).last()
                data[sym] = closes; valid.append(sym)
        except Exception as e:
            logger.warning(f"[{sym}] portfolio fetch error: {e}")
    if len(valid) < 2: return None, valid
    data = data[valid].ffill().dropna()
    if len(data) < 50: return None, valid
    return data, valid

# =============================================================================
# MC OPTIMIZATION ENGINE
# =============================================================================

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
    ann_rets = np.mean(port_returns, axis=1) * 252
    ann_vols = np.std(port_returns, axis=1, ddof=1) * np.sqrt(252)

    # Apply constraints as soft penalties — penalize violations instead of hard filter
    penalty = np.zeros(n_portfolios)
    if max_vol is not None and max_vol > 0:
        vol_excess = np.maximum(ann_vols - max_vol, 0)
        penalty += vol_excess * 50  # strong penalty: 1% over = 0.5 penalty on 0-1 score
    if min_ann_ret is not None:
        ret_shortfall = np.maximum(min_ann_ret - ann_rets, 0)
        penalty += ret_shortfall * 50

    def _vec_downside_vol(pr):
        neg = np.minimum(pr, 0)
        n_neg = np.maximum(np.sum(pr < 0, axis=1).astype(float), 1)
        return np.sqrt(np.sum(neg**2, axis=1) / n_neg) * np.sqrt(252)

    def _vec_avg_dd(pr):
        cum = np.cumprod(1 + pr, axis=1)
        peak = np.maximum.accumulate(cum, axis=1)
        dd = (cum - peak) / peak
        neg_dd = np.where(dd < 0, dd, 0)
        n_neg = np.maximum(np.sum(dd < 0, axis=1).astype(float), 1)
        return np.sum(neg_dd, axis=1) / n_neg

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
    elif score_type == 'R²':
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
        sharpes = np.where(ann_vols > 0, ann_rets / ann_vols, 0)
        dv = _vec_downside_vol(port_returns)
        sortinos = np.where(dv > 0, ann_rets / dv, 0)
        win_rates = np.mean(port_returns > 0, axis=1)
        def _rank_pct(a):
            r = a.argsort().argsort().astype(float)
            return r / max(len(r) - 1, 1)
        scores = 0.4 * _rank_pct(sharpes) + 0.3 * _rank_pct(sortinos) + 0.3 * _rank_pct(win_rates)
    else:  # Sharpe
        scores = np.where(ann_vols > 0, ann_rets / ann_vols, 0)

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

def _optimize_at_rebalance(returns_df, approach, score_type, n_portfolios, mw, mnw=0.0, allow_short=False,
                           max_vol=None, min_ann_ret=None, window_cache=None):
    n_assets = returns_df.shape[1]; data_len = len(returns_df)
    window_weights_list = []; blend_wts = []
    for wname, wdays in approach['windows'].items():
        if data_len >= wdays:
            cache_key = (wname, data_len) if window_cache is not None else None
            if cache_key and cache_key in window_cache:
                best_w = window_cache[cache_key]
            else:
                w_ret = returns_df.iloc[-wdays:]
                best_w = _optimize_window_vectorized(w_ret.values, n_portfolios, n_assets, mw, score_type, mnw, allow_short,
                                                      max_vol=max_vol, min_ann_ret=min_ann_ret)
                if cache_key and window_cache is not None:
                    window_cache[cache_key] = best_w
            window_weights_list.append(best_w); blend_wts.append(approach['blend'][wname])
    if not window_weights_list: return None
    blend_wts = np.array(blend_wts); blend_wts /= blend_wts.sum()
    all_w = np.array(window_weights_list)
    opt_w = np.average(all_w, axis=0, weights=blend_wts)
    opt_w /= opt_w.sum()
    return opt_w


def _walk_forward_single(returns_df, approach, score_type, rebal_months,
                         n_portfolios=10000, max_weight=0.50, min_weight=0.0,
                         txn_cost=0.001, allow_short=False,
                         max_vol=None, min_ann_ret=None, window_cache=None):
    n_assets = returns_df.shape[1]; mw = max_weight; mnw = min_weight
    min_is_days = max(approach['windows'].values()); dates = returns_df.index

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
    if len(rebal_dates) < 2: return None

    oos_segments = []; weight_history = []
    prev_weights = np.ones(n_assets) / n_assets

    for i, (ri, rd) in enumerate(rebal_dates):
        is_data = returns_df.iloc[:ri + 1]
        opt_w = _optimize_at_rebalance(is_data, approach, score_type, n_portfolios, mw, mnw, allow_short,
                                        max_vol=max_vol, min_ann_ret=min_ann_ret, window_cache=window_cache)
        if opt_w is None: continue
        oos_start = ri + 1
        oos_end = rebal_dates[i + 1][0] if i + 1 < len(rebal_dates) else len(dates)
        if oos_start >= oos_end: continue
        oos_data = returns_df.iloc[oos_start:oos_end]
        port_oos = oos_data.values @ opt_w
        turnover = np.sum(np.abs(opt_w - prev_weights)) / 2.0
        if txn_cost > 0 and turnover > 0: port_oos[0] -= turnover * txn_cost
        prev_weights = opt_w.copy()
        oos_segments.append(pd.Series(port_oos, index=oos_data.index))
        weight_history.append({'date': rd, 'weights': opt_w.copy(),
            'oos_start': dates[oos_start], 'oos_end': dates[min(oos_end - 1, len(dates) - 1)],
            'oos_days': len(oos_data)})

    if not oos_segments or len(weight_history) < 2: return None
    current_w = _optimize_at_rebalance(returns_df, approach, score_type, n_portfolios, mw, mnw, allow_short,
                                        max_vol=max_vol, min_ann_ret=min_ann_ret, window_cache=window_cache)
    if current_w is None: current_w = weight_history[-1]['weights']
    full_oos = pd.concat(oos_segments)
    return {'oos_returns': full_oos, 'weight_history': weight_history,
            'current_weights': current_w, 'last_rebalance': weight_history[-1]['date']}


def _calc_oos_metrics(returns_series):
    r = returns_series.values; n = len(r)
    if n < 5: return None
    cum = np.cumprod(1 + r); total = float(cum[-1] - 1)
    ann_ret = float(np.mean(r) * 252); ann_vol = float(np.std(r, ddof=1) * np.sqrt(252))
    peak = np.maximum.accumulate(cum); dd = (cum - peak) / peak
    max_dd = float(np.min(dd)); avg_dd = float(np.mean(dd[dd < 0])) if np.any(dd < 0) else 0.0
    win_rate = float(np.sum(r > 0) / n)
    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else 0.0
    neg = np.minimum(r, 0); n_neg = max(np.sum(r < 0), 1)
    down_vol = float(np.sqrt(np.sum(neg**2) / n_neg) * np.sqrt(252))
    sortino = float(ann_ret / down_vol) if down_vol > 0 else 0.0
    mar = float(ann_ret / abs(avg_dd)) if avg_dd != 0 else 0.0
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
    oos_years = (idx[-1] - idx[0]).days / 365.25
    return {'total_ret': total, 'ann_ret': ann_ret, 'ann_vol': ann_vol,
            'max_dd': max_dd, 'avg_dd': avg_dd, 'win_rate': win_rate,
            'sharpe': sharpe, 'sortino': sortino, 'mar': mar, 'r2': r2,
            'ytd': ytd, 'mtd': mtd, 'n_days': n, 'oos_years': round(oos_years, 1)}

# =============================================================================
# GRID SEARCH
# =============================================================================

def run_walkforward_grid(symbols, score_type='Win Rate', rebal_months=3, n_portfolios=10000,
                         fetch_days=1800, max_weight=0.50, min_weight=0.0,
                         txn_cost=0.001, allow_short=False, progress_bar=None,
                         max_vol=None, min_ann_ret=None):
    data, valid = fetch_symbol_history(tuple(symbols), days=fetch_days)
    if data is None or len(valid) < 2: return None
    returns = data.pct_change().dropna(); n_assets = len(valid)

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
    curr_w = eq_w.copy()
    for t in range(n_days):
        eq_daily[t] = curr_w @ ret_arr[t]
        grown = curr_w * (1 + ret_arr[t]); g_sum = grown.sum()
        curr_w = grown / g_sum if g_sum != 0 else eq_w.copy()
        if t + 1 < n_days and dates[t + 1] in eq_rebal_set:
            turnover = np.sum(np.abs(eq_w - curr_w)) / 2.0
            eq_daily[t] -= turnover * txn_cost; curr_w = eq_w.copy()
    eq_ret = pd.Series(eq_daily, index=returns.index)

    results = OrderedDict()
    window_cache = {}  # shared cache: (window_name, data_len) -> best_weights
    approach_list = list(PORTFOLIO_APPROACHES.items())
    for i, (name, approach) in enumerate(approach_list):
        if progress_bar: progress_bar.progress((i + 1) / len(approach_list), text=f'Walk-forward: {name}')
        try:
            wf = _walk_forward_single(returns, approach, score_type, rebal_months,
                                      n_portfolios, max_weight, min_weight, txn_cost, allow_short,
                                      max_vol=max_vol, min_ann_ret=min_ann_ret,
                                      window_cache=window_cache)
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
    return {'results': results, 'symbols': valid, 'returns': returns,
            'eq_returns': eq_ret, 'eq_rebal_set': eq_rebal_set,
            'score_type': score_type, 'rebal_months': rebal_months, 'txn_cost': txn_cost}

# =============================================================================
# FULL SAMPLE (IN-SAMPLE) OPTIMIZATION
# =============================================================================

def run_fullsample(symbols, score_type='Win Rate', n_portfolios=10000,
                   fetch_days=1800, max_weight=0.50, min_weight=0.0,
                   txn_cost=0.001, allow_short=False, progress_bar=None,
                   max_vol=None, min_ann_ret=None, rebal_months=3):
    """Run MC optimization on full dataset — no walk-forward split.
    Tests all PORTFOLIO_APPROACHES lookback windows that fit in the data,
    returns weights + in-sample backtest for each."""
    data, valid = fetch_symbol_history(tuple(symbols), days=fetch_days)
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
                                           window_cache=window_cache)
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
            }
        except Exception as e:
            logger.warning(f"Full sample failed for {name}: {e}")

    if not results: return None
    return {'results': results, 'symbols': valid, 'returns': returns,
            'eq_returns': eq_ret, 'eq_rebal_set': eq_rebal_set,
            'score_type': score_type, 'rebal_months': rebal_months, 'txn_cost': txn_cost}

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


def render_ranking_table(grid, rank_by='win_rate'):
    results = grid['results']
    lower_better = {'ann_vol', 'max_dd', 'avg_dd'}
    items = [(name, r['metrics'], r) for name, r in results.items()]
    reverse = rank_by not in lower_better
    items.sort(key=lambda x: x[1].get(rank_by, 0), reverse=reverse)
    best_name = items[0][0] if items else None

    html = f"<div style='overflow-x:auto;border:1px solid {C_BORDER};border-radius:6px'><table style='border-collapse:collapse;font-family:{FONTS};font-size:11px;width:100%;line-height:1.3'>"
    html += "<thead><tr>"
    for label, align in [('#','left'),('Approach','left'),('Win%','right'),('Sharpe','right'),
                          ('Sortino','right'),('MAR','right'),('R²','right'),('Total','right'),
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
        html += f"<td style='{TD}text-align:right;font-weight:700'>{_fc(m['win_rate'],'pct')}</td>"
        html += f"<td style='{TD}text-align:right'>{_fc(m['sharpe'])}</td>"
        html += f"<td style='{TD}text-align:right'>{_fc(m['sortino'])}</td>"
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

    # Equal weight row — aligned to the best approach's OOS period
    if best_name and items:
        best_r = items[0][2]
        eq = best_r.get('eq_metrics')
        eq_rebals = best_r.get('eq_n_rebals', '—')
        if eq:
            html += f"<tr><td colspan='14' style='border-bottom:1px solid {C_EW};padding:0;height:0'></td></tr>"
            html += "<tr style='background:rgba(100,116,139,0.06)'>"
            html += f"<td style='{TD}color:{C_MUTE}'>—</td>"
            html += f"<td style='{TD}color:{C_TXT};font-weight:700'>◆ Equal Weight (1/N)</td>"
            html += f"<td style='{TD}text-align:right;font-weight:700'>{_fc(eq['win_rate'],'pct')}</td>"
            html += f"<td style='{TD}text-align:right'>{_fc(eq['sharpe'])}</td>"
            html += f"<td style='{TD}text-align:right'>{_fc(eq['sortino'])}</td>"
            html += f"<td style='{TD}text-align:right'>{_fc(eq['mar'])}</td>"
            html += f"<td style='{TD}text-align:right'>{_fc(eq['r2'],'f3')}</td>"
            html += f"<td style='{TD}text-align:right;font-weight:600'>{_fc(eq['total_ret'],'pct')}</td>"
            html += f"<td style='{TD}text-align:right'>{_fc(eq['ann_ret'],'pct')}</td>"
            html += f"<td style='{TD}text-align:right'>{_fc(eq['ann_vol'],'pct',False)}</td>"
            html += f"<td style='{TD}text-align:right'>{_fc(eq['max_dd'],'pct',False)}</td>"
            html += f"<td style='{TD}text-align:right'>{_fc(eq['ytd'],'pct')}</td>"
            html += f"<td style='{TD}text-align:right;color:{C_TXT2}'>{eq['oos_years']}y</td>"
            html += f"<td style='{TD}text-align:right;color:{C_TXT2}'>{eq_rebals}</td></tr>"

            # Delta row
            best_m = items[0][1]
            higher_better = {'win_rate','sharpe','sortino','mar','r2','total_ret','ann_ret','ytd'}
            html += f"<tr><td colspan='14' style='border-bottom:1px solid {C_EW};padding:0;height:0'></td></tr>"
            html += "<tr style='background:rgba(251,191,36,0.06)'>"
            html += f"<td style='{TD}color:{C_GOLD}'>Δ</td>"
            html += f"<td style='{TD}color:{C_GOLD};font-weight:600'>★ vs Equal Weight</td>"
            for key, fmt in [('win_rate','pct'),('sharpe','f2'),('sortino','f2'),('mar','f2'),
                             ('r2','f3'),('total_ret','pct'),('ann_ret','pct'),('ann_vol','pct'),
                             ('max_dd','pct'),('ytd','pct')]:
                bv = best_m[key]; ev = eq[key]; d = bv - ev
                if key in higher_better: good = d > 0
                elif key in {'ann_vol','max_dd'}: good = abs(bv) < abs(ev)
                else: good = d > 0
                c = '#4ade80' if good else '#fb7185'; sign = '+' if d > 0 else ''
                if fmt == 'pct': ds = f"{sign}{d*100:.1f}%"
                elif fmt == 'f3': ds = f"{sign}{d:.3f}"
                else: ds = f"{sign}{d:.2f}"
                if abs(d) < 1e-6: ds = "—"; c = C_MUTE
                html += f"<td style='{TD}text-align:right;color:{c};font-weight:600'>{ds}</td>"
            html += f"<td style='{TD}text-align:right;color:{C_MUTE}'>—</td>"
            html += f"<td style='{TD}text-align:right;color:{C_MUTE}'>—</td></tr>"

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
    sorted_idx = np.argsort(-w)

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
    html += f"<td colspan='2' style='{TD}color:{C_MUTE};font-size:10px'>Optimized on all data through today</td></tr>"
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

    # Concise legend — just Sharpe + Win%
    opt_lbl = f'{approach_name} ({opt_pct:+.1f}%)  Sharpe {m["sharpe"]:.2f} · Win {m["win_rate"]*100:.0f}%'
    eq_lbl = f'Equal Weight ({eq_pct:+.1f}%)  Sharpe {eq_m["sharpe"]:.2f} · Win {eq_m["win_rate"]*100:.0f}%'

    fig = make_subplots(rows=2, cols=1, row_heights=[0.75, 0.25], shared_xaxes=True, vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=oos.index, y=opt_cum, mode='lines', line=dict(color=pos_c, width=2),
        name=opt_lbl, hovertemplate='WF: $%{y:.3f}<extra></extra>'), row=1, col=1)
    fig.add_trace(go.Scatter(x=eq_aligned.index, y=eq_cum, mode='lines',
        line=dict(color=C_EW, width=1.5), name=eq_lbl,
        hovertemplate='EW: $%{y:.3f}<extra></extra>'), row=1, col=1)
    for wh in wf['weight_history']:
        if wh['date'] >= oos.index[0]:
            fig.add_vline(x=wh['date'], line=dict(color=C_GOLD, width=0.6, dash='dot'), opacity=0.4, row=1, col=1)
    fig.add_hline(y=1.0, line=dict(color='#1f1f1f', width=0.8, dash='dash'), row=1, col=1)

    # End value annotations
    fig.add_annotation(x=oos.index[-1], y=opt_cum[-1], text=f'${opt_cum[-1]:.2f}',
        showarrow=False, xanchor='left', xshift=5,
        font=dict(size=11, color=pos_c, family=FONTS), row=1, col=1)
    fig.add_annotation(x=eq_aligned.index[-1], y=eq_cum[-1], text=f'${eq_cum[-1]:.2f}',
        showarrow=False, xanchor='left', xshift=5,
        font=dict(size=11, color=C_EW, family=FONTS), row=1, col=1)

    nr = neg_c.lstrip('#'); rv, gv, bv = int(nr[:2], 16), int(nr[2:4], 16), int(nr[4:6], 16)
    fig.add_trace(go.Scatter(x=oos.index, y=opt_dd * 100, mode='lines', fill='tozeroy',
        line=dict(color=neg_c, width=1), fillcolor=f'rgba({rv},{gv},{bv},0.2)',
        name='Drawdown', showlegend=False, hovertemplate='DD: %{y:.1f}%<extra></extra>'), row=2, col=1)
    fig.add_trace(go.Scatter(x=eq_aligned.index, y=eq_dd * 100, mode='lines',
        line=dict(color=C_EW, width=1, dash='dot'),
        name='EW Drawdown', showlegend=False, hovertemplate='EW DD: %{y:.1f}%<extra></extra>'), row=2, col=1)

    # Title — use preset name instead of symbols
    params = st.session_state.get('port_params', st.session_state.get('port_fs_params', {}))
    title_name = params.get('preset_name', 'Portfolio').upper()
    is_fullsample = 'port_fs_result' in st.session_state and st.session_state.get('port_mode') == 'Monte Carlo (Full Sample)'
    mode_tag = 'FULL SAMPLE (IN-SAMPLE)' if is_fullsample else 'OOS WALK-FORWARD'
    tag_color = '#60a5fa' if is_fullsample else C_GOLD
    fig.update_layout(template='plotly_dark', height=400, margin=dict(l=55, r=55, t=35, b=25),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', showlegend=True,
        legend=dict(x=0.01, y=0.88, bgcolor='rgba(0,0,0,0)',
                    font=dict(size=12, color='#ffffff', family=FONTS), borderwidth=0),
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

    tab_single, tab_all = st.tabs(['Single', 'All'])

    with tab_single:
        render_single_tab(is_mobile)

    with tab_all:
        render_all_tab(is_mobile)
