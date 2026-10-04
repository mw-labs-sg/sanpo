"""
SANPO — Spread Portfolio (SPREADS > Portfolio)

A walk-forward portfolio whose holdings are spreads rather than outright legs.

The Sector tab scores every pair over one window and shows the best in hindsight.
That is a ranking, not a track record: the pair that topped the year was chosen
BY the year it is being judged on. This tab answers the honest version of the
question — if, every month, you had ranked the spreads on data you actually had
and held the top few until the next rebalance, what would you have made?

Mechanics, each rebalance:
  1. Score every pair of every ticked basket on the TRAIN window only.
  2. Take the top N on the chosen metric. The in-sample sign fixes the direction,
     so a pair that fell is held the other way round rather than discarded.
  3. Hold equal-weight until the next rebalance, and charge turnover both legs.
  4. Stitch those out-of-sample stretches end to end — that curve is the result.

The benchmark is every eligible pair held equal-weight all the way through, which
is what selection has to beat to have been worth doing.
"""

import logging
from collections import OrderedDict
from itertools import combinations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import portfolio
from config import FUTURES_GROUPS, THEMES, FONTS, SYMBOL_NAMES, clean_symbol
from spreads import (fetch_sector_spread_data, COMPOSITE_METRICS,
                     MIN_DD_PCT, MAX_RATIO, basket_picker,
                     rank_by_length_adjusted)

logger = logging.getLogger(__name__)

ANN = 252.0

HISTORY_OPTIONS = OrderedDict([
    ('2 Years', 730), ('3 Years', 1095), ('5 Years', 1825), ('Max', 9999),
])

TRAIN_OPTIONS = OrderedDict([
    ('3 Months', 63), ('6 Months', 126), ('1 Year', 252), ('2 Years', 504),
])

REBAL_OPTIONS = OrderedDict([
    ('Weekly', 5), ('Monthly', 21), ('Quarterly', 63), ('Semi-Annual', 126),
])

# How the field is narrowed. 'Best of each basket' is the Scan All rule applied
# at every rebalance: each basket puts forward its own winner, so a basket of
# 465 pairs and one of 6 each get a seat instead of the big basket crowding the
# board. 'Best overall' ranks every pair against every other, ignoring baskets.
# Spreads are dollar-neutral pairs; Long only holds the outright leg. The
# difference is not just the candidate set: a spread with a negative training
# mean is the same trade reversed, so it is flipped and kept, while a single
# name with a negative mean is simply a bad holding and has to rank badly.
TRADE_OPTIONS = ['Spreads', 'Long only', 'Short only']

# How the held candidates are sized against each other. Equal dollar weight is
# not equal risk: one spread per basket across Futures, Crypto, Singapore, STI
# and US Sectors ran 10.3% to 33.9% annualised vol, and the crypto leg alone
# carried 41.7% of portfolio risk. Inverse vol flattens that to 15-25% each.
WEIGHT_OPTIONS = ['Equal', 'Inverse vol']

PICK_OPTIONS = ['Best of each basket', 'Best overall']

HOLD_OPTIONS = OrderedDict([
    ('All', None), ('Top 1', 1), ('Top 3', 3), ('Top 5', 5),
    ('Top 10', 10), ('Top 20', 20),
])

# Lower is better for all of them, because every one is scored as a rank.
SELECT_OPTIONS = ['Composite', 'Sharpe', 'Sortino', 'ROA', 'ER', 'Win Rate']

# Basket count above which the run is worth warning about. There is no limit on
# the universe: scoring is vectorised across all pairs at once, and every basket
# ticked -- about 8,800 pairs -- is ~4s and 79MB. What costs is the first fetch
# of each basket, and that is cached for half an hour.
BUSY_BASKETS = 10

# Share of the training window a pair must actually have traded to be eligible.
MIN_COVERAGE = 0.80


# =============================================================================
# CANDIDATE UNIVERSE
# =============================================================================

@st.cache_data(ttl=1800, show_spinner=False)
def _pair_returns(basket_names, history_days, trade='Spreads'):
    """Daily return series for every candidate of every ticked basket.

    'Spreads' gives every pair; 'Long only' gives every single name.

    Columns are 'basket||long||short'. Baskets are fetched separately and
    unioned, so a basket with a shorter history simply carries NaNs outside its
    own window; the walk-forward skips a pair whose window is not complete
    rather than truncating everyone else to the shortest one.
    """
    short = trade == 'Short only'
    series = {}
    for g in basket_names:
        try:
            data = fetch_sector_spread_data(g, history_days)
        except Exception as e:
            logger.warning(f"spread portfolio fetch failed for {g}: {e}")
            continue
        if data is None or len(data.columns) < 2:
            continue
        rets = data.pct_change().dropna()
        if trade in ('Long only', 'Short only'):
            # A short position is the same series with the sign flipped, so the
            # metrics downstream score what is actually held.
            for a in list(data.columns):
                series[f'{g}||{a}'] = -rets[a] if short else rets[a]
        else:
            for a, b in combinations(list(data.columns), 2):
                series[f'{g}||{a}||{b}'] = rets[a] - rets[b]
    if not series:
        return None
    df = pd.DataFrame(series).sort_index()

    # Calendar mismatch, not missing data: a 7-day basket prints Saturdays that
    # every equity pair is simply shut for. Keeping those rows would mark the
    # whole equity side untradeable for any window containing a weekend. A row
    # survives only if a STRICT majority of the ticked baskets traded on it --
    # at exactly two baskets, one crypto and one equity, 'at least half' let
    # every weekend through and left the equity side on 69% coverage. With a
    # single basket ticked there is nothing to reconcile.
    groups = pd.Index([c.split('||')[0] for c in df.columns])
    if groups.nunique() > 1:
        traded = pd.concat([df.loc[:, groups == g].notna().any(axis=1)
                            for g in groups.unique()], axis=1)
        df = df[traded.mean(axis=1) > 0.5]
    return df


# =============================================================================
# SCORING — vectorised across every pair at once
# =============================================================================

def _ranks_desc(v):
    """1 = highest value. Ties break by position, which is arbitrary but stable."""
    order = np.argsort(-v, kind='stable')
    out = np.empty(len(v), dtype=float)
    out[order] = np.arange(1, len(v) + 1, dtype=float)
    return out


def _window_stats(X):
    """Sharpe, Sortino, ROA, ER and Win% for every column of an already
    sign-corrected return window. Same definitions as the rest of SPREADS,
    computed column-wise instead of pair by pair.

    This is the vectorised twin of spreads.compute_sector_spreads, which is too
    slow to run over thousands of columns at every rebalance. The two are
    checked to agree to three decimals; change one and change the other.
    """
    mean = X.mean(0)
    sd = X.std(0, ddof=1)
    sharpe = np.divide(mean * ANN, sd * np.sqrt(ANN),
                       out=np.zeros_like(mean), where=sd > 0)

    neg = np.minimum(X, 0.0)
    dvol = np.sqrt((neg ** 2).mean(0)) * np.sqrt(ANN)
    sortino = np.divide(mean * ANN, dvol, out=np.zeros_like(mean), where=dvol > 0)

    win = (X > 0).mean(0) * 100.0

    # Curve anchored at 1.0 so an opening loss registers, as in _spread_curve.
    curve = np.vstack([np.ones((1, X.shape[1])), np.cumprod(1.0 + X, axis=0)])
    peak = np.maximum.accumulate(curve, axis=0)
    mdd = ((curve - peak) / peak).min(0) * 100.0
    total = (curve[-1] - 1.0) * 100.0

    # ROA refuses a drawdown smaller than one typical bar of the pair's own
    # movement: that is not a spread avoiding a hole, it is too few bars to
    # have had one.
    floor = np.maximum(MIN_DD_PCT, np.median(np.abs(X), axis=0) * 100.0)
    denom = np.where(np.abs(mdd) > 0, np.abs(mdd), 1.0)
    roa = np.where(np.abs(mdd) < floor, 0.0,
                   np.clip(total / denom, -MAX_RATIO, MAX_RATIO))

    d = np.diff(curve, axis=0)
    path = np.abs(d).sum(0)
    net = d.sum(0)
    er = np.where(path > 0, np.abs(net) / np.where(path > 0, path, 1.0), 0.0)
    er = np.where(net >= 0, er, -er)

    return {'Sharpe': sharpe, 'Sortino': sortino, 'ROA': roa, 'ER': er,
            'Win Rate': win, 'MDD': mdd, 'Total': total}


def _score_window(train, metric, allow_flip=True):
    """Returns (sign, score, stats). Lower score is better.

    A pair whose training mean is negative is the same spread the other way
    round, so the sign is recorded and every statistic is computed on the
    flipped series -- Sortino especially, whose denominator only looks at
    losing bars and is not simply negatable.

    ALLOW_FLIP is off for long only: there, a negative mean is not a trade to
    reverse, it is a holding to avoid, and flipping it would quietly turn the
    portfolio short.
    """
    sign = (np.where(train.mean(0) >= 0, 1.0, -1.0) if allow_flip
            else np.ones(train.shape[1]))
    X = train * sign
    stats = _window_stats(X)
    if metric == 'Composite':
        score = np.mean([_ranks_desc(stats[m]) for m in COMPOSITE_METRICS], axis=0)
    else:
        score = _ranks_desc(stats[metric])
    return sign, score, stats


# =============================================================================
# WALK-FORWARD
# =============================================================================

def _walk_forward(df, train, step, top_n, metric, cost_pct, per_basket=True,
                  allow_flip=True, weight='Equal'):
    """Out-of-sample returns, the rebalance log, and the always-on benchmark.

    PER_BASKET nominates one winner per basket before ranking, so each basket
    is represented once however many pairs it happens to contain. TOP_N then
    caps how many of those winners are held; None holds them all.
    """
    R = df.values.astype(float)
    dates = df.index
    n_bars, n_pairs = R.shape
    if n_bars <= train + 1:
        return None
    gnames, gcode = np.unique([c.split('||')[0] for c in df.columns], return_inverse=True)

    port = np.full(n_bars, np.nan)
    bench = np.full(n_bars, np.nan)
    # Per-bar weight and P&L split by basket: what was held, and what it earned.
    wt_by_g = np.zeros((n_bars, len(gnames)))
    pl_by_g = np.zeros((n_bars, len(gnames)))
    # Turnover belongs to no basket, so it gets its own line -- without it the
    # attribution silently fails to add up to the portfolio's own return.
    cost_by_bar = np.zeros(n_bars)
    prev_w = np.zeros(n_pairs)
    log = []
    cost = cost_pct / 100.0

    i = train
    while i < n_bars:
        end = min(i + step, n_bars)
        tr = R[i - train:i]
        ho = R[i:end]
        # A pair needs most of the training window and has to be tradeable on
        # the day it would be bought. Demanding a gapless window instead looked
        # tidy and was useless: one US market holiday inside the lookback made
        # every equity pair ineligible, leaving a crypto-only portfolio.
        cov = (~np.isnan(tr)).mean(0)
        ok = (cov >= MIN_COVERAGE) & ~np.isnan(ho[0])
        idx = np.flatnonzero(ok)
        if idx.size == 0:
            i = end
            continue

        # Closed days are flat days, not missing data, in training as in holding.
        sign, score, stats = _score_window(np.nan_to_num(tr[:, idx]), metric, allow_flip)

        if per_basket:
            # One nominee per basket, then rank the nominees against each other.
            nominees = []
            for code in np.unique(gcode[idx]):
                sub = np.flatnonzero(gcode[idx] == code)
                nominees.append(sub[np.argmin(score[sub])])
            nominees = np.array(nominees)
            order = nominees[np.argsort(score[nominees], kind='stable')]
        else:
            order = np.argsort(score, kind='stable')
        if top_n:
            order = order[:min(top_n, order.size)]
        take = idx[order]
        take_sign = sign[order]

        # Sizing, measured on the training window only like everything else.
        if weight == 'Inverse vol' and len(take) > 1:
            sd = np.nan_to_num(tr[:, take]).std(0, ddof=1)
            inv = np.divide(1.0, sd, out=np.zeros_like(sd), where=sd > 0)
            w_rel = inv / inv.sum() if inv.sum() > 0 else np.ones(len(take)) / len(take)
        else:
            w_rel = np.ones(len(take)) / len(take)

        w = np.zeros(n_pairs)
        w[take] = take_sign * w_rel
        held = np.nan_to_num(ho[:, take] * take_sign)   # closed day = no P&L
        r = (held * w_rel).sum(1)

        # Turnover is charged on both legs: closing one spread and opening
        # another is four trades, not two.
        turnover = np.abs(w - prev_w).sum() / 2.0
        if cost > 0 and turnover > 0:
            drag = turnover * cost * 2.0
            r = r.copy()
            r[0] -= drag
            cost_by_bar[i] -= drag
        for col_i, j in enumerate(take):
            g = gcode[j]
            wt_by_g[i:end, g] += w_rel[col_i]
            pl_by_g[i:end, g] += held[:, col_i] * w_rel[col_i]
        port[i:end] = r
        prev_w = w

        # Benchmark: every eligible pair, equal weight, same direction rule.
        b_sign = (np.where(np.nan_to_num(tr[:, idx]).mean(0) >= 0, 1.0, -1.0)
                  if allow_flip else 1.0)
        bench[i:end] = np.nan_to_num(ho[:, idx] * b_sign).mean(1)

        log.append({
            'date': dates[i], 'n_candidates': int(idx.size), 'n_held': len(take),
            'ret': float(np.prod(1 + r) - 1), 'turnover': float(turnover),
            'picks': [(df.columns[j], float(s))
                      for j, s in zip(take, take_sign, strict=True)],
            'weights': [float(x) for x in w_rel],
            'n_baskets': int(np.unique(gcode[idx]).size),
            'stats': {m: stats[m][order] for m in
                      ('Sharpe', 'Sortino', 'ROA', 'ER', 'Win Rate')},
        })
        i = end

    mask = ~np.isnan(port)
    if mask.sum() < 5:
        return None
    keep = [k for k in range(len(gnames)) if wt_by_g[mask, k].any()]
    return {
        'returns': pd.Series(port[mask], index=dates[mask]),
        'bench': pd.Series(bench[mask], index=dates[mask]),
        'log': log,
        'n_pairs': n_pairs,
        'weights_by_basket': pd.DataFrame(wt_by_g[mask][:, keep], index=dates[mask],
                                          columns=[str(gnames[k]) for k in keep]),
        # Arithmetic contributions: they sum to the portfolio's simple return
        # each bar, which is what makes them addable across baskets.
        'pnl_by_basket': pd.DataFrame(
            np.column_stack([pl_by_g[mask][:, keep], cost_by_bar[mask]]),
            index=dates[mask],
            columns=[str(gnames[k]) for k in keep] + ['Costs']),
    }


# =============================================================================
# SWEEP — the same walk-forward across every Train x Rebalance x Trade
# =============================================================================

def _run_sweep(frames, top_n, metric, cost_pct, per_basket, progress=None,
               weight='Equal'):
    """Walk forward once per (trade, train, rebalance) and collect the scores.

    FRAMES is {trade_label: returns DataFrame}. The point is not to crown a
    winner: it is to see whether a good cell has good neighbours. One bright
    square surrounded by dark ones is a config that happened to fit the sample,
    and the honest read of a grid is its shape, not its maximum.
    """
    rows = []
    combos = [(t, tr, rb) for t in frames for tr in TRAIN_OPTIONS for rb in REBAL_OPTIONS]
    for k, (trade, tr_label, rb_label) in enumerate(combos):
        if progress:
            progress.progress((k + 1) / len(combos),
                              text=f'{trade} · train {tr_label} · {rb_label}')
        df = frames[trade]
        if df is None or df.empty:
            continue
        try:
            res = _walk_forward(df, TRAIN_OPTIONS[tr_label], REBAL_OPTIONS[rb_label],
                                top_n, metric, cost_pct, per_basket,
                                allow_flip=(trade == TRADE_OPTIONS[0]), weight=weight)
        except Exception as e:
            logger.warning(f"sweep {trade}/{tr_label}/{rb_label}: {e}")
            continue
        if res is None:
            continue
        m = portfolio._calc_oos_metrics(res['returns'])
        b = portfolio._calc_oos_metrics(res['bench'])
        if m is None:
            continue
        rows.append({
            'trade': trade, 'train': tr_label, 'rebal': rb_label,
            'sharpe': m['sharpe'], 'sortino': m['sortino'], 'roa': m['roa'],
            'er': m['er'], 'win': m['win_rate'], 'total': m['total_ret'],
            'mdd': m['max_dd'], 'days': m['n_days'], 'Days': m['span_days'],
            'edge': m['sharpe'] - (b['sharpe'] if b else 0.0),
            'rebals': len(res['log']),
        })
    return rows


def _render_sweep(rows, theme, is_mobile, params):
    pos_c = theme['pos']; neg_c = theme['neg']
    _bg3 = theme.get('bg3', '#0f172a'); _bdr = theme.get('border', '#1e293b')
    _txt = theme.get('text', '#e2e8f0'); _txt2 = theme.get('text2', '#94a3b8')
    _mut = theme.get('muted', '#475569')

    if not rows:
        _note('No configuration produced enough out-of-sample history to score.', theme)
        return

    # Bottom padding, or the tight global block gap lets the first heatmap's
    # title ride up over this line.
    st.markdown(f"<div style='font-size:10px;color:{_mut};font-family:{FONTS};"
                f"padding:8px 0 16px 0'>{len(rows)} configurations · "
                f"{params['pick']} · hold {params['hold']} by {params['metric']} · "
                f"{len(params['baskets'])} baskets · cost {params['cost']:.2f}%</div>",
                unsafe_allow_html=True)

    # ------------------------------------------------------------- heatmaps
    # Sharpe over train x rebalance, one panel per trade mode. Read the shape:
    # a plateau is a setting you can live with, a lone bright square is noise.
    trains = list(TRAIN_OPTIONS.keys())
    rebals = list(REBAL_OPTIONS.keys())
    modes = [t for t in TRADE_OPTIONS if any(r['trade'] == t for r in rows)]
    lookup = {(r['trade'], r['train'], r['rebal']): r['sharpe'] for r in rows}
    vals = [v for v in lookup.values()]
    lim = max(abs(min(vals)), abs(max(vals))) or 1.0

    cols = st.columns(len(modes)) if len(modes) > 1 and not is_mobile else [st] * len(modes)
    for mode, holder in zip(modes, cols, strict=True):
        z = [[lookup.get((mode, tr, rb)) for rb in rebals] for tr in trains]
        fig = go.Figure(go.Heatmap(
            z=z, x=rebals, y=trains, zmid=0, zmin=-lim, zmax=lim,
            colorscale=[[0, '#fb7185'], [0.5, '#0b1220'], [1, '#4ade80']],
            text=[[('' if v is None else f'{v:.2f}') for v in row] for row in z],
            texttemplate='%{text}', textfont=dict(size=10, family=FONTS),
            showscale=False, hovertemplate='%{y} · %{x}<br>Sharpe %{z:.2f}<extra></extra>'))
        fig.update_layout(
            title=dict(text=mode, font=dict(size=11, family=FONTS, color=_txt), x=0.01),
            height=230, margin=dict(l=8, r=8, t=30, b=8),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family=FONTS, size=9, color=theme.get('tick', '#888')),
            xaxis=dict(title='', side='bottom'), yaxis=dict(title='', autorange='reversed'))
        holder.plotly_chart(fig, use_container_width=True,
                            config={'displayModeBar': False})

    # ------------------------------------------------------------- table
    # A longer train burns a longer warm-up, so these cells do NOT share a
    # window: the 756-bar trains score years less out of sample than the 63-bar
    # ones. Ranking on the raw Sharpe handed that difference to the short trains.
    rows = rank_by_length_adjusted(rows, 'sharpe')
    th = (f"padding:4px 8px;border-bottom:1px solid {_bdr};color:#f8fafc;"
          f"font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.06em;")
    td = f"padding:5px 8px;border-bottom:1px solid {_bdr}22;"
    h = (f"<div style='overflow-x:auto;border:1px solid {_bdr};border-radius:6px'>"
         f"<table style='border-collapse:collapse;font-family:{FONTS};font-size:11px;"
         f"width:100%;line-height:1.3'><thead style='background:{_bg3}'><tr>"
         f"<th style='{th}text-align:left'>#</th>"
         f"<th style='{th}text-align:left'>TRADE</th>"
         f"<th style='{th}text-align:left'>TRAIN</th>"
         f"<th style='{th}text-align:left'>REBALANCE</th>"
         f"<th style='{th}text-align:right'>SHARPE</th>"
         f"<th style='{th}text-align:right'>vs EW</th>"
         f"<th style='{th}text-align:right'>SORTINO</th>"
         f"<th style='{th}text-align:right'>ROA</th>"
         f"<th style='{th}text-align:right'>ER</th>"
         f"<th style='{th}text-align:right'>WIN%</th>"
         f"<th style='{th}text-align:right'>TOT%</th>"
         f"<th style='{th}text-align:right'>MDD%</th>"
         f"<th style='{th}text-align:right'>OOS</th>"
         f"</tr></thead><tbody>")
    for i, r in enumerate(rows, 1):
        sh_c = pos_c if r['sharpe'] >= 0 else neg_c
        ed_c = pos_c if r['edge'] >= 0 else neg_c
        tot_c = pos_c if r['total'] >= 0 else neg_c
        bg = 'rgba(74,222,128,0.06)' if i <= 3 else 'transparent'
        h += (f"<tr style='background:{bg}'>"
              f"<td style='{td}color:{_mut}'>{i}</td>"
              f"<td style='{td}color:{_txt};font-weight:600'>{r['trade']}</td>"
              f"<td style='{td}color:{_txt2}'>{r['train']}</td>"
              f"<td style='{td}color:{_txt2}'>{r['rebal']}</td>"
              f"<td style='{td}text-align:right;color:{sh_c};font-weight:700'>{r['sharpe']:.2f}</td>"
              f"<td style='{td}text-align:right;color:{ed_c}'>{r['edge']:+.2f}</td>"
              f"<td style='{td}text-align:right;color:{_txt2}'>{r['sortino']:.2f}</td>"
              f"<td style='{td}text-align:right;color:{_txt2}'>{r['roa']:.2f}</td>"
              f"<td style='{td}text-align:right;color:{_txt2}'>{r['er']:.2f}</td>"
              f"<td style='{td}text-align:right;color:{_txt2}'>{r['win']*100:.0f}%</td>"
              f"<td style='{td}text-align:right;color:{tot_c};font-weight:600'>{r['total']*100:+.1f}%</td>"
              f"<td style='{td}text-align:right;color:{neg_c}'>{r['mdd']*100:.1f}%</td>"
              f"<td style='{td}text-align:right;color:{_mut}'>{r['days']}d</td></tr>")
    h += "</tbody></table></div>"
    st.markdown(h, unsafe_allow_html=True)

    st.markdown(
        f"<div style='font-size:10px;color:{_mut};font-family:{FONTS};padding:8px 0 0 0'>"
        f"<b style='color:{_txt2}'>vs EW</b> is the Sharpe over holding every eligible "
        f"candidate equal-weight — the part selection actually added. Ranking "
        f"{len(rows)} configurations and taking the top one is itself a fit to this "
        f"sample: trust a cell with good neighbours in the grid above, not a lone "
        f"bright square.</div>", unsafe_allow_html=True)


# =============================================================================
# RENDER
# =============================================================================

def _pair_label(col):
    """(basket, long leg, short leg). The short leg is None for a long-only name."""
    parts = col.split('||')
    g, a = parts[0], parts[1]
    b = SYMBOL_NAMES.get(parts[2], clean_symbol(parts[2])) if len(parts) > 2 else None
    return g, SYMBOL_NAMES.get(a, clean_symbol(a)), b


def _legs(col, sgn):
    """How the row reads: a flipped spread swaps its legs, a single name has none."""
    g, a, b = _pair_label(col)
    if b is None:
        return g, a, '&mdash;'
    return g, (a if sgn > 0 else b), (b if sgn > 0 else a)


def _note(msg, theme):
    _mut = theme.get('muted', '#475569')
    st.markdown(f"<div style='padding:12px;color:{_mut};font-size:11px;"
                f"font-family:{FONTS}'>{msg}</div>", unsafe_allow_html=True)


def render_spread_portfolio_tab(is_mobile):
    theme_name = st.session_state.get('theme', 'Dark')
    theme = THEMES.get(theme_name, THEMES['Dark'])
    _mut = theme.get('muted', '#475569')

    picked = basket_picker('sp', is_mobile, theme)

    long_only = st.session_state.get('sp_trade') == TRADE_OPTIONS[1]
    if long_only:
        n_cand = sum(len(FUTURES_GROUPS[g]) for g in picked)
        what = 'names'
    else:
        n_cand = sum(len(FUTURES_GROUPS[g]) * (len(FUTURES_GROUPS[g]) - 1) // 2
                     for g in picked)
        what = 'spreads'
    # Ranking them is cheap; it is the first fetch of each basket that costs, so
    # the warning counts baskets rather than candidates.
    busy = (f' — the first run has to fetch all {len(picked)} of them, which takes '
            f'a while; after that they are cached for 30 minutes'
            if len(picked) > BUSY_BASKETS else '')
    st.markdown(f"<div style='font-size:10px;color:{_mut};font-family:{FONTS};"
                f"padding:2px 0 8px 0'>{len(picked)} baskets · {n_cand:,} candidate "
                f"{what}{busy}</div>", unsafe_allow_html=True)

    # ------------------------------------------------------------- controls
    # Two rows: seven controls abreast clipped 'Best of each basket' and
    # 'Composite' to a stub. Window first, then how it picks.
    if is_mobile:
        c0, c1 = st.columns(2); c2, c3 = st.columns(2)
        c4, c5 = st.columns(2); c6, c7 = st.columns(2)
        c8, _c9 = st.columns(2)
    else:
        c0, c1, c2, c3 = st.columns(4)
        c4, c5, c6, c7, c8 = st.columns([3, 2, 3, 2, 2])

    with c0:
        trade_label = st.selectbox('Trade', TRADE_OPTIONS, index=0,
            key='sp_trade', help='Spreads holds dollar-neutral pairs, long one leg '
                                 'and short the other. Long only holds the outright '
                                 'names, so the candidates are symbols rather than '
                                 'pairs and nothing is ever held short.')
    with c1:
        hist_label = st.selectbox('History', list(HISTORY_OPTIONS.keys()), index=1,
            key='sp_hist', help='How much price history to pull. The first TRAIN '
                                'bars of it are consumed before the first trade.')
    with c2:
        train_label = st.selectbox('Train', list(TRAIN_OPTIONS.keys()), index=1,
            key='sp_train', help='The window each ranking is computed on. Only '
                                 'data before the rebalance date is used.')
    with c3:
        rebal_label = st.selectbox('Rebalance', list(REBAL_OPTIONS.keys()), index=1,
            key='sp_rebal', help='How often the basket is re-ranked and the '
                                 'holdings swapped.')
    with c4:
        pick_label = st.selectbox('Pick', PICK_OPTIONS, index=0,
            key='sp_pick', help='Best of each basket nominates one winner per '
                                'ticked basket, the way the All tab does, so a '
                                '465-pair basket cannot crowd out a 6-pair one. '
                                'Best overall ranks every pair against every '
                                'other and ignores which basket it came from.')
    with c5:
        hold_label = st.selectbox('Hold', list(HOLD_OPTIONS.keys()), index=0,
            key='sp_hold', help='How many of those to hold, equal weight, until '
                                'the next rebalance. All holds every nominee — '
                                'one spread per basket.')
    with c6:
        metric = st.selectbox('Select by', SELECT_OPTIONS, index=0,
            key='sp_metric', help='Which metric picks the holdings, measured on '
                                  'the training window. Composite is the average '
                                  'rank across Sharpe, Sortino, ROA and ER.')
    with c7:
        weight = st.selectbox('Weight', WEIGHT_OPTIONS, index=0, key='sp_weight',
            help='Equal splits the money evenly. Inverse vol splits the RISK '
                 'evenly instead, sizing each holding by 1/volatility measured '
                 'on the training window — without it a crypto spread next to a '
                 'Singapore one carries several times its share of the risk.')
    with c8:
        cost_txt = st.text_input('Cost %', value=st.session_state.get('sp_cost', '0.10'),
            key='sp_cost', help='Round-trip cost per leg, charged on turnover at '
                                'each rebalance. A spread is two legs.')

    n_combos = len(TRADE_OPTIONS) * len(TRAIN_OPTIONS) * len(REBAL_OPTIONS)
    if is_mobile:
        c_run, c_sweep = st.columns(2)
    else:
        c_run, c_sweep, _c_rsp = st.columns([2, 2, 4])
    with c_run:
        run = st.button('▶  Run Walk-Forward', key='sp_run', type='primary',
                        use_container_width=True)
    with c_sweep:
        sweep = st.button('⌗  Sweep Settings', key='sp_sweep_btn',
                          use_container_width=True,
                          help=f'Run the same walk-forward {n_combos} times — both '
                               f'Trade modes against every Train and Rebalance — and '
                               f'show which settings held up. Pick, Hold, Select by '
                               f'and Cost stay as set above.')

    try:
        cost_pct = max(0.0, float(cost_txt))
    except ValueError:
        cost_pct = 0.10

    # ------------------------------------------------------------- sweep
    if sweep:
        if not picked:
            _note('No baskets ticked — nothing to sweep.', theme)
            return
        frames = {}
        progress = st.progress(0.0, text='Fetching…')
        for t in TRADE_OPTIONS:
            frames[t] = _pair_returns(tuple(sorted(picked)),
                                      HISTORY_OPTIONS[hist_label], t)
        rows = _run_sweep(frames, HOLD_OPTIONS[hold_label], metric, cost_pct,
                          per_basket=(pick_label == PICK_OPTIONS[0]), progress=progress,
                          weight=weight)
        progress.empty()
        st.session_state.sp_sweep = {
            'rows': rows,
            'params': {'baskets': picked, 'hold': hold_label, 'metric': metric,
                       'cost': cost_pct, 'pick': pick_label, 'hist': hist_label},
        }
        st.session_state.sp_view = 'sweep'

    # ------------------------------------------------------------- single run
    if run:
        if not picked:
            _note('No baskets ticked — nothing to build a portfolio from.', theme)
            return
        with st.spinner(f'Scoring {n_cand:,} {what} across {len(picked)} baskets…'):
            df = _pair_returns(tuple(sorted(picked)), HISTORY_OPTIONS[hist_label],
                               trade_label)
            if df is None or df.empty:
                _note('No usable history for the ticked baskets.', theme)
                return
            res = _walk_forward(df, TRAIN_OPTIONS[train_label], REBAL_OPTIONS[rebal_label],
                                HOLD_OPTIONS[hold_label], metric, cost_pct,
                                per_basket=(pick_label == PICK_OPTIONS[0]),
                                allow_flip=(trade_label == TRADE_OPTIONS[0]),
                                weight=weight)
        if res is None:
            _note('Not enough history to walk forward: the training window eats '
                  'all of it. Shorten Train or lengthen History.', theme)
            return
        res['params'] = {'baskets': picked, 'hist': hist_label, 'train': train_label,
                         'rebal': rebal_label, 'hold': hold_label, 'metric': metric,
                         'cost': cost_pct, 'pick': pick_label, 'trade': trade_label,
                         'weight': weight}
        st.session_state.sp_result = res
        st.session_state.sp_view = 'run'

    # ------------------------------------------------------------- show
    view = st.session_state.get('sp_view')
    if view == 'sweep' and st.session_state.get('sp_sweep'):
        sw = st.session_state.sp_sweep
        _render_sweep(sw['rows'], theme, is_mobile, sw['params'])
    elif view == 'run' and st.session_state.get('sp_result'):
        _render_result(st.session_state.sp_result, theme, is_mobile)
    else:
        _note('Tick the baskets to draw from, then run. Every candidate of every '
              'ticked basket is ranked at each rebalance on the training window '
              'alone, and the best are held until the next one. Sweep Settings '
              'runs the same thing across every Train and Rebalance, both long '
              'only and spreads, so you can see which settings actually hold up.',
              theme)


BASKET_COLORS = ['#4ade80', '#60a5fa', '#f59e0b', '#c084fc', '#fb7185', '#2dd4bf',
                 '#facc15', '#a78bfa', '#34d399', '#f472b6', '#38bdf8', '#fb923c']


def _rgba(hex_color, alpha):
    """'#4ade80' -> 'rgba(74,222,128,0.33)'.

    Plotly rejects 8-digit hex, so the fill opacity cannot just be appended.
    """
    h = hex_color.lstrip('#')
    return f'rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})'


def _render_attribution(res, theme, is_mobile):
    """Where the money sat, and where it came from.

    The equity curve says what the portfolio did; these say which basket did it.
    Contributions are arithmetic, so they add up: every basket plus the cost
    line equals the portfolio's own simple return.
    """
    wts = res.get('weights_by_basket')
    pnl = res.get('pnl_by_basket')
    if wts is None or pnl is None or wts.empty or len(wts.columns) < 2:
        return

    _mut = theme.get('muted', '#475569')
    _txt2 = theme.get('text2', '#94a3b8')
    grid = theme.get('grid', '#1a2740')
    cols = list(wts.columns)
    colour = {c: BASKET_COLORS[i % len(BASKET_COLORS)] for i, c in enumerate(cols)}
    colour['Costs'] = _mut

    base = dict(
        height=250 if not is_mobile else 210,
        margin=dict(l=8, r=8, t=26, b=8),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor=theme.get('plot_bg', 'rgba(0,0,0,0)'),
        font=dict(family=FONTS, size=9, color=theme.get('tick', '#888')),
        legend=dict(orientation='h', y=1.16, x=0, font=dict(size=9)),
        xaxis=dict(gridcolor=grid, zeroline=False),
        hovermode='x unified',
    )

    c1, c2 = (st.container(), st.container()) if is_mobile else st.columns(2)

    with c1:
        st.markdown(f"<div style='font-size:10px;font-weight:600;letter-spacing:0.08em;"
                    f"text-transform:uppercase;color:#cbd5e1;font-family:{FONTS};"
                    f"padding:10px 0 0 0'>Weight by basket"
                    f"<span style='color:{_mut};font-weight:400;text-transform:none;"
                    f"letter-spacing:0'> — what you held, and when</span></div>",
                    unsafe_allow_html=True)
        fig = go.Figure()
        for c in cols:
            fig.add_trace(go.Scatter(
                x=wts.index, y=wts[c] * 100, name=c, mode='lines',
                stackgroup='w', line=dict(width=0.6, color=colour[c]),
                fillcolor=_rgba(colour[c], 0.33),
                hovertemplate='%{y:.0f}%<extra>' + c + '</extra>'))
        fig.update_layout(**base)
        fig.update_yaxes(gridcolor=grid, zeroline=False, ticksuffix='%', range=[0, 100])
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with c2:
        totals = (pnl.sum() * 100).sort_values(ascending=False)
        lead = totals.index[0]
        st.markdown(f"<div style='font-size:10px;font-weight:600;letter-spacing:0.08em;"
                    f"text-transform:uppercase;color:#cbd5e1;font-family:{FONTS};"
                    f"padding:10px 0 0 0'>Contribution"
                    f"<span style='color:{_mut};font-weight:400;text-transform:none;"
                    f"letter-spacing:0'> — {lead} led with "
                    f"{totals.iloc[0]:+.1f}%, costs {totals.get('Costs', 0):+.1f}%"
                    f"</span></div>", unsafe_allow_html=True)
        cum = pnl.cumsum() * 100
        fig = go.Figure()
        for c in cum.columns:
            fig.add_trace(go.Scatter(
                x=cum.index, y=cum[c], name=c, mode='lines',
                line=dict(width=1.4, color=colour.get(c, _txt2),
                          dash='dot' if c == 'Costs' else 'solid'),
                hovertemplate='%{y:+.1f}%<extra>' + c + '</extra>'))
        fig.add_hline(y=0, line=dict(color=grid, width=0.8))
        fig.update_layout(**base)
        fig.update_yaxes(gridcolor=grid, zeroline=False, ticksuffix='%')
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def _render_result(res, theme, is_mobile):
    pos_c = theme['pos']; neg_c = theme['neg']
    _bg3 = theme.get('bg3', '#0f172a'); _bdr = theme.get('border', '#1e293b')
    _txt = theme.get('text', '#e2e8f0'); _txt2 = theme.get('text2', '#94a3b8')
    _mut = theme.get('muted', '#475569')
    p = res.get('params', {})

    m = portfolio._calc_oos_metrics(res['returns'])
    bm = portfolio._calc_oos_metrics(res['bench'])
    if m is None:
        _note('Too few out-of-sample bars to score.', theme)
        return

    # ------------------------------------------------------------- summary
    def _strip(label, mm, colour):
        tot_c = pos_c if mm['total_ret'] >= 0 else neg_c
        return (
            f"<div style='margin-top:6px;padding:5px 10px;background:{_bg3};"
            f"font-family:{FONTS};border-radius:4px;font-size:10px;color:{_txt2};"
            f"display:flex;justify-content:space-between;flex-wrap:wrap;gap:4px'>"
            f"<span><b style='color:{colour}'>{label}</b>&nbsp;·&nbsp;"
            f"{mm['n_days']} OOS days · {mm['oos_years']}y</span>"
            f"<span>Win% <b style='color:{colour}'>{mm['win_rate']*100:.1f}%</b>"
            f"&nbsp;Sharpe <b style='color:{colour}'>{mm['sharpe']:.2f}</b>"
            f"&nbsp;Sortino <b style='color:{colour}'>{mm['sortino']:.2f}</b>"
            f"&nbsp;ROA <b style='color:{colour}'>{mm['roa']:.2f}</b>"
            f"&nbsp;ER <b style='color:{colour}'>{mm['er']:.2f}</b>"
            f"&nbsp;MAR <b style='color:{colour}'>{mm['mar']:.2f}</b>"
            f"&nbsp;Tot <b style='color:{tot_c}'>{mm['total_ret']*100:+.1f}%</b>"
            f"&nbsp;MDD <b style='color:{neg_c}'>{mm['max_dd']*100:.1f}%</b></span></div>")

    hdr = (f"{p.get('trade','')} · {p.get('pick','')} · hold {p.get('hold','')} "
           f"by {p.get('metric','')} · {p.get('weight','Equal')} weight · "
           f"{p.get('rebal','')} · train {p.get('train','')} · "
           f"{len(p.get('baskets', []))} baskets · {res['n_pairs']:,} candidates · "
           f"cost {p.get('cost',0):.2f}%")
    st.markdown(f"<div style='font-size:10px;color:{_mut};font-family:{FONTS};"
                f"padding:8px 0 0 0'>{hdr}</div>", unsafe_allow_html=True)
    st.markdown(_strip('Walk-Forward', m, pos_c), unsafe_allow_html=True)
    bench_label = ('All Names (equal weight)' if p.get('trade') == TRADE_OPTIONS[1]
                   else 'All Spreads (equal weight)')
    st.markdown(_strip(bench_label, bm, _txt2), unsafe_allow_html=True)

    # ------------------------------------------------------------- curve
    cum = 100 * (1 + res['returns']).cumprod()
    bcum = 100 * (1 + res['bench']).cumprod()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum.index, y=cum.values, mode='lines', name='Walk-Forward',
                             line=dict(color=pos_c, width=1.8)))
    fig.add_trace(go.Scatter(x=bcum.index, y=bcum.values, mode='lines', name='All candidates',
                             line=dict(color=_mut, width=1.2, dash='dot')))
    for entry in res['log'][1:]:
        fig.add_vline(x=entry['date'], line=dict(color='#f59e0b', width=0.5), opacity=0.25)
    fig.update_layout(
        height=300 if not is_mobile else 240,
        margin=dict(l=8, r=8, t=24, b=8),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor=theme.get('plot_bg', 'rgba(0,0,0,0)'),
        font=dict(family=FONTS, size=10, color=theme.get('tick', '#888')),
        legend=dict(orientation='h', y=1.12, x=0, font=dict(size=9)),
        xaxis=dict(gridcolor=theme.get('grid', '#1a2740'), zeroline=False),
        yaxis=dict(gridcolor=theme.get('grid', '#1a2740'), zeroline=False, title=''),
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    _render_attribution(res, theme, is_mobile)

    # ------------------------------------------------------------- holdings
    last = res['log'][-1] if res['log'] else None
    th = (f"padding:4px 8px;border-bottom:1px solid {_bdr};color:#f8fafc;"
          f"font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.06em;")
    td = f"padding:5px 8px;border-bottom:1px solid {_bdr}22;"

    if last:
        st.markdown(f"<div style='font-size:10px;font-weight:600;letter-spacing:0.08em;"
                    f"text-transform:uppercase;color:#cbd5e1;font-family:{FONTS};"
                    f"padding:10px 0 2px 0'>Held since {last['date']:%d %b %Y}"
                    f"<span style='color:{_mut};font-weight:400;text-transform:none;"
                    f"letter-spacing:0'> — picked from {last['n_candidates']:,} eligible "
                    f"candidates on the training window</span></div>", unsafe_allow_html=True)
        h = (f"<div style='overflow-x:auto;border:1px solid {_bdr};border-radius:6px'>"
             f"<table style='border-collapse:collapse;font-family:{FONTS};font-size:11px;"
             f"width:100%;line-height:1.3'><thead style='background:{_bg3}'><tr>"
             f"<th style='{th}text-align:left'>#</th>"
             f"<th style='{th}text-align:left'>BASKET</th>"
             f"<th style='{th}text-align:left'>LONG</th>"
             f"<th style='{th}text-align:left'>SHORT</th>"
             f"<th style='{th}text-align:right'>WT%</th>"
             f"<th style='{th}text-align:right'>SHARPE</th>"
             f"<th style='{th}text-align:right'>SORTINO</th>"
             f"<th style='{th}text-align:right'>ROA</th>"
             f"<th style='{th}text-align:right'>ER</th>"
             f"<th style='{th}text-align:right'>WIN%</th>"
             f"</tr></thead><tbody>")
        for k, (col, sgn) in enumerate(last['picks']):
            # A negative training sign means the pair is held the other way round.
            g, lng, sht = _legs(col, sgn)
            s = last['stats']
            wts = last.get('weights') or []
            wt = wts[k] * 100 if k < len(wts) else 100.0 / max(len(last['picks']), 1)
            h += (f"<tr><td style='{td}color:{_mut}'>{k+1}</td>"
                  f"<td style='{td}color:{_txt};font-weight:600'>{g}</td>"
                  f"<td style='{td}color:{pos_c};font-weight:600'>{lng}</td>"
                  f"<td style='{td}color:{theme['short']};font-weight:600'>{sht}</td>"
                  f"<td style='{td}text-align:right;color:{_txt2}'>{wt:.1f}%</td>"
                  f"<td style='{td}text-align:right;color:{pos_c};font-weight:700'>{s['Sharpe'][k]:.2f}</td>"
                  f"<td style='{td}text-align:right;color:{_txt2}'>{s['Sortino'][k]:.2f}</td>"
                  f"<td style='{td}text-align:right;color:{_txt2}'>{s['ROA'][k]:.1f}</td>"
                  f"<td style='{td}text-align:right;color:{_txt2}'>{s['ER'][k]:.2f}</td>"
                  f"<td style='{td}text-align:right;color:{_txt2}'>{s['Win Rate'][k]:.0f}%</td></tr>")
        h += "</tbody></table></div>"
        st.markdown(h, unsafe_allow_html=True)

    # ------------------------------------------------------------- rebalances
    st.markdown(f"<div style='font-size:10px;font-weight:600;letter-spacing:0.08em;"
                f"text-transform:uppercase;color:#cbd5e1;font-family:{FONTS};"
                f"padding:12px 0 2px 0'>Rebalances"
                f"<span style='color:{_mut};font-weight:400;text-transform:none;"
                f"letter-spacing:0'> — each row is one out-of-sample stretch</span></div>",
                unsafe_allow_html=True)
    h = (f"<div style='overflow-x:auto;border:1px solid {_bdr};border-radius:6px'>"
         f"<table style='border-collapse:collapse;font-family:{FONTS};font-size:11px;"
         f"width:100%;line-height:1.3'><thead style='background:{_bg3}'><tr>"
         f"<th style='{th}text-align:left'>FROM</th>"
         f"<th style='{th}text-align:right'>ELIGIBLE</th>"
         f"<th style='{th}text-align:right'>BASKETS</th>"
         f"<th style='{th}text-align:right'>HELD</th>"
         f"<th style='{th}text-align:right'>TURNOVER</th>"
         f"<th style='{th}text-align:right'>OOS RETURN</th>"
         f"<th style='{th}text-align:left'>TOP PICK</th>"
         f"</tr></thead><tbody>")
    for entry in reversed(res['log']):
        rc = pos_c if entry['ret'] >= 0 else neg_c
        col, sgn = entry['picks'][0]
        g, lng, sht = _legs(col, sgn)
        h += (f"<tr><td style='{td}color:{_txt}'>{entry['date']:%d %b %Y}</td>"
              f"<td style='{td}text-align:right;color:{_mut}'>{entry['n_candidates']:,}</td>"
              f"<td style='{td}text-align:right;color:{_mut}'>{entry.get('n_baskets','—')}</td>"
              f"<td style='{td}text-align:right;color:{_txt2}'>{entry['n_held']}</td>"
              f"<td style='{td}text-align:right;color:{_mut}'>{entry['turnover']*100:.0f}%</td>"
              f"<td style='{td}text-align:right;color:{rc};font-weight:600'>{entry['ret']*100:+.2f}%</td>"
              f"<td style='{td}color:{_txt2}'>{g}: <span style='color:{pos_c}'>{lng}</span>"
              f" / <span style='color:{theme['short']}'>{sht}</span></td></tr>")
    h += "</tbody></table></div>"
    st.markdown(h, unsafe_allow_html=True)
