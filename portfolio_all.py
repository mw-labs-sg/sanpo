import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

from config import FUTURES_GROUPS, THEMES, FONTS
from portfolio import (REBAL_OPTIONS, PERIOD_OPTIONS, SCORE_TO_RANK,
                       fetch_symbol_history, _calc_oos_metrics,
                       run_walkforward_grid, run_fullsample, _section)

logger = logging.getLogger(__name__)

# =============================================================================
# SORT CONFIG
# =============================================================================

SCAN_SORT_KEYS = {
    'Win Rate': ('win_rate', True),
    'Sharpe': ('sharpe', True),
    'Sortino': ('sortino', True),
    'MAR': ('mar', True),
    'R²': ('r2', True),
    'Total Return': ('total_ret', True),
}

# =============================================================================
# EW SCAN
# =============================================================================

def _compute_ew_returns(symbols, fetch_days, rebal_months, txn_cost):
    """Compute equal-weight returns with rebalancing + txn costs."""
    data, valid = fetch_symbol_history(tuple(symbols), days=fetch_days)
    if data is None or len(valid) < 2:
        return None, None
    returns = data.pct_change().dropna()
    n_assets = len(valid)
    eq_w = np.ones(n_assets) / n_assets

    # Rebalance schedule
    if rebal_months == 0:
        dates = returns.index; rebal_set = set(); seen_weeks = set()
        for d in dates:
            yw = (d.year, d.isocalendar()[1])
            if yw not in seen_weeks: seen_weeks.add(yw); rebal_set.add(d)
    else:
        if rebal_months == 1: rebal_month_set = set(range(1, 13))
        elif rebal_months == 3: rebal_month_set = {1, 4, 7, 10}
        elif rebal_months == 6: rebal_month_set = {1, 7}
        else: rebal_month_set = {1}
        dates = returns.index; rebal_set = set(); seen = set()
        for d in dates:
            ym = (d.year, d.month)
            if ym not in seen and d.month in rebal_month_set: seen.add(ym); rebal_set.add(d)

    ret_arr = returns.values; n_days = len(ret_arr)
    ew_daily = np.zeros(n_days); curr_w = eq_w.copy()
    for t in range(n_days):
        ew_daily[t] = curr_w @ ret_arr[t]
        grown = curr_w * (1 + ret_arr[t]); curr_w = grown / grown.sum()
        if t + 1 < n_days and dates[t + 1] in rebal_set:
            turnover = np.sum(np.abs(eq_w - curr_w)) / 2.0
            ew_daily[t] -= turnover * txn_cost; curr_w = eq_w.copy()

    ew_series = pd.Series(ew_daily, index=returns.index)
    return ew_series, valid


def _run_ew_scan(period_days, rebal_months, txn_cost, rank_by, progress_bar=None):
    """Scan all groups with equal-weight returns."""
    all_results = []
    groups = list(FUTURES_GROUPS.items())

    for i, (gname, syms) in enumerate(groups):
        if progress_bar:
            progress_bar.progress((i + 1) / len(groups), text=f'EW Scan: {gname}')
        if len(syms) < 2:
            continue
        try:
            ew_ret, valid = _compute_ew_returns(syms, period_days, rebal_months, txn_cost)
            if ew_ret is None or len(ew_ret) < 20:
                continue
            metrics = _calc_oos_metrics(ew_ret)
            if metrics is None:
                continue
            metrics['group'] = gname
            metrics['n_assets'] = len(valid)
            metrics['symbols'] = valid
            metrics['ew_returns'] = ew_ret
            all_results.append(metrics)
        except Exception as e:
            logger.warning(f"EW scan error for {gname}: {e}")

    if progress_bar:
        progress_bar.empty()

    return all_results


# =============================================================================
# MC SCAN
# =============================================================================

def _run_mc_scan(period_days, rebal_months, txn_cost, score_type, n_sims,
                 max_wt, min_wt, allow_short, max_vol, min_ann_ret,
                 rank_by, progress_bar=None):
    """Scan all groups with Monte Carlo walk-forward optimization."""
    all_results = []
    groups = list(FUTURES_GROUPS.items())

    for i, (gname, syms) in enumerate(groups):
        if progress_bar:
            progress_bar.progress((i + 1) / len(groups), text=f'Optimizing: {gname}')
        if len(syms) < 2:
            continue
        try:
            grid = run_walkforward_grid(
                syms, score_type=score_type, rebal_months=rebal_months,
                fetch_days=period_days, n_portfolios=n_sims,
                max_weight=max_wt, min_weight=min_wt,
                txn_cost=txn_cost, allow_short=allow_short,
                max_vol=max_vol, min_ann_ret=min_ann_ret)
            if not grid or not grid['results']:
                continue
            # Pick best approach by rank metric
            rank_metric = SCORE_TO_RANK.get(score_type, 'win_rate')
            best_name = max(grid['results'].keys(),
                           key=lambda k: grid['results'][k]['metrics'].get(rank_metric, 0))
            best = grid['results'][best_name]
            m = best['metrics']
            m['group'] = gname
            m['n_assets'] = len(grid['symbols'])
            m['symbols'] = grid['symbols']
            m['best_approach'] = best_name
            m['ew_returns'] = best['wf']['oos_returns']
            # Store current weights from best approach
            curr_w = best['wf'].get('current_weights', None)
            if curr_w is not None:
                m['weights'] = {s: float(w) for s, w in zip(grid['symbols'], curr_w)}
            else:
                m['weights'] = {s: 1.0/len(grid['symbols']) for s in grid['symbols']}
            all_results.append(m)
        except Exception as e:
            logger.warning(f"MC scan error for {gname}: {e}")

    if progress_bar:
        progress_bar.empty()

    return all_results


# =============================================================================
# FULL SAMPLE SCAN
# =============================================================================

def _run_fs_scan(period_days, rebal_months, txn_cost, score_type, n_sims,
                 max_wt, min_wt, allow_short, max_vol, min_ann_ret,
                 rank_by, progress_bar=None):
    """Scan all groups with Monte Carlo full-sample (in-sample) optimization."""
    all_results = []
    groups = list(FUTURES_GROUPS.items())

    for i, (gname, syms) in enumerate(groups):
        if progress_bar:
            progress_bar.progress((i + 1) / len(groups), text=f'Full sample: {gname}')
        if len(syms) < 2:
            continue
        try:
            grid = run_fullsample(
                syms, score_type=score_type, n_portfolios=n_sims,
                fetch_days=period_days, max_weight=max_wt, min_weight=min_wt,
                txn_cost=txn_cost, allow_short=allow_short,
                max_vol=max_vol, min_ann_ret=min_ann_ret,
                rebal_months=rebal_months)
            if not grid or not grid['results']:
                continue
            # Pick best approach by rank metric
            rank_metric = SCORE_TO_RANK.get(score_type, 'win_rate')
            best_name = max(grid['results'].keys(),
                           key=lambda k: grid['results'][k]['metrics'].get(rank_metric, 0))
            best = grid['results'][best_name]
            m = best['metrics']
            m['group'] = gname
            m['n_assets'] = len(grid['symbols'])
            m['symbols'] = grid['symbols']
            m['best_approach'] = best_name
            m['ew_returns'] = best['wf']['oos_returns']
            curr_w = best['wf'].get('current_weights', None)
            if curr_w is not None:
                m['weights'] = {s: float(w) for s, w in zip(grid['symbols'], curr_w)}
            else:
                m['weights'] = {s: 1.0/len(grid['symbols']) for s in grid['symbols']}
            all_results.append(m)
        except Exception as e:
            logger.warning(f"FS scan error for {gname}: {e}")

    if progress_bar:
        progress_bar.empty()

    return all_results


# =============================================================================
# MAIN RENDER
# =============================================================================

def render_all_tab(is_mobile):
    import portfolio
    theme_name = st.session_state.get('theme', 'Dark')
    theme = THEMES.get(theme_name, THEMES['Dark'])
    portfolio.C_POS = theme['pos']; portfolio.C_NEG = theme['neg']
    _lbl = f"color:#f8fafc;font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;font-family:{FONTS}"

    # Row 0: Mode
    m1, m2 = st.columns([3, 5])
    with m1:
        st.markdown(f"<div style='{_lbl}'>MODE</div>", unsafe_allow_html=True)
        mode = st.selectbox("Mode", ['Monte Carlo (Walk-Forward)', 'Monte Carlo (Full Sample)', 'Equal Weight'],
                             key='portall_mode', label_visibility='collapsed')

    is_mc = mode == 'Monte Carlo (Walk-Forward)'
    is_fs = mode == 'Monte Carlo (Full Sample)'
    _dis = not (is_mc or is_fs)  # disabled for EW-only fields

    # Row 1: Objective, Rebalance, Period, Direction, Sims
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"<div style='{_lbl}'>OBJECTIVE</div>", unsafe_allow_html=True)
        score = st.selectbox("Objective", ['Win Rate', 'Composite', 'Sharpe', 'Sortino', 'MAR', 'R²', 'Total Return'],
                              key='portall_score', label_visibility='collapsed', disabled=_dis)
    with c2:
        st.markdown(f"<div style='{_lbl}'>REBALANCE</div>", unsafe_allow_html=True)
        rebal_label = st.selectbox("Rebalance", list(REBAL_OPTIONS.keys()),
                                    index=2, key='portall_rebal', label_visibility='collapsed')
    with c3:
        st.markdown(f"<div style='{_lbl}'>PERIOD</div>", unsafe_allow_html=True)
        period_label = st.selectbox("Period", list(PERIOD_OPTIONS.keys()),
                                     index=2, key='portall_period', label_visibility='collapsed')
    with c4:
        st.markdown(f"<div style='{_lbl}'>DIRECTION</div>", unsafe_allow_html=True)
        direction = st.selectbox("Direction", ['Long Only', 'Long/Short'],
                                  key='portall_direction', label_visibility='collapsed', disabled=_dis)
    with c5:
        st.markdown(f"<div style='{_lbl}'>SIMS</div>", unsafe_allow_html=True)
        if 'portall_sims' not in st.session_state: st.session_state['portall_sims'] = '5000'
        sims_str = st.text_input("Sims", key='portall_sims', label_visibility='collapsed', disabled=_dis)

    # Row 2: Constraints
    _defaults = {'portall_maxwt': '50', 'portall_minwt': '0', 'portall_cost': '0.10',
                 'portall_maxvol': '', 'portall_minret': ''}
    for k, v in _defaults.items():
        if k not in st.session_state: st.session_state[k] = v
    for k, v in [('portall_maxwt','50'),('portall_minwt','0'),('portall_cost','0.10')]:
        if not st.session_state.get(k): st.session_state[k] = v

    c6, c7, c8, c9, c10 = st.columns(5)
    with c6:
        st.markdown(f"<div style='{_lbl}'>MAX WT %</div>", unsafe_allow_html=True)
        max_wt_str = st.text_input("Max Wt", key='portall_maxwt', label_visibility='collapsed', disabled=_dis)
    with c7:
        st.markdown(f"<div style='{_lbl}'>MIN WT %</div>", unsafe_allow_html=True)
        min_wt_str = st.text_input("Min Wt", key='portall_minwt', label_visibility='collapsed', disabled=_dis)
    with c8:
        st.markdown(f"<div style='{_lbl}'>MAX VOL %</div>", unsafe_allow_html=True)
        max_vol_str = st.text_input("Max Vol", key='portall_maxvol', label_visibility='collapsed',
                                     placeholder='e.g. 15', disabled=_dis)
    with c9:
        st.markdown(f"<div style='{_lbl}'>MIN RET %</div>", unsafe_allow_html=True)
        min_ret_str = st.text_input("Min Ret", key='portall_minret', label_visibility='collapsed',
                                     placeholder='e.g. 5', disabled=_dis)
    with c10:
        st.markdown(f"<div style='{_lbl}'>COST %</div>", unsafe_allow_html=True)
        cost_str = st.text_input("Cost", key='portall_cost', label_visibility='collapsed')

    # Scan button
    scan_clicked = st.button('▶  Scan All', key='portall_scan', type='primary')

    # Parse shared params
    rebal = REBAL_OPTIONS[rebal_label]
    period_days = PERIOD_OPTIONS[period_label]
    try: txn_cost = max(0, min(5.0, float(cost_str))) / 100.0
    except (ValueError, TypeError): txn_cost = 0.001

    if scan_clicked:
        if is_mc or is_fs:
            score_type = score
            try: max_wt = max(10, min(100, float(max_wt_str))) / 100.0
            except (ValueError, TypeError): max_wt = 0.50
            try: min_wt = max(0, min(50, float(min_wt_str))) / 100.0
            except (ValueError, TypeError): min_wt = 0.0
            try: n_sims = max(1000, min(100000, int(sims_str)))
            except (ValueError, TypeError): n_sims = 5000
            try: max_vol = float(max_vol_str) / 100.0 if max_vol_str.strip() else None
            except (ValueError, TypeError): max_vol = None
            try: min_ann_ret = float(min_ret_str) / 100.0 if min_ret_str.strip() else None
            except (ValueError, TypeError): min_ann_ret = None
            allow_short = direction == 'Long/Short'
            rank_by_mc = SCORE_TO_RANK.get(score_type, 'win_rate')
            rank_display = next((k for k, v in SCAN_SORT_KEYS.items() if v[0] == rank_by_mc), 'Win Rate')

            if is_mc:
                progress = st.progress(0, text='Starting MC walk-forward scan...')
                results = _run_mc_scan(period_days, rebal, txn_cost, score_type, n_sims,
                                       max_wt, min_wt, allow_short, max_vol, min_ann_ret,
                                       rank_display, progress)
            else:
                progress = st.progress(0, text='Starting MC full-sample scan...')
                results = _run_fs_scan(period_days, rebal, txn_cost, score_type, n_sims,
                                       max_wt, min_wt, allow_short, max_vol, min_ann_ret,
                                       rank_display, progress)
            if not results:
                st.warning('No valid groups found'); return
            st.session_state.portall_results = results
            st.session_state.portall_rank_key = rank_display
            st.session_state.portall_is_mc = True
            _render_results(results, theme, rank_display, is_mobile, is_mc=True)
        else:
            # EW mode: rank by the objective dropdown
            rank_by_ew = SCORE_TO_RANK.get(score, 'win_rate')
            rank_display = next((k for k, v in SCAN_SORT_KEYS.items() if v[0] == rank_by_ew), 'Win Rate')
            progress = st.progress(0, text='Scanning groups...')
            results = _run_ew_scan(period_days, rebal, txn_cost, rank_display, progress)
            if not results:
                st.warning('No valid groups found'); return
            st.session_state.portall_results = results
            st.session_state.portall_rank_key = rank_display
            st.session_state.portall_is_mc = False
            _render_results(results, theme, rank_display, is_mobile, is_mc=False)
    elif 'portall_results' in st.session_state:
        rank_key = st.session_state.get('portall_rank_key', 'Win Rate')
        stored_mc = st.session_state.get('portall_is_mc', False)
        _render_results(st.session_state.portall_results, theme, rank_key, is_mobile,
                       is_mc=stored_mc)


# =============================================================================
# RENDER RESULTS
# =============================================================================

def _render_results(results, theme, rank_by, is_mobile, is_mc=False):
    _bg3 = theme.get('bg3', '#0f172a'); _bdr = theme.get('border', '#1e293b')
    _txt = theme.get('text', '#e2e8f0'); _txt2 = theme.get('text2', '#94a3b8')
    _mut = theme.get('muted', '#475569')

    # Compute composite score (avg rank of Sharpe, Sortino, MAR, R²)
    n = len(results)
    if n > 1:
        for metric in ['sharpe', 'sortino', 'mar', 'r2']:
            vals = [r[metric] for r in results]
            order = sorted(range(n), key=lambda i: -vals[i])
            for rank, idx in enumerate(order):
                results[idx][f'_{metric}_rank'] = rank + 1
        for r in results:
            r['_score'] = np.mean([r.get(f'_{m}_rank', n) for m in ['sharpe', 'sortino', 'mar', 'r2']])
    else:
        results[0]['_score'] = 1.0

    # Sort
    sort_key, reverse = SCAN_SORT_KEYS.get(rank_by, ('win_rate', True))
    sorted_results = sorted(results, key=lambda x: x.get(sort_key, 0), reverse=reverse)

    # Build table — same pattern as spreads_scan (works reliably)
    _render_scan_table(sorted_results, theme, is_mc)

    # Equity charts
    _render_ew_charts(sorted_results, theme, is_mobile)

    # Weights for every group, stacked top to bottom (no dropdown)
    _render_weights_all(sorted_results, theme, is_mc)


def _render_scan_table(sorted_results, theme, is_mc):
    import portfolio
    pos_c = portfolio.C_POS; neg_c = portfolio.C_NEG
    _bg3 = theme.get('bg3', '#0f172a'); _bdr = theme.get('border', '#1e293b')
    _txt = theme.get('text', '#e2e8f0'); _txt2 = theme.get('text2', '#94a3b8')
    _mut = theme.get('muted', '#475569')
    th = f"padding:4px 8px;border-bottom:1px solid {_bdr};color:#f8fafc;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.06em;"
    td = f"padding:5px 8px;border-bottom:1px solid {_bdr}22;"

    html = (f"<div style='overflow-x:auto;border:1px solid {_bdr};border-radius:6px;margin-top:8px'>"
        f"<table style='border-collapse:collapse;font-family:{FONTS};font-size:11px;width:100%;line-height:1.3'>"
        f"<thead style='background:{_bg3}'><tr>"
        f"<th style='{th}text-align:left'>#</th>"
        f"<th style='{th}text-align:left'>GROUP</th>"
        f"<th style='{th}text-align:right'>N</th>"
        f"<th style='{th}text-align:left'>APPROACH</th>"
        f"<th style='{th}text-align:right'>SCORE</th>"
        f"<th style='{th}text-align:right'>WIN%</th>"
        f"<th style='{th}text-align:right'>SHARPE</th>"
        f"<th style='{th}text-align:right'>SORTINO</th>"
        f"<th style='{th}text-align:right'>MAR</th>"
        f"<th style='{th}text-align:right'>R²</th>"
        f"<th style='{th}text-align:right'>TOT%</th>"
        f"<th style='{th}text-align:right'>ANN%</th>"
        f"<th style='{th}text-align:right'>VOL%</th>"
        f"<th style='{th}text-align:right'>MDD%</th>"
        f"<th style='{th}text-align:right'>YTD%</th>"
        f"<th style='{th}text-align:right'>MTD%</th>"
        f"</tr></thead><tbody>")

    for rank, r in enumerate(sorted_results, 1):
        sh_c = pos_c if r['sharpe'] >= 0 else neg_c
        tot_c = pos_c if r['total_ret'] >= 0 else neg_c
        tot_s = '+' if r['total_ret'] >= 0 else ''
        win_c = pos_c if r['win_rate'] >= 0.55 else (neg_c if r['win_rate'] < 0.45 else _txt2)
        ytd_c = pos_c if r.get('ytd', 0) >= 0 else neg_c
        mtd_c = pos_c if r.get('mtd', 0) >= 0 else neg_c
        score = r.get('_score', 0)
        sc_c = pos_c if score <= 3 else (_txt2 if score <= 6 else _mut)
        is_top3 = rank <= 3
        bg = 'rgba(74,222,128,0.06)' if is_top3 else 'transparent'
        fw = '700' if is_top3 else '500'
        gc = pos_c if is_top3 else _txt
        approach = r.get('best_approach', 'EW') if is_mc else 'EW'
        syms_str = ', '.join(r.get('symbols', [])[:6])
        if len(r.get('symbols', [])) > 6:
            syms_str += f" +{len(r['symbols'])-6}"
        syms_str = syms_str.replace("'", "&#39;").replace('"', '&quot;')
        html += (f"<tr style='background:{bg}' title='{syms_str}'>"
            f"<td style='{td}color:{_mut}'>{rank}</td>"
            f"<td style='{td}color:{gc};font-weight:{fw}'>{r['group']}</td>"
            f"<td style='{td}text-align:right;color:{_txt2}'>{r['n_assets']}</td>"
            f"<td style='{td}color:{_txt2};font-size:10px'>{approach}</td>"
            f"<td style='{td}text-align:right;color:{sc_c};font-weight:600'>{score:.1f}</td>"
            f"<td style='{td}text-align:right'><span style='color:{win_c};font-weight:600'>{r['win_rate']*100:.1f}%</span></td>"
            f"<td style='{td}text-align:right'><span style='color:{sh_c};font-weight:700'>{r['sharpe']:.2f}</span></td>"
            f"<td style='{td}text-align:right;color:{_txt2}'>{r['sortino']:.2f}</td>"
            f"<td style='{td}text-align:right;color:{_txt2}'>{r['mar']:.2f}</td>"
            f"<td style='{td}text-align:right;color:{_txt2}'>{r['r2']:.3f}</td>"
            f"<td style='{td}text-align:right'><span style='color:{tot_c};font-weight:600'>{tot_s}{r['total_ret']*100:.1f}%</span></td>"
            f"<td style='{td}text-align:right;color:{_txt2}'>{r['ann_ret']*100:.1f}%</td>"
            f"<td style='{td}text-align:right;color:{_txt2}'>{r['ann_vol']*100:.1f}%</td>"
            f"<td style='{td}text-align:right;color:{neg_c}'>{r['max_dd']*100:.1f}%</td>"
            f"<td style='{td}text-align:right'><span style='color:{ytd_c};font-weight:600'>{r.get('ytd',0)*100:.1f}%</span></td>"
            f"<td style='{td}text-align:right'><span style='color:{mtd_c}'>{r.get('mtd',0)*100:.1f}%</span></td>"
            f"</tr>")

    html += "</tbody></table></div>"
    st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# WEIGHTS — full weights for every group, stacked top to bottom
# =============================================================================

def _render_weights_all(sorted_results, theme, is_mc=False):
    """Render full weights for every group, stacked in ranked order.
    No dropdown — every portfolio's allocation is visible on the page so a full
    page screen capture / PDF print captures all 22 weight tables in one shot."""
    import portfolio
    pos_c = portfolio.C_POS; neg_c = portfolio.C_NEG
    _bg3 = theme.get('bg3', '#0f172a'); _bdr = theme.get('border', '#1e293b')
    _txt = theme.get('text', '#e2e8f0'); _txt2 = theme.get('text2', '#94a3b8')
    _mut = theme.get('muted', '#475569')

    label = 'WEIGHTS'
    sub = 'Optimized weights for every group' if is_mc else 'Equal weight (1/n)'
    _section(label, sub)

    if not sorted_results:
        return

    th = f"padding:4px 8px;border-bottom:1px solid {_bdr};color:#f8fafc;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.06em;"
    td = f"padding:5px 8px;border-bottom:1px solid {_bdr}22;"

    for rank, r in enumerate(sorted_results, 1):
        gname = r.get('group', '?')
        syms = r.get('symbols', [])
        n = len(syms)
        if n == 0:
            continue

        approach = r.get('best_approach', 'EW') if is_mc else 'EW'
        w = r.get('weights', None)
        if is_mc and w and isinstance(w, dict) and len(w) > 0:
            sorted_w = sorted(w.items(), key=lambda x: x[1], reverse=True)
        else:
            ew = 1.0 / n
            sorted_w = [(s, ew) for s in syms]

        # Filter near-zero weights (long-only optimizers produce many 0% slots)
        sorted_w = [(s, wt) for s, wt in sorted_w if abs(wt) > 0.001]
        if not sorted_w:
            continue

        # Summary stats for header
        sharpe = r.get('sharpe', 0)
        tot = r.get('total_ret', 0) * 100
        mdd = r.get('max_dd', 0) * 100
        sh_c = pos_c if sharpe >= 0 else neg_c
        tot_c = pos_c if tot >= 0 else neg_c

        # Header strip per group
        st.markdown(
            f"<div style='margin-top:14px;padding:6px 10px;background:{_bg3};border:1px solid {_bdr};"
            f"border-radius:6px 6px 0 0;font-family:{FONTS};font-size:11px;color:{_txt};"
            f"display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px'>"
            f"<span><b style='color:{_mut}'>#{rank}</b> <b>{gname}</b> "
            f"<span style='color:{_txt2};font-size:10px'>({approach}, N={n})</span></span>"
            f"<span style='color:{_txt2};font-size:10px'>"
            f"Sharpe <b style='color:{sh_c}'>{sharpe:.2f}</b> · "
            f"Tot <b style='color:{tot_c}'>{tot:+.1f}%</b> · "
            f"MDD <b style='color:{neg_c}'>{mdd:.1f}%</b>"
            f"</span></div>",
            unsafe_allow_html=True)

        # Weight table
        max_abs = max(abs(v) for _, v in sorted_w) if sorted_w else 1.0
        html = (f"<div style='border:1px solid {_bdr};border-top:none;border-radius:0 0 6px 6px;overflow-x:auto'>"
            f"<table style='border-collapse:collapse;font-family:{FONTS};font-size:11px;width:100%;line-height:1.3'>"
            f"<thead style='background:{_bg3}'><tr>"
            f"<th style='{th}text-align:left'>SYMBOL</th>"
            f"<th style='{th}text-align:right'>WEIGHT</th>"
            f"<th style='{th}text-align:left'>BAR</th>"
            f"</tr></thead><tbody>")

        for sym, wt in sorted_w:
            pct = wt * 100
            bar_w = (abs(wt) / max_abs) * 100 if max_abs > 0 else 0
            w_c = pos_c if wt >= 0 else neg_c
            html += (f"<tr>"
                f"<td style='{td}color:{_txt};font-weight:600'>{sym}</td>"
                f"<td style='{td}text-align:right;color:{w_c};font-weight:700'>{pct:+.1f}%</td>"
                f"<td style='{td}width:50%'>"
                f"<div style='background:{w_c}22;border-radius:2px;height:14px;width:100%'>"
                f"<div style='background:{w_c};border-radius:2px;height:14px;width:{bar_w:.1f}%'></div>"
                f"</div></td>"
                f"</tr>")

        html += "</tbody></table></div>"
        st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# EQUITY CHARTS — batches of 6 with end value labels
# =============================================================================

def _render_ew_charts(sorted_results, theme, is_mobile):
    import portfolio
    chart_data = [(r['group'], r['ew_returns']) for r in sorted_results if 'ew_returns' in r]
    if not chart_data:
        return

    pos_c = portfolio.C_POS
    _pbg = theme.get('plot_bg', '#121212'); _grd = theme.get('grid', '#1f1f1f')
    _axl = theme.get('axis_line', '#2a2a2a'); _tk = theme.get('tick', '#888888')
    _mut = theme.get('muted', '#475569')

    batch_size = 6
    for batch_start in range(0, len(chart_data), batch_size):
        batch = chart_data[batch_start:batch_start + batch_size]
        n_charts = len(batch)
        n_cols = 1 if is_mobile else min(3, n_charts)
        n_rows = (n_charts + n_cols - 1) // n_cols

        subtitles = [f"<b>{g}</b>" for g, _ in batch]
        while len(subtitles) < n_rows * n_cols:
            subtitles.append("")

        fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=subtitles,
            horizontal_spacing=0.06, vertical_spacing=0.18 if not is_mobile else 0.08)

        for i, (gname, ew_ret) in enumerate(batch):
            row = i // n_cols + 1; col = i % n_cols + 1
            cum = (1 + ew_ret).cumprod() * 100
            end_val = float(cum.iloc[-1])

            fig.add_trace(go.Scatter(
                x=cum.index, y=cum.values,
                mode='lines', line=dict(color=pos_c, width=1.5, shape='spline', smoothing=0.8),
                showlegend=False, hovertemplate='%{y:.1f}<extra></extra>'), row=row, col=col)
            fig.add_hline(y=100, line=dict(color=_grd, width=0.8, dash='dot'), row=row, col=col)

            # End value label
            axis_idx = (row - 1) * n_cols + col
            end_c = '#f8fafc' if end_val >= 100 else portfolio.C_NEG
            fig.add_annotation(
                text=f"<b>{end_val:.0f}</b>", x=0.98, y=0.95,
                xref=f"x{'' if axis_idx == 1 else axis_idx} domain",
                yref=f"y{'' if axis_idx == 1 else axis_idx} domain",
                showarrow=False, font=dict(size=11, color=end_c, family=FONTS),
                xanchor='right', yanchor='top')

            # Rank badge
            global_rank = batch_start + i + 1
            fig.add_annotation(
                text=f"<b>{global_rank}</b>", x=0.02, y=0.95,
                xref=f"x{'' if axis_idx == 1 else axis_idx} domain",
                yref=f"y{'' if axis_idx == 1 else axis_idx} domain",
                showarrow=False, font=dict(size=12, color=_mut, family=FONTS),
                xanchor='left', yanchor='top')

        for ann in fig['layout']['annotations']:
            xref_str = str(ann['xref']) if ann['xref'] else ''
            if 'domain' not in xref_str:
                ann['font'] = dict(size=10, family=FONTS)

        chart_h = 350 * n_rows if is_mobile else 220 * n_rows
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
