import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

from config import FUTURES_GROUPS, THEMES, SYMBOL_NAMES, FONTS, clean_symbol
from spreads import (LOOKBACK_OPTIONS, fetch_sector_spread_data,
                     compute_sector_spreads)

logger = logging.getLogger(__name__)

# =============================================================================
# SORT CONFIG
# =============================================================================

SCAN_SORT_KEYS = {
    'Composite': ('_score', False),
    'Sharpe': ('Sharpe', True),
    'Sortino': ('Sortino', True),
    'MAR': ('MAR', True),
    'R²': ('R²', True),
    'Total': ('Tot%', True),
    'Win Rate': ('Win%', True),
}

# =============================================================================
# MAIN RENDER
# =============================================================================

def render_scan_tab(is_mobile):
    theme_name = st.session_state.get('theme', 'Dark')
    theme = THEMES.get(theme_name, THEMES['Dark'])
    _lbl = f"color:#e2e8f0;font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;font-family:{FONTS}"
    _bg3 = theme.get('bg3', '#0f172a'); _mut = theme.get('muted', '#475569')
    ann_factor = 252

    # Controls: Lookback + Sort + Scan button
    if is_mobile:
        col_lb, col_sort = st.columns([1, 1])
    else:
        col_lb, col_sort, col_btn = st.columns([3, 3, 2])

    with col_lb:
        st.markdown(f"<div style='{_lbl}'>LOOKBACK</div>", unsafe_allow_html=True)
        lookback_label = st.selectbox("Lookback", list(LOOKBACK_OPTIONS.keys()), index=0,
            key='scan_lookback_sel', label_visibility='collapsed')
        lookback_days = LOOKBACK_OPTIONS[lookback_label]
    with col_sort:
        st.markdown(f"<div style='{_lbl}'>SORT BY</div>", unsafe_allow_html=True)
        sort_options = ['Composite', 'Sharpe', 'Sortino', 'MAR', 'R²', 'Total', 'Win Rate']
        scan_sort = st.selectbox("Sort", sort_options, index=0,
            key='scan_sort_sel', label_visibility='collapsed')

    if is_mobile:
        scan_clicked = st.button('▶  Scan All', key='spread_scan_all', type='primary')
    else:
        with col_btn:
            st.markdown(f"<div style='{_lbl}'>&nbsp;</div>", unsafe_allow_html=True)
            scan_clicked = st.button('▶  Scan All', key='spread_scan_all', type='primary')

    if scan_clicked:
        _run_scan_all(lookback_days, lookback_label, ann_factor, theme, scan_sort, is_mobile)
    elif 'spread_scan_results' in st.session_state:
        _render_scan_all(st.session_state.spread_scan_results, theme, scan_sort,
                         lookback_days, ann_factor, is_mobile)

# =============================================================================
# SCAN ENGINE
# =============================================================================

def _run_scan_all(lookback_days, lookback_label, ann_factor, theme, scan_sort, is_mobile):
    all_top = []
    groups = list(FUTURES_GROUPS.items())
    progress = st.progress(0, text='Scanning groups...')

    for i, (gname, syms) in enumerate(groups):
        progress.progress((i + 1) / len(groups), text=f'Scanning: {gname}')
        if len(syms) < 2:
            continue
        try:
            data = fetch_sector_spread_data(gname, lookback_days)
            if data is None or len(data.columns) < 2:
                continue
            pairs = compute_sector_spreads(data, ann_factor)
            if not pairs:
                continue
            pairs.sort(key=lambda x: x.get('_score', 999))
            top = pairs[0]
            all_top.append({
                'group': gname,
                'long': top['long'], 'short': top['short'],
                'Sharpe': top['Sharpe'], 'Sortino': top['Sortino'],
                'MAR': top['MAR'], 'R²': top['R²'], 'Win%': top['Win%'],
                'Tot%': top['Tot%'], 'Vol%': top['Vol%'],
                'MDD%': top['MDD%'], 'Corr': top['Corr'],
                '_score': top['_score'],
            })
        except Exception as e:
            logger.warning(f"Scan error for {gname}: {e}")

    progress.empty()

    if not all_top:
        st.warning('No valid spreads found across groups')
        return

    # Recompute global ranks
    n = len(all_top)
    if n > 1:
        for metric in ['Sharpe', 'Sortino', 'MAR', 'R²']:
            vals = [p[metric] for p in all_top]
            order = sorted(range(n), key=lambda i: -vals[i])
            for rank, idx in enumerate(order):
                all_top[idx][f'_{metric}_rank'] = rank + 1
        for p in all_top:
            p['_score'] = np.mean([p[f'_{m}_rank'] for m in ['Sharpe', 'Sortino', 'MAR', 'R²']])
    else:
        all_top[0]['_score'] = 1.0

    st.session_state.spread_scan_results = all_top
    _render_scan_all(all_top, theme, scan_sort, lookback_days, ann_factor, is_mobile)

# =============================================================================
# RENDER RESULTS
# =============================================================================

def _render_scan_all(all_top, theme, scan_sort, lookback_days, ann_factor, is_mobile):
    key, reverse = SCAN_SORT_KEYS.get(scan_sort, ('_score', False))
    sorted_results = sorted(all_top, key=lambda x: x.get(key, 0), reverse=reverse)

    _render_scan_table(sorted_results, theme)
    _render_scan_charts(sorted_results, lookback_days, ann_factor, theme, is_mobile)

# =============================================================================
# SCAN TABLE
# =============================================================================

def _render_scan_table(sorted_results, theme):
    pos_c = theme['pos']; neg_c = theme['neg']; short_c = theme['short']
    _bg3 = theme.get('bg3', '#0f172a'); _bdr = theme.get('border', '#1e293b')
    _txt = theme.get('text', '#e2e8f0'); _txt2 = theme.get('text2', '#94a3b8')
    _mut = theme.get('muted', '#475569')
    th = f"padding:4px 8px;border-bottom:1px solid {_bdr};color:#f8fafc;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.06em;"
    td = f"padding:5px 8px;border-bottom:1px solid {_bdr}22;"

    html = f"""<div style='overflow-x:auto;border:1px solid {_bdr};border-radius:6px;margin-top:8px'>
    <table style='border-collapse:collapse;font-family:{FONTS};font-size:11px;width:100%;line-height:1.3'>
        <thead style='background:{_bg3}'><tr>
            <th style='{th}text-align:left'>#</th>
            <th style='{th}text-align:left'>GROUP</th>
            <th style='{th}text-align:left'>LONG</th>
            <th style='{th}text-align:left'>SHORT</th>
            <th style='{th}text-align:right'>SCORE</th>
            <th style='{th}text-align:right'>SHARPE</th>
            <th style='{th}text-align:right'>SORTINO</th>
            <th style='{th}text-align:right'>MAR</th>
            <th style='{th}text-align:right'>R²</th>
            <th style='{th}text-align:right'>WIN%</th>
            <th style='{th}text-align:right'>TOT%</th>
            <th style='{th}text-align:right'>VOL%</th>
            <th style='{th}text-align:right'>MDD%</th>
            <th style='{th}text-align:right'>CORR</th>
        </tr></thead><tbody>"""

    for rank, p in enumerate(sorted_results, 1):
        ln = SYMBOL_NAMES.get(p['long'], clean_symbol(p['long']))
        sn = SYMBOL_NAMES.get(p['short'], clean_symbol(p['short']))
        sh_c = pos_c if p['Sharpe'] >= 0 else neg_c
        tot_c = pos_c if p['Tot%'] >= 0 else neg_c
        tot_s = '+' if p['Tot%'] >= 0 else ''
        win_c = pos_c if p['Win%'] >= 55 else (neg_c if p['Win%'] < 45 else _txt2)
        score = p.get('_score', 0)
        sc_c = pos_c if score <= 3 else (_txt2 if score <= 6 else _mut)
        is_top3 = rank <= 3
        bg = 'rgba(74,222,128,0.06)' if is_top3 else 'transparent'
        fw = '700' if is_top3 else '500'
        gc = pos_c if is_top3 else _txt
        html += f"""<tr style='background:{bg}'>
            <td style='{td}color:{_mut}'>{rank}</td>
            <td style='{td}color:{gc};font-weight:{fw}'>{p['group']}</td>
            <td style='{td}color:{pos_c};font-weight:600'>{ln}</td>
            <td style='{td}color:{short_c};font-weight:600'>{sn}</td>
            <td style='{td}text-align:right;color:{sc_c};font-weight:600'>{score:.1f}</td>
            <td style='{td}text-align:right'><span style='color:{sh_c};font-weight:700'>{p["Sharpe"]:.2f}</span></td>
            <td style='{td}text-align:right;color:{_txt2}'>{p["Sortino"]:.2f}</td>
            <td style='{td}text-align:right;color:{_txt2}'>{p["MAR"]:.2f}</td>
            <td style='{td}text-align:right;color:{_txt2}'>{p["R²"]:.3f}</td>
            <td style='{td}text-align:right'><span style='color:{win_c};font-weight:600'>{p["Win%"]:.0f}%</span></td>
            <td style='{td}text-align:right'><span style='color:{tot_c};font-weight:600'>{tot_s}{p["Tot%"]:.1f}%</span></td>
            <td style='{td}text-align:right;color:{_txt2}'>{p["Vol%"]:.1f}%</td>
            <td style='{td}text-align:right;color:{neg_c}'>{p["MDD%"]:.1f}%</td>
            <td style='{td}text-align:right;color:{_txt2}'>{p["Corr"]:.2f}</td>
        </tr>"""

    html += "</tbody></table></div>"
    st.markdown(html, unsafe_allow_html=True)

# =============================================================================
# SCAN CHARTS — batches of 6
# =============================================================================

def _render_scan_charts(sorted_results, lookback_days, ann_factor, theme, is_mobile):
    if not sorted_results:
        return

    chart_pairs = []
    for r in sorted_results:
        try:
            data = fetch_sector_spread_data(r['group'], lookback_days)
            if data is None or len(data.columns) < 2:
                continue
            pairs = compute_sector_spreads(data, ann_factor)
            if not pairs:
                continue
            pairs.sort(key=lambda x: x.get('_score', 999))
            top = pairs[0]
            top['_group'] = r['group']
            chart_pairs.append({'pair': top, 'data': data})
        except Exception:
            continue

    if not chart_pairs:
        return

    _pbg = theme.get('plot_bg', '#121212'); _grd = theme.get('grid', '#1f1f1f')
    _axl = theme.get('axis_line', '#2a2a2a'); _tk = theme.get('tick', '#888888')
    _mut = theme.get('muted', '#475569')

    batch_size = 6
    for batch_start in range(0, len(chart_pairs), batch_size):
        batch = chart_pairs[batch_start:batch_start + batch_size]
        n_charts = len(batch)
        n_cols = 1 if is_mobile else min(3, n_charts)
        n_rows = (n_charts + n_cols - 1) // n_cols

        subtitles = []
        for cp in batch:
            p = cp['pair']; g = p.get('_group', '')
            ln = SYMBOL_NAMES.get(p['long'], clean_symbol(p['long']))
            sn = SYMBOL_NAMES.get(p['short'], clean_symbol(p['short']))
            lc = theme['long']; sc = theme['short']
            subtitles.append(
                f"<b>{g}</b>  <span style='color:{lc}'>■</span> {ln}  "
                f"<span style='color:{sc}'>■</span> {sn}  "
                f"<span style='color:#ffffff'>■</span> Spread"
            )
        while len(subtitles) < n_rows * n_cols:
            subtitles.append("")

        fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=subtitles,
            horizontal_spacing=0.06, vertical_spacing=0.18 if not is_mobile else 0.08)

        for i, cp in enumerate(batch):
            p = cp['pair']; data = cp['data']
            row = i // n_cols + 1; col = i % n_cols + 1

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
            global_rank = batch_start + i + 1
            fig.add_annotation(
                text=f"<b>{global_rank}</b>", x=0.02, y=0.95,
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
