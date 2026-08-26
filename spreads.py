import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import combinations
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

from config import FUTURES_GROUPS, SYMBOL_NAMES, FONTS, clean_symbol

logger = logging.getLogger(__name__)

# =============================================================================
# SPREAD STATISTICS
# =============================================================================

def _spread_sharpe(returns, ann_factor=252):
    if returns.std() == 0 or len(returns) < 5: return 0.0
    return float((returns.mean() / returns.std()) * np.sqrt(ann_factor))

def _spread_sortino(returns, ann_factor=252):
    if returns.std() == 0 or len(returns) < 5: return 0.0
    ann_ret = returns.mean() * ann_factor
    down = returns[returns < 0]
    down_std = np.sqrt(np.mean(down**2)) * np.sqrt(ann_factor) if len(down) > 0 else 0
    return float(ann_ret / down_std) if down_std else 0.0

def _spread_drawdowns(returns):
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    mdd = float(dd.min() * 100)
    add = float(dd[dd < 0].mean() * 100) if (dd < 0).any() else 0.0
    return mdd, add

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
            hist = ticker.history(start=start)
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

def compute_sector_spreads(data, ann_factor=252):
    if data is None or len(data.columns) < 2: return []

    asset_sharpes = {}
    for sym in data.columns:
        ret = data[sym].pct_change().dropna()
        asset_sharpes[sym] = _spread_sharpe(ret, ann_factor)
    best_long_sym = max(asset_sharpes, key=asset_sharpes.get)
    best_long_sharpe = asset_sharpes[best_long_sym]

    pairs = []
    for s1, s2 in combinations(data.columns.tolist(), 2):
        r1 = data[s1].pct_change().dropna()
        r2 = data[s2].pct_change().dropna()
        spread_ret = (r1 - r2).dropna()

        sh = _spread_sharpe(spread_ret, ann_factor)
        so = _spread_sortino(spread_ret, ann_factor)

        if sh < 0:
            spread_ret = -spread_ret
            sh, so = -sh, -so
            s1, s2 = s2, s1

        mdd, add = _spread_drawdowns(spread_ret)
        cum_spread = (1 + spread_ret).cumprod()
        total = float((cum_spread.iloc[-1] - 1) * 100)
        ann = float(spread_ret.mean() * ann_factor * 100)
        vol = float(spread_ret.std() * np.sqrt(ann_factor) * 100)
        mar = float(ann / abs(add)) if add != 0 else 0.0
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
            'Sharpe': sh, 'Sortino': so, 'MAR': mar, 'R²': r2_val,
            'Tot%': total, 'Ann%': ann, 'Vol%': vol, 'MDD%': mdd, 'ADD%': add,
            'Corr': corr, 'Win%': win_rate, 'beats_long': sh > best_long_sharpe,
            'cum_long': cum1, 'cum_short': cum2, 'cum_spread': cum_sp,
        })

    n = len(pairs)
    if n == 0: return []
    for metric in ['Sharpe', 'Sortino', 'MAR', 'R²']:
        vals = [p[metric] for p in pairs]
        order = sorted(range(n), key=lambda i: -vals[i])
        for rank, idx in enumerate(order): pairs[idx][f'_{metric}_rank'] = rank + 1
    for p in pairs:
        p['_score'] = np.mean([p[f'_{m}_rank'] for m in ['Sharpe', 'Sortino', 'MAR', 'R²']])
    pairs.sort(key=lambda x: -x['Sharpe'])

    for p in pairs:
        p['best_long_sym'] = best_long_sym
        p['best_long_sharpe'] = best_long_sharpe

    return pairs

# =============================================================================
# SORTING
# =============================================================================

SORT_KEYS = {
    'Composite': '_score', 'Sharpe': 'Sharpe', 'Sortino': 'Sortino',
    'MAR': 'MAR', 'R²': 'R²', 'Total': 'Tot%', 'Win Rate': 'Win%'
}

def sort_spread_pairs(pairs, sort_key='Composite', ascending=False):
    key = SORT_KEYS.get(sort_key, sort_key)
    default_reverse = (key != '_score')
    reverse = not default_reverse if ascending else default_reverse
    return sorted(pairs, key=lambda x: x.get(key, 0), reverse=reverse)

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
            <th style='{th}text-align:center'>vs LONG</th>
        </tr></thead><tbody>"""

    for rank, p in enumerate(show, 1):
        ln = SYMBOL_NAMES.get(p['long'], clean_symbol(p['long']))
        sn = SYMBOL_NAMES.get(p['short'], clean_symbol(p['short']))
        sh_c = pos_c if p['Sharpe'] >= 0 else neg_c
        tot_c = pos_c if p['Tot%'] >= 0 else neg_c
        tot_s = '+' if p['Tot%'] >= 0 else ''
        win_c = pos_c if p['Win%'] >= 55 else (neg_c if p['Win%'] < 45 else _txt2)
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
    from spreads_sector import render_sector_tab
    from spreads_scan import render_scan_tab

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
    </style>""", unsafe_allow_html=True)

    tab_sector, tab_scan = st.tabs(['Sector', 'All'])

    with tab_sector:
        render_sector_tab(is_mobile)

    with tab_scan:
        render_scan_tab(is_mobile)
