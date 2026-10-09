import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

from config import FUTURES_GROUPS, THEMES, SYMBOL_NAMES, FONTS, clean_symbol
from spreads import (LOOKBACK_OPTIONS, INTERVAL_CONFIG, fetch_sector_spread_data,
                     lookback_note,
                     fetch_interval_data, compute_sector_spreads,
                     compute_basket_singles, annualization_factor,
                     composite_ranks, sort_spread_pairs, basket_picker,
                     rank_by_length_adjusted)

logger = logging.getLogger(__name__)

# =============================================================================
# SORT CONFIG
# =============================================================================

# What gets ranked inside the ticked scope.
RANK_MODES = ['Best per basket', 'All pairs']

# Spreads are pairs; the other two rank the outright names. A spread can be
# flipped to face the right way, an outright cannot, so the direction is chosen
# here rather than inferred.
TRADE_MODES = ['Spreads', 'Long only', 'Short only']

SCAN_SORT_KEYS = {
    'Composite': ('_score', False),
    'Sharpe': ('Sharpe', True),
    'Sortino': ('Sortino', True),
    'ROA': ('ROA', True),
    'ER': ('ER', True),
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
    _mut = theme.get('muted', '#475569')
    ann_factor = 252

    # Scope is ticked, not typed. Sector and All were the same computation over
    # different scopes -- one basket, or every basket -- so the scope is a
    # checkbox list and RANK says what gets ranked within it. Tick one basket
    # and rank its pairs, and this is the old Sector tab.
    picked = basket_picker('sc', is_mobile, theme, label='Baskets to scan')

    # One narrow column of inputs, each with its own label above it. The
    # previous label-left/input-right split needed hand-built markdown for the
    # labels, and that markdown is what the Scan button kept riding over.
    if is_mobile:
        col_in = st.container()
    else:
        col_in, _col_rest = st.columns([2, 5])

    with col_in:
        trade = st.selectbox("Trade", TRADE_MODES,
            index=TRADE_MODES.index('Long only'), key='scan_trade_sel',
            help='Spreads ranks every pair, long one leg against the other. '
                 'Long only and Short only rank the outright names instead, so '
                 'the candidates are symbols and only one side is traded.')
        iv_keys = list(INTERVAL_CONFIG.keys())
        interval = st.selectbox("Interval", iv_keys, index=iv_keys.index('1d'),
            key='scan_interval_sel',
            help='Bar size everything is measured on. Intraday reaches back only '
                 'so far: 15m to 60 days, 1h and 4h to 730. It changes the shape '
                 'you see more than the ranking — measured across Futures, Crypto '
                 'and US Sectors, 15m/1h/4h rank +0.90 to +1.00 with daily and '
                 'pick the same leader.')
        # YTD stays the default now that shorter anchors sit above it.
        _lb_keys = list(LOOKBACK_OPTIONS.keys())
        lookback_label = st.selectbox("Lookback", _lb_keys, index=_lb_keys.index('YTD'),
            key='scan_lookback_sel',
            help='How far back to score. Today, WTD, MTD and YTD are calendar '
                 'anchors — they start at the session, the Monday, the 1st or '
                 'January and run to now, so they get longer as the period does. '
                 'The rest are fixed counts of trading days. Today and WTD only '
                 'amount to a window on an intraday Interval: at 1d they are one '
                 'bar and five.')
        lookback_days = LOOKBACK_OPTIONS[lookback_label]
        scan_sort = st.selectbox("Optimize by", list(SCAN_SORT_KEYS.keys()), index=0,
            key='scan_sort_sel',
            help='Not just a sort: it chooses what you see as well as the order. '
                 'Composite is the average rank across Sharpe, Sortino, ROA and '
                 'ER. Every metric here, Composite included, is discounted by the '
                 'square root of the window length before it is ranked, so a short '
                 'history cannot win on a small sample. The columns still show the '
                 'real numbers \u2014 only the order is adjusted.')
        rank_mode = st.selectbox("Rank", RANK_MODES, index=0, key='scan_rank_sel',
            help='Best per basket gives each ticked basket one row — its own '
                 'strongest candidate — so a 465-pair basket cannot crowd out a '
                 '6-pair one. All pairs pools everything from every ticked '
                 'basket and ranks it together; tick a single basket and that '
                 'is its internal ranking.')

        # Padding, not margin: a margin on the inner div collapses out of
        # Streamlit's block, so the button measured 4px ABOVE this line and sat
        # on top of it.
        n_sym = sum(len(FUTURES_GROUPS[g]) for g in picked)
        note = lookback_note(lookback_label, lookback_days, interval)
        warn = f" · <span style='color:#fbbf24'>{note}</span>" if note else ''
        st.markdown(f"<div style='font-size:10px;color:{_mut};font-family:{FONTS};"
                    f"line-height:1.6;padding:14px 0 22px 2px'>{len(picked)} baskets · "
                    f"{n_sym:,} symbols{warn}</div>", unsafe_allow_html=True)
        scan_clicked = st.button('▶  Scan', key='spread_scan_all', type='primary',
                                 use_container_width=True)

    if scan_clicked:
        if not picked:
            st.markdown(f"<div style='padding:12px;color:{_mut};font-size:11px;"
                        f"font-family:{FONTS}'>No baskets ticked — nothing to scan.</div>",
                        unsafe_allow_html=True)
            return
        _run_scan_all(tuple(picked), lookback_days, lookback_label, ann_factor,
                      theme, scan_sort, rank_mode, is_mobile, interval, trade)
    elif 'spread_scan_pairs' in st.session_state:
        # Cached from the last scan. Every pair of every basket is kept, so
        # Optimize by and Rank both re-pick without a refetch.
        _render_scan_all(st.session_state.spread_scan_pairs, theme, scan_sort,
                         lookback_days, ann_factor, is_mobile, rank_mode)

# =============================================================================
# SCAN ENGINE
# =============================================================================

def _basket_candidates(gname, interval, lookback_days, ann_factor, trade):
    """(candidates, price frame) for one basket: pairs, or the outright names.

    The frame comes back too because the charts date their x-axis off it.
    """
    if interval == '1d':
        data = fetch_sector_spread_data(gname, lookback_days)
        af = annualization_factor(data.index, ann_factor) if data is not None else ann_factor
    else:
        data, af = fetch_interval_data(tuple(FUTURES_GROUPS.get(gname, ())),
                                       interval, lookback_days)
    if data is None or data.empty:
        return None, None
    if trade == 'Spreads':
        if len(data.columns) < 2:
            return None, None
        return compute_sector_spreads(data, af), data
    return compute_basket_singles(data, af, trade), data

def _run_scan_all(picked, lookback_days, lookback_label, ann_factor, theme,
                  scan_sort, rank_mode, is_mobile, interval='1d', trade='Spreads'):
    """Score every pair of every group, and keep them all.

    The scan used to store one pair per group, picked on Composite. Sort by then
    could only reorder those, so asking for ROA got the composite-best pair
    ranked by ROA rather than the group's best ROA pair. Keeping every pair lets
    the pick follow Sort by, and lets it change without a refetch.
    """
    by_group = {}
    groups = [(g, FUTURES_GROUPS[g]) for g in picked if g in FUTURES_GROUPS]
    progress = st.progress(0, text='Scanning groups...')

    for i, (gname, syms) in enumerate(groups):
        progress.progress((i + 1) / len(groups), text=f'Scanning: {gname}')
        if len(syms) < 2:
            continue
        try:
            # Per group: a crypto group prints 365 bars a year, an equity one 252.
            pairs, _frame = _basket_candidates(gname, interval, lookback_days,
                                               ann_factor, trade)
            if not pairs:
                continue
            # The cum_* curves are the heavy part and the charts refetch anyway.
            slim = []
            for p in pairs:
                row = {k: v for k, v in p.items() if not k.startswith('cum_')}
                row['group'] = gname
                slim.append(row)
            by_group[gname] = slim
        except Exception as e:
            logger.warning(f"Scan error for {gname}: {e}")

    progress.empty()

    if not by_group:
        st.warning('No valid spreads found across groups')
        return

    st.session_state.spread_scan_pairs = by_group
    st.session_state.spread_scan_ctx = {'interval': interval, 'trade': trade}
    _render_scan_all(by_group, theme, scan_sort, lookback_days, ann_factor,
                     is_mobile, rank_mode)

# =============================================================================
# RENDER RESULTS
# =============================================================================

def _pick_tops(by_group, scan_sort, rank_mode='Best per basket', top_n=25):
    """One representative pair per group, chosen on the sort metric.

    Each pick is a copy, so the group's own Composite -- ranked against its
    siblings -- survives in the cache for the next re-pick, while the copies get
    a fresh Composite ranked across groups.
    """
    if rank_mode == 'All pairs':
        # Every pair of every ticked basket, ranked against each other. With a
        # single basket ticked this is exactly the old Sector ranking.
        pool = [p for pairs in by_group.values() for p in pairs]
        tops = [dict(p) for p in sort_spread_pairs(pool, scan_sort)[:top_n]]
    else:
        tops = []
        for pairs in by_group.values():
            if not pairs:
                continue
            tops.append(dict(sort_spread_pairs(pairs, scan_sort)[0]))
    composite_ranks(tops)
    return tops


def _render_scan_all(by_group, theme, scan_sort, lookback_days, ann_factor,
                     is_mobile, rank_mode='Best per basket'):
    tops = _pick_tops(by_group, scan_sort, rank_mode)
    key, reverse = SCAN_SORT_KEYS.get(scan_sort, ('_score', False))
    # Rows here come from baskets with genuinely different histories, so the
    # order is taken on the length-adjusted value whatever metric is chosen.
    sorted_results = rank_by_length_adjusted(tops, key, reverse)

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
    # Every row one line: 'MSTR Options Income' used to wrap a row to three
    # lines and 'Cell Therapy / In-Vivo CAR-T' to two, which broke the scan.
    td = (f"padding:5px 8px;border-bottom:1px solid {_bdr}22;white-space:nowrap;"
          f"overflow:hidden;text-overflow:ellipsis;max-width:150px;")

    html = f"""<div style='overflow-x:auto;border:1px solid {_bdr};border-radius:6px;margin-top:8px'>
    <table style='border-collapse:collapse;font-family:{FONTS};font-size:11px;width:100%;line-height:1.3'>
        <thead style='background:{_bg3}'><tr>
            <th style='{th}text-align:left'>#</th>
            <th style='{th}text-align:left'>GROUP</th>
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
            <th style='{th}text-align:right'>DAYS</th>
            <th style='{th}text-align:center'>vs LONG</th>
        </tr></thead><tbody>"""

    for rank, p in enumerate(sorted_results, 1):
        # An outright trades one side only; the other prints a dash.
        ln = (SYMBOL_NAMES.get(p['long'], clean_symbol(p['long']))
              if p['long'] else '&mdash;')
        sn = (SYMBOL_NAMES.get(p['short'], clean_symbol(p['short']))
              if p['short'] else '&mdash;')
        sh_c = pos_c if p['Sharpe'] >= 0 else neg_c
        tot_c = pos_c if p['Tot%'] >= 0 else neg_c
        tot_s = '+' if p['Tot%'] >= 0 else ''
        win_c = pos_c if p['Win%'] >= 55 else (neg_c if p['Win%'] < 45 else _txt2)
        _er = p.get('ER', 0)
        er_c = pos_c if _er >= 0.30 else (_mut if _er < 0.10 else _txt2)
        _roa = p.get('ROA', 0)
        roa_c = pos_c if _roa >= 3 else (_mut if _roa <= 0 else _txt2)
        score = p.get('_score', 0)
        sc_c = pos_c if score <= 3 else (_txt2 if score <= 6 else _mut)
        # An outright has no second leg to correlate against.
        _corr = p.get('Corr', float('nan'))
        corr_s = '&mdash;' if _corr != _corr else f'{_corr:.2f}'
        is_top3 = rank <= 3
        # Same marker the sector table carries: did the spread beat simply being
        # long the best single leg in its group?
        beats = p.get('beats_long', False)
        vs = (f"<span style='color:{pos_c};font-weight:700'>&#9650;</span>" if beats
              else f"<span style='color:{_mut}'>&mdash;</span>")
        if beats:
            bg = f'linear-gradient(90deg,{pos_c}08,{_bg3},{pos_c}08)'
        else:
            bg = 'rgba(74,222,128,0.06)' if is_top3 else 'transparent'
        fw = '700' if is_top3 else '500'
        gc = pos_c if is_top3 else _txt
        html += f"""<tr style='background:{bg}'>
            <td style='{td}color:{_mut}'>{rank}</td>
            <td style='{td}color:{gc};font-weight:{fw}' title='{p['group']}'>{p['group']}</td>
            <td style='{td}color:{pos_c};font-weight:600' title='{ln}'>{ln}</td>
            <td style='{td}color:{short_c};font-weight:600' title='{sn}'>{sn}</td>
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
            <td style='{td}text-align:right;color:{_txt2}'>{corr_s}</td>
            <td style='{td}text-align:right;color:{_mut}'>{p.get("Days", 0)}</td>
            <td style='{td}text-align:center'>{vs}</td>
        </tr>"""

    html += "</tbody></table></div>"
    st.markdown(html, unsafe_allow_html=True)

# =============================================================================
# SCAN CHARTS — batches of 6
# =============================================================================

def _render_scan_charts(sorted_results, lookback_days, ann_factor, theme, is_mobile):
    ctx = st.session_state.get('spread_scan_ctx', {})
    interval = ctx.get('interval', '1d')
    trade = ctx.get('trade', 'Spreads')
    if not sorted_results:
        return

    chart_pairs = []
    for r in sorted_results:
        try:
            pairs, data = _basket_candidates(r['group'], interval, lookback_days,
                                             ann_factor, trade)
            if not pairs or data is None:
                continue
            # Match the row's legs: once Sort by drives the pick, re-choosing
            # on Composite here would chart a different pair than the table.
            top = next((q for q in pairs
                        if q['long'] == r['long'] and q['short'] == r['short']), None)
            if top is None:
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
            ln = (SYMBOL_NAMES.get(p['long'], clean_symbol(p['long']))
                  if p['long'] else '')
            sn = (SYMBOL_NAMES.get(p['short'], clean_symbol(p['short']))
                  if p['short'] else '')
            lc = theme['long']; sc = theme['short']
            if ln and sn:
                subtitles.append(
                    f"<b>{g}</b>  <span style='color:{lc}'>■</span> {ln}  "
                    f"<span style='color:{sc}'>■</span> {sn}  "
                    f"<span style='color:#ffffff'>■</span> Spread")
            else:
                # An outright: one leg, and the dotted line is the position
                # itself -- inverted already when it is held short.
                side = 'Long' if ln else 'Short'
                col_ = lc if ln else sc
                subtitles.append(
                    f"<b>{g}</b>  <span style='color:{col_}'>■</span> {ln or sn}  "
                    f"<span style='color:#ffffff'>■</span> {side}")
        while len(subtitles) < n_rows * n_cols:
            subtitles.append("")

        fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=subtitles,
            horizontal_spacing=0.06, vertical_spacing=0.18 if not is_mobile else 0.08)

        for i, cp in enumerate(batch):
            p = cp['pair']; data = cp['data']
            row = i // n_cols + 1; col = i % n_cols + 1

            # Legs the trade actually has. An outright carries one.
            if p.get('cum_long') is not None:
                fig.add_trace(go.Scatter(x=list(range(len(p['cum_long']))), y=p['cum_long'].values,
                    mode='lines', line=dict(color=theme['long'], width=1.3, shape='spline', smoothing=1.0),
                    showlegend=False, hovertemplate='Long: %{y:.1f}<extra></extra>'), row=row, col=col)
            if p.get('cum_short') is not None:
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
