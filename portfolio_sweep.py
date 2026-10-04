"""
SANPO — PORTFOLIO / Optimal

Single fixes the objective and the rebalance by hand and sweeps only the
lookback. This sweeps all three: every objective against every rebalance
frequency, each one ranking its own eleven lookbacks internally, and reports the
configuration that came out best.

What it is for is NOT reading off the single brightest cell. One good square
surrounded by poor ones is a configuration that happened to fit the sample; the
honest read of a grid is its shape. The heatmap is there so the shape is visible
-- the same reason SPREADS draws one over train x rebalance.
"""

import logging

import plotly.graph_objects as go
import streamlit as st

from config import THEMES, FONTS, filter_listings
from portfolio import (OBJECTIVES, REBAL_OPTIONS, PERIOD_OPTIONS, SCORE_TO_RANK,
                       LOWER_IS_BETTER, composite_ranks, best_approach, rank_rows,
                       run_walkforward_grid, _section, C_MUTE)
from portfolio_single import LISTINGS, _pool, _resolve_min_hist
from spreads import basket_picker

logger = logging.getLogger(__name__)

# Beyond this the sweep is minutes rather than seconds, and it is worth saying so
# before the click rather than after.
BUSY_RUNS = 24


def _run_sweep(symbols, objectives, rebalances, period_days, n_sims, max_wt, min_wt,
               txn_cost, allow_short, max_pos, min_hist_days, progress=None):
    """One walk-forward grid per (objective, rebalance). Each call sweeps the
    eleven lookbacks itself, so the third dimension comes free with the second.

    Rows carry the span they were scored over, because they genuinely differ:
    a weekly rebalance starts trading sooner than an annual one, so it banks a
    longer out-of-sample record on identical data.
    """
    rows = []
    combos = [(o, r) for o in objectives for r in rebalances]
    for i, (obj, rebal_label) in enumerate(combos):
        if progress:
            progress.progress((i + 1) / len(combos), text=f'{obj} · {rebal_label}')
        try:
            grid = run_walkforward_grid(
                symbols, score_type=obj, rebal_months=REBAL_OPTIONS[rebal_label],
                fetch_days=period_days, n_portfolios=n_sims,
                max_weight=max_wt, min_weight=min_wt, txn_cost=txn_cost,
                allow_short=allow_short, max_pos=max_pos,
                min_history_days=min_hist_days)
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


def _heatmap(rows, objectives, rebalances, metric, theme):
    """Objective (rows) against rebalance (columns), coloured on METRIC.

    Drawn so the SHAPE is readable. A lone bright square with dark neighbours is
    a configuration that fitted the sample; a bright region is a setting that
    holds up as you move around it.
    """
    lookup = {(r['objective'], r['rebal']): r for r in rows}
    z, text = [], []
    for obj in objectives:
        zr, tr = [], []
        for rb in rebalances:
            r = lookup.get((obj, rb))
            v = None if r is None else float(r.get(metric, 0) or 0)
            zr.append(v)
            tr.append('' if v is None else f'{v:.2f}')
        z.append(zr); text.append(tr)

    flat = [v for row in z for v in row if v is not None]
    if not flat:
        return None
    lim = max(abs(min(flat)), abs(max(flat))) or 1.0
    reverse = metric in LOWER_IS_BETTER
    scale = ([[0, '#4ade80'], [0.5, '#0b1220'], [1, '#fb7185']] if reverse
             else [[0, '#fb7185'], [0.5, '#0b1220'], [1, '#4ade80']])
    fig = go.Figure(go.Heatmap(
        z=z, x=rebalances, y=objectives, zmid=0 if not reverse else None,
        zmin=-lim if not reverse else min(flat), zmax=lim if not reverse else max(flat),
        colorscale=scale, text=text, texttemplate='%{text}',
        textfont=dict(size=10, family=FONTS), showscale=False,
        hovertemplate='%{y} · %{x}<br>%{z:.3f}<extra></extra>'))
    fig.update_layout(
        height=60 + 30 * len(objectives), margin=dict(l=8, r=8, t=10, b=8),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family=FONTS, size=9, color=theme.get('tick', '#888')),
        xaxis=dict(title='', side='bottom'), yaxis=dict(title='', autorange='reversed'))
    return fig


def _render_table(rows, theme, sort_key, is_mobile):
    pos_c = theme['pos']; neg_c = theme['neg']
    _bg3 = theme.get('bg3', '#0f172a'); _bdr = theme.get('border', '#1e293b')
    _txt = theme.get('text', '#e2e8f0'); _txt2 = theme.get('text2', '#94a3b8')
    _mut = theme.get('muted', '#475569')
    th = (f"padding:4px 8px;border-bottom:1px solid {_bdr};color:#f8fafc;font-weight:600;"
          f"font-size:9px;text-transform:uppercase;letter-spacing:0.06em;")
    td = f"padding:5px 8px;border-bottom:1px solid {_bdr}22;"

    cols = [('#', 'left'), ('OBJECTIVE', 'left'), ('REBALANCE', 'left'), ('LOOKBACK', 'left'),
            ('SCORE', 'right'), ('WIN%', 'right'), ('SHARPE', 'right'), ('SORTINO', 'right'),
            ('ROA', 'right'), ('ER', 'right'), ('MAR', 'right'), ('R²', 'right'),
            ('TOT%', 'right'), ('VOL%', 'right'), ('MDD%', 'right'), ('OOS', 'right'),
            ('REBALS', 'right')]
    html = (f"<div style='overflow-x:auto;border:1px solid {_bdr};border-radius:6px;margin-top:8px'>"
            f"<table style='border-collapse:collapse;font-family:{FONTS};font-size:11px;"
            f"width:100%;line-height:1.3'><thead style='background:{_bg3}'><tr>")
    for label, align in cols:
        html += f"<th style='{th}text-align:{align}'>{label}</th>"
    html += "</tr></thead><tbody>"

    for rank, r in enumerate(rows, 1):
        top = rank <= 3
        bg = 'rgba(74,222,128,0.06)' if top else 'transparent'
        fw = '700' if top else '500'
        oc = pos_c if top else _txt
        sc = r.get('_score', 0)
        sc_c = pos_c if sc <= 3 else (_txt2 if sc <= 8 else _mut)
        sh_c = pos_c if r['sharpe'] >= 0 else neg_c
        tot_c = pos_c if r['total_ret'] >= 0 else neg_c
        html += (f"<tr style='background:{bg}'>"
                 f"<td style='{td}color:{_mut}'>{rank}</td>"
                 f"<td style='{td}color:{oc};font-weight:{fw}'>{r['objective']}</td>"
                 f"<td style='{td}color:{_txt2}'>{r['rebal']}</td>"
                 f"<td style='{td}color:{_txt2};font-size:10px'>{r['lookback']}</td>"
                 f"<td style='{td}text-align:right;color:{sc_c};font-weight:600'>{sc:.2f}</td>"
                 f"<td style='{td}text-align:right'>{r['win_rate']*100:.1f}%</td>"
                 f"<td style='{td}text-align:right;color:{sh_c};font-weight:700'>{r['sharpe']:.2f}</td>"
                 f"<td style='{td}text-align:right;color:{_txt2}'>{r['sortino']:.2f}</td>"
                 f"<td style='{td}text-align:right;color:{_txt2}'>{r['roa']:.2f}</td>"
                 f"<td style='{td}text-align:right;color:{_txt2}'>{r['er']:.3f}</td>"
                 f"<td style='{td}text-align:right;color:{_txt2}'>{r['mar']:.2f}</td>"
                 f"<td style='{td}text-align:right;color:{_txt2}'>{r['r2']:.3f}</td>"
                 f"<td style='{td}text-align:right;color:{tot_c};font-weight:600'>{r['total_ret']*100:+.1f}%</td>"
                 f"<td style='{td}text-align:right;color:{_txt2}'>{r['ann_vol']*100:.1f}%</td>"
                 f"<td style='{td}text-align:right;color:{_txt2}'>{r['max_dd']*100:.1f}%</td>"
                 f"<td style='{td}text-align:right;color:{_mut}'>{r['oos_years']}y</td>"
                 f"<td style='{td}text-align:right;color:{_mut}'>{r.get('n_rebalances', '—')}</td>"
                 f"</tr>")
    html += "</tbody></table></div>"
    st.markdown(html, unsafe_allow_html=True)


def render_sweep_tab(is_mobile):
    import portfolio
    theme_name = st.session_state.get('theme', 'Dark')
    theme = THEMES.get(theme_name, THEMES['Dark'])
    portfolio.C_POS = theme['pos']; portfolio.C_NEG = theme['neg']

    picked = basket_picker('pw', is_mobile, theme, label='Baskets in play')
    pooled = _pool(picked)
    listings = st.session_state.get('sweep_listings', LISTINGS[0])
    symbols = filter_listings(pooled, listings)

    _defaults = {'sweep_sims': '2000', 'sweep_maxwt': '50', 'sweep_minwt': '0',
                 'sweep_cost': '0.10', 'sweep_maxpos': '', 'sweep_minhist': 'auto'}
    for k, v in _defaults.items():
        st.session_state.setdefault(k, v)

    if is_mobile:
        a1, a2 = st.columns(2); b1, b2 = st.columns(2)
    else:
        a1, a2, b1, b2 = st.columns(4)
    with a1:
        st.selectbox('Listings', LISTINGS, key='sweep_listings',
                     help='Which exchanges to draw from. Same test the Single tab uses.')
    with a2:
        period_label = st.selectbox('Period', list(PERIOD_OPTIONS.keys()), index=2,
                                    key='sweep_period',
                                    help='How much price history to pull. Every cell in the sweep '
                                         'sees the same history, so this is the one setting that '
                                         'is not being searched over.')
    with b1:
        direction = st.selectbox('Direction', ['Long Only', 'Long/Short'], key='sweep_direction')
    with b2:
        rank_by = st.selectbox('Rank by', ['Composite'] + OBJECTIVES[1:], key='sweep_rank',
                               help='What orders the table and colours the grid. Composite is the '
                                    'average rank across Sharpe, Sortino, ROA and ER, discounted '
                                    'for how long each configuration actually traded — which '
                                    'matters here, because a weekly rebalance banks more '
                                    'out-of-sample days than an annual one on the same data.')

    sweep_objs = st.multiselect('Objectives to sweep', OBJECTIVES, default=OBJECTIVES,
                                key='sweep_objs',
                                help='Each is optimised for, then judged on its own terms. Drop the '
                                     'ones you would never trade on and the sweep gets shorter.')
    sweep_rebals = st.multiselect('Rebalance frequencies to sweep', list(REBAL_OPTIONS.keys()),
                                  default=list(REBAL_OPTIONS.keys()), key='sweep_rebals')

    with st.expander('More settings — constraints and cost'):
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            sims_str = st.text_input('Sims', key='sweep_sims',
                                     help='Per lookback window, per cell. The sweep runs a lot of '
                                          'them, so this starts lower than the Single tab: 2,000 '
                                          'is enough to rank configurations even where it is not '
                                          'enough to settle the last basis point of a weight.')
        with c2:
            max_pos_str = st.text_input('Max Pos', key='sweep_maxpos', placeholder='e.g. 20')
        with c3:
            max_wt_str = st.text_input('Max Wt %', key='sweep_maxwt')
        with c4:
            min_wt_str = st.text_input('Min Wt %', key='sweep_minwt')
        with c5:
            min_hist_str = st.text_input('Min Hist Y', key='sweep_minhist',
                                         placeholder='auto, or e.g. 2')
        cost_str = st.text_input('Cost %', key='sweep_cost')

    n_runs = len(sweep_objs) * len(sweep_rebals)
    busy = (' — a few minutes; drop some objectives to shorten it'
            if n_runs > BUSY_RUNS else '')
    st.markdown(f"<div style='font-size:10px;color:{C_MUTE};font-family:{FONTS};"
                f"padding:2px 0 8px 2px'>{len(picked)} baskets · {len(symbols):,} symbols "
                f"· {n_runs} configurations × {len(portfolio.PORTFOLIO_APPROACHES)} "
                f"lookbacks{busy}</div>", unsafe_allow_html=True)

    go_clicked = st.button('▶  Sweep', key='sweep_run', type='primary')

    period_days = PERIOD_OPTIONS[period_label]
    try: txn_cost = max(0, min(5.0, float(cost_str))) / 100.0
    except (ValueError, TypeError): txn_cost = 0.001

    if go_clicked:
        if len(symbols) < 2:
            st.warning('Tick baskets holding at least 2 symbols between them.'); return
        if not sweep_objs or not sweep_rebals:
            st.warning('Pick at least one objective and one rebalance frequency.'); return
        try: max_wt = max(10, min(100, float(max_wt_str))) / 100.0
        except (ValueError, TypeError): max_wt = 0.50
        try: min_wt = max(0, min(50, float(min_wt_str))) / 100.0
        except (ValueError, TypeError): min_wt = 0.0
        try: n_sims = max(1000, min(100000, int(sims_str)))
        except (ValueError, TypeError): n_sims = 2000
        try: max_pos = max(2, int(float(max_pos_str))) if max_pos_str.strip() else 0
        except (ValueError, TypeError): max_pos = 0
        min_hist_days, auto_pick = _resolve_min_hist(symbols, period_days, min_hist_str)
        if auto_pick is not None:
            dropped, kept, total, days, cutoff = auto_pick
            st.caption(f'ⓘ Auto left out the {dropped} newest listing'
                       f'{"s" if dropped != 1 else ""} of {total} — anything after '
                       f'{cutoff.date()} — for a shared window of {days:,} trading days.')

        progress = st.progress(0, text='Sweeping...')
        rows = _run_sweep(symbols, sweep_objs, sweep_rebals, period_days, n_sims,
                          max_wt, min_wt, txn_cost, direction == 'Long/Short',
                          max_pos, min_hist_days, progress)
        if not rows:
            st.warning('No configuration produced a usable walk-forward. Try a longer Period, '
                       'or set Min Hist Y to auto.')
            return
        composite_ranks(rows)
        st.session_state.sweep_rows = rows
        st.session_state.sweep_meta = {'objs': sweep_objs, 'rebals': sweep_rebals,
                                       'n_syms': len(symbols)}

    rows = st.session_state.get('sweep_rows')
    if not rows:
        st.markdown(f"<div style='padding:20px;color:{C_MUTE};font-size:11px;font-family:{FONTS}'>"
                    f"Tick baskets and click Sweep to search objective × rebalance × "
                    f"lookback.</div>", unsafe_allow_html=True)
        return

    meta = st.session_state.get('sweep_meta', {})
    key = SCORE_TO_RANK.get(rank_by, '_score') if rank_by != 'Composite' else '_score'
    ordered = rank_rows(list(rows), key, reverse=key not in LOWER_IS_BETTER)

    best = ordered[0]
    _section('BEST CONFIGURATION',
             f"{meta.get('n_syms', '?')} symbols · {len(rows)} configurations searched")
    st.markdown(
        f"<div style='padding:10px 12px;background:rgba(4,8,16,0.46);border-radius:4px;"
        f"font-family:{FONTS};font-size:12px;color:{theme.get('text', '#e2e8f0')}'>"
        f"<b style='color:{theme['pos']}'>{best['objective']}</b> · "
        f"rebalanced <b>{best['rebal']}</b> · lookback <b>{best['lookback']}</b>"
        f"<span style='color:{C_MUTE}'> &nbsp;|&nbsp; Sharpe {best['sharpe']:.2f} · "
        f"Sortino {best['sortino']:.2f} · ROA {best['roa']:.2f} · ER {best['er']:.3f} · "
        f"{best['oos_years']}y out of sample</span></div>", unsafe_allow_html=True)

    _section('THE GRID', f'{rank_by} across objective and rebalance — read the shape, '
                         f'not the brightest square')
    fig = _heatmap(rows, meta.get('objs', OBJECTIVES), meta.get('rebals', list(REBAL_OPTIONS)),
                   key if key != '_score' else 'sharpe', theme)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    _section('EVERY CONFIGURATION', f'ranked by {rank_by}, length-adjusted')
    _render_table(ordered, theme, key, is_mobile)
