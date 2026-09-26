import html
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from config import FUTURES_GROUPS, THEMES, SYMBOL_NAMES, FONTS, clean_symbol
from portfolio import (C_MUTE, C_BG, C_TXT, C_TXT2, C_GOLD, BENCH_COLORS, MAX_BENCHMARKS, _tint,
                       REBAL_OPTIONS, PERIOD_OPTIONS, SCORE_TO_RANK,
                       fetch_symbol_history, benchmark_series, _bench_metrics, _calc_oos_metrics,
                       run_walkforward_grid, run_fullsample,
                       render_ranking_table,
                       render_weights_table, render_oos_chart,
                       render_monthly_table, _section)


def _clean_benchmarks(raw):
    """Benchmark tickers go into yfinance URLs and into HTML labels, so keep only
    the characters real tickers use. 'SPY, XLV' -> ['SPY', 'XLV']."""
    if not raw: return []
    out = []
    for part in raw.replace(';', ',').split(','):
        sym = ''.join(c for c in part.strip().upper() if c.isalnum() or c in '.^=-:')
        if sym and sym not in out: out.append(sym)
    return out[:MAX_BENCHMARKS]


def _warn_failed(failed):
    if failed:
        st.warning(f"No usable history for {', '.join(failed)} — left out of the comparison")


def render_single_tab(is_mobile):
    import portfolio
    theme_name = st.session_state.get('theme', 'Dark')
    theme = THEMES.get(theme_name, THEMES['Dark'])
    portfolio.C_POS = theme['pos']; portfolio.C_NEG = theme['neg']
    _lbl = f"color:#f8fafc;font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;font-family:{FONTS}"

    def _fld(text, tip):
        """Field label with a hover description — the ⓘ marks that one exists."""
        t = html.escape(tip, quote=True)
        return (f"<div style='{_lbl}' title=\"{t}\">{text}"
                f"<span style='color:#64748b;font-weight:400;margin-left:3px' title=\"{t}\">ⓘ</span></div>")

    # Consistent input styling
    st.markdown(f"""<style>
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] div[aria-selected] {{
            font-family: {FONTS} !important; font-size: 13px !important; letter-spacing: 0.01em !important;
        }}
        div[data-baseweb="menu"] li, ul[role="listbox"] li {{
            font-family: {FONTS} !important; font-size: 13px !important;
        }}
        .stTextInput input {{ font-family: {FONTS} !important; font-size: 13px !important; letter-spacing: 0.01em !important; }}
        .stTextInput input::placeholder {{ font-family: {FONTS} !important; font-size: 13px !important; }}
    </style>""", unsafe_allow_html=True)

    # Row 0: Mode + Portfolio + Symbols
    group_names = ['Custom'] + list(FUTURES_GROUPS.keys())
    if 'port_preset_name' not in st.session_state:
        st.session_state.port_preset_name = 'Custom'
    if 'port_sym_input' not in st.session_state:
        st.session_state.port_sym_input = ''

    def _on_portfolio_change():
        sel = st.session_state.port_selector
        if sel != 'Custom':
            syms = FUTURES_GROUPS.get(sel, [])
            st.session_state.port_sym_input = ', '.join(syms)
        st.session_state.port_preset_name = sel

    m0, p1, p2, p3 = st.columns([3, 2, 3.4, 2.2])
    with m0:
        st.markdown(_fld('MODE', 'How the weights are chosen. Walk-Forward: optimise on past data only, then score the period that follows (out-of-sample, the honest test). Full Sample: optimise on all the data and score the same data (in-sample, flattering). Equal Weight: no optimisation at all, every asset gets 1/N.'), unsafe_allow_html=True)
        mode = st.selectbox("Mode", ['Monte Carlo (Walk-Forward)', 'Monte Carlo (Full Sample)', 'Equal Weight'],
                             key='port_mode', label_visibility='collapsed')
    with p1:
        st.markdown(_fld('PORTFOLIO', 'Load a saved basket of symbols into the Symbols box, or pick Custom and type your own.'), unsafe_allow_html=True)
        current_idx = group_names.index(st.session_state.port_preset_name) if st.session_state.port_preset_name in group_names else 0
        st.selectbox("Portfolio", group_names, index=current_idx,
                     key='port_selector', label_visibility='collapsed', on_change=_on_portfolio_change)
    with p2:
        st.markdown(_fld('SYMBOLS', 'The tickers to allocate between, comma-separated (Yahoo Finance symbols). These are what the optimiser splits money across. Need at least 2.'), unsafe_allow_html=True)
        sym_input = st.text_input("Symbols", key='port_sym_input', label_visibility='collapsed',
                                   placeholder='AAPL, MSFT, GOOG, ...')
    with p3:
        st.markdown(_fld('BENCHMARK', 'Optional comparison tickers, comma-separated, up to 4 (e.g. SPY, XLV, XLB). They are NOT part of the portfolio and get no weight — each one is drawn on the chart and added to the ranking table so you can see whether the portfolio actually beat it.'), unsafe_allow_html=True)
        bench_input = st.text_input("Benchmark", key='port_bench', label_visibility='collapsed',
                                     placeholder=f'optional, e.g. SPY, XLV (max {MAX_BENCHMARKS})')

    is_mc = mode == 'Monte Carlo (Walk-Forward)'
    is_fs = mode == 'Monte Carlo (Full Sample)'
    _dis = not (is_mc or is_fs)

    # Row 1: Objective, Rebalance, Period, Direction, Sims
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(_fld('OBJECTIVE', 'The number the optimiser tries to maximise when it picks weights: Win Rate = share of up days, Sharpe = return per unit of volatility, Sortino = return per unit of downside volatility, MAR = return per unit of average drawdown, R² = how straight the equity curve is, Total Return = raw growth.'), unsafe_allow_html=True)
        score = st.selectbox("Objective", ['Win Rate', 'Composite', 'Sharpe', 'Sortino', 'MAR', 'R²', 'Total Return'],
                              key='port_score', label_visibility='collapsed', disabled=_dis)
    with c2:
        st.markdown(_fld('REBALANCE', 'How often holdings are reset back to target weights. Every reset pays the Cost % on whatever it has to trade, so more frequent is not automatically better.'), unsafe_allow_html=True)
        rebal_label = st.selectbox("Rebalance", list(REBAL_OPTIONS.keys()),
                                    index=2, key='port_rebal', label_visibility='collapsed')
    with c3:
        st.markdown(_fld('PERIOD', 'How much price history to pull. Longer means more data to learn from and a longer backtest, but it also drags in older market regimes.'), unsafe_allow_html=True)
        period_label = st.selectbox("Period", list(PERIOD_OPTIONS.keys()),
                                     index=2, key='port_period', label_visibility='collapsed')
    with c4:
        st.markdown(_fld('DIRECTION', 'Long Only: every weight is 0 or positive. Long/Short: negative weights are allowed, so the portfolio can short (note that Min Wt % is ignored in this mode).'), unsafe_allow_html=True)
        direction = st.selectbox("Direction", ['Long Only', 'Long/Short'],
                                  key='port_direction', label_visibility='collapsed', disabled=_dis)
    with c5:
        st.markdown(_fld('SIMS', 'How many random weight combinations to test per lookback window. Higher gives a steadier answer but takes longer. 10,000 is a good default.'), unsafe_allow_html=True)
        if 'port_sims' not in st.session_state: st.session_state['port_sims'] = '10000'
        if not st.session_state.get('port_sims'): st.session_state['port_sims'] = '10000'
        sims_str = st.text_input("Sims", key='port_sims', label_visibility='collapsed', disabled=_dis)

    # Row 2: Max Wt, Min Wt, Max Vol, Min Ret, Cost
    _defaults2 = {'port_maxwt': '50', 'port_minwt': '0', 'port_cost': '0.10',
                   'port_maxvol': '', 'port_minret': '', 'port_minpos': ''}
    for k, v in _defaults2.items():
        if k not in st.session_state: st.session_state[k] = v
    for k, v in [('port_maxwt','50'),('port_minwt','0'),('port_cost','0.10')]:
        if not st.session_state.get(k): st.session_state[k] = v

    c6, c7, c11, c8, c9, c10 = st.columns(6)
    with c6:
        st.markdown(_fld('MAX WT %', 'Ceiling on any single asset, so nothing can dominate. 50 means no holding above 50%.'), unsafe_allow_html=True)
        max_wt_str = st.text_input("Max Wt", key='port_maxwt', label_visibility='collapsed', disabled=_dis)
    with c7:
        st.markdown(_fld('MIN WT %', 'Floor on EVERY asset — it forces each one into the portfolio at this weight or more. 0 means no floor. This keeps assets in; it does not round anything.'), unsafe_allow_html=True)
        min_wt_str = st.text_input("Min Wt", key='port_minwt', label_visibility='collapsed', disabled=_dis)
    with c11:
        st.markdown(_fld('MIN POS %', 'Dust cut. After the weights are chosen, anything smaller than this is set to 0 and the remaining positions are rescaled back to 100%. Use it to avoid trading pointless slivers — e.g. 1 turns a 0.4% position into nothing. Leave blank to keep every sliver. This is the opposite of Min Wt %: it throws assets out rather than forcing them in.'), unsafe_allow_html=True)
        min_pos_str = st.text_input("Min Pos", key='port_minpos', label_visibility='collapsed',
                                     placeholder='drop <1%', disabled=_dis)
    with c8:
        st.markdown(_fld('MAX VOL %', 'Soft cap on annualised volatility. Portfolios above it are penalised in the search rather than banned outright, so the result can still exceed it if nothing else works. Blank = no cap.'), unsafe_allow_html=True)
        max_vol_str = st.text_input("Max Vol", key='port_maxvol', label_visibility='collapsed',
                                     placeholder='e.g. 15', disabled=_dis)
    with c9:
        st.markdown(_fld('MIN RET %', 'Soft floor on annualised return. Portfolios below it are penalised in the search rather than banned. Blank = no floor.'), unsafe_allow_html=True)
        min_ret_str = st.text_input("Min Ret", key='port_minret', label_visibility='collapsed',
                                     placeholder='e.g. 5', disabled=_dis)
    with c10:
        st.markdown(_fld('COST %', 'Round-trip transaction cost charged on turnover at every rebalance, in percent. 0.10 = 10 basis points.'), unsafe_allow_html=True)
        cost_str = st.text_input("Cost", key='port_cost', label_visibility='collapsed')

    # Run button
    if is_mc:
        btn_label = '▶  Optimize (WF)'
    elif is_fs:
        btn_label = '▶  Optimize (Full)'
    else:
        btn_label = '▶  Run EW'
    run_clicked = st.button(btn_label, key='port_run', type='primary')

    # Determine session key based on mode to avoid cross-contamination
    if is_mc:
        result_key = 'port_grid'
    elif is_fs:
        result_key = 'port_fs_result'
    else:
        result_key = 'port_ew_result'

    if not run_clicked and result_key not in st.session_state:
        if is_mc:
            hint = 'walk-forward optimization'
        elif is_fs:
            hint = 'full-sample optimization (in-sample)'
        else:
            hint = 'equal-weight backtest'
        st.markdown(f"<div style='padding:20px;color:{C_MUTE};font-size:11px;font-family:{FONTS}'>Configure parameters and click to start {hint}</div>", unsafe_allow_html=True)
        return

    # Parse shared params
    rebal = REBAL_OPTIONS[rebal_label]
    fetch_days = PERIOD_OPTIONS[period_label]
    try: txn_cost = max(0, min(5.0, float(cost_str))) / 100.0
    except (ValueError, TypeError): txn_cost = 0.001

    if run_clicked:
        raw = sym_input.strip()
        if not raw:
            st.warning('Enter symbols'); return
        symbols = [s.strip().upper() for s in raw.replace(';', ',').split(',') if s.strip()]
        symbols = list(dict.fromkeys(symbols))

        bench = _clean_benchmarks(bench_input)
        if is_mc:
            _run_mc(symbols, score, rebal_label, rebal, period_label, fetch_days,
                    direction, sims_str, max_wt_str, min_wt_str, max_vol_str, min_ret_str,
                    txn_cost, bench, min_pos_str)
        elif is_fs:
            _run_fs(symbols, score, rebal_label, rebal, period_label, fetch_days,
                    direction, sims_str, max_wt_str, min_wt_str, max_vol_str, min_ret_str,
                    txn_cost, bench, min_pos_str)
        else:
            _run_ew(symbols, rebal, fetch_days, txn_cost, rebal_label, period_label, bench)

    # Display results
    if is_mc:
        _display_mc(is_mobile, _lbl)
    elif is_fs:
        _display_fs(is_mobile, _lbl)
    else:
        _display_ew(is_mobile, theme)


# =============================================================================
# MC RUN + DISPLAY
# =============================================================================

def _run_mc(symbols, score, rebal_label, rebal, period_label, fetch_days,
            direction, sims_str, max_wt_str, min_wt_str, max_vol_str, min_ret_str,
            txn_cost, benchmark=(), min_pos_str=''):
    try: max_wt = max(10, min(100, float(max_wt_str))) / 100.0
    except (ValueError, TypeError): max_wt = 0.50
    try: min_wt = max(0, min(50, float(min_wt_str))) / 100.0
    except (ValueError, TypeError): min_wt = 0.0
    try: n_sims = max(1000, min(100000, int(sims_str)))
    except (ValueError, TypeError): n_sims = 10000
    try: max_vol = float(max_vol_str) / 100.0 if max_vol_str.strip() else None
    except (ValueError, TypeError): max_vol = None
    try: min_ann_ret = float(min_ret_str) / 100.0 if min_ret_str.strip() else None
    except (ValueError, TypeError): min_ann_ret = None
    try: min_pos = max(0, min(50, float(min_pos_str))) / 100.0 if min_pos_str.strip() else 0.0
    except (ValueError, TypeError): min_pos = 0.0

    allow_short = direction == 'Long/Short'
    n_syms = len(symbols)
    if min_wt > 0 and min_wt * n_syms > 1.0: min_wt = round(1.0 / n_syms, 4)
    if min_wt >= max_wt: min_wt = 0.0

    progress = st.progress(0, text='Starting walk-forward...')
    grid = run_walkforward_grid(symbols, score_type=score, rebal_months=rebal,
                                 fetch_days=fetch_days, n_portfolios=n_sims,
                                 max_weight=max_wt, min_weight=min_wt,
                                 txn_cost=txn_cost, allow_short=allow_short,
                                 progress_bar=progress,
                                 max_vol=max_vol, min_ann_ret=min_ann_ret,
                                 benchmarks=benchmark, min_pos=min_pos)
    progress.empty()

    if not grid or not grid['results']:
        st.warning('Need ≥2 assets with sufficient history for walk-forward'); return
    _warn_failed(grid.get('bench_failed'))

    st.session_state.port_grid = grid
    preset_name = st.session_state.get('port_preset_name', 'Custom')
    if preset_name == 'Custom' or not preset_name:
        sym_set = set(symbols)
        for pname, psyms in FUTURES_GROUPS.items():
            if set(psyms) == sym_set:
                preset_name = pname; break
        else:
            preset_name = 'Portfolio'
    st.session_state.port_params = {
        'score': score, 'rebal_label': st.session_state.get('port_rebal', 'Quarterly'),
        'period_label': st.session_state.get('port_period', '5 Years'),
        'direction': 'L/S' if allow_short else 'Long',
        'min_wt': min_wt, 'max_wt': max_wt, 'n_sims': n_sims, 'txn_cost': txn_cost,
        'max_vol': max_vol, 'min_ann_ret': min_ann_ret, 'min_pos': min_pos,
        'preset_name': preset_name,
    }
    if 'port_view_approach' in st.session_state:
        del st.session_state.port_view_approach


def _display_mc(is_mobile, _lbl):
    import portfolio
    if 'port_grid' not in st.session_state: return
    grid = st.session_state.port_grid
    params = st.session_state.port_params
    rank_metric = SCORE_TO_RANK.get(params['score'], 'win_rate')
    n_app = len(grid['results'])

    constraints_str = ''
    if params.get('max_vol'): constraints_str += f" · max vol {params['max_vol']*100:.0f}%"
    if params.get('min_ann_ret'): constraints_str += f" · min ret {params['min_ann_ret']*100:.0f}%"
    if params.get('min_pos'): constraints_str += f" · drop <{params['min_pos']*100:g}%"
    bench_syms = grid.get('bench_symbols') or []
    if bench_syms: constraints_str += f" · vs {', '.join(bench_syms)}"
    _section('APPROACH RANKING',
             f"{n_app} lookbacks · {params['rebal_label']} · {params['period_label']} · "
             f"{params['direction']} · wt {params['min_wt']*100:.0f}–{params['max_wt']*100:.0f}% · "
             f"cost {params['txn_cost']*100:.2f}% · {params['n_sims']:,} sims · max {params['score']}{constraints_str} · all OOS")
    result = render_ranking_table(grid, rank_metric)
    best_name, sorted_names = result
    if not best_name or not sorted_names: return

    cur = st.session_state.get('port_view_approach')
    if cur not in sorted_names:
        st.session_state.port_view_approach = sorted_names[0]
    sel_col, _ = st.columns([3, 5])
    with sel_col:
        st.markdown(f"""<div style='{_lbl};margin-top:8px'>VIEW APPROACH
            <span style='color:{C_MUTE};font-weight:400;text-transform:none;letter-spacing:0'>
            — which lookback to show below</span></div>""",
            unsafe_allow_html=True)
        selected_approach = st.selectbox("Approach", sorted_names,
                                          key='port_view_approach', label_visibility='collapsed',
                                          help='Each entry is a different lookback recipe for choosing weights — '
                                               '"12mo" optimises on the last 12 months, "12mo Recency" blends '
                                               '3/6/9/12-month windows with more weight on the recent ones. Ranked '
                                               'best-first by your Objective; the list re-sorts when you re-run.')

    sel = grid['results'][selected_approach]; sm = sel['metrics']; swf = sel['wf']
    is_best = selected_approach == best_name
    star = f"<span style='color:{C_GOLD}'>★</span> " if is_best else ""

    st.markdown(f"""<div style='margin-top:12px;padding:5px 10px;background:{C_BG};font-family:{FONTS};border-radius:4px;
        font-size:10px;color:{C_TXT2};display:flex;justify-content:space-between;flex-wrap:wrap;gap:4px'>
        <span>{star}<b style='color:{C_TXT}'>{selected_approach}</b>
        &nbsp;·&nbsp;{sm['n_days']} OOS days · {sm['oos_years']}y · {sm['n_rebalances']} rebalances</span>
        <span>Win% <b style='color:{portfolio.C_POS}'>{sm["win_rate"]*100:.1f}%</b>
        &nbsp;Sharpe <b style='color:{portfolio.C_POS}'>{sm["sharpe"]:.2f}</b>
        &nbsp;Sortino <b style='color:{portfolio.C_POS}'>{sm["sortino"]:.2f}</b>
        &nbsp;MAR <b style='color:{portfolio.C_POS}'>{sm["mar"]:.2f}</b></span>
    </div>""", unsafe_allow_html=True)

    _section('OOS EQUITY CURVE', f'{selected_approach} · {params["rebal_label"]} · yellow = rebalance dates')
    render_oos_chart(grid, selected_approach)

    _section('OOS MONTHLY RETURNS', f'{selected_approach} · walk-forward out-of-sample only')
    render_monthly_table(swf['oos_returns'])

    _section('CURRENT / NEXT WEIGHTS', f'{selected_approach} · optimized on all data through today · trade these')
    render_weights_table(grid, selected_approach)


# =============================================================================
# FULL SAMPLE RUN + DISPLAY
# =============================================================================

def _run_fs(symbols, score, rebal_label, rebal, period_label, fetch_days,
            direction, sims_str, max_wt_str, min_wt_str, max_vol_str, min_ret_str,
            txn_cost, benchmark=(), min_pos_str=''):
    try: max_wt = max(10, min(100, float(max_wt_str))) / 100.0
    except (ValueError, TypeError): max_wt = 0.50
    try: min_wt = max(0, min(50, float(min_wt_str))) / 100.0
    except (ValueError, TypeError): min_wt = 0.0
    try: n_sims = max(1000, min(100000, int(sims_str)))
    except (ValueError, TypeError): n_sims = 10000
    try: max_vol = float(max_vol_str) / 100.0 if max_vol_str.strip() else None
    except (ValueError, TypeError): max_vol = None
    try: min_ann_ret = float(min_ret_str) / 100.0 if min_ret_str.strip() else None
    except (ValueError, TypeError): min_ann_ret = None
    try: min_pos = max(0, min(50, float(min_pos_str))) / 100.0 if min_pos_str.strip() else 0.0
    except (ValueError, TypeError): min_pos = 0.0

    allow_short = direction == 'Long/Short'
    n_syms = len(symbols)
    if min_wt > 0 and min_wt * n_syms > 1.0: min_wt = round(1.0 / n_syms, 4)
    if min_wt >= max_wt: min_wt = 0.0

    progress = st.progress(0, text='Starting full-sample optimization...')
    grid = run_fullsample(symbols, score_type=score, n_portfolios=n_sims,
                          fetch_days=fetch_days, max_weight=max_wt, min_weight=min_wt,
                          txn_cost=txn_cost, allow_short=allow_short,
                          progress_bar=progress,
                          max_vol=max_vol, min_ann_ret=min_ann_ret,
                          rebal_months=rebal, benchmarks=benchmark, min_pos=min_pos)
    progress.empty()

    if not grid or not grid['results']:
        st.warning('Need ≥2 assets with sufficient history for full-sample optimization'); return
    _warn_failed(grid.get('bench_failed'))

    st.session_state.port_fs_result = grid
    preset_name = st.session_state.get('port_preset_name', 'Custom')
    if preset_name == 'Custom' or not preset_name:
        sym_set = set(symbols)
        for pname, psyms in FUTURES_GROUPS.items():
            if set(psyms) == sym_set:
                preset_name = pname; break
        else:
            preset_name = 'Portfolio'
    st.session_state.port_fs_params = {
        'score': score, 'rebal_label': st.session_state.get('port_rebal', 'Quarterly'),
        'period_label': st.session_state.get('port_period', '5 Years'),
        'direction': 'L/S' if allow_short else 'Long',
        'min_wt': min_wt, 'max_wt': max_wt, 'n_sims': n_sims, 'txn_cost': txn_cost,
        'max_vol': max_vol, 'min_ann_ret': min_ann_ret, 'min_pos': min_pos,
        'preset_name': preset_name,
    }
    if 'port_fs_view_approach' in st.session_state:
        del st.session_state.port_fs_view_approach


def _display_fs(is_mobile, _lbl):
    import portfolio
    if 'port_fs_result' not in st.session_state: return
    grid = st.session_state.port_fs_result
    params = st.session_state.port_fs_params
    rank_metric = SCORE_TO_RANK.get(params['score'], 'win_rate')
    n_app = len(grid['results'])

    constraints_str = ''
    if params.get('max_vol'): constraints_str += f" · max vol {params['max_vol']*100:.0f}%"
    if params.get('min_ann_ret'): constraints_str += f" · min ret {params['min_ann_ret']*100:.0f}%"
    if params.get('min_pos'): constraints_str += f" · drop <{params['min_pos']*100:g}%"
    bench_syms = grid.get('bench_symbols') or []
    if bench_syms: constraints_str += f" · vs {', '.join(bench_syms)}"
    _section('APPROACH RANKING (IN-SAMPLE)',
             f"{n_app} lookbacks · {params['rebal_label']} · {params['period_label']} · "
             f"{params['direction']} · wt {params['min_wt']*100:.0f}–{params['max_wt']*100:.0f}% · "
             f"cost {params['txn_cost']*100:.2f}% · {params['n_sims']:,} sims · max {params['score']}{constraints_str} · FULL SAMPLE")
    result = render_ranking_table(grid, rank_metric)
    best_name, sorted_names = result
    if not best_name or not sorted_names: return

    cur = st.session_state.get('port_fs_view_approach')
    if cur not in sorted_names:
        st.session_state.port_fs_view_approach = sorted_names[0]
    sel_col, _ = st.columns([3, 5])
    with sel_col:
        st.markdown(f"""<div style='{_lbl};margin-top:8px'>VIEW APPROACH
            <span style='color:{C_MUTE};font-weight:400;text-transform:none;letter-spacing:0'>
            — which lookback to show below</span></div>""",
            unsafe_allow_html=True)
        selected_approach = st.selectbox("Approach", sorted_names,
                                          key='port_fs_view_approach', label_visibility='collapsed',
                                          help='Each entry is a different lookback recipe for choosing weights — '
                                               '"12mo" optimises on the last 12 months, "12mo Recency" blends '
                                               '3/6/9/12-month windows with more weight on the recent ones. Ranked '
                                               'best-first by your Objective; the list re-sorts when you re-run.')

    sel = grid['results'][selected_approach]; sm = sel['metrics']; swf = sel['wf']
    is_best = selected_approach == best_name
    star = f"<span style='color:{C_GOLD}'>★</span> " if is_best else ""

    st.markdown(f"""<div style='margin-top:12px;padding:5px 10px;background:{C_BG};font-family:{FONTS};border-radius:4px;
        font-size:10px;color:{C_TXT2};display:flex;justify-content:space-between;flex-wrap:wrap;gap:4px'>
        <span>{star}<b style='color:{C_TXT}'>{selected_approach}</b>
        &nbsp;·&nbsp;{sm['n_days']} days · {sm['oos_years']}y · IN-SAMPLE</span>
        <span>Win% <b style='color:{portfolio.C_POS}'>{sm["win_rate"]*100:.1f}%</b>
        &nbsp;Sharpe <b style='color:{portfolio.C_POS}'>{sm["sharpe"]:.2f}</b>
        &nbsp;Sortino <b style='color:{portfolio.C_POS}'>{sm["sortino"]:.2f}</b>
        &nbsp;MAR <b style='color:{portfolio.C_POS}'>{sm["mar"]:.2f}</b></span>
    </div>""", unsafe_allow_html=True)

    _section('IN-SAMPLE EQUITY CURVE', f'{selected_approach} · {params["rebal_label"]} · full sample (not OOS)')
    render_oos_chart(grid, selected_approach)

    _section('MONTHLY RETURNS (IN-SAMPLE)', f'{selected_approach} · full sample')
    render_monthly_table(swf['oos_returns'])

    _section('OPTIMIZED WEIGHTS', f'{selected_approach} · trade these')
    render_weights_table(grid, selected_approach)


# =============================================================================
# EW RUN + DISPLAY
# =============================================================================

def _run_ew(symbols, rebal_months, fetch_days, txn_cost, rebal_label, period_label, benchmark=()):
    """Compute equal-weight returns with rebalancing + txn costs."""
    data, valid = fetch_symbol_history(tuple(symbols), days=fetch_days)
    if data is None or len(valid) < 2:
        st.warning('Need ≥2 assets with data'); return

    returns = data.pct_change().dropna()
    n_assets = len(valid)
    eq_w = np.ones(n_assets) / n_assets

    # Rebalance schedule
    if rebal_months == -1:
        dates = returns.index; rebal_set = set()
    elif rebal_months == 0:
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
    metrics = _calc_oos_metrics(ew_series)
    if metrics is None:
        st.warning('Insufficient data for metrics'); return

    bench, bench_failed = benchmark_series(benchmark, fetch_days, data.index)
    _warn_failed(bench_failed)

    st.session_state.port_ew_result = {
        'ew_returns': ew_series, 'metrics': metrics, 'symbols': valid,
        'rebal_label': rebal_label, 'period_label': period_label,
        'rebal_set': rebal_set, 'txn_cost': txn_cost,
        'bench': _bench_metrics(bench),
    }


def _display_ew(is_mobile, theme):
    import portfolio
    if 'port_ew_result' not in st.session_state: return
    res = st.session_state.port_ew_result
    m = res['metrics']; ew_ret = res['ew_returns']
    benches = res.get('bench') or []
    pos_c = portfolio.C_POS; neg_c = portfolio.C_NEG
    _bg3 = theme.get('bg3', '#0f172a'); _bdr = theme.get('border', '#1e293b')
    _txt2 = theme.get('text2', '#94a3b8'); _mut = theme.get('muted', '#475569')

    # Summary bar
    st.markdown(f"""<div style='margin-top:12px;padding:5px 10px;background:{C_BG};font-family:{FONTS};border-radius:4px;
        font-size:10px;color:{C_TXT2};display:flex;justify-content:space-between;flex-wrap:wrap;gap:4px'>
        <span><b style='color:{C_TXT}'>Equal Weight</b>
        &nbsp;·&nbsp;{m['n_days']} days · {m['oos_years']}y · {res['rebal_label']} · {res['period_label']}</span>
        <span>Win% <b style='color:{pos_c}'>{m["win_rate"]*100:.1f}%</b>
        &nbsp;Sharpe <b style='color:{pos_c}'>{m["sharpe"]:.2f}</b>
        &nbsp;Sortino <b style='color:{pos_c}'>{m["sortino"]:.2f}</b>
        &nbsp;MAR <b style='color:{pos_c}'>{m["mar"]:.2f}</b>
        &nbsp;Tot <b style='color:{pos_c if m["total_ret"]>=0 else neg_c}'>{m["total_ret"]*100:.1f}%</b>
        &nbsp;MDD <b style='color:{neg_c}'>{m["max_dd"]*100:.1f}%</b></span>
    </div>""", unsafe_allow_html=True)

    for i, (b_sym, _b_ret, b_m) in enumerate(benches):
        bc = BENCH_COLORS[i % len(BENCH_COLORS)]
        gap = m['total_ret'] - b_m['total_ret']
        st.markdown(f"""<div style='margin-top:4px;padding:5px 10px;background:{C_BG};font-family:{FONTS};border-radius:4px;
            font-size:10px;color:{C_TXT2};display:flex;justify-content:space-between;flex-wrap:wrap;gap:4px'>
            <span><b style='color:{bc}'>{b_sym}</b>
            &nbsp;·&nbsp;benchmark · {b_m['n_days']} days · {b_m['oos_years']}y</span>
            <span>Win% <b style='color:{bc}'>{b_m["win_rate"]*100:.1f}%</b>
            &nbsp;Sharpe <b style='color:{bc}'>{b_m["sharpe"]:.2f}</b>
            &nbsp;Sortino <b style='color:{bc}'>{b_m["sortino"]:.2f}</b>
            &nbsp;MAR <b style='color:{bc}'>{b_m["mar"]:.2f}</b>
            &nbsp;Tot <b style='color:{bc}'>{b_m["total_ret"]*100:.1f}%</b>
            &nbsp;MDD <b style='color:{bc}'>{b_m["max_dd"]*100:.1f}%</b>
            &nbsp;<span style='color:{C_MUTE}'>EW − {b_sym}</span>
            <b style='color:{pos_c if gap >= 0 else neg_c}'>{gap*100:+.1f}%</b></span>
        </div>""", unsafe_allow_html=True)

    # Equity curve + Drawdown (2-panel like MC chart)
    _section('EW EQUITY CURVE', f'{res["rebal_label"]} · cost {res["txn_cost"]*100:.2f}%')
    _pbg = theme.get('plot_bg', '#121212'); _grd = theme.get('grid', '#1f1f1f')
    _axl = theme.get('axis_line', '#2a2a2a'); _tk = theme.get('tick', '#888888')

    cum_arr = (1 + ew_ret).cumprod().values
    cum = (1 + ew_ret).cumprod() * 100
    end_val = float(cum.iloc[-1])
    end_c = '#f8fafc' if end_val >= 100 else neg_c

    # Drawdown
    peak = np.maximum.accumulate(cum_arr)
    dd = (cum_arr - peak) / peak * 100  # percentage

    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=1, row_heights=[0.75, 0.25],
                        shared_xaxes=True, vertical_spacing=0.04)

    # Equity curve
    tot_pct = (cum_arr[-1] - 1) * 100
    eq_lbl = f'Equal Weight ({tot_pct:+.1f}%)  Sharpe {m["sharpe"]:.2f} · Win {m["win_rate"]*100:.0f}%'
    fig.add_trace(go.Scatter(x=cum.index, y=cum.values,
        mode='lines', line=dict(color=pos_c, width=2.2),
        name=eq_lbl, hovertemplate='%{y:.1f}<extra></extra>'), row=1, col=1)
    fig.add_hline(y=100, line=dict(color=_grd, width=0.8, dash='dot'), row=1, col=1)

    # Benchmark lines, aligned to the EW window
    b_plots = []
    for i, (b_sym, b_ret, _bm) in enumerate(benches):
        b_al = b_ret.loc[b_ret.index >= ew_ret.index[0]]
        b_m = _calc_oos_metrics(b_al) if len(b_al) >= 5 else None
        if b_m is None: continue
        b_cum = np.cumprod(1 + b_al.values) * 100
        b_pk = np.maximum.accumulate(b_cum)
        bc = BENCH_COLORS[i % len(BENCH_COLORS)]
        b_plots.append({'sym': b_sym, 'idx': b_al.index, 'cum': b_cum,
                        'dd': (b_cum - b_pk) / b_pk * 100, 'color': bc})
        b_lbl = (f'{b_sym} ({b_cum[-1]-100:+.1f}%)  Sharpe {b_m["sharpe"]:.2f} '
                 f'· Win {b_m["win_rate"]*100:.0f}%')
        fig.add_trace(go.Scatter(x=b_al.index, y=b_cum, mode='lines',
            line=dict(color=_tint(bc, 0.78), width=1.2), name=b_lbl,
            hovertemplate=f'{b_sym}: %{{y:.1f}}<extra></extra>'), row=1, col=1)

    # Rebalance markers
    rebal_dates = sorted(res['rebal_set'])
    for rd in rebal_dates:
        if rd in cum.index:
            fig.add_vline(x=rd, line=dict(color='#fbbf24', width=0.5), opacity=0.25, row=1, col=1)

    # End value
    fig.add_annotation(text=f"<b>{end_val:.0f}</b>", x=cum.index[-1], y=end_val,
        showarrow=False, font=dict(size=12, color=end_c, family=FONTS),
        xanchor='left', xshift=6, yanchor='middle', row=1, col=1)

    # Drawdown panel
    nr = neg_c.lstrip('#'); rv, gv, bv = int(nr[:2], 16), int(nr[2:4], 16), int(nr[4:6], 16)
    fig.add_trace(go.Scatter(x=cum.index, y=dd, mode='lines', fill='tozeroy',
        line=dict(color=neg_c, width=1), fillcolor=f'rgba({rv},{gv},{bv},0.2)',
        name='Drawdown', showlegend=False,
        hovertemplate='DD: %{y:.1f}%<extra></extra>'), row=2, col=1)

    for b in b_plots:
        fig.add_trace(go.Scatter(x=b['idx'], y=b['dd'], mode='lines',
            line=dict(color=_tint(b['color'], 0.40), width=0.9),
            name=f"{b['sym']} Drawdown", showlegend=False,
            hovertemplate=f"{b['sym']} DD: %{{y:.1f}}%<extra></extra>"), row=2, col=1)

    fig.update_layout(template='plotly_dark', height=400,
        margin=dict(l=40, r=60, t=35, b=25),
        plot_bgcolor=_pbg, paper_bgcolor=_pbg,
        showlegend=True,
        legend=dict(x=0.01, y=0.88, bgcolor='rgba(0,0,0,0)',
                    font=dict(size=12, color='#ffffff', family=FONTS), borderwidth=0),
        hovermode='x unified', font=dict(family=FONTS))
    fig.update_xaxes(gridcolor=_grd, linecolor=_axl,
        tickfont=dict(color=_tk, size=9, family=FONTS), showgrid=False)
    fig.update_yaxes(gridcolor=_grd, linecolor=_axl,
        tickfont=dict(color=_tk, size=9, family=FONTS), side='right')
    fig.update_yaxes(ticksuffix='%', row=2, col=1)
    st.plotly_chart(fig, width='stretch', config={
        'scrollZoom': True, 'displayModeBar': False, 'responsive': True})

    # Monthly returns
    _section('MONTHLY RETURNS', 'Equal weight')
    render_monthly_table(ew_ret)

    # Weights table
    _section('WEIGHTS', 'Equal weight allocation')
    symbols = res['symbols']; n = len(symbols); wt = 1.0 / n
    th = f"padding:4px 8px;border-bottom:1px solid {_bdr};color:#f8fafc;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.06em;"
    td = f"padding:5px 8px;border-bottom:1px solid {_bdr}22;"
    html = f"""<div style='overflow-x:auto;border:1px solid {_bdr};border-radius:6px'>
    <table style='border-collapse:collapse;font-family:{FONTS};font-size:11px;width:100%;line-height:1.3'>
        <thead style='background:{_bg3}'><tr>
            <th style='{th}text-align:left'>SYMBOL</th>
            <th style='{th}text-align:left'>NAME</th>
            <th style='{th}text-align:right'>WEIGHT</th>
        </tr></thead><tbody>"""
    for sym in symbols:
        name = SYMBOL_NAMES.get(sym, clean_symbol(sym))
        html += f"""<tr>
            <td style='{td}color:{pos_c};font-weight:600'>{sym}</td>
            <td style='{td}color:{_txt2}'>{name}</td>
            <td style='{td}text-align:right;color:#f8fafc;font-weight:600'>{wt*100:.1f}%</td>
        </tr>"""
    html += "</tbody></table></div>"
    st.markdown(html, unsafe_allow_html=True)
