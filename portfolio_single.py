import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from config import FUTURES_GROUPS, THEMES, SYMBOL_NAMES, FONTS, clean_symbol
from portfolio import (C_MUTE, C_BG, C_TXT, C_TXT2, C_GOLD, BENCH_COLORS, MAX_BENCHMARKS, _tint,
                       REBAL_OPTIONS, PERIOD_OPTIONS, SCORE_TO_RANK, OBJECTIVES,
                       fetch_symbol_history, fetch_notes, benchmark_series, _bench_metrics, _calc_oos_metrics,
                       run_walkforward_grid, run_fullsample,
                       render_ranking_table,
                       render_weights_table, render_oos_chart,
                       render_monthly_table, _section)
# The universe is ticked, not typed, and it is the same checkbox list SPREADS
# uses -- one picker, one meaning of "basket", wherever you are.
from spreads import basket_picker


def _pool(picked):
    """Every symbol in the ticked baskets, deduplicated, in basket order.

    Baskets overlap on purpose -- GC=F sits in Metals and in Inflation Hedge --
    and a symbol listed twice would take two slices of the same portfolio."""
    out = []
    for g in picked:
        for sym in FUTURES_GROUPS.get(g, []):
            if sym not in out:
                out.append(sym)
    return out


def _clean_benchmarks(raw):
    """Benchmark tickers go into yfinance URLs and into HTML labels, so keep only
    the characters real tickers use. 'SPY, XLV' -> ['SPY', 'XLV']."""
    if not raw: return []
    out = []
    for part in raw.replace(';', ',').split(','):
        sym = ''.join(c for c in part.strip().upper() if c.isalnum() or c in '.^=-:')
        if sym and sym not in out: out.append(sym)
    return out[:MAX_BENCHMARKS]


def _fetch_failure(symbols, fetch_days, what, min_hist_days=0):
    """Explain a failed run instead of shrugging. A big basket usually fails for
    one of three reasons: Yahoo refused some symbols, the symbols barely overlap,
    or they overlap but no lookback window fits inside that overlap."""
    notes = fetch_notes(symbols, fetch_days, min_hist_days)
    if not notes:
        st.warning(f'Need ≥2 assets with sufficient history for {what}')
        return
    ok, bad = notes['n_ok'], notes['no_data']
    if ok < 2 and len(notes['too_new']) > notes['n_requested'] - 2:
        st.warning(f"Min Hist excluded {len(notes['too_new'])} of {notes['n_requested']} symbols "
                   f"(nothing listed before {notes['cutoff'].date()}). Lower Min Hist Y, or raise Period "
                   f"— a symbol cannot show more history than the Period fetches.")
        return
    if notes['too_new']:
        st.caption(f"Min Hist excluded {len(notes['too_new'])} symbol(s) listed after "
                   f"{notes['cutoff'].date()}: {', '.join(notes['too_new'][:15])}"
                   + (f" and {len(notes['too_new']) - 15} more" if len(notes['too_new']) > 15 else ''))
    if ok < 2:
        msg = f"Only {ok} of {notes['n_requested']} symbols returned usable history."
        if bad:
            shown = ', '.join(bad[:15]) + (f" and {len(bad) - 15} more" if len(bad) > 15 else '')
            msg += f" No data for: {shown}."
        st.warning(msg)
        if len(bad) > 5:
            st.caption('Yahoo rate-limits large batches — that many failures usually means throttling '
                       'rather than bad tickers. Wait a minute and run again.')
    elif notes['common_rows'] < 50:
        lim = ', '.join(f"{sym} ({d.date()})" for sym, d in notes['limiters'])
        st.warning(f"{ok} symbols fetched, but they only overlap for {notes['common_rows']} trading days. "
                   f"Latest listings: {lim}. Set Min Hist Y to exclude them automatically, or drop them by hand.")
    else:
        span = f"{notes['start'].date()} to {notes['end'].date()}" if notes['start'] is not None else 'the shared window'
        lim = ', '.join(f"{sym} ({d.date()})" for sym, d in notes['limiters'])
        st.warning(f"{ok} symbols share only {notes['common_rows']} trading days ({span}), which is too short for "
                   f"any lookback in this mode. Latest listings: {lim} — set Min Hist Y to exclude symbols that "
                   f"new, or use a shorter lookback.")


def _note_window(symbols, fetch_days, min_hist_days=0):
    """A couple of recent IPOs can quietly cut a 5-year request down to 2 years,
    taking the longer lookbacks with them. Say so rather than let it pass."""
    notes = fetch_notes(symbols, fetch_days, min_hist_days)
    if not notes or not notes['limiters'] or notes['start'] is None: return
    if notes['too_new']:
        st.caption(f"ⓘ Min Hist excluded {len(notes['too_new'])} symbol(s) listed after "
                   f"{notes['cutoff'].date()}: {', '.join(notes['too_new'][:15])}"
                   + (f" and {len(notes['too_new']) - 15} more" if len(notes['too_new']) > 15 else ''))
    if notes['common_rows'] >= notes['union_rows'] * 0.9: return
    lim = ', '.join(f"{sym} ({d.date()})" for sym, d in notes['limiters'][:2])
    lost = notes['union_rows'] - notes['common_rows']
    st.caption(f"ⓘ Shared history starts {notes['start'].date()} — {notes['common_rows']} of "
               f"{notes['union_rows']} trading days, {lost} lost to the latest listings ({lim}). "
               f"Backtests and the longer lookbacks only see the shared window.")


def _warn_failed(failed):
    if failed:
        st.warning(f"No usable history for {', '.join(failed)} — left out of the comparison")


def _validate(symbols, max_wt_str, min_wt_str, min_pos_str, round_str, min_hist_str,
              cost_str, sims_str, fetch_days):
    """Catch the settings that quietly fight each other before a run burns a minute
    on them. Returns (errors, notes): errors stop the run, notes are just FYI."""
    errors = []; notes = []

    def num(s, default=None):
        try: return float(s) if str(s).strip() else default
        except (ValueError, TypeError): return None

    if len(symbols) < 2:
        errors.append('Tick baskets holding at least 2 symbols between them — '
                      'a portfolio needs two.')

    max_wt, min_wt = num(max_wt_str, 50), num(min_wt_str, 0)
    min_pos, step = num(min_pos_str), num(round_str)
    min_hist, cost, sims = num(min_hist_str), num(cost_str, 0.10), num(sims_str, 10000)

    if max_wt is None: errors.append('Max Wt % must be a number.')
    if min_wt is None: errors.append('Min Wt % must be a number.')
    if num(min_pos_str, 0) is None: errors.append('Min Pos % must be a number, or blank.')
    if num(round_str, 0) is None: errors.append('Round % must be a number, or blank.')
    if num(min_hist_str, 0) is None: errors.append('Min Hist Y must be a number, or blank.')
    if cost is None: errors.append('Cost % must be a number.')
    if sims is None: errors.append('Sims must be a number.')
    if errors: return errors, notes

    n = max(len(symbols), 1)
    if min_wt and max_wt and min_wt >= max_wt:
        notes.append(f'Min Wt {min_wt:g}% is not below Max Wt {max_wt:g}% \u2014 the floor will be ignored.')
    if min_wt and min_wt * n > 100:
        notes.append(f'Min Wt {min_wt:g}% across {n} symbols needs {min_wt * n:.0f}% \u2014 '
                     f'it will be cut to {100 / n:.2f}%, the most that fits.')
    if max_wt and max_wt * n < 100:
        errors.append(f'Max Wt {max_wt:g}% across {n} symbols caps the portfolio at {max_wt * n:.0f}%. '
                      f'Raise it to at least {100 / n:.1f}%.')
    if min_pos and max_wt and min_pos >= max_wt:
        errors.append(f'Min Pos {min_pos:g}% is at or above Max Wt {max_wt:g}%, so every position would be dropped.')
    if step and max_wt and step > max_wt:
        errors.append(f'Round {step:g}% is coarser than Max Wt {max_wt:g}% \u2014 nothing could round to a valid weight.')
    if min_pos and step and step > min_pos:
        notes.append(f'Round {step:g}% is coarser than Min Pos {min_pos:g}%, so rounding decides what survives.')
    if min_hist and min_hist * 365 >= fetch_days:
        errors.append(f'Min Hist {min_hist:g}y needs more history than Period fetches '
                      f'({fetch_days / 365:.1f}y), so every symbol would be excluded.')
    if sims and sims < 1000:
        notes.append(f'Sims {sims:.0f} is below the 1,000 minimum \u2014 it will be raised to 1,000.')
    if cost and cost > 5:
        notes.append(f'Cost {cost:g}% is above the 5% cap \u2014 it will be clamped.')
    return errors, notes


def render_single_tab(is_mobile):
    import portfolio
    theme_name = st.session_state.get('theme', 'Dark')
    theme = THEMES.get(theme_name, THEMES['Dark'])
    portfolio.C_POS = theme['pos']; portfolio.C_NEG = theme['neg']
    _lbl = f"color:#f8fafc;font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;font-family:{FONTS}"

    # Native Streamlit labels, restyled to the SANPO scale. They were hand-rolled
    # markdown before, which the tab container clipped to a sliver -- the fields
    # ended up effectively unlabelled.
    st.markdown(f"""<style>
        .stSelectbox label p, .stTextInput label p {{
            font-size: 10px !important; font-weight: 600 !important; text-transform: uppercase;
            letter-spacing: 0.08em; color: #cbd5e1 !important; font-family: {FONTS} !important;
        }}
        .stSelectbox label, .stTextInput label {{ margin-bottom: 1px !important; }}
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

    def _group(title, blurb):
        st.markdown(f"<div style='margin:16px 0 7px;font-family:{FONTS}'>"
                    f"<span style='color:#f8fafc;font-size:10px;font-weight:700;letter-spacing:0.1em'>{title}</span>"
                    f"<span style='color:#64748b;font-size:10px;margin-left:8px'>{blurb}</span></div>",
                    unsafe_allow_html=True)

    _defaults = {'port_sims': '10000',
                 'port_maxwt': '50', 'port_minwt': '0', 'port_cost': '0.10', 'port_maxvol': '',
                 'port_minret': '', 'port_minpos': '', 'port_round': '', 'port_minhist': ''}
    for k, v in _defaults.items():
        if k not in st.session_state: st.session_state[k] = v
    for k, v in [('port_maxwt', '50'), ('port_minwt', '0'), ('port_cost', '0.10'), ('port_sims', '10000')]:
        if not st.session_state.get(k): st.session_state[k] = v

    # ---------------------------------------------------------------- what to trade
    # No group header above this one: the picker's own BASKETS IN PLAY label
    # already says what it is, and a second heading over it was just noise.
    # Ticked, not typed. The Preset selectbox could only load one basket and the
    # Symbols box was the real input after that, so a two-basket universe meant
    # pasting tickers by hand; the picker makes it a pair of clicks. Ticking
    # several pools their symbols into ONE portfolio -- the All sub-tab is the
    # view that keeps each basket separate.
    picked = basket_picker('po', is_mobile, theme, label='Baskets in play')
    symbols = _pool(picked)
    dupes = sum(len(FUTURES_GROUPS.get(g, [])) for g in picked) - len(symbols)
    overlap = f' \u00b7 {dupes} duplicate{"s" if dupes != 1 else ""} dropped' if dupes else ''
    st.markdown(f"<div style='font-size:10px;color:{C_MUTE};font-family:{FONTS};"
                f"padding:2px 0 8px 2px'>{len(picked)} baskets \u00b7 {len(symbols):,} "
                f"symbols{overlap}</div>", unsafe_allow_html=True)

    # The name the results carry. One basket names itself; several are only
    # honestly described by their count.
    if len(picked) == 1:
        st.session_state.port_preset_name = picked[0]
    elif picked:
        st.session_state.port_preset_name = f'{len(picked)} baskets'
    else:
        st.session_state.port_preset_name = 'Portfolio'

    a1, a2 = st.columns(2)
    with a1:
        mode = st.selectbox('Mode', ['Monte Carlo (Walk-Forward)', 'Monte Carlo (Full Sample)', 'Equal Weight'],
                            key='port_mode',
                            help='How the weights are chosen. Walk-Forward optimises on past data only and scores the '
                                 'period that follows, which is the honest test. Full Sample optimises on all the data '
                                 'and scores the same data, which flatters. Equal Weight skips optimisation: every '
                                 'asset gets 1/N.')
    with a2:
        bench_input = st.text_input('Benchmark (optional)', key='port_bench',
                                    placeholder=f'e.g. SPY, XLV (max {MAX_BENCHMARKS})',
                                    help='Comparison tickers, comma-separated. They get no weight and are not part of '
                                         'the portfolio \u2014 each is drawn on the chart and added to the ranking '
                                         'table so you can see whether the portfolio beat it.')

    is_mc = mode == 'Monte Carlo (Walk-Forward)'
    is_fs = mode == 'Monte Carlo (Full Sample)'
    _dis = not (is_mc or is_fs)

    # ---------------------------------------------------------------- how to test
    _group('HOW TO TEST', 'what the optimiser aims at, and over what history')
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        score = st.selectbox('Objective', OBJECTIVES, key='port_score', disabled=_dis,
                             help='What the optimiser maximises, and what the ranking table then sorts on \u2014 the '
                                  'same nine SPREADS offers. Composite is the average rank across Sharpe, Sortino, '
                                  'ROA and ER, each discounted by the square root of the window length so an '
                                  'approach that burned a long warm-up cannot win on the short sample left over. '
                                  'Sharpe = return per unit of volatility. Sortino = same but only downside '
                                  'volatility. ROA = total return over the worst drawdown. ER = how straight the '
                                  'equity curve is, 1.0 being a straight line. MAR = return per unit of average '
                                  'drawdown. R\u00b2 = straightness again, fitted. Total Return = raw growth. '
                                  'Win Rate = share of up days.')
    with b2:
        rebal_label = st.selectbox('Rebalance', list(REBAL_OPTIONS.keys()), index=2, key='port_rebal',
                                   help='How often holdings are reset to target weights. Every reset pays Cost % on what '
                                        'it trades, so more frequent is not automatically better.')
    with b3:
        period_label = st.selectbox('Period', list(PERIOD_OPTIONS.keys()), index=2, key='port_period',
                                    help='How much price history to pull. Longer gives more to learn from and a longer '
                                         'backtest, but drags in older market regimes.')
    with b4:
        direction = st.selectbox('Direction', ['Long Only', 'Long/Short'], key='port_direction', disabled=_dis,
                                 help='Long Only keeps every weight at 0 or above. Long/Short allows negative weights, '
                                      'so the portfolio can short (Min Wt % is ignored then).')

    # ---------------------------------------------------------------- how to execute
    _group('HOW TO EXECUTE', 'shape the weights into something you can actually trade')
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        min_pos_str = st.text_input('Min Pos % (drop below)', key='port_minpos', placeholder='e.g. 1', disabled=_dis,
                                    help='Dust cut. After the weights are chosen, anything under this goes to 0 and the '
                                         'rest are rescaled to 100%. Set 1 and a 0.4% sliver becomes nothing. Blank '
                                         'keeps every sliver. The opposite of Min Wt %: this throws assets out.')
    with c2:
        round_str = st.text_input('Round % (step)', key='port_round', placeholder='e.g. 1', disabled=_dis,
                                  help='Snap the final weights to a clean step: 1 gives whole percents (34%, 29%, 0%), '
                                       '0.5 gives half percents. Under half a step rounds to 0, and the weights still '
                                       'add to exactly 100%. Pair with Min Pos % to be sure slivers are gone.')
    with c3:
        max_wt_str = st.text_input('Max Wt %', key='port_maxwt', disabled=_dis,
                                   help='Ceiling on any single asset, so nothing dominates. 50 means no holding above 50%.')
    with c4:
        min_wt_str = st.text_input('Min Wt %', key='port_minwt', disabled=_dis,
                                   help='Floor on EVERY asset \u2014 it forces each one in at this weight or more. '
                                        '0 means no floor. It keeps assets in; it does not round anything.')

    # ---------------------------------------------------------------- the rest
    with st.expander('More settings \u2014 data, constraints and cost'):
        d1, d2, d3, d4, d5 = st.columns(5)
        with d1:
            min_hist_str = st.text_input('Min Hist Y', key='port_minhist', placeholder='e.g. 2',
                                         help='Leave out symbols listed more recently than this many years. Every asset '
                                              'needs a price on every day of the backtest, so one recent IPO drags the '
                                              'whole basket down to its listing date. The run says what it dropped.')
        with d2:
            sims_str = st.text_input('Sims', key='port_sims', disabled=_dis,
                                     help='Random weight combinations tested per lookback window. Higher is steadier '
                                          'but slower. 10,000 is a good default.')
        with d3:
            max_vol_str = st.text_input('Max Vol %', key='port_maxvol', placeholder='e.g. 15', disabled=_dis,
                                        help='Soft cap on annualised volatility. Portfolios above it are penalised in '
                                             'the search rather than banned, so the result can still exceed it.')
        with d4:
            min_ret_str = st.text_input('Min Ret %', key='port_minret', placeholder='e.g. 5', disabled=_dis,
                                        help='Soft floor on annualised return. Portfolios below it are penalised in the '
                                             'search rather than banned.')
        with d5:
            cost_str = st.text_input('Cost %', key='port_cost',
                                     help='Transaction cost charged on turnover at every rebalance. 0.10 = 10 basis points.')

    # Run button
    if is_mc:
        btn_label = '\u25b6  Optimize (WF)'
    elif is_fs:
        btn_label = '\u25b6  Optimize (Full)'
    else:
        btn_label = '\u25b6  Run EW'
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
        if not symbols:
            st.warning('No baskets ticked \u2014 nothing to run.'); return

        errors, notes = _validate(symbols, max_wt_str, min_wt_str, min_pos_str, round_str,
                                  min_hist_str, cost_str, sims_str, fetch_days)
        for e in errors: st.error(e)
        for note in notes: st.caption(note)
        if errors: return

        bench = _clean_benchmarks(bench_input)
        try: min_hist_days = int(max(0, min(20, float(min_hist_str))) * 365) if min_hist_str.strip() else 0
        except (ValueError, TypeError): min_hist_days = 0
        if is_mc:
            _run_mc(symbols, score, rebal_label, rebal, period_label, fetch_days,
                    direction, sims_str, max_wt_str, min_wt_str, max_vol_str, min_ret_str,
                    txn_cost, bench, min_pos_str, round_str, min_hist_days)
        elif is_fs:
            _run_fs(symbols, score, rebal_label, rebal, period_label, fetch_days,
                    direction, sims_str, max_wt_str, min_wt_str, max_vol_str, min_ret_str,
                    txn_cost, bench, min_pos_str, round_str, min_hist_days)
        else:
            _run_ew(symbols, rebal, fetch_days, txn_cost, rebal_label, period_label, bench, min_hist_days)

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
            txn_cost, benchmark=(), min_pos_str='', round_str='', min_hist_days=0):
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
    try: round_step = max(0.1, min(25, float(round_str))) / 100.0 if round_str.strip() else 0.0
    except (ValueError, TypeError): round_step = 0.0

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
                                 benchmarks=benchmark, min_pos=min_pos, round_step=round_step,
                                 min_history_days=min_hist_days)
    progress.empty()

    if not grid or not grid['results']:
        _fetch_failure(symbols, fetch_days, 'walk-forward', min_hist_days)
        fetch_symbol_history.clear()  # don't serve the failure from cache for 30 min
        return
    _warn_failed(grid.get('bench_failed'))

    _note_window(symbols, fetch_days, min_hist_days)
    st.session_state.port_grid = grid
    # Named from the ticked baskets before the run started, so the reverse lookup
    # that used to put a name to a typed symbol list has nothing left to guess at.
    preset_name = st.session_state.get('port_preset_name') or 'Portfolio'
    st.session_state.port_params = {
        'score': score, 'rebal_label': st.session_state.get('port_rebal', 'Quarterly'),
        'period_label': st.session_state.get('port_period', '5 Years'),
        'direction': 'L/S' if allow_short else 'Long',
        'min_wt': min_wt, 'max_wt': max_wt, 'n_sims': n_sims, 'txn_cost': txn_cost,
        'max_vol': max_vol, 'min_ann_ret': min_ann_ret, 'min_pos': min_pos, 'round_step': round_step,
        'min_hist_days': min_hist_days,
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
    if params.get('round_step'): constraints_str += f" · round to {params['round_step']*100:g}%"
    if params.get('min_hist_days'): constraints_str += f" · min hist {params['min_hist_days']/365:g}y"
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
        &nbsp;ROA <b style='color:{portfolio.C_POS}'>{sm["roa"]:.2f}</b>
        &nbsp;ER <b style='color:{portfolio.C_POS}'>{sm["er"]:.2f}</b>
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
            txn_cost, benchmark=(), min_pos_str='', round_str='', min_hist_days=0):
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
    try: round_step = max(0.1, min(25, float(round_str))) / 100.0 if round_str.strip() else 0.0
    except (ValueError, TypeError): round_step = 0.0

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
                          rebal_months=rebal, benchmarks=benchmark, min_pos=min_pos,
                          round_step=round_step, min_history_days=min_hist_days)
    progress.empty()

    if not grid or not grid['results']:
        _fetch_failure(symbols, fetch_days, 'full-sample optimization', min_hist_days)
        fetch_symbol_history.clear()
        return
    _warn_failed(grid.get('bench_failed'))

    _note_window(symbols, fetch_days, min_hist_days)
    st.session_state.port_fs_result = grid
    # Named from the ticked baskets before the run started, so the reverse lookup
    # that used to put a name to a typed symbol list has nothing left to guess at.
    preset_name = st.session_state.get('port_preset_name') or 'Portfolio'
    st.session_state.port_fs_params = {
        'score': score, 'rebal_label': st.session_state.get('port_rebal', 'Quarterly'),
        'period_label': st.session_state.get('port_period', '5 Years'),
        'direction': 'L/S' if allow_short else 'Long',
        'min_wt': min_wt, 'max_wt': max_wt, 'n_sims': n_sims, 'txn_cost': txn_cost,
        'max_vol': max_vol, 'min_ann_ret': min_ann_ret, 'min_pos': min_pos, 'round_step': round_step,
        'min_hist_days': min_hist_days,
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
    if params.get('round_step'): constraints_str += f" · round to {params['round_step']*100:g}%"
    if params.get('min_hist_days'): constraints_str += f" · min hist {params['min_hist_days']/365:g}y"
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
        &nbsp;ROA <b style='color:{portfolio.C_POS}'>{sm["roa"]:.2f}</b>
        &nbsp;ER <b style='color:{portfolio.C_POS}'>{sm["er"]:.2f}</b>
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

def _run_ew(symbols, rebal_months, fetch_days, txn_cost, rebal_label, period_label, benchmark=(),
            min_hist_days=0):
    """Compute equal-weight returns with rebalancing + txn costs."""
    data, valid = fetch_symbol_history(tuple(symbols), days=fetch_days, min_history_days=min_hist_days)
    if data is None or len(valid) < 2:
        _fetch_failure(symbols, fetch_days, 'an equal-weight backtest', min_hist_days)
        fetch_symbol_history.clear()
        return

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

    _note_window(symbols, fetch_days, min_hist_days)
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
        &nbsp;ROA <b style='color:{pos_c}'>{m["roa"]:.2f}</b>
        &nbsp;ER <b style='color:{pos_c}'>{m["er"]:.2f}</b>
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
            &nbsp;ROA <b style='color:{bc}'>{b_m["roa"]:.2f}</b>
            &nbsp;ER <b style='color:{bc}'>{b_m["er"]:.2f}</b>
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
