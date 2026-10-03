import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
import logging

from config import FUTURES_GROUPS, THEMES, SYMBOL_NAMES, FONTS, clean_symbol
from spreads import (compute_sector_spreads, sort_spread_pairs,
                     render_spread_table, render_spread_charts)

logger = logging.getLogger(__name__)

# yf interval, resample target, bars per trading day, max calendar days yfinance allows
INTERVAL_CONFIG = {
    '15m': {'yf': '15m', 'resample': None, 'bars_per_day': 26,  'max_cal_days': 59},
    '1h':  {'yf': '1h',  'resample': None, 'bars_per_day': 7,   'max_cal_days': 729},
    '4h':  {'yf': '1h',  'resample': '4h', 'bars_per_day': 2,   'max_cal_days': 729},
    '1d':  {'yf': '1d',  'resample': None, 'bars_per_day': 1,   'max_cal_days': None},
    '1wk': {'yf': '1wk', 'resample': None, 'bars_per_day': 0.2, 'max_cal_days': None},
}

# Universal lookback — trading days (0 = YTD)
LOOKBACK_OPTIONS = {
    'YTD':      0,
    '1 Day':    1,
    '2 Days':   2,
    '5 Days':   5,
    '10 Days':  10,
    '30 Days':  30,
    '60 Days':  60,
    '120 Days': 120,
    '240 Days': 240,
    '520 Days': 520,
}

ANN_FACTORS = {
    '15m': 26 * 252,
    '1h':  7 * 252,
    '4h':  2 * 252,
    '1d':  252,
    '1wk': 52,
}

# Every pair gets scored, so N symbols means N*(N-1)/2 spreads. 40 is already
# 780 pairs; past that a typo in the box turns into a very long fetch.
MAX_BASKET = 40


def _parse_basket(raw):
    """'gc=f, si=f ; si=f' -> ['GC=F', 'SI=F']. These symbols go into yfinance
    URLs and into HTML labels, so keep only the characters real tickers use."""
    if not raw:
        return []
    out = []
    for part in raw.replace(';', ',').replace('\n', ',').split(','):
        sym = ''.join(c for c in part.strip().upper() if c.isalnum() or c in '.^=-:')
        if sym and sym not in out:
            out.append(sym)
    return out


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_interval_data(symbols, interval_key, lookback_days):
    cfg = INTERVAL_CONFIG[interval_key]
    symbols = list(symbols or ())
    if len(symbols) < 2:
        return None

    if lookback_days == 0:  # YTD
        start = datetime.now().replace(month=1, day=1).strftime('%Y-%m-%d')
    else:
        cal_days = int(lookback_days * 1.6)
        if cfg['max_cal_days']:
            cal_days = min(cal_days, cfg['max_cal_days'])
        start = (datetime.now() - pd.Timedelta(days=max(cal_days, 2))).strftime('%Y-%m-%d')

    data = pd.DataFrame()
    for sym in symbols:
        try:
            hist = yf.Ticker(sym).history(start=start, interval=cfg['yf'])
            if hist.empty:
                continue
            closes = hist['Close'].copy()
            if closes.index.tz is not None:
                closes.index = closes.index.tz_convert('UTC').tz_localize(None)
            if cfg.get('resample'):
                closes = closes.resample(cfg['resample']).last().dropna()
            if interval_key in ('1d', '1wk'):
                closes.index = closes.index.normalize()
                closes = closes.groupby(closes.index).last()
            data[sym] = closes
        except Exception as e:
            logger.debug(f"[{sym}] fetch error ({interval_key}): {e}")

    if data.empty or len(data.columns) < 2:
        return None
    data = data.ffill().dropna()

    if lookback_days > 0:
        bars = max(int(lookback_days * cfg['bars_per_day']), 5)
        if len(data) > bars:
            data = data.iloc[-bars:]

    if len(data) < 5:
        return None
    return 100 * (data / data.iloc[0])


def render_sector_tab(is_mobile):
    theme_name = st.session_state.get('theme', 'Dark')
    theme = THEMES.get(theme_name, THEMES['Dark'])
    pos_c = theme['pos']
    _bg3 = theme.get('bg3', '#0f172a'); _mut = theme.get('muted', '#475569'); _txt2 = theme.get('text2', '#94a3b8')

    def _note(msg):
        st.markdown(f"<div style='padding:12px;color:{_mut};font-size:11px;font-family:{FONTS}'>{msg}</div>",
                    unsafe_allow_html=True)

    preset_names = ['Custom'] + list(FUTURES_GROUPS.keys())

    # First load: sit on Custom, but pre-fill the box with the first real group so
    # the tab has something to compute instead of an empty basket.
    if 'spread_sym_input' not in st.session_state:
        first_group = next(iter(FUTURES_GROUPS))
        st.session_state.spread_sym_input = ', '.join(FUTURES_GROUPS[first_group])
        st.session_state.spread_sector = 'Custom'
    if st.session_state.get('spread_sector') not in preset_names:
        st.session_state.spread_sector = 'Custom'

    def _on_preset_change():
        sel = st.session_state.spread_sector_sel
        if sel != 'Custom':
            st.session_state.spread_sym_input = ', '.join(FUTURES_GROUPS.get(sel, []))
        st.session_state.spread_sector = sel

    def _on_symbols_change():
        # Edit a loaded preset and it is no longer that preset. Say so, rather
        # than leaving the box labelled 'Metals' over a basket of tech names.
        sel = st.session_state.get('spread_sector_sel', 'Custom')
        if sel != 'Custom' and _parse_basket(st.session_state.spread_sym_input) != list(FUTURES_GROUPS.get(sel, [])):
            st.session_state.spread_sector_sel = 'Custom'
            st.session_state.spread_sector = 'Custom'

    # Controls — row 1: the basket
    if is_mobile:
        col_sec, col_sym = st.container(), st.container()
    else:
        col_sec, col_sym = st.columns([2, 6])

    with col_sec:
        st.selectbox("Preset", preset_names,
            index=preset_names.index(st.session_state.spread_sector),
            key='spread_sector_sel', on_change=_on_preset_change,
            help='Load a saved group into Symbols, or pick Custom and type your own basket.')

    with col_sym:
        st.text_input("Symbols", key='spread_sym_input', on_change=_on_symbols_change,
            placeholder='GC=F, SI=F, HG=F, ...',
            help=f'The basket to spread against itself, comma-separated (Yahoo symbols). '
                 f'Every pair is scored, so {MAX_BASKET} symbols is the cap. Edit a preset '
                 f'freely — what is in this box is what gets computed.')

    preset = st.session_state.spread_sector
    symbols = _parse_basket(st.session_state.spread_sym_input)
    over_cap = len(symbols) - MAX_BASKET
    symbols = symbols[:MAX_BASKET]

    # Controls — row 2: how to measure it
    if is_mobile:
        col_iv, col_lb = st.columns([1, 1])
        col_sort, col_dir = st.columns([1, 1])
    else:
        col_iv, col_lb, col_sort, col_dir = st.columns([2, 3, 3, 2])

    with col_iv:
        iv_keys = list(INTERVAL_CONFIG.keys())
        interval_key = st.selectbox("Interval", iv_keys,
            index=iv_keys.index(st.session_state.get('spread_interval', '1d')),
            key='spread_interval_sel',
            help='Bar size the spreads are measured on. Intraday intervals only reach '
                 'back so far: 15m to 60 days, 1h and 4h to 730.')
        st.session_state.spread_interval = interval_key
        ann_factor = ANN_FACTORS[interval_key]

    with col_lb:
        lb_keys = list(LOOKBACK_OPTIONS.keys())
        default_lb = st.session_state.get('spread_lookback', 'YTD')
        if default_lb not in lb_keys:
            default_lb = 'YTD'
        lookback_label = st.selectbox("Lookback", lb_keys,
            index=lb_keys.index(default_lb),
            key='spread_lookback_sel',
            help='How far back to score the spreads, in trading days.')
        st.session_state.spread_lookback = lookback_label
        lookback_days = LOOKBACK_OPTIONS[lookback_label]

    with col_sort:
        sort_options = ['Composite', 'Sharpe', 'Sortino', 'MAR', 'R²', 'Total', 'Win Rate']
        sort_key = st.selectbox("Sort by", sort_options, index=0,
            key='spread_sort_sel',
            help='Which metric ranks the pairs. Composite is the average rank across '
                 'Sharpe, Sortino, MAR and R², so a 1.0 is best on all four.')

    with col_dir:
        sort_dir = st.selectbox("Order", ['Desc', 'Asc'], index=0, key='spread_dir_sel')
        ascending = sort_dir == 'Asc'

    if len(symbols) < 2:
        _note('A spread needs two legs — put at least 2 symbols in the box, or load a preset.')
        return
    if over_cap > 0:
        st.caption(f"ⓘ Using the first {MAX_BASKET} symbols and ignoring {over_cap} more "
                   f"— every pair is scored, so the basket is capped at {MAX_BASKET}.")

    # Fetch and compute
    with st.spinner(f'Computing {preset} spreads ({interval_key} · {lookback_label})...'):
        data = _fetch_interval_data(tuple(symbols), interval_key, lookback_days)

    if data is None or len(data.columns) < 2:
        limits = {'15m': '60 days', '1h': '730 days', '4h': '730 days'}
        note = f" ({interval_key} is limited to the last {limits[interval_key]})" if interval_key in limits else ''
        got = 0 if data is None else len(data.columns)
        _note(f'Only {got} of {len(symbols)} symbols returned usable {interval_key} history, '
              f'and a spread needs 2{note}. Check the symbols, or try a longer interval.')
        return

    missing = [s for s in symbols if s not in data.columns]
    pairs = compute_sector_spreads(data, ann_factor)
    if not pairs:
        _note('No spreads computed')
        return

    sorted_pairs = sort_spread_pairs(pairs, sort_key, ascending)

    # Info bar
    best_long_sym = pairs[0].get('best_long_sym', '')
    best_long_sharpe = pairs[0].get('best_long_sharpe', 0)
    best_long_name = SYMBOL_NAMES.get(best_long_sym, clean_symbol(best_long_sym))
    n_combos = len(pairs)
    n_beats = sum(1 for p in pairs if p['beats_long'])
    fmt = '%d %b %H:%M' if interval_key in ('15m', '1h', '4h') else '%d %b %Y'
    start_date = data.index[0].strftime(fmt)
    end_date = data.index[-1].strftime(fmt)

    beats_c = pos_c if n_beats > 0 else _mut
    st.markdown(f"""
        <div style='padding:5px 10px;background-color:{_bg3};font-family:{FONTS};display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:4px;border-radius:4px'>
            <span style='color:{_mut};font-size:10px'>{len(data.columns)} legs · {n_combos} pairs · {start_date} → {end_date} · {interval_key}</span>
            <span style='color:{_txt2};font-size:10px'>
                Best long: <span style='color:{pos_c};font-weight:600'>{best_long_name}</span>
                <span style='color:{_mut}'>Sharpe {best_long_sharpe:.2f}</span>
                &nbsp;·&nbsp;
                <span style='color:{beats_c}'>{n_beats} spread{"s" if n_beats != 1 else ""} beat{"s" if n_beats != 1 else ""} it</span>
            </span>
        </div>""", unsafe_allow_html=True)

    if missing:
        st.caption(f"ⓘ Dropped {len(missing)} symbol(s) with no overlapping {interval_key} history: "
                   f"{', '.join(missing[:12])}{' …' if len(missing) > 12 else ''}")

    # Charts (top 6)
    render_spread_charts(sorted_pairs, data, theme, mobile=is_mobile)

    # Table (top 10)
    render_spread_table(sorted_pairs, theme, top_n=10)
