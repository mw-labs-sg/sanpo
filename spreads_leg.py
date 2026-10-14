"""
SANPO — SPREADS / Leg

The Sector sub-tab asks which pair in a basket is best. This asks the question
you actually have when you already hold a view: I want to be long (or short)
THIS name — what is the other side?

So one leg is fixed and typed, the other is searched for across every ticked
basket. It is one spread per candidate rather than every pair against every
other, which is what makes a three-hundred-name universe answerable, and the
direction you asked for is never quietly reversed to make the numbers look
better.
"""

import logging

import pandas as pd
import streamlit as st

from config import FUTURES_GROUPS, THEMES, FONTS, SYMBOL_NAMES, clean_symbol
from spreads import (LOOKBACK_OPTIONS, INTERVAL_CONFIG, SORT_KEYS, SORT_OPTIONS,
                     fetch_interval_data, compute_leg_spreads, lookback_note,
                     rank_by_length_adjusted, basket_picker, render_spread_table,
                     render_spread_charts)

logger = logging.getLogger(__name__)

SIDES = ['Long', 'Short']

# Yahoo symbols only, and not too many: the fetch is one request per name.
MAX_CANDIDATES = 400


def _clean(sym):
    return ''.join(c for c in (sym or '').strip().upper() if c.isalnum() or c in '.^=-:')


def render_leg_tab(is_mobile):
    theme_name = st.session_state.get('theme', 'Dark')
    theme = THEMES.get(theme_name, THEMES['Dark'])
    _mut = theme.get('muted', '#475569')

    picked = basket_picker('lg', is_mobile, theme, label='Universe to search')

    if is_mobile:
        col_in = st.container()
    else:
        col_in, _rest = st.columns([2, 5])

    with col_in:
        leg_raw = st.text_input('Leg', key='leg_symbol', placeholder='e.g. NQ=F',
                                help='The name you have already decided to trade. It does not have '
                                     'to be in any of the ticked baskets — it is fetched '
                                     'alongside them either way.')
        side = st.selectbox('Side', SIDES, key='leg_side',
                            help='Which way you are trading the leg. Long means the leg is bought '
                                 'and the candidate sold; Short is the reverse. The direction is '
                                 'held to — a pair that loses money this way round is allowed '
                                 'to rank badly rather than being flipped until it looks good.')
        iv_keys = list(INTERVAL_CONFIG.keys())
        interval = st.selectbox('Interval', iv_keys, index=iv_keys.index('1d'), key='leg_interval',
                                help='Bar size everything is measured on.')
        lb_keys = list(LOOKBACK_OPTIONS.keys())
        lookback_label = st.selectbox('Lookback', lb_keys, index=lb_keys.index('YTD'),
                                      key='leg_lookback',
                                      help='How far back to score. Today, WTD, MTD and YTD are '
                                           'calendar anchors; the rest are fixed counts of bars.')
        lookback_days = LOOKBACK_OPTIONS[lookback_label]
        sort_by = st.selectbox('Optimize by', SORT_OPTIONS, index=0, key='leg_sort',
                               help='What ranks the counter-legs. Composite is the average rank '
                                    'across Sharpe, Sortino, ROA and ER. Every metric here is '
                                    'discounted by the square root of the window length before '
                                    'ranking, so a candidate with less history cannot win on a '
                                    'small sample.')

        leg = _clean(leg_raw)
        candidates = []
        for g in picked:
            for sym in FUTURES_GROUPS.get(g, []):
                if sym != leg and sym not in candidates:
                    candidates.append(sym)
        truncated = len(candidates) > MAX_CANDIDATES
        if truncated:
            candidates = candidates[:MAX_CANDIDATES]

        note = lookback_note(lookback_label, lookback_days, interval)
        bits = [f'{len(picked)} baskets', f'{len(candidates):,} candidates']
        if truncated:
            bits.append(f'capped at {MAX_CANDIDATES}')
        warn = f" · <span style='color:#fbbf24'>{note}</span>" if note else ''
        st.markdown(f"<div style='font-size:10px;color:{_mut};font-family:{FONTS};"
                    f"line-height:1.6;padding:14px 0 22px 2px'>{' · '.join(bits)}{warn}</div>",
                    unsafe_allow_html=True)
        go_clicked = st.button('▶  Find the other side', key='leg_run', type='primary',
                               use_container_width=True)

    if go_clicked:
        if not leg:
            st.markdown(f"<div style='padding:12px;color:{_mut};font-size:11px;"
                        f"font-family:{FONTS}'>Type the leg you want to trade.</div>",
                        unsafe_allow_html=True)
            return
        if not candidates:
            st.markdown(f"<div style='padding:12px;color:{_mut};font-size:11px;"
                        f"font-family:{FONTS}'>No baskets ticked — nothing to search.</div>",
                        unsafe_allow_html=True)
            return
        with st.spinner(f'Scoring {len(candidates):,} counter-legs against {leg}...'):
            data, ann_factor = fetch_interval_data(tuple([leg] + candidates), interval,
                                                   lookback_days)
            if data is None or leg not in data.columns:
                st.markdown(
                    f"<div style='padding:12px;color:{_mut};font-size:11px;font-family:{FONTS}'>"
                    f"No usable history for <b>{leg}</b> at {interval} over {lookback_label}. "
                    f"Check the symbol, or widen the Lookback.</div>", unsafe_allow_html=True)
                return
            pairs = compute_leg_spreads(data, leg, side, ann_factor)
        if not pairs:
            st.markdown(f"<div style='padding:12px;color:{_mut};font-size:11px;"
                        f"font-family:{FONTS}'>Nothing overlapped {leg} for long enough to "
                        f"score.</div>", unsafe_allow_html=True)
            return
        st.session_state.leg_pairs = pairs
        st.session_state.leg_index = data.index
        st.session_state.leg_meta = {'leg': leg, 'side': side, 'interval': interval,
                                     'lookback': lookback_label, 'n': len(pairs)}

    pairs = st.session_state.get('leg_pairs')
    if not pairs:
        st.markdown(f"<div style='padding:20px;color:{_mut};font-size:11px;font-family:{FONTS}'>"
                    f"Type a leg, tick the baskets to search, and find the other side.</div>",
                    unsafe_allow_html=True)
        return

    meta = st.session_state.get('leg_meta', {})
    key, reverse = SORT_KEYS.get(sort_by, '_score'), sort_by != 'Composite'
    ordered = rank_by_length_adjusted(list(pairs), key, reverse)

    name = SYMBOL_NAMES.get(meta.get('leg', ''), clean_symbol(meta.get('leg', '')))
    side_c = theme['pos'] if meta.get('side') == 'Long' else theme.get('short', '#fb7185')
    st.markdown(
        f"<div style='margin:14px 0 8px;padding:10px 12px;background:rgba(4,8,16,0.46);"
        f"border-radius:4px;font-family:{FONTS};font-size:12px;color:{theme.get('text','#e2e8f0')}'>"
        f"<b style='color:{side_c}'>{meta.get('side','')} {name}</b> against "
        f"<b>{meta.get('n',0):,}</b> candidates · {meta.get('interval','')} · "
        f"{meta.get('lookback','')}"
        f"<span style='color:{_mut}'> &nbsp;|&nbsp; ranked by {sort_by}, length-adjusted</span>"
        f"</div>", unsafe_allow_html=True)

    render_spread_table(ordered, theme, top_n=25)
    # The charts read the index off this frame for their date ticks; any column
    # of the fetch would do, so the stored index is wrapped back into one.
    idx = st.session_state.get('leg_index')
    if idx is not None and len(idx):
        render_spread_charts(ordered[:6], pd.Series(0.0, index=idx), theme, is_mobile)
