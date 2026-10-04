"""
SANPO — Baskets tab

The master list of named symbol lists, shared by every tab with a preset
dropdown: SPREADS, PORTFOLIO, MARKETS and CHARTS. Every basket here can be
edited, renamed or deleted — the ones seeded from config.py are no different
from your own. The whole list is written to baskets.json and loaded into
config.FUTURES_GROUPS in place, so an edit reaches those dropdowns immediately.
"""

import html
import logging
from collections import OrderedDict

import pandas as pd
import streamlit as st
import yfinance as yf

from config import (DEFAULT_CATEGORY, FONTS, FUTURES_GROUPS, basket_category,
                    get_theme, parse_symbols, save_baskets, st_html, surface)

logger = logging.getLogger(__name__)

NEW = '+ New basket'
MAX_NAME = 28


# =============================================================================
# VALIDATION
# =============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def _dead_symbols(symbols):
    """Symbols Yahoo returned no recent data for — i.e. probably typos.

    Advisory only: a save is never blocked on this. Yahoo rate-limits and goes
    down, and a brand-new listing can be legitimately empty.
    """
    syms = list(symbols)
    if not syms:
        return []
    try:
        raw = yf.download(syms, period='5d', interval='1d', auto_adjust=True,
                          progress=False, threads=True, group_by='ticker')
    except Exception as e:
        logger.warning(f"symbol check failed: {e}")
        return []
    if raw is None or raw.empty:
        return []

    bad = []
    multi = isinstance(raw.columns, pd.MultiIndex)
    for s in syms:
        try:
            if multi:
                if s not in raw.columns.get_level_values(0):
                    bad.append(s); continue
                ser = raw[s]['Close'].dropna()
            else:
                ser = raw['Close'].dropna()
            if ser.empty:
                bad.append(s)
        except Exception:
            bad.append(s)
    return bad


# =============================================================================
# TABLE
# =============================================================================

def _build_table(baskets):
    s = surface()
    bdr = s['border']

    HDR = ("font-size:9px;font-weight:700;letter-spacing:0.1em;"
           "text-transform:uppercase;color:#f8fafc")

    body = [
        f"<div style='font-family:{FONTS};min-width:720px'>",
        f"<div style='display:flex;align-items:center;padding:5px 12px;"
        f"border-bottom:1px solid {bdr};gap:10px'>"
        f"<div style='width:170px;flex-shrink:0;{HDR}'>BASKET</div>"
        f"<div style='width:90px;flex-shrink:0;{HDR}'>CATEGORY</div>"
        f"<div style='width:34px;flex-shrink:0;{HDR};text-align:right'>N</div>"
        f"<div style='flex:1;{HDR}'>SYMBOLS</div></div>",
    ]
    for i, (name, syms) in enumerate(baskets.items()):
        alt = s['row_alt'] if i % 2 else 'transparent'
        shown = ', '.join(syms[:14]) + (f' +{len(syms) - 14} more' if len(syms) > 14 else '')
        body.append(
            f"<div style='display:flex;align-items:center;padding:5px 12px;gap:10px;"
            f"background:{alt};border-bottom:1px solid {bdr}33'>"
            f"<div style='width:170px;flex-shrink:0;font-size:11px;font-weight:600;"
            f"color:{s['text']};overflow:hidden;text-overflow:ellipsis;white-space:nowrap'>"
            f"{html.escape(name)}</div>"
            f"<div style='width:90px;flex-shrink:0;font-size:10px;color:{s['muted']};"
            f"overflow:hidden;text-overflow:ellipsis;white-space:nowrap'>"
            f"{html.escape(basket_category(name))}</div>"
            f"<div style='width:34px;flex-shrink:0;font-size:11px;color:{s['text2']};"
            f"text-align:right'>{len(syms)}</div>"
            f"<div style='flex:1;font-size:10px;color:{s['muted']};overflow:hidden;"
            f"text-overflow:ellipsis;white-space:nowrap'>{html.escape(shown)}</div></div>")
    body.append("</div>")
    return ''.join(body), len(baskets)


def _wrap(body, height):
    s = surface()
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<link href='https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap' rel='stylesheet'>"
        "<style>* { margin:0; padding:0; box-sizing:border-box; }"
        f"body {{ background:transparent; font-family:{FONTS}; color:{s['text']}; overflow:hidden; }}"
        "::-webkit-scrollbar { width:4px; height:4px; }"
        f"::-webkit-scrollbar-track {{ background:{s['bg2']}; }}"
        f"::-webkit-scrollbar-thumb {{ background:{s['border']}; border-radius:2px; }}"
        "</style></head><body>"
        f"<div style='height:{height}px;overflow-y:auto;overflow-x:auto'>{body}</div>"
        "</body></html>")


# =============================================================================
# TAB
# =============================================================================

def render_baskets_tab(is_mobile):
    t = get_theme()
    acc = t.get('accent', '#4ade80')

    # FUTURES_GROUPS is the live list — save_baskets() reloads it in place, so
    # there is no second copy to keep in step.
    baskets = OrderedDict((k, list(v)) for k, v in FUTURES_GROUPS.items())

    # Streamlit refuses writes to a widget's key once that widget exists this
    # run, so Save and Delete park their changes here and they land on the way
    # in, before the boxes below are built.
    if 'bk_pending' in st.session_state:
        pending = st.session_state.bk_pending
        del st.session_state.bk_pending
        for k, v in pending.items():
            st.session_state[k] = v

    for k, v in (('bk_name', ''), ('bk_syms', ''), ('bk_cat', ''), ('bk_pick', NEW)):
        if k not in st.session_state:
            st.session_state[k] = v

    options = [NEW, *baskets.keys()]
    if st.session_state.bk_pick not in options:
        st.session_state.bk_pick = NEW

    def _on_pick():
        sel = st.session_state.bk_pick
        if sel == NEW:
            st.session_state.bk_name = ''
            st.session_state.bk_syms = ''
            st.session_state.bk_cat = ''
        else:
            st.session_state.bk_name = sel
            st.session_state.bk_syms = ', '.join(baskets.get(sel, []))
            st.session_state.bk_cat = basket_category(sel)

    def _forget(name):
        """Stop the other tabs remembering a basket that no longer exists."""
        for k in ('sector', 'spread_sector', 'port_preset_name'):
            if st.session_state.get(k) == name:
                del st.session_state[k]

    # ---------------------------------------------------------------- editor
    if is_mobile:
        col_pick, col_name, col_cat = st.container(), st.container(), st.container()
    else:
        col_pick, col_name, col_cat = st.columns([3, 3, 2])

    with col_pick:
        st.selectbox('Edit', options, key='bk_pick', on_change=_on_pick,
                     help='Start a new basket, or load one from the list below '
                          'to change its symbols, rename it or delete it.')
    with col_name:
        st.text_input('Name', key='bk_name', placeholder='Biotech',
                      help=f'What the preset dropdowns will call it. Change it '
                           f'while a basket is loaded to rename that basket. '
                           f'Up to {MAX_NAME} characters.')
    with col_cat:
        st.text_input('Category', key='bk_cat', placeholder=DEFAULT_CATEGORY,
                      help='Which section this basket sits under in the SPREADS '
                           'Portfolio picker — Macro, AI, Healthcare, or anything '
                           'you type. Blank files it under ' + DEFAULT_CATEGORY + '.')

    st.text_area('Symbols', key='bk_syms', height=92,
                 placeholder='XBI, IBB, MRNA, VRTX, REGN',
                 help='Yahoo symbols, as you would type them on finance.yahoo.com '
                      '— ES=F, ^VIX, D05.SI, BTC-USD. Duplicates are dropped and '
                      'order is kept. SPREADS scores every pair, so it uses the '
                      'first 40.')

    picked = st.session_state.bk_pick
    loaded = picked in baskets
    is_last = loaded and len(baskets) == 1

    if is_mobile:
        col_add, col_edit, col_del = st.columns([1, 1, 1])
    else:
        col_add, col_edit, col_del, _sp = st.columns([1, 1, 1, 4])

    with col_add:
        do_add = st.button('Add', key='bk_add', use_container_width=True,
                           help='Create a new basket from the Name and Symbols above. '
                                'Works with a basket loaded too — change the name and '
                                'Add saves it as a second basket, leaving the original.')
    with col_edit:
        do_edit = st.button('Edit', key='bk_edit', disabled=not loaded,
                            use_container_width=True,
                            help=('Apply the Name and Symbols above to the loaded '
                                  'basket. Change the symbols, change the name to '
                                  'rename it, or both.') if loaded
                                 else 'Load a basket above first.')
    with col_del:
        if is_last:
            del_help = 'This is the last basket — the other tabs need one to work from.'
        elif not loaded:
            del_help = 'Load a basket above first.'
        else:
            del_help = None
        do_del = st.button('Delete', key='bk_del', disabled=not loaded or is_last,
                           use_container_width=True, help=del_help)

    # ---------------------------------------------------------------- add / edit
    if do_add or do_edit:
        name = ' '.join(st.session_state.bk_name.split())
        syms = parse_symbols(st.session_state.bk_syms)
        # Add always writes a new basket; Edit always rewrites the loaded one.
        target = None if do_add else picked

        if not name:
            st.session_state.bk_msg = ('err', 'Give the basket a name first.')
        elif len(name) > MAX_NAME:
            st.session_state.bk_msg = ('err', f'Name is longer than {MAX_NAME} characters.')
        elif name == NEW or name.lower() == 'custom':
            st.session_state.bk_msg = ('err', f'"{name}" is reserved — the preset '
                                              f'dropdowns use it already.')
        elif not syms:
            st.session_state.bk_msg = ('err', 'No symbols in the box.')
        elif target is None and name in baskets:
            st.session_state.bk_msg = ('err', f'"{name}" already exists — load it in '
                                              f'the box above and press Edit, or give '
                                              f'this one a different name.')
        elif target is not None and name != target and name in baskets:
            st.session_state.bk_msg = ('err', f'Cannot rename to "{name}" — a basket '
                                              f'already has that name.')
        else:
            new = OrderedDict(baskets)
            renamed = target is not None and target != name
            if renamed:
                # Keep the basket where it sits in the list.
                new = OrderedDict((name if k == target else k, v) for k, v in new.items())
            new[name] = syms

            cats = {n: basket_category(n) for n in baskets}
            if renamed:
                cats.pop(target, None)
            cats[name] = (' '.join(st.session_state.bk_cat.split())
                          or (basket_category(target) if target else DEFAULT_CATEGORY))

            with st.spinner('Checking symbols…'):
                dead = _dead_symbols(tuple(syms))
            save_baskets(new, cats)
            if renamed:
                _forget(target)
            st.session_state.bk_pending = {'bk_pick': name, 'bk_name': name,
                                           'bk_syms': ', '.join(syms),
                                           'bk_cat': cats[name]}

            if target is None:
                verb = f'Added "{name}"'
            elif renamed:
                verb = f'Renamed "{target}" to "{name}"'
            else:
                verb = f'Edited "{name}"'
            msg = f'{verb} — {len(syms)} symbols.'
            if dead:
                msg += (' Yahoo returned nothing for ' + ', '.join(dead) +
                        ' — check the spelling.')
            st.session_state.bk_msg = ('warn' if dead else 'ok', msg)
        st.rerun()

    # ---------------------------------------------------------------- delete
    if do_del and loaded and not is_last:
        new = OrderedDict((k, v) for k, v in baskets.items() if k != picked)
        save_baskets(new, {n: basket_category(n) for n in new})
        _forget(picked)
        st.session_state.bk_pending = {'bk_pick': NEW, 'bk_name': '', 'bk_syms': '',
                                       'bk_cat': ''}
        st.session_state.bk_msg = ('ok', f'Deleted "{picked}".')
        st.rerun()

    # ---------------------------------------------------------------- status
    kind, text = st.session_state.get('bk_msg', (None, None))
    if 'bk_msg' in st.session_state:
        del st.session_state.bk_msg
    if text:
        col = {'ok': acc, 'warn': '#f59e0b', 'err': t.get('neg', '#ef4444')}[kind]
        st.markdown(
            f"<div style='font-size:11px;color:{col};font-family:{FONTS};"
            f"padding:6px 0 2px 0'>{html.escape(text)}</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------------- table
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    body, n = _build_table(baskets)
    height = min(620, 34 + n * 27)
    st_html(_wrap(body, height), height=height)
