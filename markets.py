"""
SANPO — Markets tab
Unified FX + World Indices + custom group browser.
- Group selector: FX, World Indices (regional layout), plus all FUTURES_GROUPS keys (flat).
- Period selector: Day / WTD / MTD / QTD / YTD %.
- Single centre-axis bar column showing the selected period.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
import pytz
import logging

from config import THEMES, FONTS, FUTURES_GROUPS, sym_name, st_html, sort_val

logger = logging.getLogger(__name__)


def get_theme():
    tn = st.session_state.get('theme', 'Dark')
    return THEMES.get(tn, THEMES['Dark'])


# =============================================================================
# REGIONAL LAYOUTS (preserved from fx.py + worldindices.py)
# =============================================================================

FX_REGIONS = {
    'Asia': [
        ('JPY=X', 'USD/JPY'), ('AUD=X', 'USD/AUD'), ('NZD=X', 'USD/NZD'),
        ('SGD=X', 'USD/SGD'), ('HKD=X', 'USD/HKD'), ('CNY=X', 'USD/CNY'),
        ('MYR=X', 'USD/MYR'), ('INR=X', 'USD/INR'), ('KRW=X', 'USD/KRW'),
    ],
    'Europe': [
        ('EUR=X', 'USD/EUR'), ('GBP=X', 'USD/GBP'), ('CHF=X', 'USD/CHF'),
        ('SEK=X', 'USD/SEK'), ('NOK=X', 'USD/NOK'), ('PLN=X', 'USD/PLN'),
        ('TRY=X', 'USD/TRY'),
    ],
    'Africa / ME': [('ZAR=X', 'USD/ZAR')],
    'Americas': [
        ('CAD=X', 'USD/CAD'), ('MXN=X', 'USD/MXN'), ('BRL=X', 'USD/BRL'),
    ],
}

WORLD_REGIONS = {
    'Asia': [
        ('^STI', 'Singapore STI'), ('^HSI', 'Hang Seng'),
        ('000001.SS', 'Shanghai'), ('^N225', 'Nikkei 225'),
        ('^AXJO', 'ASX 200'), ('^BSESN', 'India Sensex'),
        ('^KS11', 'KOSPI'), ('^KLSE', 'Malaysia KLCI'),
        ('^TWII', 'Taiwan TWSE'),
    ],
    'Europe': [
        ('^FTSE', 'FTSE 100'), ('^GDAXI', 'Germany DAX'),
        ('^FCHI', 'France CAC 40'), ('^STOXX50E', 'Euro Stoxx 50'),
    ],
    'Americas': [
        ('^GSPC', 'S&P 500'), ('^IXIC', 'Nasdaq'),
        ('^DJI', 'Dow Jones'), ('^RUT', 'Russell 2000'),
        ('^GSPTSE', 'Canada TSX'), ('^BVSP', 'Brazil IBOVESPA'),
        ('^MXX', 'Mexico IPC'),
    ],
}

REGIONAL_GROUPS = {
    'World Indices': WORLD_REGIONS,
    'FX': FX_REGIONS,
}


# =============================================================================
# PERIOD CONFIG
# =============================================================================

PERIOD_OPTIONS = {
    'Day %': 'day',
    'WTD %': 'wtd',
    'MTD %': 'mtd',
    'QTD %': 'qtd',
    'YTD %': 'ytd',
}


def _period_starts(now):
    """Naive Timestamps for WTD/MTD/QTD/YTD anchors."""
    today = pd.Timestamp(now.date())
    week_start = today - pd.Timedelta(days=now.weekday())   # Monday
    month_start = today.replace(day=1)
    q_month = ((now.month - 1) // 3) * 3 + 1
    quarter_start = today.replace(month=q_month, day=1)
    year_start = today.replace(month=1, day=1)
    return week_start, month_start, quarter_start, year_start


# =============================================================================
# DATA FETCH (cached per-group)
# =============================================================================

@st.cache_data(ttl=300, max_entries=20, show_spinner=False)
def fetch_periods(tickers_tuple):
    tickers = list(tickers_tuple)
    results = {}
    if not tickers:
        return results
    empty = {'price': None, 'day': None, 'wtd': None, 'mtd': None, 'qtd': None, 'ytd': None}
    try:
        raw = yf.download(
            tickers, period='1y', interval='1d',
            auto_adjust=True, progress=False, threads=True,
        )
        close = raw['Close'] if 'Close' in raw else raw
        if isinstance(close, pd.Series):
            close = close.to_frame(name=tickers[0])

        now = datetime.now()
        week_start, month_start, quarter_start, year_start = _period_starts(now)

        for ticker in tickers:
            try:
                if ticker not in close.columns:
                    results[ticker] = dict(empty); continue
                s = close[ticker].dropna()
                if len(s) < 2:
                    results[ticker] = dict(empty); continue
                if s.index.tz is not None:
                    s.index = s.index.tz_localize(None)
                price = float(s.iloc[-1])
                prev = float(s.iloc[-2])
                day = ((price - prev) / prev * 100) if prev else None

                def pct_from(start_ts):
                    # Base is the last close BEFORE the period opens, so a
                    # WTD/MTD/QTD/YTD figure includes the period's first day.
                    before = s[s.index < start_ts]
                    if len(before):
                        base = float(before.iloc[-1])
                    else:
                        sub = s[s.index >= start_ts]
                        if len(sub) < 1:
                            return None
                        base = float(sub.iloc[0])
                    return ((price - base) / base * 100) if base else None

                results[ticker] = {
                    'price': price, 'day': day,
                    'wtd': pct_from(week_start),
                    'mtd': pct_from(month_start),
                    'qtd': pct_from(quarter_start),
                    'ytd': pct_from(year_start),
                }
            except Exception as e:
                logger.warning(f"[{ticker}] period calc error: {e}")
                results[ticker] = dict(empty)
    except Exception as e:
        logger.warning(f"Markets fetch error: {e}")
        for ticker in tickers:
            results[ticker] = dict(empty)
    return results


# =============================================================================
# FORMATTING HELPERS
# =============================================================================

def _format_price(p, ticker):
    if p is None or pd.isna(p):
        return '—'
    if ticker.endswith('=X'):
        if p > 100:
            return f"{p:,.2f}"
        elif p > 10:
            return f"{p:.4f}"
        else:
            return f"{p:.5f}"
    if p > 10:
        return f"{p:,.2f}"
    return f"{p:.4f}"


def _centre_bar(val, max_abs, pos_c, neg_c, bar_bg):
    centre_line = (
        "<div style='position:absolute;left:50%;top:0;width:1px;"
        "height:100%;background:#2d3f55;z-index:2'></div>"
    )
    if val is None or pd.isna(val) or max_abs == 0:
        return (
            f"<div style='flex:1;position:relative;height:20px;"
            f"background:{bar_bg};border-radius:2px;overflow:visible'>"
            f"{centre_line}</div>"
        )
    is_pos = val >= 0
    color = pos_c if is_pos else neg_c
    pct = min(abs(val) / max_abs * 46, 46)
    sign = '+' if is_pos else ''
    label = f"{sign}{val:.2f}%"
    INSIDE_THRESH = 18
    if is_pos:
        bar_css = f"left:50%;width:{pct:.1f}%;background:linear-gradient(90deg,{color}30,{color}65)"
        lbl_css = (
            f"right:{50 - pct + 2:.1f}%;color:{color}"
            if pct >= INSIDE_THRESH
            else f"left:{50 + pct + 1:.1f}%;color:{color}"
        )
    else:
        bar_css = f"right:50%;width:{pct:.1f}%;background:linear-gradient(270deg,{color}30,{color}65)"
        lbl_css = (
            f"left:{50 - pct + 2:.1f}%;color:{color}"
            if pct >= INSIDE_THRESH
            else f"right:{50 + pct + 1:.1f}%;color:{color}"
        )
    return (
        f"<div style='flex:1;position:relative;height:20px;"
        f"background:{bar_bg};border-radius:2px;overflow:hidden'>"
        f"<div style='position:absolute;top:0;{bar_css};height:100%;border-radius:2px'></div>"
        f"{centre_line}"
        f"<span style='position:absolute;top:50%;transform:translateY(-50%);{lbl_css};"
        f"font-size:9.5px;font-weight:700;white-space:nowrap;"
        f"font-variant-numeric:tabular-nums;z-index:3'>{label}</span>"
        f"</div>"
    )


def _resolve_layout(group_name):
    """Return (is_regional, structure)."""
    if group_name in REGIONAL_GROUPS:
        return True, REGIONAL_GROUPS[group_name]
    syms = FUTURES_GROUPS.get(group_name, [])
    flat = [(s, sym_name(s)) for s in syms]
    return False, flat


def _build_panel(data, period_key, group_name):
    t = get_theme()
    bg2 = t.get('bg2', '#0a0f1a')
    bg3 = t.get('bg3', '#0f172a')
    bdr = t.get('border', '#1e293b')
    pos_c = t.get('pos', '#4ade80')
    neg_c = t.get('neg', '#f59e0b')
    bar_bg = bg3
    blue = '#60a5fa'

    is_regional, layout = _resolve_layout(group_name)

    # Global scaling across the visible group
    all_vals = []
    if is_regional:
        for region_pairs in layout.values():
            for sym, _ in region_pairs:
                v = data.get(sym, {}).get(period_key)
                if v is not None and not pd.isna(v):
                    all_vals.append(abs(v))
    else:
        for sym, _ in layout:
            v = data.get(sym, {}).get(period_key)
            if v is not None and not pd.isna(v):
                all_vals.append(abs(v))
    max_abs = max(all_vals) if all_vals else 1.0

    period_label = next(k for k, v in PERIOD_OPTIONS.items() if v == period_key).upper()
    HDR = "font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#f8fafc"

    html = f"""<div style='font-family:{FONTS}'>
        <div style='display:flex;align-items:center;padding:6px 12px 5px 12px;
                    border-bottom:1px solid {bdr};gap:8px'>
            <div style='width:170px;flex-shrink:0;{HDR}'>INSTRUMENT</div>
            <div style='width:90px;flex-shrink:0;{HDR};text-align:right'>PRICE</div>
            <div style='flex:1;{HDR};text-align:center'>{period_label}</div>
        </div>
        <div style='height:1px;background:{bdr};margin:0 12px 2px 12px'></div>"""

    def render_row(i, ticker, name):
        d = data.get(ticker, {})
        price_str = _format_price(d.get('price'), ticker)
        row_bg = bg2 if i % 2 == 0 else bg3
        bar = _centre_bar(d.get(period_key), max_abs, pos_c, neg_c, bar_bg)
        return (
            f"<div style='display:flex;align-items:center;background:{row_bg};"
            f"padding:3px 12px;border-bottom:1px solid {bdr}10;gap:8px'>"
            f"<div style='width:170px;font-size:11px;font-weight:600;"
            f"color:#f8fafc;flex-shrink:0;white-space:nowrap;"
            f"overflow:hidden;text-overflow:ellipsis'>{name}</div>"
            f"<div style='width:90px;font-size:10px;font-weight:500;"
            f"color:#ffffff;flex-shrink:0;text-align:right;"
            f"font-variant-numeric:tabular-nums'>{price_str}</div>"
            f"{bar}"
            f"</div>"
        )

    if is_regional:
        first = True
        for region, pairs in layout.items():
            if not first:
                html += f"<div style='height:1px;background:{blue};opacity:0.5;margin:4px 0'></div>"
            first = False
            html += (
                f"<div style='display:flex;align-items:center;gap:8px;padding:5px 12px 3px 12px'>"
                f"<div style='font-size:9px;font-weight:700;letter-spacing:0.12em;"
                f"text-transform:uppercase;color:{blue}'>{region}</div>"
                f"<div style='flex:1;height:1px;background:{bdr}'></div></div>"
            )
            sorted_pairs = sorted(
                pairs,
                key=lambda x: sort_val(data.get(x[0], {}).get(period_key)),
                reverse=True,
            )
            for i, (ticker, name) in enumerate(sorted_pairs):
                html += render_row(i, ticker, name)
    else:
        sorted_layout = sorted(
            layout,
            key=lambda x: sort_val(data.get(x[0], {}).get(period_key)),
            reverse=True,
        )
        for i, (ticker, name) in enumerate(sorted_layout):
            html += render_row(i, ticker, name)

    html += "</div>"
    return html


def _wrap(body, height):
    t = get_theme()
    bg2 = t.get('bg2', '#0a0f1a')
    bdr = t.get('border', '#1e293b')
    txt = t.get('text', '#e2e8f0')
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<link href='https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap' rel='stylesheet'>"
        "<style>* { margin:0; padding:0; box-sizing:border-box; }"
        f"body {{ background:transparent; font-family:{FONTS}; color:{txt}; overflow:auto; }}"
        f"::-webkit-scrollbar {{ width:4px; }}"
        f"::-webkit-scrollbar-track {{ background:{bg2}; }}"
        f"::-webkit-scrollbar-thumb {{ background:{bdr}; border-radius:2px; }}"
        "</style></head><body>"
        f"{body}"
        "</body></html>"
    )


# =============================================================================
# MAIN ENTRY
# =============================================================================

def render_markets_tab(is_mobile):
    sgt = pytz.timezone('Asia/Singapore')
    now_str = datetime.now(sgt).strftime('%d %b %Y %H:%M SGT')

    regional_keys = list(REGIONAL_GROUPS.keys())
    flat_keys = [k for k in FUTURES_GROUPS.keys() if k not in REGIONAL_GROUPS]
    group_options = regional_keys + flat_keys

    col_grp, col_per, col_spacer, col_ts = st.columns([1.4, 1, 2.4, 1.3])
    with col_grp:
        st.markdown(
            f"<div style='font-size:9px;font-weight:700;color:#e2e8f0;font-family:{FONTS};"
            f"text-transform:uppercase;letter-spacing:0.08em;margin-bottom:-18px'>GROUP</div>",
            unsafe_allow_html=True,
        )
        group = st.selectbox(
            'Group', group_options,
            index=0, key='mkt_group_sel', label_visibility='collapsed',
        )
    with col_per:
        st.markdown(
            f"<div style='font-size:9px;font-weight:700;color:#e2e8f0;font-family:{FONTS};"
            f"text-transform:uppercase;letter-spacing:0.08em;margin-bottom:-18px'>PERIOD</div>",
            unsafe_allow_html=True,
        )
        period_label = st.selectbox(
            'Period', list(PERIOD_OPTIONS.keys()),
            index=0, key='mkt_period_sel', label_visibility='collapsed',
        )
    period_key = PERIOD_OPTIONS[period_label]
    with col_ts:
        st.markdown(
            f"<div style='font-size:9px;color:#f8fafc;font-family:{FONTS};"
            f"padding:28px 0 0 0;text-align:right'>Updated: {now_str}</div>",
            unsafe_allow_html=True,
        )

    is_regional, layout = _resolve_layout(group)
    if is_regional:
        tickers = [s for region in layout.values() for s, _ in region]
    else:
        tickers = [s for s, _ in layout]

    if not tickers:
        st.warning(f"No tickers in group '{group}'")
        return

    with st.spinner(f'Loading {group}...'):
        data = fetch_periods(tuple(tickers))

    n_rows = len(tickers) + (len(layout) if is_regional else 0)
    height = min(820, max(220, 70 + 26 * n_rows))
    st_html(_wrap(_build_panel(data, period_key, group), height), height=height)
