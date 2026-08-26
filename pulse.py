import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from collections import OrderedDict
import pytz
import logging

from config import THEMES, FONTS, clean_symbol, st_html

logger = logging.getLogger(__name__)


def get_theme():
    tn = st.session_state.get('theme', 'Dark')
    return THEMES.get(tn, THEMES['Dark'])


def _s():
    """Return surface palette dict based on current theme."""
    t = get_theme()
    is_light = t.get('mode') == 'light'
    bg = t.get('bg', '#1e1e1e')
    bg2 = t.get('bg2', '#0a0f1a')
    bg3 = t.get('bg3', '#0f172a')
    bdr = t.get('border', '#1e293b')
    txt = t.get('text', '#e2e8f0')
    txt2 = t.get('text2', '#94a3b8')
    muted = t.get('muted', '#475569')
    if is_light:
        return dict(
            bg=bg, bg2=bg2, bg3=bg3, card=bg2,
            border=bdr, text=txt, text2=txt2, muted=muted,
            off_dot='#d1d5db', off_name='#9ca3af', link='#334155',
            bar_bg=bdr, row_alt=bg3, hm_txt=txt,
        )
    else:
        return dict(
            bg=bg, bg2=bg2, bg3=bg3, card=bg3,
            border=bdr, text=txt, text2=txt2, muted=muted,
            off_dot='#3a3a3a', off_name='#4a5568', link='#c9d1d9',
            bar_bg=bg3, row_alt='#131d2e', hm_txt=txt,
        )


def _wrap(body, height):
    s = _s()
    page = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<link href='https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap' rel='stylesheet'>"
        "<style>"
        "* { margin:0; padding:0; box-sizing:border-box; }"
        f"body {{ background:transparent; font-family:{FONTS}; color:{s['text']}; overflow:auto; }}"
        f"a {{ color:{s['link']}; text-decoration:none; }}"
        "a:hover { text-decoration:underline; }"
        f"::-webkit-scrollbar {{ width:4px; }}"
        f"::-webkit-scrollbar-track {{ background:{s['bg2']}; }}"
        f"::-webkit-scrollbar-thumb {{ background:{s['border']}; border-radius:2px; }}"
        "</style></head><body>"
        f"{body}"
        "</body></html>"
    )
    st_html(page, height=height)


# ── DATA ─────────────────────────────────────────────────────────────────────

HERO_SYMBOLS = OrderedDict([
    ('ES=F',     {'label': 'S&P 500',  'fmt': ',.0f'}),
    ('NQ=F',     {'label': 'NASDAQ',   'fmt': ',.0f'}),
    ('BTC-USD',  {'label': 'BITCOIN',  'fmt': ',.0f'}),
    ('GC=F',     {'label': 'GOLD',     'fmt': ',.0f'}),
    ('CL=F',     {'label': 'CRUDE',    'fmt': ',.2f'}),
    ('ZN=F',     {'label': '10Y NOTE', 'fmt': ',.2f'}),
    ('USDSGD=X', {'label': 'USD/SGD',  'fmt': ',.4f'}),
    ('^STI',     {'label': 'STI',      'fmt': ',.0f'}),
])

HEATMAP_SECTORS = OrderedDict([
    ('Indices',    ['ES=F', 'NQ=F', 'NKD=F', 'RTY=F']),
    ('Crypto',     ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD']),
    ('Energy',     ['CL=F', 'NG=F', 'RB=F', 'HO=F']),
    ('Metals',     ['GC=F', 'SI=F', 'HG=F', 'PL=F']),
    ('Grains',     ['ZC=F', 'ZW=F', 'ZS=F', 'ZM=F']),
    ('Softs',      ['CC=F', 'KC=F', 'SB=F', 'CT=F']),
    ('Rates',      ['ZB=F', 'ZN=F', 'ZF=F', 'ZT=F']),
    ('FX',         ['6J=F', '6E=F', '6B=F', '6A=F']),
    ('US Sectors', ['XLE', 'XLF', 'XLK', 'XLV']),
    ('Shipping',   ['ZIM', 'SBLK', 'STNG', 'FRO']),
    ('Strategy',   ['MSTR', 'MSTU', 'MSTY', 'MSTX']),
    ('Singapore',  ['^STI', 'ES3.SI', 'S68.SI', 'MBH.SI']),
])

SPARKLINE_SYMBOLS = ['ES=F', 'BTC-USD', 'GC=F', 'CL=F', 'USDSGD=X', '^STI']


# ── FETCH ────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_pulse_batch():
    all_syms = list(HERO_SYMBOLS.keys())
    for syms in HEATMAP_SECTORS.values():
        all_syms.extend(syms)
    all_syms = list(OrderedDict.fromkeys(all_syms))
    batch_syms = [s for s in all_syms if not s.startswith('^')]
    result = {}
    try:
        batch = yf.download(batch_syms, period='5d', group_by='ticker', threads=True, progress=False)
    except Exception as e:
        logger.warning(f"Pulse batch failed: {e}")
        batch = pd.DataFrame()
    for sym in all_syms:
        try:
            if sym.startswith('^'):
                # Index symbols with ^ often fail in batch download
                hist = yf.Ticker(sym).history(period='5d')
            elif batch.empty:
                hist = yf.Ticker(sym).history(period='5d')
            elif len(all_syms) == 1:
                hist = batch.copy()
            elif sym in batch.columns.get_level_values(0):
                hist = batch[sym].dropna(how='all')
            else:
                hist = yf.Ticker(sym).history(period='5d')
            if hist.empty:
                continue
            current = float(hist['Close'].iloc[-1])
            prev_close = float(hist['Close'].iloc[-2]) if len(hist) >= 2 else current
            change = ((current - prev_close) / prev_close) * 100 if prev_close else 0
            result[sym] = {'price': current, 'change': round(change, 2)}
        except Exception as e:
            logger.debug(f"[{sym}] pulse fetch error: {e}")
    return result


@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_sparklines():
    result = {}
    for sym in SPARKLINE_SYMBOLS:
        try:
            hist = yf.Ticker(sym).history(period='1mo')
            if not hist.empty:
                result[sym] = hist['Close'].values.tolist()
        except Exception:
            pass
    return result


# ── BREAKOUT SCANNER ─────────────────────────────────────────────────────────

# Symbols shown in the breakout panel — curated cross-asset watchlist
# Derived from HEATMAP_SECTORS — single source of truth, symbols only
BREAKOUT_SYMBOLS = OrderedDict()
for _syms in HEATMAP_SECTORS.values():
    for _sym in _syms:
        if _sym not in BREAKOUT_SYMBOLS:
            BREAKOUT_SYMBOLS[_sym] = clean_symbol(_sym)


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_breakout_data():
    """
    Fetch 2 years of daily data per symbol.
    2y gives ~500 trading days — enough for:
      - full previous calendar year (up to ~252 bars)
      - full previous calendar month (up to ~23 bars)
      - full previous ISO week (up to 5 bars)
      - current period bars
    Returns dict: sym -> DataFrame with OHLC columns
    """
    syms = list(BREAKOUT_SYMBOLS.keys())
    batch_syms = [s for s in syms if not s.startswith('^')]
    result = {}

    # Batch download for non-index symbols
    try:
        batch = yf.download(batch_syms, period='2y', group_by='ticker',
                            threads=True, progress=False)
    except Exception as e:
        logger.warning(f"Breakout batch download failed: {e}")
        batch = pd.DataFrame()

    for sym in syms:
        try:
            if sym.startswith('^'):
                hist = yf.Ticker(sym).history(period='2y')
            elif batch.empty:
                hist = yf.Ticker(sym).history(period='2y')
            elif len(batch_syms) == 1:
                hist = batch.copy()
            elif sym in batch.columns.get_level_values(0):
                hist = batch[sym].dropna(how='all')
            else:
                hist = yf.Ticker(sym).history(period='2y')

            if hist.empty or len(hist) < 10:
                continue
            # Ensure required columns exist
            for col in ('Open', 'High', 'Low', 'Close'):
                if col not in hist.columns:
                    break
            else:
                result[sym] = hist
        except Exception as e:
            logger.debug(f"[{sym}] breakout fetch error: {e}")

    return result


def _compute_breakout_status(hist, period_type):
    """
    Reuses charts.py _slice_period logic exactly — imported at call time
    to avoid circular import at module load.

    Returns dict with:
      prev_high, prev_low, prev_mid, prev_close  — the reference levels
      curr_price                                  — latest close
      pct_r    : Williams-style %R (0=at low, 100=at high, >100=breakout above, <0=breakdown)
      status   : 'above_high' | 'above_mid' | 'below_mid' | 'below_low'
      reversal : 'buy' | 'sell' | ''
      curr_high, curr_low                         — this period's high/low so far
    """
    from charts import _slice_period  # deferred to avoid circular import

    try:
        prev_period, current_bars = _slice_period(hist, period_type)
        if prev_period is None or prev_period.empty:
            return None

        ph = float(prev_period['High'].max())
        pl = float(prev_period['Low'].min())
        pc = float(prev_period['Close'].iloc[-1])
        pm = (ph + pl) / 2.0
        rng = ph - pl

        # Use latest close as current price
        if current_bars is not None and not current_bars.empty:
            curr_price = float(current_bars['Close'].iloc[-1])
            curr_high  = float(current_bars['High'].max())
            curr_low   = float(current_bars['Low'].min())
        else:
            # Period just started — no current bars yet (e.g. first bar of new week)
            # Fall back to the very last bar in hist
            curr_price = float(hist['Close'].iloc[-1])
            curr_high  = curr_price
            curr_low   = curr_price

        # %R: position within prev period range
        # >100 means above prev high (breakout), <0 means below prev low (breakdown)
        pct_r = ((curr_price - pl) / rng * 100) if rng > 0 else 50.0

        # Status — exact same 4-state logic as charts.py
        if curr_price > ph:
            status = 'above_high'
        elif curr_price < pl:
            status = 'below_low'
        elif curr_price > pm:
            status = 'above_mid'
        else:
            status = 'below_mid'

        # Reversal — exact same logic as charts.py _check_reversal
        reversal = ''
        if current_bars is not None and not current_bars.empty:
            if curr_high > ph and curr_price <= ph:
                reversal = 'sell'   # rejected at high
            elif curr_low < pl and curr_price >= pl:
                reversal = 'buy'    # bounced off low

        return {
            'prev_high':  ph,
            'prev_low':   pl,
            'prev_mid':   pm,
            'prev_close': pc,
            'curr_price': curr_price,
            'curr_high':  curr_high,
            'curr_low':   curr_low,
            'pct_r':      pct_r,
            'status':     status,
            'reversal':   reversal,
        }
    except Exception as e:
        logger.debug(f"_compute_breakout_status ({period_type}) error: {e}")
        return None


# ── SVG SPARKLINE ────────────────────────────────────────────────────────────

def _svg_sparkline(data, width=100, height=28, pos_color='#4ade80', neg_color='#f59e0b'):
    if not data or len(data) < 2:
        return ''
    vals = np.array(data)
    mn, mx = vals.min(), vals.max()
    rng = mx - mn if mx != mn else 1
    n = len(vals)
    xs = [round(i * width / (n - 1), 1) for i in range(n)]
    ys = [round(height - 2 - (v - mn) / rng * (height - 4), 1) for v in vals]
    color = pos_color if vals[-1] >= vals[0] else neg_color
    points = ' '.join(f'{x},{y}' for x, y in zip(xs, ys))
    fill_points = f'0,{height} {points} {width},{height}'
    uid = f'sp{abs(hash(tuple(data))) % 99999}'
    return (
        f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}'>"
        f"<defs><linearGradient id='{uid}' x1='0' y1='0' x2='0' y2='1'>"
        f"<stop offset='0%' stop-color='{color}' stop-opacity='0.3'/>"
        f"<stop offset='100%' stop-color='{color}' stop-opacity='0'/>"
        f"</linearGradient></defs>"
        f"<polygon points='{fill_points}' fill='url(#{uid})'/>"
        f"<polyline points='{points}' fill='none' stroke='{color}' stroke-width='1.5' stroke-linejoin='round' stroke-linecap='round'/>"
        f"<circle cx='{xs[-1]}' cy='{ys[-1]}' r='2' fill='{color}'/>"
        f"</svg>"
    )


# ── MARKET STATUS ────────────────────────────────────────────────────────────

# SGX market holidays 2026 (excludes weekends)
_SG_HOLIDAYS = [
    (2026,1,1,'New Year'),  (2026,2,17,'CNY'), (2026,2,18,'CNY'),
    (2026,3,21,'Hari Raya Puasa'), (2026,4,3,'Good Friday'),
    (2026,5,1,'Labour Day'), (2026,5,27,'Hari Raya Haji'),
    (2026,6,1,'Vesak Day'), (2026,8,10,'National Day'),
    (2026,11,9,'Deepavali'), (2026,12,25,'Christmas'),
]
# NYSE market holidays 2026
_US_HOLIDAYS = [
    (2026,1,1,'New Year'), (2026,1,19,'MLK Day'), (2026,2,16,"Presidents'"),
    (2026,4,3,'Good Friday'), (2026,5,25,'Memorial Day'), (2026,6,19,'Juneteenth'),
    (2026,7,3,'Independence'), (2026,9,7,'Labor Day'), (2026,11,26,'Thanksgiving'),
    (2026,12,25,'Christmas'),
]

def _next_holiday(holidays, today):
    from datetime import date
    for y, m, d, name in holidays:
        h = date(y, m, d)
        if h >= today:
            delta = (h - today).days
            return name, h.strftime('%d %b'), delta
    return None, None, None

def _market_status():
    now_utc = datetime.now(pytz.utc)
    today = now_utc.astimezone(pytz.timezone('Asia/Singapore')).date()
    results = []
    markets = [
        ('SG',  'Asia/Singapore', 9, 0,  17, 0,  _SG_HOLIDAYS, '#fb7185'),  # rose
        ('US',  'US/Eastern',     9, 30, 16, 0,  _US_HOLIDAYS, '#60a5fa'),  # blue
    ]
    for name, tz_str, oh, om, ch, cm, holidays, dot_color in markets:
        tz = pytz.timezone(tz_str)
        local = now_utc.astimezone(tz)
        is_weekend = local.weekday() >= 5
        open_time = local.replace(hour=oh, minute=om, second=0)
        close_time = local.replace(hour=ch, minute=cm, second=0)
        is_open = not is_weekend and open_time <= local <= close_time
        hol_name, hol_date, hol_days = _next_holiday(holidays, today)
        results.append({
            'name': name, 'time': local.strftime('%H:%M'), 'open': is_open,
            'dot_color': dot_color,
            'hol_name': hol_name, 'hol_date': hol_date, 'hol_days': hol_days,
        })
    return results


# ── RENDER ───────────────────────────────────────────────────────────────────

def _render_market_status_bar():
    markets = _market_status()
    t = get_theme()
    s = _s()
    is_light = t.get('mode') == 'light'
    dots = ''
    for m in markets:
        # Dot color: red for SG, blue for US (always their color, brighter when open)
        dot_c = m['dot_color']
        # Country name: strong text
        nc = ('#0f172a' if is_light else '#f8fafc') if m['open'] else ('#64748b' if is_light else '#f8fafc')
        # Time: always visible
        tc = ('#334155' if is_light else '#f8fafc') if m['open'] else ('#94a3b8' if is_light else '#f8fafc')
        dots += (
            f"<div style='display:flex;align-items:center;gap:4px'>"
            f"<div style='width:6px;height:6px;border-radius:50%;background:{dot_c}'></div>"
            f"<span style='color:{nc};font-size:9px;font-weight:600;letter-spacing:0.06em'>{m['name']}</span>"
            f"<span style='color:{tc};font-size:9px;font-weight:600'>{m['time']}</span>"
            f"</div>"
        )

    # Next holidays
    hols = ''
    for m in markets:
        if m['hol_name']:
            dot_c = m['dot_color']
            hols += (
                f"<div style='display:flex;align-items:center;gap:4px'>"
                f"<div style='width:6px;height:6px;border-radius:2px;background:{dot_c}'></div>"
                f"<span style='color:#e2e8f0;font-size:8px;font-weight:600;letter-spacing:0.06em;text-transform:uppercase'>"
                f"{m['name']} {m['hol_name']} {m['hol_date']}</span>"
                f"</div>"
            )

    html = (
        "<style>@keyframes pulse-dot{0%,100%{opacity:1}50%{opacity:0.4}}</style>"
        f"<div style='display:flex;gap:16px;flex-wrap:wrap;padding:6px 12px;background:{s['bg2']};"
        f"border:1px solid {s['border']};border-radius:4px;align-items:center'>"
        f"<span style='color:#f8fafc;font-size:8px;font-weight:600;letter-spacing:0.1em;"
        f"align-self:center'>MARKETS</span>"
        f"{dots}"
        f"<span style='color:{s['border']};font-size:10px;align-self:center'>│</span>"
        f"{hols}</div>"
    )
    _wrap(html, 28)


def _render_hero_row(data):
    t = get_theme()
    s = _s()
    pos_c, neg_c = t['pos'], t['neg']
    is_light = get_theme().get('mode') == 'light'

    cards = ''
    for sym, cfg in HERO_SYMBOLS.items():
        d = data.get(sym)
        if not d:
            continue
        price, change = d['price'], d['change']
        color = pos_c if change >= 0 else neg_c
        sign = '+' if change >= 0 else ''
        arrow = '&#9650;' if change > 0 else '&#9660;' if change < 0 else '&#8211;'
        price_str = f'{price:{cfg["fmt"]}}'

        if is_light:
            card_bg = f'background:{s["bg2"]};'
            glow = ''
        else:
            card_bg = f'background:linear-gradient(135deg,{s["bg3"]},{s["bg2"]});'
            glow = f'text-shadow:0 0 20px {color}40;'

        # Force high contrast price color
        price_c = '#0f172a' if is_light else '#f8fafc'
        label_c = '#475569' if is_light else '#f8fafc'

        cards += (
            f"<div style='flex:1;min-width:110px;padding:10px 12px;"
            f"{card_bg}"
            f"border:1px solid {s['border']};border-radius:6px;position:relative;overflow:hidden'>"
            f"<div style='position:absolute;top:0;left:0;right:0;height:2px;"
            f"background:linear-gradient(90deg,transparent,{color}40,transparent)'></div>"
            f"<div style='color:{label_c};font-size:8px;font-weight:600;letter-spacing:0.12em;"
            f"text-transform:uppercase;margin-bottom:4px'>{cfg['label']}</div>"
            f"<div style='color:{price_c};font-size:17px;font-weight:700;"
            f"font-variant-numeric:tabular-nums;letter-spacing:-0.02em;"
            f"{glow}'>{price_str}</div>"
            f"<div style='margin-top:3px;display:flex;align-items:center;gap:4px'>"
            f"<span style='color:{color};font-size:10px'>{arrow}</span>"
            f"<span style='color:{color};font-size:11px;font-weight:700'>{sign}{change:.2f}%</span>"
            f"</div></div>"
        )
    html = f"<div style='display:flex;gap:6px;flex-wrap:wrap'>{cards}</div>"
    _wrap(html, 80)


def _render_sparkline_row(spark_data, pulse_data):
    t = get_theme(); s = _s()
    pos_c, neg_c = t['pos'], t['neg']
    cards = ''
    for sym in SPARKLINE_SYMBOLS:
        sdata = spark_data.get(sym)
        pdata = pulse_data.get(sym)
        if not sdata or not pdata:
            continue
        short = clean_symbol(sym)
        change = pdata['change']
        color = pos_c if change >= 0 else neg_c
        sign = '+' if change >= 0 else ''
        svg = _svg_sparkline(sdata, width=80, height=22, pos_color=pos_c, neg_color=neg_c)
        cards += (
            f"<div style='flex:1;min-width:120px;padding:5px 8px;background:{s['card']};"
            f"border:1px solid {s['border']};border-radius:4px'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:2px'>"
            f"<span style='color:{s['text2']};font-size:8px;font-weight:600;letter-spacing:0.06em'>{short}</span>"
            f"<span style='color:{color};font-size:8px;font-weight:700'>{sign}{change:.2f}%</span>"
            f"</div>"
            f"{svg}"
            f"</div>"
        )
    html = f"<div style='display:flex;gap:5px;flex-wrap:wrap'>{cards}</div>"
    _wrap(html, 50)

def _render_heatmap_grid(data):
    t = get_theme()
    s = _s()
    pos_c, neg_c = t['pos'], t['neg']
    is_light = get_theme().get('mode') == 'light'

    def _bg(change, pos, neg):
        abs_c = min(abs(change), 5)
        opacity = 0.15 + (abs_c / 5) * 0.55 if is_light else 0.25 + (abs_c / 5) * 0.75
        base = pos if change >= 0 else neg
        r, g, b = int(base[1:3], 16), int(base[3:5], 16), int(base[5:7], 16)
        return f'rgba({r},{g},{b},{opacity:.2f})', abs_c

    sectors_html = ''
    active_sectors = 0
    for sector, syms in HEATMAP_SECTORS.items():
        # top 2 gainers + top 2 losers
        sector_data = [(sym, data.get(sym, {}).get('change', 0)) for sym in syms if data.get(sym)]
        if not sector_data:
            continue
        sector_data.sort(key=lambda x: x[1], reverse=True)
        display = sector_data if len(sector_data) <= 4 else sector_data[:2] + sector_data[-2:]
        cells = ''
        for sym, change in display:
            name = clean_symbol(sym)
            bg, intensity = _bg(change, pos_c, neg_c)
            if intensity > 3:
                name_c = '#ffffff' if not is_light else '#1e293b'
                val_c = '#ffffff' if not is_light else '#1e293b'
            else:
                name_c = s['hm_txt']
                val_c = pos_c if change >= 0 else neg_c
            sign = '+' if change >= 0 else ''
            cell_border = 'rgba(0,0,0,0.06)' if is_light else 'rgba(255,255,255,0.04)'
            cells += (
                f"<div style='flex:1;min-width:60px;padding:6px 8px;background:{bg};"
                f"border-radius:3px;border:1px solid {cell_border};text-align:center'>"
                f"<div style='color:{name_c};font-size:10px;font-weight:600'>{name}</div>"
                f"<div style='color:{val_c};font-size:11px;font-weight:700;margin-top:1px;"
                f"font-variant-numeric:tabular-nums'>{sign}{change:.2f}%</div>"
                f"</div>"
            )
        active_sectors += 1
        sectors_html += (
            f"<div style='margin-bottom:6px'>"
            f"<div style='color:#f8fafc;font-size:8px;font-weight:600;letter-spacing:0.1em;"
            f"text-transform:uppercase;margin-bottom:4px;padding-left:2px'>{sector}</div>"
            f"<div style='display:flex;gap:4px;flex-wrap:wrap'>{cells}</div>"
            f"</div>"
        )

    html = (
        f"<div style='padding:10px 12px;background:{s['bg2']};border:1px solid {s['border']};border-radius:6px'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px'>"
        f"<span style='color:#f8fafc;font-size:9px;font-weight:600;letter-spacing:0.1em;"
        f"text-transform:uppercase'>MARKET HEATMAP</span>"
        f"<span style='color:{s['muted']};font-size:8px'>Day Change %</span>"
        f"</div>"
        f"{sectors_html}</div>"
    )
    _wrap(html, 52 * active_sectors + 50)


def _render_movers(data):
    t = get_theme()
    s = _s()
    pos_c, neg_c = t['pos'], t['neg']
    all_items = [(sym, d['change']) for sym, d in data.items()]
    all_items.sort(key=lambda x: x[1], reverse=True)
    gainers = [(sc, c) for sc, c in all_items if c > 0][:5]
    losers = [(sc, c) for sc, c in all_items if c < 0][-5:]
    losers.reverse()

    def _rows(items, color, is_gain):
        if not items:
            return f"<div style='color:{s['muted']};font-size:10px;padding:8px'>No data</div>"
        max_abs = max(abs(c) for _, c in items) or 1
        html = ''
        for sym, change in items:
            short = clean_symbol(sym)
            bar_pct = max(abs(change) / max_abs * 85, 8)
            sign = '+' if change >= 0 else ''
            align = 'left' if is_gain else 'right'
            grad_dir = '90deg' if is_gain else '270deg'
            html += (
                f"<div style='display:flex;align-items:center;padding:5px 0;gap:6px'>"
                f"<div style='width:45px;flex-shrink:0'>"
                f"<span style='color:{s['text']};font-size:10px;font-weight:600'>{short}</span></div>"
                f"<div style='flex:1;position:relative;height:18px;background:{s['bar_bg']};border-radius:2px;overflow:hidden'>"
                f"<div style='position:absolute;top:0;{align}:0;height:100%;width:{bar_pct}%;"
                f"background:linear-gradient({grad_dir},{color}20,{color}60);border-radius:2px'></div>"
                f"<span style='position:absolute;top:50%;transform:translateY(-50%);{align}:6px;"
                f"color:{color};font-size:10px;font-weight:700;font-variant-numeric:tabular-nums'>"
                f"{sign}{change:.2f}%</span></div></div>"
            )
        return html

    gain_html = _rows(gainers, pos_c, True)
    lose_html = _rows(losers, neg_c, False)
    n_rows = max(len(gainers), len(losers), 1)

    html = (
        f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:16px;"
        f"padding:10px 12px;background:{s['bg2']};border:1px solid {s['border']};border-radius:6px'>"
        f"<div>"
        f"<div style='color:#f8fafc;font-size:9px;font-weight:600;letter-spacing:0.1em;"
        f"margin-bottom:6px;display:flex;align-items:center;gap:4px'>"
        f"<span style='color:{pos_c};font-size:12px'>&#9650;</span> TOP GAINERS</div>"
        f"{gain_html}</div>"
        f"<div>"
        f"<div style='color:#f8fafc;font-size:9px;font-weight:600;letter-spacing:0.1em;"
        f"margin-bottom:6px;display:flex;align-items:center;gap:4px'>"
        f"<span style='color:{neg_c};font-size:12px'>&#9660;</span> TOP LOSERS</div>"
        f"{lose_html}</div></div>"
    )
    h = 28 * n_rows + 50
    _wrap(html, h)
    return h



def _render_pulse_news(iframe_height=600):
    """News panel — stretches to match left column height."""
    from news import fetch_rss_feed
    t = get_theme(); s = _s()
    pos_c = t['pos']

    feeds = [
        ('CNA',           'https://www.channelnewsasia.com/api/v1/rss-outbound-feed?_format=xml&category=6511'),
        ('Straits Times', 'https://www.straitstimes.com/news/business/rss.xml'),
        ('Bloomberg',     'https://feeds.bloomberg.com/markets/news.rss'),
        ('FT',            'https://www.ft.com/rss/home'),
    ]

    all_items = []
    for name, url in feeds:
        all_items.extend(fetch_rss_feed(name, url))

    all_items.sort(key=lambda x: x.get('sort_key', ''), reverse=True)
    all_items = all_items[:40]
    if not all_items:
        return

    rows = ''
    for i, item in enumerate(all_items):
        bg = s['bg2'] if i % 2 == 0 else s['row_alt']
        src_col = item.get('source', '')
        dt_col  = item.get('date', '')
        rows += (
            "<div style='padding:4px 10px;background:" + bg + ";border-bottom:1px solid " + s['border'] + "18;"
            "display:flex;align-items:baseline;gap:6px;font-family:" + FONTS + ";white-space:nowrap;overflow:hidden'>"
            "<span style='flex-shrink:0;width:115px;display:flex;gap:5px;align-items:baseline'>"
            "<span style='color:" + pos_c + ";font-weight:600;font-size:9px'>" + src_col + "</span>"
            "<span style='color:" + s['muted'] + ";font-size:9px'>" + dt_col + "</span></span>"
            "<a href='" + item.get('url', '#') + "' target='_blank' style='color:" + s['link'] + ";text-decoration:none;"
            "font-size:10.5px;font-weight:500;overflow:hidden;text-overflow:ellipsis'>" + item.get('title', '') + "</a>"
            "</div>"
        )

    html = (
        "<div style='background:" + s['bg2'] + ";border:1px solid " + s['border'] + ";border-radius:6px;"
        "overflow:hidden;font-family:" + FONTS + ";height:" + str(iframe_height) + "px;"
        "display:flex;flex-direction:column'>"
        "<div style='padding:6px 10px;display:flex;justify-content:space-between;align-items:center;"
        "border-bottom:1px solid " + s['border'] + ";flex-shrink:0'>"
        "<span style='color:#f8fafc;font-size:9px;font-weight:600;letter-spacing:0.1em'>LATEST</span>"
        "<span style='color:" + s['muted'] + ";font-size:9px;font-weight:500'>" + str(len(all_items)) + "</span></div>"
        "<div style='overflow-y:auto;flex:1'>" + rows + "</div>"
        "</div>"
    )
    _wrap(html, iframe_height)


# ── BREAKOUT TABLES (week + month) ───────────────────────────────────────────

TOP_N_BREAKOUTS = 5   # max rows per side (above/below) per period


def _render_breakout_tables(breakout_data, pulse_data=None):
    """
    Two compact tables stacked:
      Table 1 — WEEK BREAKOUTS  : ▲ above prev week high | ▼ below prev week low
      Table 2 — MONTH BREAKOUTS : ▲ above prev month high | ▼ below prev month low

    Only TRUE breakouts shown (price outside the prev period range).
    Sorted by distance from the level (furthest first).
    Format identical to _render_movers: left=green bars, right=amber bars.
    Top 5 per side.
    Returns total iframe height used so news panel can match.
    """
    t     = get_theme()
    s     = _s()
    pos_c = t['pos']
    neg_c = t['neg']
    bdr   = s['border']
    bg2   = s['bg2']
    muted = s['muted']
    bar_bg = s['bar_bg']

    # ── Collect and classify ─────────────────────────────────────────────────
    def _classify(period_type):
        """
        Returns (above_list, below_list) where each item is (label, pct_beyond).
        pct_beyond = how far price has moved beyond the level, as % of that level.
          above: (curr - prev_high) / prev_high * 100  — always >= 0
          below: (prev_low - curr)  / prev_low  * 100  — always >= 0
        Sorted descending (furthest break first). Capped at TOP_N_BREAKOUTS.
        """
        above_list = []
        below_list = []
        for sym in BREAKOUT_SYMBOLS:
            hist = breakout_data.get(sym)
            if hist is None:
                continue
            res = _compute_breakout_status(hist, period_type)
            if res is None:
                continue
            status = res['status']
            rev = res['reversal']
            display = clean_symbol(sym)

            if status == 'above_high':
                ph = res['prev_high']
                pct = (res['curr_price'] - ph) / ph * 100 if ph else 0
                above_list.append((display, pct, rev))
            elif status == 'below_low':
                pl = res['prev_low']
                pct = (pl - res['curr_price']) / pl * 100 if pl else 0
                below_list.append((display, pct, rev))
            # inside range → not shown

        above_list.sort(key=lambda x: x[1], reverse=True)
        below_list.sort(key=lambda x: x[1], reverse=True)
        return above_list[:TOP_N_BREAKOUTS], below_list[:TOP_N_BREAKOUTS]

    w_above, w_below = _classify('week')
    m_above, m_below = _classify('month')
    y_above, y_below = _classify('year')

    # ── Bar row builder ──────────────────────────────────────────────────────
    def _rows(items, color, is_above):
        if not items:
            return "<div style='color:" + muted + ";font-size:10px;padding:6px 2px'>No breakouts</div>"
        max_pct = max(x[1] for x in items) or 1
        html = ''
        for label, pct, rev in items:
            bar_pct = max(pct / max_pct * 85, 8)
            grad_dir = '90deg' if is_above else '270deg'
            align    = 'left'  if is_above else 'right'
            pct_str  = ("+" if is_above else "-") + f"{pct:.0f}%"
            rev_dot  = ''
            if rev == 'buy':
                rev_dot = "<span style='color:" + pos_c + ";font-size:8px;margin-left:2px'>&#9679;</span>"
            elif rev == 'sell':
                rev_dot = "<span style='color:" + neg_c + ";font-size:8px;margin-left:2px'>&#9679;</span>"
            html += (
                "<div style='display:flex;align-items:center;padding:5px 0;gap:6px'>"
                "<div style='width:45px;flex-shrink:0'>"
                "<span style='color:#e2e8f0;font-size:10px;font-weight:600'>" + label + "</span>"
                + rev_dot +
                "</div>"
                "<div style='flex:1;position:relative;height:18px;background:" + bar_bg + ";border-radius:2px;overflow:hidden'>"
                "<div style='position:absolute;top:0;" + align + ":0;height:100%;width:" + f"{bar_pct:.0f}" + "%;background:linear-gradient(" + grad_dir + "," + color + "20," + color + "60);border-radius:2px'></div>"
                "<span style='position:absolute;top:50%;transform:translateY(-50%);" + align + ":6px;"
                "color:" + color + ";font-size:10px;font-weight:700;font-variant-numeric:tabular-nums'>"
                + pct_str + "</span>"
                "</div>"
                "</div>"
            )
        return html

    # ── Single breakout table ────────────────────────────────────────────────
    TABLE_H = 200  # fixed scrollable body height per table

    def _table(above, below, period_label):
        above_html = _rows(above, pos_c, True)
        below_html = _rows(below, neg_c, False)

        html = (
            "<div style='background:" + bg2 + ";border:1px solid " + bdr + ";border-radius:6px;"
            "font-family:" + FONTS + ";flex-shrink:0'>"
            # Header
            "<div style='color:" + muted + ";font-size:8px;font-weight:600;letter-spacing:0.12em;"
            "text-transform:uppercase;padding:7px 10px 5px 10px;"
            "border-bottom:1px solid " + bdr + "'>"
            "PREV " + period_label + " BREAKOUTS</div>"
            # Scrollable body
            "<div style='overflow-y:auto;max-height:" + str(TABLE_H) + "px;padding:4px 10px 6px 10px'>"
            "<div style='display:grid;grid-template-columns:1fr 1fr;gap:16px'>"
            # Above col
            "<div>"
            "<div style='color:#f8fafc;font-size:9px;font-weight:600;letter-spacing:0.1em;"
            "margin-bottom:4px;margin-top:4px;display:flex;align-items:center;gap:4px'>"
            "<span style='color:" + pos_c + ";font-size:11px'>&#9650;</span> ABOVE HIGH"
            "<span style='margin-left:auto;color:" + muted + ";font-size:8px;font-weight:500'>YTD</span></div>"
            + above_html +
            "</div>"
            # Below col
            "<div>"
            "<div style='color:#f8fafc;font-size:9px;font-weight:600;letter-spacing:0.1em;"
            "margin-bottom:4px;margin-top:4px;display:flex;align-items:center;gap:4px'>"
            "<span style='color:" + neg_c + ";font-size:11px'>&#9660;</span> BELOW LOW"
            "<span style='margin-left:auto;color:" + muted + ";font-size:8px;font-weight:500'>YTD</span></div>"
            + below_html +
            "</div>"
            "</div>"
            "</div>"
            "</div>"
        )
        height = TABLE_H + 42  # header + scrollable body + padding
        return html, height

    week_html,  week_h  = _table(w_above, w_below,  'WEEK')
    month_html, month_h = _table(m_above, m_below, 'MONTH')
    year_html,  year_h  = _table(y_above, y_below,  'YEAR')

    total_height = week_h + month_h + year_h + 16
    combined = (
        "<div style='display:flex;flex-direction:column;gap:8px;font-family:" + FONTS + "'>"
        + week_html + month_html + year_html +
        "</div>"
    )
    _wrap(combined, total_height)
    return total_height


# ── MAIN ─────────────────────────────────────────────────────────────────────

def render_pulse_tab(is_mobile):
    with st.spinner('Scanning markets...'):
        data          = _fetch_pulse_batch()
        spark_data    = _fetch_sparklines()
        breakout_data = _fetch_breakout_data()

    if not data:
        st.info('Markets data loading — try refreshing.')
        return

    _render_market_status_bar()
    _render_hero_row(data)
    if spark_data:
        _render_sparkline_row(spark_data, data)

    if is_mobile:
        _render_movers(data)
        _render_breakout_tables(breakout_data, pulse_data=data)
        _render_pulse_news(iframe_height=400)
        _render_heatmap_grid(data)
    else:
        col_left, col_right = st.columns([55, 45])

        with col_left:
            movers_height = _render_movers(data)
            bo_height = _render_breakout_tables(breakout_data, pulse_data=data)

        with col_right:
            news_height = movers_height + 8 + bo_height
            _render_pulse_news(iframe_height=news_height)

        _render_heatmap_grid(data)
