"""
SANPO — Rates tab
US Treasury yield curve (Treasury.gov XML feed) + SG SGS yield curve (MAS API).
Overlay both curves, show 2s10s spread, SORA vs SOFR.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import urllib.request
import json
import xml.etree.ElementTree as ET
import logging
from datetime import datetime, timedelta

from config import get_theme, FONTS

logger = logging.getLogger(__name__)

_UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'

# =============================================================================
# US TREASURY YIELD CURVE
# =============================================================================

US_TENORS = ['1 Mo', '2 Mo', '3 Mo', '4 Mo', '6 Mo', '1 Yr', '2 Yr', '3 Yr',
             '5 Yr', '7 Yr', '10 Yr', '20 Yr', '30 Yr']
US_TENOR_MONTHS = [1, 2, 3, 4, 6, 12, 24, 36, 60, 84, 120, 240, 360]
US_XML_FIELDS = ['d:BC_1MONTH', 'd:BC_2MONTH', 'd:BC_3MONTH', 'd:BC_4MONTH',
                 'd:BC_6MONTH', 'd:BC_1YEAR', 'd:BC_2YEAR', 'd:BC_3YEAR',
                 'd:BC_5YEAR', 'd:BC_7YEAR', 'd:BC_10YEAR', 'd:BC_20YEAR', 'd:BC_30YEAR']


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_us_curve():
    """Fetch US Treasury par yield curves — current year + previous year for comparisons."""
    try:
        now = datetime.now()
        all_rows = []
        for yr in [now.year - 1, now.year]:
            url = (f'https://home.treasury.gov/resource-center/data-chart-center/interest-rates/'
                   f'pages/xml?data=daily_treasury_yield_curve&field_tdr_date_value={yr}')
            req = urllib.request.Request(url, headers={'User-Agent': _UA})
            resp = urllib.request.urlopen(req, timeout=15)
            raw = resp.read()

            ns = {
                'a': 'http://www.w3.org/2005/Atom',
                'm': 'http://schemas.microsoft.com/ado/2007/08/dataservices/metadata',
                'd': 'http://schemas.microsoft.com/ado/2007/08/dataservices',
            }
            root = ET.fromstring(raw)
            entries = root.findall('.//a:entry', ns)
            for entry in entries:
                props = entry.find('.//m:properties', ns)
                if props is None:
                    continue
                date_el = props.find('d:NEW_DATE', ns)
                if date_el is None:
                    continue
                date_str = date_el.text[:10] if date_el.text else ''
                yields = []
                for field in US_XML_FIELDS:
                    el = props.find(field, ns)
                    val = float(el.text) if el is not None and el.text else None
                    yields.append(val)
                all_rows.append({'date': date_str, 'yields': yields})

        if not all_rows:
            return None, None

        all_rows.sort(key=lambda x: x['date'])
        latest = all_rows[-1]

        return {
            'date': latest['date'],
            'tenors': US_TENORS,
            'months': US_TENOR_MONTHS,
            'yields': latest['yields'],
            'all_rows': all_rows,
        }, None
    except Exception as e:
        logger.warning(f"US curve fetch error: {e}")
        return None, str(e)


# =============================================================================
# SG SGS YIELD CURVE (MAS API primary, yfinance fallback)
# =============================================================================

SG_TENORS = ['6 Mo', '1 Yr', '2 Yr', '5 Yr', '10 Yr', '15 Yr', '20 Yr', '30 Yr']
SG_TENOR_MONTHS = [6, 12, 24, 60, 120, 180, 240, 360]
SG_FIELDS = ['sgs_6m_bid_yield', 'sgs_1y_bid_yield', 'sgs_2y_bid_yield',
             'sgs_5y_bid_yield', 'sgs_10y_bid_yield', 'sgs_15y_bid_yield',
             'sgs_20y_bid_yield', 'sgs_30y_bid_yield']

# SORA resource
SORA_RESOURCE = '9a0bf149-308c-4bd2-832d-76c8e6cb47ed'
# SGS benchmark yields resource
SGS_RESOURCE = '5f2b18a8-0883-4769-a635-879c63d3caac'

# Yahoo Finance tickers for SG bonds (fallback)
SG_YF_TICKERS = {
    '6 Mo': 'SG6MT=RR',
    '1 Yr': 'SG1YT=RR',
    '2 Yr': 'SG2YT=RR',
    '5 Yr': 'SG5YT=RR',
    '10 Yr': 'SG10YT=RR',
    '15 Yr': 'SG15YT=RR',
    '20 Yr': 'SG20YT=RR',
    '30 Yr': 'SG30YT=RR',
}


def _fetch_sg_mas():
    """Try MAS API — avoid sort param (causes issues), use between filter + client-side sort."""
    import urllib.parse
    from datetime import date

    end = date.today()
    # date(year-2, ...) raises on 29 Feb; step back in days instead.
    start = end - timedelta(days=730)

    params = urllib.parse.urlencode({
        'resource_id': SGS_RESOURCE,
        'limit': 800,
    })
    between = f"&between[end_of_day]={start.isoformat()},{end.isoformat()}"
    url = f'https://eservices.mas.gov.sg/api/action/datastore/search.json?{params}{between}'

    # Try requests first (better header handling), fall back to urllib
    try:
        import requests as _req
        resp = _req.get(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json',
        }, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        req = urllib.request.Request(url, headers={'User-Agent': _UA})
        resp = urllib.request.urlopen(req, timeout=15)
        data = json.loads(resp.read())

    records = data.get('result', {}).get('records', [])
    if not records:
        return None

    all_rows = []
    for r in records:
        date_str = r.get('end_of_day', '')
        yields = []
        for f in SG_FIELDS:
            val = r.get(f)
            try:
                yields.append(float(val) if val else None)
            except (ValueError, TypeError):
                yields.append(None)
        all_rows.append({'date': date_str, 'yields': yields})

    all_rows.sort(key=lambda x: x['date'])
    latest = all_rows[-1]

    return {
        'date': latest['date'],
        'tenors': SG_TENORS,
        'months': SG_TENOR_MONTHS,
        'yields': latest['yields'],
        'all_rows': all_rows,
    }


def _fetch_sg_yfinance():
    """Fallback: use yfinance for SG bond yields with ~2Y history."""
    import yfinance as yf

    tickers_str = ' '.join(SG_YF_TICKERS.values())
    data = yf.download(tickers_str, period='2y', interval='1d', progress=False, threads=True)

    if data.empty:
        return None

    close = data['Close'] if 'Close' in data.columns else data
    # Build all_rows from the DataFrame
    all_rows = []
    for date_idx in close.index:
        date_str = date_idx.strftime('%Y-%m-%d')
        yields = []
        for tenor in SG_TENORS:
            ticker = SG_YF_TICKERS.get(tenor)
            try:
                val = float(close.loc[date_idx, ticker])
                yields.append(val if not np.isnan(val) else None)
            except Exception:
                yields.append(None)
        # Only include rows where at least some data exists
        if any(y is not None for y in yields):
            all_rows.append({'date': date_str, 'yields': yields})

    if not all_rows:
        return None

    all_rows.sort(key=lambda x: x['date'])
    latest = all_rows[-1]

    return {
        'date': latest['date'],
        'tenors': SG_TENORS,
        'months': SG_TENOR_MONTHS,
        'yields': latest['yields'],
        'all_rows': all_rows,
    }


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_sg_curve():
    """Fetch SGS yields — try MAS API first, fall back to yfinance."""
    # Try MAS API
    try:
        result = _fetch_sg_mas()
        if result:
            return result, None
    except Exception as e:
        logger.warning(f"MAS API error: {e}")

    # Fallback to yfinance
    try:
        result = _fetch_sg_yfinance()
        if result:
            logger.info("SG rates loaded via yfinance fallback")
            return result, None
    except Exception as e:
        logger.warning(f"yfinance SG fallback error: {e}")

    return None, "Both MAS API and yfinance failed"


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_sora():
    """Fetch latest SORA from MAS API."""
    try:
        import urllib.parse
        params = urllib.parse.urlencode({
            'resource_id': SORA_RESOURCE,
            'limit': 10,
        })
        url = f'https://eservices.mas.gov.sg/api/action/datastore/search.json?{params}'

        try:
            import requests as _req
            resp = _req.get(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'application/json',
            }, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            req = urllib.request.Request(url, headers={'User-Agent': _UA})
            resp = urllib.request.urlopen(req, timeout=15)
            data = json.loads(resp.read())

        records = data.get('result', {}).get('records', [])
        if not records:
            return None
        # Sort by date descending, pick latest
        records.sort(key=lambda x: x.get('end_of_day', ''), reverse=True)
        r = records[0]
        return {
            'date': r.get('end_of_day', ''),
            'sora': r.get('sora', ''),
            'comp_1m': r.get('comp_sora_1m', ''),
            'comp_3m': r.get('comp_sora_3m', ''),
            'comp_6m': r.get('comp_sora_6m', ''),
        }
    except Exception:
        return None


# =============================================================================
# COMPARISON HELPERS
# =============================================================================

COMPARE_OPTIONS = {
    'None': 0,
    '1 Week': 7,
    '1 Month': 30,
    '3 Months': 91,
    '6 Months': 182,
    '1 Year': 365,
}

def _find_curve_near_date(all_rows, target_date_str):
    """Find the row closest to (but not after) target date."""
    if not all_rows:
        return None
    best = None
    for r in all_rows:
        if r['date'] <= target_date_str:
            best = r
    return best


def _get_comparison_curves(data, compare_days_list):
    """Return dict of {label: {date, yields}} for each comparison period."""
    if not data or 'all_rows' not in data:
        return {}
    now_str = data['date']
    try:
        now_dt = datetime.strptime(now_str[:10], '%Y-%m-%d')
    except Exception:
        return {}

    comps = {}
    for label, days in compare_days_list:
        if days == 0:
            continue
        target_dt = now_dt - timedelta(days=days)
        target_str = target_dt.strftime('%Y-%m-%d')
        row = _find_curve_near_date(data['all_rows'], target_str)
        if row:
            comps[label] = row
    return comps


def _fmt_date(d):
    """Format '2026-03-04' as '04-Mar-2026'."""
    try:
        dt = datetime.strptime(d[:10], '%Y-%m-%d')
        return dt.strftime('%d-%b-%Y')
    except Exception:
        return d


# =============================================================================
# RENDER
# =============================================================================

def _render_curve_chart(us, theme):
    """US yield curve — today (green) + 1M/3M/1Y ago (orange dotted gradient)."""
    _pbg = theme.get('plot_bg', '#0f1117'); _grd = theme.get('grid', '#1a1f2e')
    pos_c = theme['pos']

    if not us:
        return

    compare_days = [
        ('1 Year',   365),
        ('3 Months', 91),
    ]

    # Orange gradient dotted: oldest = darkest, newest = lightest
    comp_styles = [
        {'color': '#7c2d12', 'width': 1.3},   # 1Y — darkest orange
        {'color': '#fb923c', 'width': 1.3},   # 3M — lighter orange
    ]

    fig = go.Figure()
    us_x = [m / 12 for m in us['months']]

    # Historical curves — dotted orange gradient
    comps = _get_comparison_curves(us, compare_days)
    for i, (label, row) in enumerate(comps.items()):
        style = comp_styles[i % len(comp_styles)]
        fig.add_trace(go.Scatter(x=us_x, y=row['yields'], mode='lines',
            name=f"{label} ({_fmt_date(row['date'])})",
            line=dict(color=style['color'], width=style['width'], dash='dot'),
            hovertemplate='%{y:.2f}%<extra>' + label + '</extra>'))

    # Today — green, thick, solid, markers
    fig.add_trace(go.Scatter(x=us_x, y=us['yields'], mode='lines+markers',
        name=f"Today ({_fmt_date(us['date'])})",
        line=dict(color=pos_c, width=3),
        marker=dict(size=6, color=pos_c),
        hovertemplate='%{text}<br>%{y:.2f}%<extra>Today</extra>',
        text=us['tenors']))

    # Tenor labels on today's curve — 1M, 2Y, 10Y, 30Y
    tenor_labels = {0: '1M', 6: '2Y', 10: '10Y', 12: '30Y'}
    for idx, label in tenor_labels.items():
        if idx < len(us['yields']) and us['yields'][idx] is not None:
            fig.add_annotation(x=us_x[idx], y=us['yields'][idx],
                text=f"<b>{label}</b> {us['yields'][idx]:.2f}%",
                showarrow=True, arrowhead=0, arrowcolor='rgba(34,197,94,0.25)', ax=0, ay=-25,
                font=dict(size=10, color=pos_c, family=FONTS),
                bgcolor='rgba(15,17,23,0.5)', borderpad=2)

    fig.update_layout(template='plotly_dark', height=450,
        margin=dict(l=40, r=20, t=30, b=40), plot_bgcolor=_pbg, paper_bgcolor=_pbg,
        showlegend=True, legend=dict(x=0.01, y=0.99, font=dict(size=10, family=FONTS),
                                     bgcolor='rgba(15,17,23,0.85)'),
        hovermode='x unified', font=dict(family=FONTS),
        xaxis_title='Maturity (Years)', yaxis_title='Yield %')
    fig.update_xaxes(gridcolor=_grd, tickfont=dict(color='#888', size=9, family=FONTS),
                     tickvals=[0.08, 0.17, 0.25, 0.33, 0.5, 1, 2, 3, 5, 7, 10, 20, 30],
                     ticktext=['1M', '2M', '3M', '4M', '6M', '1Y', '2Y', '3Y', '5Y', '7Y', '10Y', '20Y', '30Y'])
    fig.update_yaxes(gridcolor=_grd, tickfont=dict(color='#888', size=9, family=FONTS), side='right')
    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})


def _render_us_table(us, theme):
    """US rates table with 3M and 1Y deltas."""
    _bg3 = theme.get('bg3', '#0f172a'); _bdr = theme.get('border', '#1e293b')
    _txt = theme.get('text', '#e2e8f0')
    pos_c = theme['pos']; neg_c = theme['neg']

    th = f"padding:4px 8px;border-bottom:1px solid {_bdr};color:#f8fafc;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.05em;"
    td = f"padding:4px 8px;border-bottom:1px solid {_bdr}22;font-size:11px;"

    # Get 3M and 1Y ago yields
    comps = _get_comparison_curves(us, [('3M', 91), ('1Y', 365)])
    comp_3m = comps.get('3M', {}).get('yields', [None] * len(us['yields']))
    comp_1y = comps.get('1Y', {}).get('yields', [None] * len(us['yields']))

    html = f"""<div style='overflow-x:auto;border:1px solid {_bdr};border-radius:4px'>
    <table style='border-collapse:collapse;font-family:{FONTS};width:100%;line-height:1.3'>
    <thead style='background:{_bg3}'><tr>
        <th style='{th}text-align:left'>TENOR</th>
        <th style='{th}text-align:right;color:{pos_c}'>YIELD</th>
        <th style='{th}text-align:right'>Δ 3M</th>
        <th style='{th}text-align:right'>Δ 1Y</th>
    </tr></thead><tbody>"""

    for i, (tenor, y) in enumerate(zip(us['tenors'], us['yields'])):
        y_str = f'{y:.2f}%' if y is not None else '—'

        d3_str = ''
        if y is not None and i < len(comp_3m) and comp_3m[i] is not None:
            d = y - comp_3m[i]
            c = pos_c if d <= 0 else neg_c
            d3_str = f"<span style='color:{c}'>{d:+.2f}</span>"

        d1y_str = ''
        if y is not None and i < len(comp_1y) and comp_1y[i] is not None:
            d = y - comp_1y[i]
            c = pos_c if d <= 0 else neg_c
            d1y_str = f"<span style='color:{c}'>{d:+.2f}</span>"

        html += f"""<tr>
            <td style='{td}color:{_txt};font-weight:600'>{tenor}</td>
            <td style='{td}text-align:right;color:{pos_c}'>{y_str}</td>
            <td style='{td}text-align:right'>{d3_str}</td>
            <td style='{td}text-align:right'>{d1y_str}</td>
        </tr>"""

    html += "</tbody></table></div>"
    st.markdown(html, unsafe_allow_html=True)


def _render_sg_curve_chart(sg, theme):
    """SG SGS yield curve — today (green) + 1M/3M/1Y ago (orange dotted gradient)."""
    _pbg = theme.get('plot_bg', '#0f1117'); _grd = theme.get('grid', '#1a1f2e')
    pos_c = theme['pos']

    if not sg:
        return

    compare_days = [
        ('1 Year',   365),
        ('3 Months', 91),
    ]

    comp_styles = [
        {'color': '#7c2d12', 'width': 1.3},
        {'color': '#fb923c', 'width': 1.3},
    ]

    fig = go.Figure()
    sg_x = [m / 12 for m in sg['months']]

    comps = _get_comparison_curves(sg, compare_days)
    for i, (label, row) in enumerate(comps.items()):
        style = comp_styles[i % len(comp_styles)]
        fig.add_trace(go.Scatter(x=sg_x, y=row['yields'], mode='lines',
            name=f"{label} ({_fmt_date(row['date'])})",
            line=dict(color=style['color'], width=style['width'], dash='dot'),
            hovertemplate='%{y:.2f}%<extra>' + label + '</extra>'))

    fig.add_trace(go.Scatter(x=sg_x, y=sg['yields'], mode='lines+markers',
        name=f"Today ({_fmt_date(sg['date'])})",
        line=dict(color=pos_c, width=3),
        marker=dict(size=6, color=pos_c),
        hovertemplate='%{text}<br>%{y:.2f}%<extra>Today</extra>',
        text=sg['tenors']))

    # Tenor labels: 2Y (idx 2), 10Y (idx 4), 30Y (idx 7)
    tenor_labels = {2: '2Y', 4: '10Y', 7: '30Y'}
    for idx, label in tenor_labels.items():
        if idx < len(sg['yields']) and sg['yields'][idx] is not None:
            fig.add_annotation(x=sg_x[idx], y=sg['yields'][idx],
                text=f"<b>{label}</b> {sg['yields'][idx]:.2f}%",
                showarrow=True, arrowhead=0, arrowcolor='rgba(34,197,94,0.25)', ax=0, ay=-25,
                font=dict(size=10, color=pos_c, family=FONTS),
                bgcolor='rgba(15,17,23,0.5)', borderpad=2)

    fig.update_layout(template='plotly_dark', height=450,
        margin=dict(l=40, r=20, t=30, b=40), plot_bgcolor=_pbg, paper_bgcolor=_pbg,
        showlegend=True, legend=dict(x=0.01, y=0.99, font=dict(size=10, family=FONTS),
                                     bgcolor='rgba(15,17,23,0.85)'),
        hovermode='x unified', font=dict(family=FONTS),
        xaxis_title='Maturity (Years)', yaxis_title='Yield %')
    fig.update_xaxes(gridcolor=_grd, tickfont=dict(color='#888', size=9, family=FONTS),
                     tickvals=[0.5, 1, 2, 5, 10, 15, 20, 30],
                     ticktext=['6M', '1Y', '2Y', '5Y', '10Y', '15Y', '20Y', '30Y'])
    fig.update_yaxes(gridcolor=_grd, tickfont=dict(color='#888', size=9, family=FONTS), side='right')
    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})


def _render_sg_table(sg, sora, theme):
    """SG rates table with 3M and 1Y deltas + SORA."""
    pos_c = theme['pos']; neg_c = theme['neg']
    _bg3 = theme.get('bg3', '#0f172a'); _bdr = theme.get('border', '#1e293b')
    _txt = theme.get('text', '#e2e8f0'); _txt2 = theme.get('text2', '#94a3b8')

    th = f"padding:4px 8px;border-bottom:1px solid {_bdr};color:#f8fafc;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.05em;"
    td = f"padding:4px 8px;border-bottom:1px solid {_bdr}22;font-size:11px;"

    comps = _get_comparison_curves(sg, [('3M', 91), ('1Y', 365)])
    comp_3m = comps.get('3M', {}).get('yields', [None] * len(sg['yields']))
    comp_1y = comps.get('1Y', {}).get('yields', [None] * len(sg['yields']))

    html = f"""<div style='overflow-x:auto;border:1px solid {_bdr};border-radius:4px'>
    <table style='border-collapse:collapse;font-family:{FONTS};width:100%;line-height:1.3'>
    <thead style='background:{_bg3}'><tr>
        <th style='{th}text-align:left'>TENOR</th>
        <th style='{th}text-align:right;color:{pos_c}'>YIELD</th>
        <th style='{th}text-align:right'>Δ 3M</th>
        <th style='{th}text-align:right'>Δ 1Y</th>
    </tr></thead><tbody>"""

    for i, (tenor, y) in enumerate(zip(sg['tenors'], sg['yields'])):
        y_str = f'{y:.2f}%' if y is not None else '—'

        d3_str = ''
        if y is not None and i < len(comp_3m) and comp_3m[i] is not None:
            d = y - comp_3m[i]
            c = pos_c if d <= 0 else neg_c
            d3_str = f"<span style='color:{c}'>{d:+.2f}</span>"

        d1y_str = ''
        if y is not None and i < len(comp_1y) and comp_1y[i] is not None:
            d = y - comp_1y[i]
            c = pos_c if d <= 0 else neg_c
            d1y_str = f"<span style='color:{c}'>{d:+.2f}</span>"

        html += f"""<tr>
            <td style='{td}color:{_txt};font-weight:600'>{tenor}</td>
            <td style='{td}text-align:right;color:{pos_c}'>{y_str}</td>
            <td style='{td}text-align:right'>{d3_str}</td>
            <td style='{td}text-align:right'>{d1y_str}</td>
        </tr>"""

    html += "</tbody></table></div>"

    # SORA below
    if sora:
        html += (f"<div style='margin-top:6px;padding:5px 10px;background:{_bg3};border-radius:4px;"
                 f"font-family:{FONTS};font-size:10px;color:{_txt2}'>"
                 f"SORA ({_fmt_date(sora['date'])}): "
                 f"O/N <b style='color:{pos_c}'>{sora.get('sora','—')}%</b> · "
                 f"1M <b style='color:{pos_c}'>{sora.get('comp_1m','—')}%</b> · "
                 f"3M <b style='color:{pos_c}'>{sora.get('comp_3m','—')}%</b> · "
                 f"6M <b style='color:{pos_c}'>{sora.get('comp_6m','—')}%</b>"
                 f"</div>")

    st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# MAIN
# =============================================================================

def render_rates_tab(is_mobile):
    t = get_theme()
    _mut = t.get('muted', '#475569')
    _lbl = f"color:#f8fafc;font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;font-family:{FONTS}"

    with st.spinner('Fetching rates...'):
        us, us_err = _fetch_us_curve()
        sg, sg_err = _fetch_sg_curve()
        sora = _fetch_sora()

    # Sub-tabs: US | SG
    tab_us, tab_sg = st.tabs(['US RATES', 'SG RATES'])

    with tab_us:
        if not us:
            st.warning('Could not fetch US rate data')
        else:
            # Summary bar
            us_1m = us['yields'][0] if us['yields'][0] else None
            us_10y = us['yields'][10] if us['yields'][10] else None
            us_2y = us['yields'][6] if us['yields'][6] else None
            _bg3 = t.get('bg3', '#0f172a'); _txt2 = t.get('text2', '#94a3b8')
            parts = [f"US Treasury {_fmt_date(us['date'])}"]
            if us_1m:
                parts.append(f"1M <b style='color:{t['pos']}'>{us_1m:.2f}%</b>")
            if us_2y:
                parts.append(f"2Y <b style='color:{t['pos']}'>{us_2y:.2f}%</b>")
            if us_10y:
                parts.append(f"10Y <b style='color:{t['pos']}'>{us_10y:.2f}%</b>")
            if us_2y and us_10y:
                s = us_10y - us_2y
                c = t['pos'] if s >= 0 else t['neg']
                parts.append(f"2s10s <b style='color:{c}'>{s*100:+.0f}bp</b>")
            st.markdown(f"""<div style='padding:5px 10px;background:{_bg3};font-family:{FONTS};
                font-size:10px;color:{_txt2};border-radius:4px;margin-bottom:6px'>
                {' &nbsp;·&nbsp; '.join(parts)}</div>""", unsafe_allow_html=True)

            if is_mobile:
                _render_curve_chart(us, t)
                _render_us_table(us, t)
            else:
                left, right = st.columns([6, 4])
                with left:
                    _render_curve_chart(us, t)
                with right:
                    st.markdown(f"<div style='{_lbl};margin-bottom:4px'>RATES TABLE</div>", unsafe_allow_html=True)
                    _render_us_table(us, t)

    with tab_sg:
        if not sg:
            st.warning('Could not fetch SG rate data')
        else:
            _bg3 = t.get('bg3', '#0f172a'); _txt2 = t.get('text2', '#94a3b8')
            pos_c = t['pos']
            sg_10y = sg['yields'][4] if sg['yields'][4] else None
            sg_2y = sg['yields'][2] if sg['yields'][2] else None
            parts = [f"SGS Benchmarks {_fmt_date(sg['date'])}"]
            if sg_10y:
                parts.append(f"10Y <b style='color:{pos_c}'>{sg_10y:.2f}%</b>")
            if sg_2y:
                parts.append(f"2Y <b style='color:{pos_c}'>{sg_2y:.2f}%</b>")
            if sg_2y and sg_10y:
                s = sg_10y - sg_2y
                c = t['pos'] if s >= 0 else t['neg']
                parts.append(f"2s10s <b style='color:{c}'>{s*100:+.0f}bp</b>")
            if sora:
                parts.append(f"SORA <b style='color:{pos_c}'>{sora.get('sora','—')}%</b>")
            st.markdown(f"""<div style='padding:5px 10px;background:{_bg3};font-family:{FONTS};
                font-size:10px;color:{_txt2};border-radius:4px;margin-bottom:6px'>
                {' &nbsp;·&nbsp; '.join(parts)}</div>""", unsafe_allow_html=True)

            if is_mobile:
                _render_sg_curve_chart(sg, t)
                _render_sg_table(sg, sora, t)
            else:
                left, right = st.columns([6, 4])
                with left:
                    _render_sg_curve_chart(sg, t)
                with right:
                    st.markdown(f"<div style='{_lbl};margin-bottom:4px'>RATES TABLE</div>", unsafe_allow_html=True)
                    _render_sg_table(sg, sora, t)
