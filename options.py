"""
SANPO — Options tab
Live options chain, IV skew, term structure, OI analysis, max pain.
Data via yfinance (equity/ETF options — no futures options).
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

from config import get_theme, FONTS

logger = logging.getLogger(__name__)

# =============================================================================
# DATA
# =============================================================================

@st.cache_data(ttl=300, show_spinner=False)
@st.cache_data(ttl=600, show_spinner=False)
def _fetch_expiries(symbol):
    try:
        return list(yf.Ticker(symbol).options)
    except Exception:
        return []


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_chain(symbol, expiry):
    try:
        tk = yf.Ticker(symbol)
        chain = tk.option_chain(expiry)
        calls = chain.calls.copy(); puts = chain.puts.copy()
        hist = tk.history(period='2d')
        price = float(hist['Close'].iloc[-1]) if len(hist) > 0 else None
        return calls, puts, price
    except Exception as e:
        logger.warning(f"Options fetch error [{symbol} {expiry}]: {e}")
        return None, None, None


@st.cache_data(ttl=600, show_spinner=False)
def _fetch_term_structure(symbol):
    try:
        tk = yf.Ticker(symbol)
        expiries = tk.options
        hist = tk.history(period='2d')
        price = float(hist['Close'].iloc[-1]) if len(hist) > 0 else None
        if not price or not expiries:
            return None
        rows = []
        for exp in expiries[:15]:
            try:
                c = tk.option_chain(exp)
                if len(c.calls) == 0:
                    continue
                atm_c = c.calls.iloc[(c.calls['strike'] - price).abs().argsort()[:1]]
                atm_p = c.puts.iloc[(c.puts['strike'] - price).abs().argsort()[:1]]
                civ = float(atm_c['impliedVolatility'].iloc[0]) * 100
                piv = float(atm_p['impliedVolatility'].iloc[0]) * 100
                dte = max(0, (pd.Timestamp(exp) - pd.Timestamp.now()).days)
                rows.append({'expiry': exp, 'dte': dte, 'call_iv': civ, 'put_iv': piv,
                             'avg_iv': (civ + piv) / 2})
            except Exception:
                continue
        return pd.DataFrame(rows) if rows else None
    except Exception:
        return None


def _max_pain(calls, puts):
    strikes = sorted(set(calls['strike'].tolist() + puts['strike'].tolist()))
    if not strikes:
        return None
    call_oi = calls.set_index('strike')['openInterest'].fillna(0)
    put_oi = puts.set_index('strike')['openInterest'].fillna(0)
    best_s, best_v = strikes[0], float('inf')
    for s in strikes:
        pain = 0
        for k, oi in call_oi.items():
            if s > k:
                pain += (s - k) * oi * 100
        for k, oi in put_oi.items():
            if s < k:
                pain += (k - s) * oi * 100
        if pain < best_v:
            best_v = pain; best_s = s
    return best_s


# =============================================================================
# CHAIN TABLE
# =============================================================================

def _render_chain_table(calls, puts, price, theme):
    pos_c = theme['pos']; neg_c = theme['neg']
    _bg3 = theme.get('bg3', '#0f172a'); _bdr = theme.get('border', '#1e293b')
    _txt2 = theme.get('text2', '#94a3b8'); _mut = theme.get('muted', '#475569')

    c = calls[['strike', 'bid', 'ask', 'lastPrice', 'volume', 'openInterest', 'impliedVolatility']].copy()
    p = puts[['strike', 'bid', 'ask', 'lastPrice', 'volume', 'openInterest', 'impliedVolatility']].copy()
    c.columns = ['strike', 'c_bid', 'c_ask', 'c_last', 'c_vol', 'c_oi', 'c_iv']
    p.columns = ['strike', 'p_bid', 'p_ask', 'p_last', 'p_vol', 'p_oi', 'p_iv']
    merged = pd.merge(c, p, on='strike', how='outer').sort_values('strike').fillna(0)
    lo, hi = price * 0.70, price * 1.30
    merged = merged[(merged['strike'] >= lo) & (merged['strike'] <= hi)]

    th = f"padding:3px 6px;border-bottom:1px solid {_bdr};color:#f8fafc;font-weight:600;font-size:8px;text-transform:uppercase;letter-spacing:0.05em;"
    td = f"padding:3px 6px;border-bottom:1px solid {_bdr}22;font-size:10px;"

    html = f"""<div style='overflow-x:auto;border:1px solid {_bdr};border-radius:4px;max-height:55vh;overflow-y:auto'>
    <table style='border-collapse:collapse;font-family:{FONTS};width:100%;line-height:1.2'>
    <thead style='background:{_bg3};position:sticky;top:0;z-index:1'><tr>
        <th style='{th}text-align:right;color:{pos_c}' colspan='4'>CALLS</th>
        <th style='{th}text-align:center;color:#f8fafc'>STRIKE</th>
        <th style='{th}text-align:left;color:{neg_c}' colspan='4'>PUTS</th>
    </tr><tr>
        <th style='{th}text-align:right'>IV</th>
        <th style='{th}text-align:right'>OI</th>
        <th style='{th}text-align:right'>Vol</th>
        <th style='{th}text-align:right'>Bid/Ask</th>
        <th style='{th}text-align:center'></th>
        <th style='{th}text-align:left'>Bid/Ask</th>
        <th style='{th}text-align:left'>Vol</th>
        <th style='{th}text-align:left'>OI</th>
        <th style='{th}text-align:left'>IV</th>
    </tr></thead><tbody>"""

    for _, r in merged.iterrows():
        s = r['strike']
        itm_call = s <= price; itm_put = s >= price
        is_atm = abs(s - price) / price < 0.005
        row_bg = 'rgba(74,222,128,0.08)' if is_atm else 'transparent'
        c_bg = 'rgba(74,222,128,0.04)' if itm_call else 'transparent'
        p_bg = 'rgba(245,158,11,0.04)' if itm_put else 'transparent'
        strike_c = '#f8fafc' if is_atm else _txt2

        c_iv = f"{r['c_iv']*100:.0f}%" if r['c_iv'] > 0 else '—'
        p_iv = f"{r['p_iv']*100:.0f}%" if r['p_iv'] > 0 else '—'
        c_ba = f"{r['c_bid']:.2f}/{r['c_ask']:.2f}" if r['c_ask'] > 0 else '—'
        p_ba = f"{r['p_bid']:.2f}/{r['p_ask']:.2f}" if r['p_ask'] > 0 else '—'
        c_vol = f"{int(r['c_vol']):,}" if r['c_vol'] > 0 else '—'
        p_vol = f"{int(r['p_vol']):,}" if r['p_vol'] > 0 else '—'
        c_oi = f"{int(r['c_oi']):,}" if r['c_oi'] > 0 else '—'
        p_oi = f"{int(r['p_oi']):,}" if r['p_oi'] > 0 else '—'

        html += f"""<tr style='background:{row_bg}'>
            <td style='{td}text-align:right;background:{c_bg};color:{_txt2}'>{c_iv}</td>
            <td style='{td}text-align:right;background:{c_bg};color:{_txt2}'>{c_oi}</td>
            <td style='{td}text-align:right;background:{c_bg};color:{_txt2}'>{c_vol}</td>
            <td style='{td}text-align:right;background:{c_bg};color:{pos_c};font-weight:500'>{c_ba}</td>
            <td style='{td}text-align:center;color:{strike_c};font-weight:700;font-size:11px'>{s:.0f}</td>
            <td style='{td}text-align:left;background:{p_bg};color:{neg_c};font-weight:500'>{p_ba}</td>
            <td style='{td}text-align:left;background:{p_bg};color:{_txt2}'>{p_vol}</td>
            <td style='{td}text-align:left;background:{p_bg};color:{_txt2}'>{p_oi}</td>
            <td style='{td}text-align:left;background:{p_bg};color:{_txt2}'>{p_iv}</td>
        </tr>"""

    html += "</tbody></table></div>"
    st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# CHARTS
# =============================================================================

def _render_iv_skew(calls, puts, price, theme):
    pos_c = theme['pos']; neg_c = theme['neg']
    _pbg = theme.get('plot_bg', '#0f1117'); _grd = theme.get('grid', '#1a1f2e')

    lo, hi = price * 0.75, price * 1.25
    c = calls[(calls['strike'] >= lo) & (calls['strike'] <= hi) & (calls['impliedVolatility'] > 0)]
    p = puts[(puts['strike'] >= lo) & (puts['strike'] <= hi) & (puts['impliedVolatility'] > 0)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=c['strike'], y=c['impliedVolatility'] * 100,
        mode='lines+markers', name='Call IV', line=dict(color=pos_c, width=2),
        marker=dict(size=4), hovertemplate='Strike %{x:.0f}<br>IV %{y:.1f}%<extra>Call</extra>'))
    fig.add_trace(go.Scatter(x=p['strike'], y=p['impliedVolatility'] * 100,
        mode='lines+markers', name='Put IV', line=dict(color=neg_c, width=2),
        marker=dict(size=4), hovertemplate='Strike %{x:.0f}<br>IV %{y:.1f}%<extra>Put</extra>'))
    fig.add_vline(x=price, line=dict(color='#ffffff', width=1, dash='dot'),
                  annotation_text=f'${price:.0f}', annotation_font=dict(color='#ffffff', size=10))

    fig.update_layout(template='plotly_dark', height=280,
        margin=dict(l=40, r=20, t=30, b=30), plot_bgcolor=_pbg, paper_bgcolor=_pbg,
        showlegend=True, legend=dict(x=0.01, y=0.99, font=dict(size=10, family=FONTS)),
        hovermode='x unified', font=dict(family=FONTS),
        yaxis_title='IV %', xaxis_title='Strike')
    fig.update_xaxes(gridcolor=_grd, tickfont=dict(color='#888', size=9, family=FONTS))
    fig.update_yaxes(gridcolor=_grd, tickfont=dict(color='#888', size=9, family=FONTS), side='right')
    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})


def _render_oi_volume(calls, puts, price, theme):
    pos_c = theme['pos']; neg_c = theme['neg']
    _pbg = theme.get('plot_bg', '#0f1117'); _grd = theme.get('grid', '#1a1f2e')

    lo, hi = price * 0.80, price * 1.20
    c = calls[(calls['strike'] >= lo) & (calls['strike'] <= hi)]
    p = puts[(puts['strike'] >= lo) & (puts['strike'] <= hi)]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=['Open Interest', 'Volume'])

    fig.add_trace(go.Bar(x=c['strike'], y=c['openInterest'].fillna(0), name='Call OI',
        marker_color=pos_c, opacity=0.7), row=1, col=1)
    fig.add_trace(go.Bar(x=p['strike'], y=-p['openInterest'].fillna(0), name='Put OI',
        marker_color=neg_c, opacity=0.7), row=1, col=1)
    fig.add_trace(go.Bar(x=c['strike'], y=c['volume'].fillna(0), name='Call Vol',
        marker_color=pos_c, opacity=0.5, showlegend=False), row=2, col=1)
    fig.add_trace(go.Bar(x=p['strike'], y=-p['volume'].fillna(0), name='Put Vol',
        marker_color=neg_c, opacity=0.5, showlegend=False), row=2, col=1)
    fig.add_vline(x=price, line=dict(color='#ffffff', width=1, dash='dot'))

    fig.update_layout(template='plotly_dark', height=350, barmode='overlay',
        margin=dict(l=40, r=20, t=30, b=30), plot_bgcolor=_pbg, paper_bgcolor=_pbg,
        showlegend=True, legend=dict(x=0.01, y=0.99, font=dict(size=10, family=FONTS)),
        hovermode='x unified', font=dict(family=FONTS))
    fig.update_xaxes(gridcolor=_grd, tickfont=dict(color='#888', size=9, family=FONTS))
    fig.update_yaxes(gridcolor=_grd, tickfont=dict(color='#888', size=9, family=FONTS), side='right')
    for ann in fig['layout']['annotations']:
        ann['font'] = dict(size=10, family=FONTS, color='#94a3b8')
    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})


def _render_term_structure(symbol, theme):
    pos_c = theme['pos']; neg_c = theme['neg']
    _pbg = theme.get('plot_bg', '#0f1117'); _grd = theme.get('grid', '#1a1f2e')
    _mut = theme.get('muted', '#475569')

    ts = _fetch_term_structure(symbol)
    if ts is None or len(ts) < 2:
        st.markdown(f"<div style='color:{_mut};font-size:10px;font-family:{FONTS};padding:8px'>Insufficient expiries for term structure</div>",
                    unsafe_allow_html=True)
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts['dte'], y=ts['call_iv'], mode='lines+markers',
        name='Call IV', line=dict(color=pos_c, width=2), marker=dict(size=5),
        hovertemplate='DTE %{x}<br>IV %{y:.1f}%<extra>Call</extra>'))
    fig.add_trace(go.Scatter(x=ts['dte'], y=ts['put_iv'], mode='lines+markers',
        name='Put IV', line=dict(color=neg_c, width=2), marker=dict(size=5),
        hovertemplate='DTE %{x}<br>IV %{y:.1f}%<extra>Put</extra>'))
    fig.add_trace(go.Scatter(x=ts['dte'], y=ts['avg_iv'], mode='lines',
        name='Avg', line=dict(color='#60a5fa', width=1.5, dash='dot'),
        hovertemplate='DTE %{x}<br>IV %{y:.1f}%<extra>Avg</extra>'))

    for _, row in ts.iterrows():
        fig.add_annotation(x=row['dte'], y=row['avg_iv'], text=row['expiry'][5:],
            showarrow=False, yshift=12, font=dict(size=8, color='#64748b', family=FONTS))

    fig.update_layout(template='plotly_dark', height=280,
        margin=dict(l=40, r=20, t=30, b=30), plot_bgcolor=_pbg, paper_bgcolor=_pbg,
        showlegend=True, legend=dict(x=0.01, y=0.99, font=dict(size=10, family=FONTS)),
        hovermode='x unified', font=dict(family=FONTS),
        yaxis_title='IV %', xaxis_title='Days to Expiry')
    fig.update_xaxes(gridcolor=_grd, tickfont=dict(color='#888', size=9, family=FONTS))
    fig.update_yaxes(gridcolor=_grd, tickfont=dict(color='#888', size=9, family=FONTS), side='right')
    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})


# =============================================================================
# MAIN RENDER
# =============================================================================

def render_options_tab(is_mobile):
    t = get_theme()
    pos_c = t['pos']; neg_c = t['neg']
    _mut = t.get('muted', '#475569'); _txt2 = t.get('text2', '#94a3b8')
    _bg3 = t.get('bg3', '#0f172a'); _bdr = t.get('border', '#1e293b')
    _lbl = f"color:#f8fafc;font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;font-family:{FONTS}"

    # Controls
    c1, c2, c3 = st.columns([2, 3, 3])
    with c1:
        st.markdown(f"<div style='{_lbl}'>SYMBOL</div>", unsafe_allow_html=True)
        symbol = st.text_input("Symbol", value='AAPL', key='opt_symbol',
                               label_visibility='collapsed').strip().upper()
    with c2:
        st.markdown(f"<div style='{_lbl}'>EXPIRY</div>", unsafe_allow_html=True)
        expiries = _fetch_expiries(symbol)
        if not expiries:
            st.markdown(f"<div style='color:{_mut};font-size:11px;font-family:{FONTS}'>No options for {symbol}</div>",
                        unsafe_allow_html=True)
            return
        expiry = st.selectbox("Expiry", expiries, key='opt_expiry', label_visibility='collapsed')

    # Fetch
    with st.spinner('Loading options...'):
        calls, puts, price = _fetch_chain(symbol, expiry)

    if calls is None or puts is None or price is None:
        st.warning(f'No options data for {symbol}'); return

    dte = max(0, (pd.Timestamp(expiry) - pd.Timestamp.now()).days)
    max_pain = _max_pain(calls, puts)

    # Summary stats
    total_call_oi = int(calls['openInterest'].fillna(0).sum())
    total_put_oi = int(puts['openInterest'].fillna(0).sum())
    pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 0
    total_call_vol = int(calls['volume'].fillna(0).sum())
    total_put_vol = int(puts['volume'].fillna(0).sum())

    atm_c = calls.iloc[(calls['strike'] - price).abs().argsort()[:1]]
    atm_p = puts.iloc[(puts['strike'] - price).abs().argsort()[:1]]
    atm_iv = (float(atm_c['impliedVolatility'].iloc[0]) + float(atm_p['impliedVolatility'].iloc[0])) / 2 * 100

    pcr_c = pos_c if pcr < 0.7 else (neg_c if pcr > 1.3 else _txt2)
    mp_c = pos_c if max_pain and max_pain > price else neg_c

    st.markdown(f"""
    <div style='padding:5px 10px;background:{_bg3};font-family:{FONTS};display:flex;justify-content:space-between;
        align-items:center;flex-wrap:wrap;gap:6px;border-radius:4px;margin-bottom:6px'>
        <span style='color:{_txt2};font-size:10px'>
            <b style='color:#f8fafc'>{symbol}</b> ${price:.2f}
            &nbsp;·&nbsp; {expiry} ({dte}d)
            &nbsp;·&nbsp; ATM IV <b style='color:#60a5fa'>{atm_iv:.1f}%</b>
        </span>
        <span style='color:{_txt2};font-size:10px'>
            P/C OI <b style='color:{pcr_c}'>{pcr:.2f}</b>
            &nbsp;·&nbsp; Call OI <b style='color:{pos_c}'>{total_call_oi:,}</b>
            Put OI <b style='color:{neg_c}'>{total_put_oi:,}</b>
            &nbsp;·&nbsp; Call Vol <b style='color:{pos_c}'>{total_call_vol:,}</b>
            Put Vol <b style='color:{neg_c}'>{total_put_vol:,}</b>
            &nbsp;·&nbsp; Max Pain <b style='color:{mp_c}'>{f"${max_pain:.0f}" if max_pain is not None else "N/A"}</b>
        </span>
    </div>""", unsafe_allow_html=True)

    # Layout
    if is_mobile:
        _render_chain_table(calls, puts, price, t)
        _render_iv_skew(calls, puts, price, t)
        _render_oi_volume(calls, puts, price, t)
        _render_term_structure(symbol, t)
    else:
        left, right = st.columns([5, 5])
        with left:
            st.markdown(f"<div style='{_lbl};margin-bottom:4px'>OPTIONS CHAIN</div>", unsafe_allow_html=True)
            _render_chain_table(calls, puts, price, t)
        with right:
            tab_skew, tab_oi, tab_term = st.tabs(['IV SKEW', 'OI / VOLUME', 'TERM STRUCTURE'])
            with tab_skew:
                _render_iv_skew(calls, puts, price, t)
            with tab_oi:
                _render_oi_volume(calls, puts, price, t)
            with tab_term:
                _render_term_structure(symbol, t)
