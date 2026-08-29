import streamlit as st
import pytz
import re
import logging
import warnings

from config import get_theme, st_html

# =============================================================================
# SETUP
# =============================================================================

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

st.set_page_config(page_title="SANPO", layout="wide", initial_sidebar_state="collapsed")



def _inject_theme_css():
    t = get_theme()
    is_light = t.get('mode') == 'light'
    bg = t.get('bg', '#212121'); bg2 = t.get('bg2', '#1a1a1a'); bg3 = t.get('bg3', '#2a2a2a')
    bdr = t.get('border', '#333333')
    grad = t.get('bg_gradient', 'none')
    txt = t.get('text', '#ececec'); txt2 = t.get('text2', '#afafaf')
    accent = t.get('accent', '#4ade80')
    # Widget chrome reads from the theme rather than hardcoded hexes, so the
    # palette really does live in one place (these were navy #1a2744).
    sb_bg = '#f1f5f9' if is_light else bg2
    sel_bg = '#f1f5f9' if is_light else bg3
    sel_c = '#334155' if is_light else txt2
    sel_bdr = '#e2e8f0' if is_light else bdr
    tab_bdr = bdr
    tab_c = '#64748b' if is_light else txt2
    tab_sel_c = '#0f172a' if is_light else txt
    radio_bg = bg3
    radio_bdr = bdr
    btn_bg = bg3
    btn_c = txt
    st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Orbitron:wght@500;700&display=swap');
    .stApp {{
        background-color: {bg};
        background-image: {grad};
        background-attachment: fixed;
        background-repeat: no-repeat;
        background-size: cover;
        font-family: 'Inter', sans-serif;
    }}
    header[data-testid="stHeader"] {{ background: transparent; }}
    .main .block-container, [data-testid="stAppViewContainer"] {{ background: transparent; }}
    /* Plotly panels are transparent now, so they read as part of the backdrop. */
    .js-plotly-plot .plot-container, .stPlotlyChart {{ background: transparent !important; }}
    [data-testid="stSidebar"] {{ background-color: {sb_bg}; }}
    .stSelectbox > div > div {{ background-color: {sel_bg}; color: {sel_c}; font-family: 'Inter', sans-serif; border: 1px solid {sel_bdr}; }}
    .stTextInput > div > div > input {{ font-family: 'Inter', sans-serif; }}
    div[data-testid="stHorizontalBlock"] {{ gap: 0.3rem; }}
    div[data-testid="stVerticalBlock"] {{ gap: 0.25rem !important; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 0; background-color: transparent; padding: 0; border-radius: 0; border-bottom: 1px solid {tab_bdr}; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent; color: {tab_c}; border: none;
        border-bottom: 2px solid transparent;
        padding: 8px 20px; font-size: 12px; font-weight: 600;
        letter-spacing: 0.1em; text-transform: uppercase;
        font-family: 'Inter', sans-serif;
    }}
    .stTabs [aria-selected="true"] {{ background-color: transparent; color: {tab_sel_c}; border-bottom: 2px solid {accent}; font-weight: 700; }}
    .stRadio > div {{ flex-direction: row; gap: 8px; }}
    .stRadio > div > label {{ background-color: {radio_bg}; padding: 4px 12px; border-radius: 3px;
        border: 1px solid {radio_bdr}; color: {sel_c}; font-size: 12px; }}
    div[data-testid="stMarkdownContainer"] p {{ margin-bottom: 0; }}
    .block-container {{ padding-top: 2.5rem; padding-bottom: 0rem; }}
    button[kind="secondary"] {{ background-color: {btn_bg}; color: {btn_c}; border: 1px solid {tab_bdr}; font-family: 'Inter', sans-serif; }}
    .stButton > button {{ font-size: 11px !important; padding: 4px 8px !important; min-height: 30px !important; font-family: 'Inter', sans-serif !important; }}
    @media (max-width: 768px) {{
        .block-container {{ padding: 2.5rem 0.5rem 0 0.5rem !important; }}
        .stButton > button {{ font-size: 9px !important; padding: 2px 4px !important; min-height: 24px !important; }}
        /* iOS Safari mishandles background-attachment:fixed (blank or jumping
           backdrop on scroll), so pin the gradient to the page instead. */
        .stApp {{ background-attachment: scroll; }}
    }}
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    [data-testid="stStatusWidget"] {{visibility: hidden;}}
    [data-testid="stAppDeployButton"] {{display: none;}}
    [data-testid="stToolbar"] {{display: none;}}
    header[data-testid="stHeader"] {{height: 0; min-height: 0;}}
</style>
""", unsafe_allow_html=True)


def _detect_mobile():
    try:
        ua = st.context.headers.get('User-Agent', '')
        return bool(re.search(r'iPhone|Android.*Mobile|Windows Phone', ua, re.I))
    except Exception:
        return False


def main():
    from pulse import render_pulse_tab
    from charts import render_charts_tab, render_scanner_charts_tab
    from spreads import render_spreads_tab
    from portfolio import render_portfolio_tab
    from news import render_news_tab
    from research import render as render_research_tab
    from options import render_options_tab
    from rates import render_rates_tab
    from markets import render_markets_tab
    from private import render_private_tab
    from predictions import render_predictions_tab

    # Init session state
    if 'sector' not in st.session_state: st.session_state.sector = 'Futures'
    if 'symbol' not in st.session_state: st.session_state.symbol = 'ES=F'
    if 'chart_type' not in st.session_state: st.session_state.chart_type = 'bars'

    st.session_state.theme = 'Dark'

    is_mobile = _detect_mobile()
    est = pytz.timezone('US/Eastern')

    _inject_theme_css()

    # SANPO logo header
    t = get_theme()
    pos_c = t['pos']
    neg_c = t['neg']
    title_c = '#f8fafc'

    st.markdown(f"""
        <style>
            @keyframes sanpo-sweep {{ from {{ transform: rotate(0deg); }} to {{ transform: rotate(360deg); }} }}
            @keyframes sanpo-glow {{ 0%,100% {{ filter: drop-shadow(0 0 3px {pos_c}40); }} 50% {{ filter: drop-shadow(0 0 8px {pos_c}90); }} }}
            @keyframes sanpo-core {{ 0%,100% {{ r: 2.5; }} 35%,50% {{ r: 3.2; }} }}
            @keyframes sanpo-halo {{ 0%,100% {{ opacity: 0.06; }} 35%,50% {{ opacity: 0.22; }} }}
            @keyframes sanpo-ripple {{ 0%,100% {{ opacity: 0.18; }} 35%,50% {{ opacity: 0.65; }} }}
            @keyframes sanpo-breathe {{ 0%,100% {{ opacity: 0.08; }} 40%,60% {{ opacity: 0.95; }} }}
        </style>
        <div style='display:flex;align-items:center;gap:14px;padding:6px 0 16px 0'>
            <svg width="56" height="56" viewBox="0 0 40 40" fill="none" style="animation:sanpo-glow 3s ease-in-out infinite">
                <!-- Breathing rings (ripple outward from center) -->
                <circle cx="20" cy="20" r="6"  stroke="#334155" stroke-width="0.5" style="animation:sanpo-ripple 4s ease-in-out infinite 0.3s"/>
                <circle cx="20" cy="20" r="12" stroke="#334155" stroke-width="0.5" style="animation:sanpo-ripple 4s ease-in-out infinite 1.0s"/>
                <circle cx="20" cy="20" r="18" stroke="#334155" stroke-width="0.5" style="animation:sanpo-ripple 4s ease-in-out infinite 1.7s"/>
                <!-- Heartbeat core -->
                <circle cx="20" cy="20" r="4.5" fill="{pos_c}" style="animation:sanpo-halo 4s ease-in-out infinite 0.0s"/>
                <circle cx="20" cy="20" fill="{pos_c}" style="animation:sanpo-core 4s ease-in-out infinite 0.0s"/>
                <!-- Sweep line -->
                <line x1="20" y1="20" x2="20" y2="2" stroke="url(#sanpoSweepG)" stroke-width="1.2" stroke-linecap="round" style="animation:sanpo-sweep 4s linear infinite;transform-origin:20px 20px"/>
                <!-- 5 dots: 2 mid, 3 outer — each with own cycle -->
                <circle cx="29.2" cy="27.7" r="1.0" fill="{pos_c}" style="animation:sanpo-breathe 4.1s ease-in-out infinite 1.1s"/>
                <circle cx="10.8" cy="12.3" r="1.1" fill="{neg_c}" style="animation:sanpo-breathe 3.6s ease-in-out infinite 0.8s"/>
                <circle cx="30.3" cy="5.3"  r="1.7" fill="{pos_c}" style="animation:sanpo-breathe 4.4s ease-in-out infinite 1.6s"/>
                <circle cx="23.1" cy="37.7" r="1.5" fill="{neg_c}" style="animation:sanpo-breathe 3.8s ease-in-out infinite 2.0s"/>
                <circle cx="2.1"  cy="18.4" r="1.4" fill="{pos_c}" style="animation:sanpo-breathe 4.2s ease-in-out infinite 1.3s"/>
                <defs><linearGradient id="sanpoSweepG" x1="20" y1="20" x2="20" y2="3">
                    <stop offset="0%" stop-color="{pos_c}" stop-opacity="0.6"/>
                    <stop offset="100%" stop-color="{pos_c}" stop-opacity="0"/>
                </linearGradient></defs>
            </svg>
            <span style='font-family:Orbitron,sans-serif;font-size:24px;font-weight:700;letter-spacing:0.08em;color:{title_c};line-height:1'>SANPO</span>
        </div>
    """, unsafe_allow_html=True)

    # Tabs
    tab_pulse, tab_news, tab_markets, tab_pred, tab_private, tab_portfolio, tab_spreads, tab_charts, tab_options, tab_rates, tab_research = st.tabs(["PULSE", "NEWS", "MARKETS", "PREDICT", "PRIVATE", "PORTFOLIO", "SPREADS", "CHARTS", "OPTIONS", "RATES", "RESEARCH"])

    with tab_pulse:
        render_pulse_tab(is_mobile)

    with tab_news:
        render_news_tab(is_mobile)

    with tab_markets:
        render_markets_tab(is_mobile)

    with tab_pred:
        render_predictions_tab(is_mobile)

    with tab_private:
        render_private_tab(is_mobile)

    with tab_portfolio:
        render_portfolio_tab(is_mobile)

    with tab_spreads:
        render_spreads_tab(is_mobile)

    with tab_charts:
        sub_asset, sub_scanner = st.tabs(["BY ASSET", "BY TIMEFRAME"])
        with sub_asset:
            render_charts_tab(is_mobile, est)
        with sub_scanner:
            render_scanner_charts_tab(is_mobile, est)

    with tab_options:
        render_options_tab(is_mobile)

    with tab_rates:
        render_rates_tab(is_mobile)

    with tab_research:
        render_research_tab()

    # Global auto-refresh aligned to :00 :30
    st_html("""<script>
    (function(){
        var now=new Date(), m=now.getMinutes(), s=now.getSeconds(), ms=now.getMilliseconds();
        var next30=30-m%30;
        var delay=(next30*60-s)*1000-ms;
        if(delay<5000) delay+=1800000;
        setTimeout(function(){window.parent.location.reload()}, delay);
    })();
    </script>""", height=0)


if __name__ == "__main__":
    main()
