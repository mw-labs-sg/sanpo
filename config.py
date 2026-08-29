import streamlit as st
from collections import OrderedDict

FONTS = 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'

# =============================================================================
# SYMBOL GROUPS
# =============================================================================

FUTURES_GROUPS = OrderedDict([
    ('Futures',   ['ES=F', 'NQ=F', 'GC=F', 'SI=F', 'CL=F', 'NG=F', 'ZC=F', 'ZS=F', 'BTC=F', 'ETH=F', 'SB=F', 'KC=F']),
    ('Indices',   ['ES=F', 'NQ=F', 'YM=F', 'RTY=F', 'NKD=F']),
    ('Rates',     ['ZB=F', 'ZN=F', 'ZF=F', 'ZT=F']),
    ('FX', [
        # Asia
        'JPY=X', 'AUD=X', 'NZD=X', 'SGD=X', 'HKD=X',
        'CNY=X', 'MYR=X', 'INR=X', 'KRW=X',
        # Europe
        'EUR=X', 'GBP=X', 'CHF=X', 'SEK=X', 'NOK=X', 'PLN=X', 'TRY=X',
        # Africa/ME
        'ZAR=X',
        # Americas
        'CAD=X', 'MXN=X', 'BRL=X',
    ]),
    ('Crypto',    ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD']),
    ('Energy',    ['CL=F', 'NG=F', 'RB=F', 'HO=F']),
    ('Metals',    ['GC=F', 'SI=F', 'PL=F', 'HG=F']),
    ('Grains',    ['ZC=F', 'ZS=F', 'ZW=F', 'ZM=F']),
    ('Softs',     ['SB=F', 'KC=F', 'CC=F', 'CT=F']),
    ('Singapore', ['ES3.SI', 'S68.SI', 'MBH.SI', 'MMS.SI']),
    ('STI', [
        # Banks
        'D05.SI', 'O39.SI', 'U11.SI',
        # Telco / Conglomerates / Industrials
        'Z74.SI', 'S63.SI', 'S68.SI', 'F34.SI', 'J36.SI', 'C6L.SI', 'BN4.SI',
        'BS6.SI', 'H78.SI', 'U96.SI', 'Y92.SI', 'U14.SI', 'G13.SI',
        '5E2.SI', 'C09.SI', 'D01.SI', 'S58.SI', 'V03.SI',
        # REITs
        'C38U.SI', 'A17U.SI', 'N2IU.SI', 'M44U.SI', 'AJBU.SI',
        'ME8U.SI', 'J69U.SI', 'BUOU.SI',
        # Other (CapitaLand Investment)
        '9CI.SI',
    ]),
    ('US Sectors',['XLB', 'XLC', 'XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLK', 'XLU', 'XLRE', 'SPY']),
    ('Countries', [
        # Asia Pacific
        'EWJ', 'EWY', 'EWT', 'EWH', 'EWS', 'EWM', 'THD', 'EIDO', 'VNM', 'EWA',
        # Europe
        'EWU', 'EWG', 'EWQ', 'EWI', 'EWP', 'EWL', 'EWD', 'EWN', 'EPOL', 'TUR',
        # Americas
        'EWC', 'EWZ', 'EWW', 'ARGT', 'ECH',
        # Middle East / Africa
        'KSA', 'EIS', 'EZA',
        # EM Broad
        'GXC', 'PIN',
    ]),
    ('Macro',     ['DBC', 'USO', 'GLD', 'SLV', 'CPER', 'BIL', 'HYG', 'LQD', 'TLT', 'BND', 'EMB', 'EEM', 'SPY', 'BTC-USD', 'ETH-USD']),
    ('Core 4',    ['SHV', 'VT', 'GLD', 'BTC-USD']),
    ('Core 5',    ['IAU', 'VOO', 'VTI', 'SHV', 'BTC-USD']),
    ('Exchanges', ['ICE', 'NDAQ', 'CME', 'CBOE', 'X.TO', 'LSEG.L', 'DB1.DE', 'ENX.PA', '8697.T', '0388.HK', 'ASX.AX', 'S68.SI']),
    ('Shipping',  ['BOAT', 'SEA', 'ZIM', 'MATX', 'DAC', 'CMRE', 'FRO', 'STNG', 'INSW', 'TK', 'GOGL', 'SBLK', 'GNK', 'GSL', 'AMKBY']),
    ('Strategy',  ['MSTR', 'STRK', 'STRF', 'STRC', 'STRD', 'MSTU', 'MSTX', 'MSTZ', 'MSTY']),
    ('Crypto Co', ['COIN', 'MSTR', 'MARA', 'RIOT', 'HOOD', 'CORZ', 'CLSK', 'HUT', 'GLXY.TO', 'CRCL']),
    ('Volatility',    ['^VIX', '^VVIX', '^GVZ', '^OVX', '^VXN', '^SKEW', '^MOVE']),
    ('World Indices', ['^STI', '^HSI', '000001.SS', '^N225', '^AXJO',
                       '^BSESN', '^KS11', '^KLSE', '^TWII',
                       '^FTSE', '^GDAXI', '^FCHI', '^STOXX50E',
                       '^GSPC', '^IXIC', '^DJI', '^RUT',
                       '^GSPTSE', '^BVSP', '^MXX']),
])

SYMBOL_NAMES = {
    # Futures — Indices
    'ES=F': 'E-mini S&P 500', 'NQ=F': 'E-mini Nasdaq 100', 'YM=F': 'E-mini Dow',
    'RTY=F': 'E-mini Russell 2000', 'NKD=F': 'Nikkei 225',
    # Futures — Rates
    'ZB=F': '30Y T-Bond', 'ZN=F': '10Y T-Note', 'ZF=F': '5Y T-Note', 'ZT=F': '2Y T-Note',
    # Futures — Metals
    'GC=F': 'Gold', 'SI=F': 'Silver', 'PL=F': 'Platinum', 'HG=F': 'Copper',
    # Futures — Energy
    'CL=F': 'Crude Oil WTI', 'NG=F': 'Natural Gas', 'RB=F': 'RBOB Gasoline', 'HO=F': 'Heating Oil',
    # Futures — Grains
    'ZS=F': 'Soybeans', 'ZC=F': 'Corn', 'ZW=F': 'Wheat', 'ZM=F': 'Soybean Meal',
    # Futures — Softs
    'SB=F': 'Sugar', 'KC=F': 'Coffee', 'CC=F': 'Cocoa', 'CT=F': 'Cotton',
    # Crypto spot
    'BTC-USD': 'Bitcoin', 'ETH-USD': 'Ethereum', 'SOL-USD': 'Solana', 'XRP-USD': 'XRP',
    # Crypto futures
    'BTC=F': 'Bitcoin Futures', 'ETH=F': 'Ethereum Futures',
    # FX futures
    '6E=F': 'Euro FX', '6J=F': 'Japanese Yen', '6B=F': 'British Pound', '6A=F': 'Australian Dollar',
    # FX spot — USD base
    'USDSGD=X': 'USD/SGD', 'USDJPY=X': 'USD/JPY', 'USDHKD=X': 'USD/HKD',
    'USDCNY=X': 'USD/CNY', 'USDINR=X': 'USD/INR', 'USDMYR=X': 'USD/MYR', 'USDKRW=X': 'USD/KRW',
    # FX spot — =X format (USD quoted)
    'EUR=X': 'USD/EUR', 'GBP=X': 'USD/GBP', 'JPY=X': 'USD/JPY', 'AUD=X': 'USD/AUD',
    'NZD=X': 'USD/NZD', 'CHF=X': 'USD/CHF', 'CAD=X': 'USD/CAD', 'SGD=X': 'USD/SGD',
    'HKD=X': 'USD/HKD', 'CNY=X': 'USD/CNY', 'MYR=X': 'USD/MYR', 'INR=X': 'USD/INR',
    'KRW=X': 'USD/KRW', 'SEK=X': 'USD/SEK', 'NOK=X': 'USD/NOK', 'PLN=X': 'USD/PLN',
    'TRY=X': 'USD/TRY', 'ZAR=X': 'USD/ZAR', 'MXN=X': 'USD/MXN', 'BRL=X': 'USD/BRL',
    # FX cross rates
    'EURUSD=X': 'EUR/USD', 'GBPUSD=X': 'GBP/USD', 'AUDUSD=X': 'AUD/USD', 'NZDUSD=X': 'NZD/USD',
    # Volatility
    '^VIX': 'VIX', '^VVIX': 'VIX of VIX', '^GVZ': 'Gold Volatility',
    '^OVX': 'Crude Oil Volatility', '^VXN': 'Nasdaq Volatility',
    '^SKEW': 'CBOE Skew Index', '^MOVE': 'Bond Volatility (MOVE)',
    # Singapore
    'ES3.SI': 'STI ETF', 'S68.SI': 'SGX', 'MBH.SI': 'Amova IG Bond', 'MMS.SI': 'SGD Money Mkt',
    # STI components
    'D05.SI': 'DBS', 'O39.SI': 'OCBC', 'U11.SI': 'UOB',
    'Z74.SI': 'Singtel', 'S63.SI': 'ST Engineering', 'F34.SI': 'Wilmar',
    'J36.SI': 'Jardine Matheson', 'C6L.SI': 'Singapore Airlines', 'BN4.SI': 'Keppel',
    'C38U.SI': 'CICT', 'BS6.SI': 'Yangzijiang', 'H78.SI': 'Hongkong Land',
    '9CI.SI': 'CapitaLand Investment', 'A17U.SI': 'Ascendas REIT', 'U96.SI': 'Sembcorp Industries',
    'Y92.SI': 'ThaiBev', 'U14.SI': 'UOL Group', 'G13.SI': 'Genting Singapore',
    '5E2.SI': 'Seatrium', 'C09.SI': 'City Developments', 'N2IU.SI': 'Mapletree PACT',
    'M44U.SI': 'Mapletree Logistics', 'AJBU.SI': 'Keppel DC REIT', 'D01.SI': 'DFI Retail',
    'ME8U.SI': 'Mapletree Industrial', 'S58.SI': 'SATS', 'J69U.SI': 'Frasers Centrepoint',
    'V03.SI': 'Venture Corp', 'BUOU.SI': 'Frasers Log & Comm',
    # US Sectors
    'XLB': 'Materials', 'XLC': 'Comms', 'XLY': 'Cons Disc', 'XLP': 'Cons Staples',
    'XLE': 'Energy', 'XLF': 'Financials', 'XLV': 'Healthcare', 'XLI': 'Industrials',
    'XLK': 'Technology', 'XLU': 'Utilities', 'XLRE': 'Real Estate', 'SPY': 'S&P 500',
    # Country ETFs
    'EWA': 'Australia', 'EWZ': 'Brazil', 'EWC': 'Canada', 'GXC': 'China',
    'EWQ': 'France', 'EWG': 'Germany', 'EWH': 'Hong Kong', 'PIN': 'India',
    'EWI': 'Italy', 'EWJ': 'Japan', 'EWM': 'Malaysia', 'EWW': 'Mexico',
    'EWS': 'Singapore', 'EWY': 'South Korea', 'EWP': 'Spain', 'EWT': 'Taiwan',
    'EWU': 'UK', 'VNM': 'Vietnam', 'KSA': 'Saudi Arabia', 'ARGT': 'Argentina',
    'ECH': 'Chile', 'EIS': 'Israel', 'EZA': 'South Africa', 'EPOL': 'Poland', 'TUR': 'Turkey',
    # Macro ETFs
    'DBC': 'Commodities', 'USO': 'Oil ETF', 'GLD': 'Gold ETF', 'SLV': 'Silver ETF',
    'CPER': 'Copper ETF', 'BIL': 'T-Bills', 'HYG': 'High Yield', 'LQD': 'IG Corp',
    'TLT': '20Y+ Treasury', 'BND': 'Total Bond', 'EMB': 'EM Bonds', 'EEM': 'EM Equity',
    'IAU': 'iShares Gold', 'VOO': 'Vanguard S&P 500', 'VTI': 'Vanguard Total US Mkt', 'VT': 'Vanguard Total World', 'SHV': 'Short Treasury',
    # Exchanges
    'ICE': 'ICE', 'NDAQ': 'Nasdaq Inc', 'CME': 'CME Group', 'CBOE': 'Cboe Global',
    'X.TO': 'TMX Group', 'LSEG.L': 'LSEG', 'DB1.DE': 'Deutsche Börse',
    'ENX.PA': 'Euronext', '8697.T': 'JPX', '0388.HK': 'HKEX', 'ASX.AX': 'ASX Ltd',
    # Shipping
    'BOAT': 'Global Shipping ETF', 'SEA': 'Sea to Sky Cargo', 'ZIM': 'ZIM Shipping',
    'MATX': 'Matson', 'DAC': 'Danaos', 'CMRE': 'Costamare', 'FRO': 'Frontline',
    'STNG': 'Scorpio Tankers', 'INSW': 'Intl Seaways', 'TK': 'Teekay',
    'GOGL': 'Golden Ocean', 'SBLK': 'Star Bulk', 'GNK': 'Genco Shipping',
    'GSL': 'Global Ship Lease', 'AMKBY': 'Maersk',
    # Strategy / MSTR
    'MSTR': 'Strategy', 'STRK': 'Strike Pref', 'STRF': 'Strife Pref',
    'STRC': 'Stretch Pref', 'STRD': 'Stride Pref',
    'MSTU': '2x Long MSTR', 'MSTX': '2x Long MSTR', 'MSTZ': '2x Inv MSTR',
    'MSTY': 'MSTR Options Income',
    # World indices
    '^STI': 'Singapore STI', '^HSI': 'Hang Seng', '000001.SS': 'Shanghai Composite',
    '^N225': 'Nikkei 225', '^AXJO': 'ASX 200', '^BSESN': 'India Sensex',
    '^KS11': 'KOSPI', '^KLSE': 'Malaysia KLCI', '^TWII': 'Taiwan TWSE',
    '^FTSE': 'FTSE 100', '^GDAXI': 'Germany DAX', '^FCHI': 'France CAC 40',
    '^STOXX50E': 'Euro Stoxx 50', '^GSPC': 'S&P 500 Index', '^IXIC': 'Nasdaq Composite',
    '^DJI': 'Dow Jones', '^RUT': 'Russell 2000', '^GSPTSE': 'Canada TSX',
    '^BVSP': 'Brazil IBOVESPA', '^MXX': 'Mexico IPC',
    # Crypto companies
    'COIN': 'Coinbase', 'MARA': 'Marathon Digital', 'RIOT': 'Riot Platforms',
    'HOOD': 'Robinhood', 'CORZ': 'Core Scientific', 'CLSK': 'CleanSpark',
    'HUT': 'Hut 8', 'GLXY.TO': 'Galaxy Digital', 'CRCL': 'Circle',
}

# =============================================================================
# THEMES
# =============================================================================

THEMES = {
    'Dark': {
        'mode': 'dark',
        'pos': '#4ade80', 'neg': '#f59e0b',
        'zone_hi': '#4ade80', 'zone_amid': '#86efac', 'zone_bmid': '#fbbf24', 'zone_lo': '#f59e0b',
        'str_up': '#4ade80', 'str_dn': '#f59e0b', 'pull': '#fbbf24', 'bnce': '#93c5fd',
        'long': '#4ade80', 'short': '#f59e0b',
        # Cool faded blue. `bg` is the flat fallback painted under the
        # gradient (and mirrored in .streamlit/config.toml).
        'bg': '#101a31',
        # Panels and rows are SEMI-TRANSPARENT on purpose: opaque cards would
        # cover the gradient and it would only show in the margins. These
        # composite over the backdrop so the fade reads through the whole page.
        'bg2': 'rgba(9,15,29,0.58)',      # recessed panels
        'bg3': 'rgba(28,45,76,0.52)',     # raised rows, cards
        'widget_bg': '#1b2c49',           # selects/buttons — kept opaque
        # `border` must stay a 6-digit hex: several tabs append an alpha pair
        # to it (e.g. f"1px solid {bdr}22").
        'border': '#28395a',
        'text': '#e2e8f0', 'text2': '#9fb2ca', 'muted': '#6b7f9c',
        'accent': '#4ade80',
        'bg_gradient': (
            'radial-gradient(1200px 720px at 18% -8%, rgba(59,130,246,0.20), transparent 62%),'
            'radial-gradient(900px 620px at 92% 12%, rgba(56,189,248,0.12), transparent 58%),'
            'linear-gradient(180deg, #0a0f1e 0%, #111c33 45%, #17263f 78%, #0e1626 100%)'
        ),
        # Transparent so Plotly panels sit on the gradient rather than punching
        # opaque rectangles through it.
        'plot_bg': 'rgba(0,0,0,0)', 'grid': '#22304a', 'axis_line': '#2c3d5c', 'tick': '#9fb2ca',
    },
}

# =============================================================================
# SHARED HELPERS
# =============================================================================

def clean_symbol(sym):
    return (sym.replace('=F', '').replace('=X', '').replace('.SI', '')
               .replace('^', '').replace('-USD', '').replace('-GBP', '').replace('-EUR', ''))

def sym_name(sym):
    """Friendly name: SYMBOL_NAMES lookup with clean_symbol fallback."""
    return SYMBOL_NAMES.get(sym, clean_symbol(sym))

def get_theme():
    """Single source of truth for current theme — used by all tabs."""
    name = st.session_state.get('theme', 'Dark')
    return THEMES.get(name, THEMES['Dark'])

def st_html(page, height=0):
    """Embed a raw HTML string in an iframe.

    Prefers st.iframe (st.components.v1.html is deprecated for removal); falls
    back to the legacy component on older Streamlit builds.
    """
    if hasattr(st, 'iframe'):
        # st.iframe rejects height=0, which the legacy component allowed for
        # script-only embeds.
        return st.iframe(page, height=max(1, int(height)))
    from streamlit.components.v1 import html as _legacy_html
    return _legacy_html(page, height=height)

def sort_val(v, default=float('-inf')):
    """Sort key for optional numerics.

    `v or default` is wrong here: it sends an exactly-flat 0.0 to the bottom of
    the table and leaves NaN as the key, which makes the ordering arbitrary.
    """
    if v is None:
        return default
    try:
        if v != v:  # NaN
            return default
    except TypeError:
        return default
    return v

def surface():
    """Derived surface palette for HTML rendering — used by all tabs."""
    t = get_theme()
    is_light = t.get('mode') == 'light'
    bg  = t.get('bg', '#1e1e1e');  bg2 = t.get('bg2', '#0a0f1a')
    bg3 = t.get('bg3', '#0f172a'); bdr = t.get('border', '#1e293b')
    txt = t.get('text', '#e2e8f0'); txt2 = t.get('text2', '#94a3b8')
    muted = t.get('muted', '#475569')
    if is_light:
        return dict(bg=bg, bg2=bg2, bg3=bg3, card=bg2,
            border=bdr, text=txt, text2=txt2, muted=muted,
            off_dot='#d1d5db', off_name='#9ca3af', link='#334155',
            bar_bg=bdr, row_alt=bg3, hm_txt=txt)
    return dict(bg=bg, bg2=bg2, bg3=bg3, card=bg3,
        border=bdr, text=txt, text2=txt2, muted=muted,
        off_dot='#3a4a63', off_name='#7d8fa8', link=txt,
        bar_bg=bg3, row_alt='rgba(20,32,55,0.45)', hm_txt=txt)
