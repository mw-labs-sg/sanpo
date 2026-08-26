"""
SANPO — Private Companies tab
Full list from Yahoo Finance (~100 companies)
"""

import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
import pytz
import logging
import base64
import requests
import json

from config import THEMES, FONTS, st_html, sort_val

logger = logging.getLogger(__name__)


def get_theme():
    tn = st.session_state.get('theme', 'Dark')
    return THEMES.get(tn, THEMES['Dark'])


# (ticker, company, valuation_b, total_funding_b, latest_date, latest_amt, round_class, sector)
PRIVATE_COMPANIES = [
    ('SPAX.PVT', 'SpaceX',              1427.000,   8.923,  '2026-02-02', '250B',      'Corporate Round',    'Space'),
    ('OPAI.PVT', 'OpenAI',               840.000, 166.070,  '2026-02-26', '110B',      'Series C',           'AI'),
    ('ANTH.PVT', 'Anthropic',            380.005,  53.150,  '2026-02-11', '15.045B',   'Series G-1',         'AI'),
    ('STRI.PVT', 'Stripe',               184.401,   8.733,  '2026-02-23', '--',        'Tender Offer 3',     'Fintech'),
    ('DATB.PVT', 'Databricks',           138.436,  20.551,  '2026-02-08', '5B',        'Series L',           'Data & Analytics'),
    ('ANIN.PVT', 'Anduril',              84.495,    6.955,  '2026-01-13', '500M',      'Series G-1',         'Defense'),
    ('RAMP.PVT', 'Ramp',                 32.722,    2.191,  '2025-11-16', '300M',      'Series E-3',         'Fintech'),
    ('CESY.PVT', 'Cerebras',             25.929,    2.923,  '2026-02-03', '1.014B',    'Series H',           'Hardware'),
    ('PEAI.PVT', 'Perplexity',           20.391,    2.254,  '2025-09-25', '525M',      'Series E-6',         'AI'),
    ('RIPP.PVT', 'Rippling',             19.386,    1.414,  '2025-05-08', '460.183M',  'Series G',           'HR/Admin'),
    ('EPGA.PVT', 'Epic Games',           17.991,    7.530,  '2024-02-07', '1.5B',      'Corporate Round',    'Gaming'),
    ('RIPL.PVT', 'Ripple',               17.710,    0.575,  '2016-09-14', '44.477M',   'Series B',           'Crypto'),
    ('FANA.PVT', 'Fanatics',             17.093,    4.404,  '2022-12-05', '700M',      'Private Equity 3',   'Commerce'),
    ('NEUR.PVT', 'Neuralink',            14.865,    1.337,  '2025-06-01', '650M',      'Series E',           'Biotech'),
    ('CUES.PVT', 'Crusoe Energy',        14.647,    2.620,  '2025-10-23', '1.375B',    'Series E',           'Energy'),
    ('SKIA.PVT', 'Skild AI',             13.999,    2.217,  '2026-01-13', '694.067M',  'Series C',           'Hardware'),
    ('POLA.PVT', 'Polymarket',           12.295,    1.270,  '2025-10-06', '1B',        'Series D',           'Crypto'),
    ('KLSH.PVT', 'Kalshi',               11.115,    1.531,  '2025-11-19', '999.997M',  'Series E',           'Fintech'),
    ('LAMD.PVT', 'Lambda',               11.053,    2.364,  '2025-11-17', '410.284M',  'Series E-3',         'Data & Analytics'),
    ('KRAK.PVT', 'Kraken',               10.581,    0.965,  '2025-11-17', '200M',      'Series D',           'Crypto'),
    ('SHAI.PVT', 'Shield AI',            10.432,    1.032,  '2025-03-05', '240M',      'Series F1',          'Defense'),
    ('REPI.PVT', 'Replit',                9.720,    0.854,  '2026-03-10', '400M',      'Series D',           'Software'),
    ('WHOO.PVT', 'WHOOP',                 9.071,    0.899,  '2026-02-25', '440M',      'Series G-1',         'Consumer'),
    ('DISO.PVT', 'Discord',               8.543,    1.005,  '2021-08-14', '499.999M',  'Series I',           'Messaging'),
    ('NETS.PVT', 'Netskope',              8.249,    1.050,  '2021-07-08', '300M',      'Series H',           'Security'),
    ('VERC.PVT', 'Vercel',                8.104,    0.863,  '2025-09-29', '299.999M',  'Series F',           'Software'),
    ('GLEA.PVT', 'Glean',                 7.845,    0.775,  '2025-06-09', '150M',      'Series F',           'Data & Analytics'),
    ('ZIPL.PVT', 'Zipline',               6.879,    1.753,  '2026-01-19', '600M',      'Series H',           'Logistics'),
    ('GROQ.PVT', 'Groq',                  6.387,    1.798,  '2025-09-16', '750M',      'Series D-3',         'Hardware'),
    ('PSIQ.PVT', 'PsiQuantum',            5.748,    1.668,  '2025-09-09', '969.825M',  'Series E',           'Hardware'),
    ('ABRI.PVT', 'Abridge',               5.686,    0.758,  '2025-06-23', '300M',      'Series E',           'Software'),
    ('SAAQ.PVT', 'SandboxAQ',             5.121,    0.916,  '2025-04-03', '450M',      'Series E',           'Security'),
    ('REMA.PVT', 'Redwood Materials',     5.061,    2.175,  '2025-10-22', '350M',      'Series E',           'Energy'),
    ('UPGR.PVT', 'Upgrade',               5.048,    0.796,  '2025-11-30', '49.139M',   'Tender Offer 1',     'Fintech'),
    ('VERK.PVT', 'Verkada',               4.866,    0.656,  '2025-12-02', '80M',       'Late Stage Venture', 'Security'),
    ('COHS.PVT', 'Cohesity',              4.690,    1.959,  '2024-12-09', '27.5M',     'Series H-1',         'Data & Analytics'),
    ('CRIB.PVT', 'Cribl',                 4.405,    0.725,  '2024-08-26', '199.036M',  'Series E',           'Data & Analytics'),
    ('BREX.PVT', 'Brex',                  4.306,    1.246,  '2022-01-10', '300M',      'Series D-2',         'Fintech'),
    ('HARN.PVT', 'Harness',               4.293,    0.716,  '2025-12-10', '200M',      'Tender Offer 1',     'Software'),
    ('TANI.PVT', 'Tanium',                4.265,    0.471,  '2020-06-24', '150M',      'Series H',           'Security'),
    ('THFD.PVT', "Farmer's Dog",          4.214,    0.268,  '2022-06-07', '100M',      'Series E',           'Consumer'),
    ('ARWO.PVT', 'Arctic Wolf',           4.210,    0.501,  '2021-07-12', '75M',       'Series F',           'Security'),
    ('NURO.PVT', 'Nuro',                  3.954,    2.206,  '2025-08-20', '203.188M',  'Series E',           'Transport'),
    ('MERC.PVT', 'Mercury',               3.924,    0.452,  '2025-03-25', '242.549M',  'Series C',           'Fintech'),
    ('AIR.PVT',  'Airtable',              3.764,    4.114,  '2021-12-12', '735M',      'Series F',           'Software'),
    ('STOS.PVT', 'Stoke Space',           3.419,    1.194,  '2026-02-09', '350M',      'Series D-2',         'Space'),
    ('ADDE.PVT', 'Addepar',               3.196,    0.721,  '2025-05-12', '230.711M',  'Series G',           'Fintech'),
    ('POSM.PVT', 'Postman',               3.158,    0.352,  '2021-08-17', '145M',      'Series D',           'Software'),
    ('LIG.PVT',  'Lightmatter',           3.028,    0.822,  '2024-10-15', '13.483M',   'Series C-3',         'Hardware'),
    ('TATE.PVT', 'TAE Technologies',      2.472,    1.187,  '2025-06-01', '150M',      'Series 12',          'Energy'),
    ('AGRO.PVT', 'Agility Robotics',      2.206,    0.641,  '2025-06-24', '400M',      'Series C-3',         'Hardware'),
    ('STRV.PVT', 'Strava',                2.200,    0.228,  '2025-05-21', '50M',       'Series F-1',         'Apps'),
    ('WORA.PVT', 'Workato',               2.115,    0.441,  '2021-11-09', '200M',      'Series E',           'Software'),
    ('MOTV.PVT', 'Motive',                2.073,    0.435,  '2025-07-29', '169.945M',  'Series F Senior',    'Logistics'),
    ('AUAN.PVT', 'Automation Anywhere',   2.026,    0.815,  '2019-11-20', '174M',      'Series B',           'Data & Analytics'),
    ('INTC.PVT', 'Intercom',              1.980,    0.241,  '2018-03-16', '125M',      'Series D',           'Sales & Mktg'),
    ('DITD.PVT', 'Divergent 3D',          1.933,    0.885,  '2025-09-14', '153.103M',  'Series E',           'Manufacturing'),
    ('RAPP.PVT', 'Rappi',                 1.774,    5.871,  '2022-09-14', '608.6M',    'Series F',           'Transport'),
    ('SINA.PVT', 'Sila Nanotechnologies', 1.708,    3.239,  '2024-06-26', '375M',      'Series G',           'Energy'),
    ('CHAA.PVT', 'Chainalysis',           1.547,    0.587,  '2022-05-11', '150M',      'Series F',           'Security'),
    ('CONS.PVT', 'ConsenSys',             1.383,    0.705,  '2022-03-14', '450M',      'Series D',           'Crypto'),
    ('SANS.PVT', 'SambaNova',             1.376,    1.431,  '2026-02-03', '300M',      'Series E-1',         'Hardware'),
    ('TURO.PVT', 'Turo',                  1.259,    0.558,  '2024-09-29', '457K',      'Series 1',           'Transport'),
    ('NEFJ.PVT', 'Neo4j',                 1.239,    0.712,  '2021-11-08', '197.031M',  'Series F',           'Data & Analytics'),
    ('CIHE.PVT', 'Cityblock Health',      1.228,    1.827,  '2024-06-17', '39M',       'Series X',           'Health Care'),
    ('THMA.PVT', 'Thrive Market',         1.157,    0.603,  '2021-07-07', '86.377M',   'Series C',           'Food & Bev'),
    ('GERO.PVT', 'Gecko Robotics',        1.142,    0.349,  '2025-06-11', '116.065M',  'Series D-1',         'Hardware'),
    ('INNA.PVT', 'Innovaccer',            1.117,    0.654,  '2025-01-08', '153.044M',  'Series F-1',         'Health Care'),
    ('EPIR.PVT', 'Epirus',                1.025,    0.541,  '2025-03-03', '250M',      'Series D',           'Defense'),
    ('FLEP.PVT', 'Flexport',              1.012,    2.134,  '2022-02-06', '935M',      'Series E',           'Logistics'),
    ('LIDE.PVT', 'Liquid Death',          0.952,    2.484,  '2024-03-10', '67.61M',    'Series F-1',         'Food & Bev'),
    ('ATTE.PVT', 'Attentive',             0.913,    0.866,  '2021-03-23', '470.1M',    'Series E',           'Sales & Mktg'),
    ('FLOQ.PVT', 'FloQast',               0.889,    0.304,  '2024-04-09', '60.294M',   'Series E',           'Software'),
    ('PATR.PVT', 'Patreon',               0.865,    0.408,  '2021-04-05', '155M',      'Series F',           'Media'),
    ('VECR.PVT', 'Vectra AI',             0.840,    0.352,  '2021-04-28', '130M',      'Series F',           'Security'),
    ('DTMR.PVT', 'Dataminr',              0.832,    1.035,  '2021-03-22', '475M',      'Series F',           'Software'),
    ('INSA.PVT', 'Instabase',             0.802,    0.276,  '2025-01-16', '100M',      'Series D',           'Data & Analytics'),
    ('EGNY.PVT', 'Egnyte',                0.773,    0.414,  '2018-10-09', '75M',       'Series E',           'Security'),
    ('DRUV.PVT', 'Druva',                 0.773,    0.147,  '2021-04-18', '147M',      'Series H',           'Security'),
    ('DRAG.PVT', 'Dragos',                0.763,    0.382,  '2021-10-27', '210M',      'Series D',           'Security'),
    ('CRES.PVT', 'Cresta',                0.747,    0.279,  '2024-11-18', '125M',      'Series D',           'Software'),
    ('WORR.PVT', 'Workrise',              0.736,    0.787,  '2021-05-19', '300M',      'Series E',           'HR/Admin'),
    ('APOL.PVT', 'Apollo.io',             0.723,    0.657,  '2023-08-28', '100.65M',   'Series D',           'Sales & Mktg'),
    ('EISL.PVT', 'Eight Sleep',           0.654,    0.270,  '2026-03-03', '69.122M',   'Series D',           'Consumer'),
    ('LOOR.PVT', 'Loft Orbital',          0.609,    0.316,  '2025-01-13', '170M',      'Series C',           'Space'),
    ('GREE.PVT', 'Greenlight',            0.601,    1.110,  '2021-04-26', '260M',      'Series D',           'Fintech'),
    ('ENRX.PVT', 'EnergyX',              0.572,    0.106,  '2023-06-29', '37.433M',   'Series B',           'Energy'),
    ('BLOO.PVT', 'Bloomreach',            0.567,    0.444,  '2022-02-22', '175M',      'Series F',           'Sales & Mktg'),
    ('SIXS.PVT', '6sense',                0.556,    0.526,  '2022-01-19', '180M',      'Series E-1',         'Software'),
    ('THSP.PVT', 'ThoughtSpot',           0.537,    0.803,  '2023-07-17', '124.638M',  'Series F-1',         'Software'),
    ('BIGI.PVT', 'BigID',                 0.532,    1.216,  '2024-03-17', '61.44M',    'Series E',           'Security'),
    ('KORE.PVT', 'Kore.ai',               0.531,    0.248,  '2024-01-29', '150M',      'Series D',           'Software'),
    ('ACOR.PVT', 'Acorns',                0.467,    0.839,  '2023-03-30', '111.358M',  'Series G-E',         'Fintech'),
    ('IMPF.PVT', 'Impossible Foods',      0.453,    1.961,  '2021-11-22', '500.145M',  'Series H-1',         'Food & Bev'),
    ('ZOCD.PVT', 'Zocdoc',                0.446,    1.149,  '2021-02-10', '100M',      'Series D-2',         'Health Care'),
    ('AVIA.PVT', 'Aviatrix',              0.411,    0.383,  '2021-09-07', '200M',      'Series E',           'Data & Analytics'),
    ('STDA.PVT', 'Starburst Data',        0.408,    0.432,  '2022-02-08', '250M',      'Series D',           'Data & Analytics'),
    ('FIRB.PVT', 'Firebolt',              0.371,    0.609,  '2022-01-25', '100M',      'Series C',           'Data & Analytics'),
    ('COLH.PVT', 'Collective Health',     0.367,    0.713,  '2021-05-03', '280M',      'Series F',           'Health Care'),
    ('AVTN.PVT', 'Avathon',               0.335,    0.653,  '2022-01-24', '123M',      'Series D',           'AI'),
    ('APSC.PVT', 'Apeel Sciences',        0.325,    0.647,  '2021-08-17', '250M',      'Series E',           'Agriculture'),
    ('SIST.PVT', 'SingleStore',           0.310,    0.389,  '2022-07-11', '116M',      'Series F-2',         'Data & Analytics'),
    ('OUTR.PVT', 'Outreach',              0.303,    0.491,  '2021-05-26', '201.027M',  'Series G',           'Software'),
    ('LEOL.PVT', 'LeoLabs',               0.303,    0.423,  '2024-02-11', '94M',       'Series B',           'Space'),
    ('NEWS.PVT', 'Newsela',               0.297,    0.189,  '2021-02-24', '100M',      'Series D',           'Education'),
    ('WYLA.PVT', 'Wyze Labs',             0.232,    0.146,  '2021-07-29', '100M',      'Series B',           'Hardware'),
    ('DATO.PVT', 'DataRobot',             0.170,    3.361,  '2021-07-26', '330M',      'Series G',           'Data & Analytics'),
    ('HTWO.PVT', 'H2O.ai',               0.138,    0.744,  '2021-11-06', '91.031M',   'Series E',           'Software'),
    ('SISE.PVT', 'Sisense',               0.000,    0.000,  '2020-01-12', '--',        'Series F',           'Data & Analytics'),
]


def _export_to_github(companies, prices):
    """Write private.json to GitHub — only when the data actually changed.

    Streamlit re-runs every tab on every interaction, so calling the API
    directly here meant a GET + PUT (and a commit) per script run. The payload
    is built without a timestamp so it can key a cache, and the push only fires
    on a genuine content change (or hourly).
    """
    try:
        token = st.secrets.get("GITHUB_TOKEN", "")
        if not token:
            return

        data = {
            'source': 'SANPO Private Companies',
            'count': len(companies),
            'companies': []
        }

        for row in companies:
            ticker, company, val_b, raised_b, latest_date, latest_amt, round_class, sector = row
            d = prices.get(ticker, {})
            price = d.get('price')
            ytd = d.get('ytd')
            data['companies'].append({
                'symbol': ticker,
                'name': company,
                'price': round(price, 4) if price is not None else None,
                'price_52w_pct': round(ytd, 2) if ytd is not None else None,
                'valuation_b': val_b,
                'total_raised_b': raised_b,
                'latest_date': latest_date,
                'latest_amt': latest_amt,
                'round': round_class,
                'sector': sector,
            })

        _push_private_json(json.dumps(data, indent=2, ensure_ascii=False))

    except Exception as e:
        logger.warning(f"GitHub export error: {e}")


@st.cache_data(ttl=3600, show_spinner=False)
def _push_private_json(body_json):
    """Commit private.json. Cached on body_json so unchanged data is a no-op.

    The token is read here rather than passed in, to keep it out of the cache key.
    """
    try:
        token = st.secrets.get("GITHUB_TOKEN", "")
        if not token:
            return
        sgt = pytz.timezone('Asia/Singapore')
        data = json.loads(body_json)
        data['updated'] = datetime.now(sgt).isoformat()

        url = "https://api.github.com/repos/YureiKara/sanpo/contents/private.json"
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
        sha = None
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            sha = r.json().get("sha")

        content = base64.b64encode(
            json.dumps(data, indent=2, ensure_ascii=False).encode()
        ).decode()
        payload = {"message": "data: update private.json", "content": content}
        if sha:
            payload["sha"] = sha
        requests.put(url, headers=headers, json=payload, timeout=10)
    except Exception as e:
        logger.warning(f"GitHub push error: {e}")


@st.cache_data(ttl=600, max_entries=3, show_spinner=False)
def fetch_prices():
    results = {}
    tickers = [row[0] for row in PRIVATE_COMPANIES]

    try:
        raw = yf.download(
            tickers, period='1y', interval='1d',
            auto_adjust=True, progress=False, threads=True,
            group_by='ticker'
        )
        for ticker in tickers:
            try:
                if isinstance(raw.columns, pd.MultiIndex) and ticker in raw.columns.get_level_values(0):
                    df = raw[ticker]['Close'].dropna()
                elif not isinstance(raw.columns, pd.MultiIndex) and 'Close' in raw.columns:
                    df = raw['Close'].dropna()
                else:
                    df = pd.Series(dtype=float)

                if len(df) >= 2:
                    price = float(df.iloc[-1])
                    year_start = float(df.iloc[0])
                    ytd = (price - year_start) / year_start * 100 if year_start else None
                    results[ticker] = {'price': price, 'ytd': ytd}
                else:
                    results[ticker] = {'price': None, 'ytd': None}
            except Exception:
                results[ticker] = {'price': None, 'ytd': None}
    except Exception as e:
        logger.warning(f"Bulk download error: {e}")
        for ticker in tickers:
            results[ticker] = {'price': None, 'ytd': None}

    missing = [t for t in tickers if results.get(t, {}).get('price') is None]
    for ticker in missing:
        try:
            fi = yf.Ticker(ticker).fast_info
            price = fi.get('lastPrice')
            yc = fi.get('yearChange')
            results[ticker] = {
                'price': price,
                'ytd': (yc * 100) if yc is not None else None
            }
        except Exception:
            results[ticker] = {'price': None, 'ytd': None}

    return results


def _fmt_pct(val):
    if val is None: return '—', '#475569'
    sign = '+' if val >= 0 else ''
    color = '#4ade80' if val >= 0 else '#f59e0b'
    return f"{sign}{val:.2f}%", color


def _fmt_val(b):
    if not b: return '—'
    if b >= 1000: return f"${b/1000:.3f}T"
    if b >= 1: return f"${b:.3f}B"
    return f"${b*1000:.0f}M"


def _build_table(data, sort_by):
    t   = get_theme()
    bg2 = t.get('bg2', '#0a0f1a')
    bg3 = t.get('bg3', '#0f172a')
    bdr = t.get('border', '#1e293b')
    acc = t.get('accent', '#4ade80')

    rows = list(PRIVATE_COMPANIES)
    if sort_by == '52W %':
        rows.sort(key=lambda x: sort_val(data.get(x[0], {}).get('ytd')), reverse=True)
    elif sort_by == 'Valuation':
        rows.sort(key=lambda x: x[2], reverse=True)
    elif sort_by == 'Total Raised':
        rows.sort(key=lambda x: x[3], reverse=True)
    elif sort_by == 'Latest Date':
        rows.sort(key=lambda x: x[4], reverse=True)
    elif sort_by == 'Round':
        rows.sort(key=lambda x: x[6])
    elif sort_by == 'Sector':
        rows.sort(key=lambda x: x[7])

    HDR = "font-size:9px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#f8fafc"

    html = f"""<div style='font-family:{FONTS};min-width:900px'>
        <div style='display:flex;align-items:center;padding:5px 12px;
                    border-bottom:1px solid {bdr};gap:8px'>
            <div style='width:85px;flex-shrink:0;{HDR}'>SYMBOL</div>
            <div style='width:140px;flex-shrink:0;{HDR}'>COMPANY</div>
            <div style='width:65px;flex-shrink:0;{HDR};text-align:right'>PRICE</div>
            <div style='width:70px;flex-shrink:0;{HDR};text-align:right'>52W %</div>
            <div style='width:80px;flex-shrink:0;{HDR};text-align:right'>VALUATION</div>
            <div style='width:85px;flex-shrink:0;{HDR};text-align:right'>TOTAL RAISED</div>
            <div style='width:90px;flex-shrink:0;{HDR};text-align:center'>LATEST DATE</div>
            <div style='width:75px;flex-shrink:0;{HDR};text-align:right'>LATEST AMT</div>
            <div style='width:120px;flex-shrink:0;{HDR}'>ROUND</div>
            <div style='flex:1;{HDR}'>SECTOR</div>
        </div>"""

    for i, row in enumerate(rows):
        ticker, company, val_b, raised_b, latest_date, latest_amt, round_class, sector = row
        d = data.get(ticker, {})
        price = d.get('price')
        ytd   = d.get('ytd')

        price_str      = f"${price:,.2f}" if price is not None else '—'
        ytd_str, ytd_c = _fmt_pct(ytd)
        val_str        = _fmt_val(val_b)
        raised_str     = _fmt_val(raised_b)
        row_bg = bg2 if i % 2 == 0 else bg3

        html += f"""
        <div style='display:flex;align-items:center;background:{row_bg};
                    padding:5px 12px;border-bottom:1px solid {bdr}12;gap:8px'>
            <div style='width:85px;flex-shrink:0;font-size:10px;font-weight:600;color:{acc}'>{ticker}</div>
            <div style='width:140px;flex-shrink:0;font-size:11px;font-weight:600;color:#f8fafc;
                        overflow:hidden;text-overflow:ellipsis;white-space:nowrap'>{company}</div>
            <div style='width:65px;flex-shrink:0;font-size:10px;color:#ffffff;text-align:right;
                        font-variant-numeric:tabular-nums'>{price_str}</div>
            <div style='width:70px;flex-shrink:0;font-size:10px;font-weight:700;color:{ytd_c};
                        text-align:right;font-variant-numeric:tabular-nums'>{ytd_str}</div>
            <div style='width:80px;flex-shrink:0;font-size:10px;font-weight:600;color:{acc};
                        text-align:right'>{val_str}</div>
            <div style='width:85px;flex-shrink:0;font-size:10px;color:#94a3b8;text-align:right'>{raised_str}</div>
            <div style='width:90px;flex-shrink:0;font-size:10px;color:#94a3b8;text-align:center'>{latest_date}</div>
            <div style='width:75px;flex-shrink:0;font-size:10px;color:#94a3b8;text-align:right'>{latest_amt}</div>
            <div style='width:120px;flex-shrink:0;font-size:10px;color:#94a3b8;
                        overflow:hidden;text-overflow:ellipsis;white-space:nowrap'>{round_class}</div>
            <div style='flex:1;font-size:10px;color:#475569;
                        overflow:hidden;text-overflow:ellipsis;white-space:nowrap'>{sector}</div>
        </div>"""

    html += "</div>"
    return html


def _wrap(body, height):
    t   = get_theme()
    bg2 = t.get('bg2', '#0a0f1a')
    bdr = t.get('border', '#1e293b')
    txt = t.get('text', '#e2e8f0')
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<link href='https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap' rel='stylesheet'>"
        "<style>* { margin:0; padding:0; box-sizing:border-box; }"
        f"body {{ background:transparent; font-family:{FONTS}; color:{txt}; overflow:hidden; margin:0; }}"
        f"::-webkit-scrollbar {{ width:4px; height:4px; }}"
        f"::-webkit-scrollbar-track {{ background:{bg2}; }}"
        f"::-webkit-scrollbar-thumb {{ background:{bdr}; border-radius:2px; }}"
        "</style></head><body>"
        f"<div style='height:800px;overflow-y:auto;overflow-x:auto'>{body}</div>"
        "</body></html>"
    )


def render_private_tab(is_mobile):
    sgt = pytz.timezone('Asia/Singapore')
    now_str = datetime.now(sgt).strftime('%d %b %Y %H:%M SGT')

    col_sort, col_spacer, col_ts = st.columns([1, 4, 1])
    with col_sort:
        st.markdown(f"<div style='font-size:9px;font-weight:700;color:#e2e8f0;font-family:{FONTS};text-transform:uppercase;letter-spacing:0.08em;margin-bottom:-18px'>SORT BY</div>", unsafe_allow_html=True)
        sort_by = st.selectbox('Sort by', ['Valuation', '52W %', 'Total Raised', 'Latest Date', 'Round', 'Sector'], key='private_sort', label_visibility='collapsed')
    with col_ts:
        st.markdown(f"<div style='font-size:9px;color:#f8fafc;font-family:{FONTS};padding:28px 0 0 0;text-align:right'>Updated: {now_str}</div>", unsafe_allow_html=True)

    with st.spinner('Loading private companies...'):
        data = fetch_prices()

    # Export to GitHub for Yurei (silent, background)
    _export_to_github(PRIVATE_COMPANIES, data)

    height = 820
    st_html(_wrap(_build_table(data, sort_by), height), height=height)
