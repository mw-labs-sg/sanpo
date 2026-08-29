"""
SANPO — Predictions tab
Polymarket Gamma API - /events endpoint with tag_id filtering
"""

import streamlit as st
import requests
import base64
import json
from datetime import datetime
import pytz
import logging
import re

from config import THEMES, FONTS, st_html

logger = logging.getLogger(__name__)


def get_theme():
    tn = st.session_state.get('theme', 'Dark')
    return THEMES.get(tn, THEMES['Dark'])


def _export_to_github(markets):
    """Write predictions.json to GitHub — only when the data actually changed.

    Streamlit re-runs every tab on every interaction, so calling the API here
    directly meant a GET + PUT (and a commit) per script run. The payload is
    built without a timestamp so it can key a cache, and the push only fires on
    a genuine content change (or hourly).
    """
    try:
        if not st.secrets.get("GITHUB_TOKEN", ""):
            return
        data = {
            'source': 'SANPO Predictions - Polymarket',
            'count': len(markets),
            'markets': markets
        }
        _push_predictions_json(json.dumps(data, indent=2, ensure_ascii=False))
    except Exception as e:
        logger.warning(f"GitHub export error: {e}")


@st.cache_data(ttl=3600, show_spinner=False)
def _push_predictions_json(body_json):
    """Commit predictions.json. Cached on body_json so unchanged data is a no-op.

    The token is read here rather than passed in, to keep it out of the cache key.
    """
    try:
        token = st.secrets.get("GITHUB_TOKEN", "")
        if not token:
            return
        sgt = pytz.timezone('Asia/Singapore')
        data = json.loads(body_json)
        data['updated'] = datetime.now(sgt).isoformat()

        url = "https://api.github.com/repos/YureiKara/sanpo/contents/predictions.json"
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
        sha = None
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            sha = r.json().get("sha")
        content_b64 = base64.b64encode(
            json.dumps(data, indent=2, ensure_ascii=False).encode()
        ).decode()
        payload = {"message": "data: update predictions.json", "content": content_b64}
        if sha:
            payload["sha"] = sha
        requests.put(url, headers=headers, json=payload, timeout=10)
    except Exception as e:
        logger.warning(f"GitHub push error: {e}")


CATEGORIES = {
    'All':         None,
    'Finance':     120,
    'Crypto':      21,
    'Politics':    2,
    'Tech':        1401,
    'Geopolitics': 100265,
    'Culture':     596,
}


def _strip_question_prefix(label, event_title):
    """Remove repeated question prefix from outcome label."""
    # Remove common prefixes like "Will X win..." -> "X"
    label = label.strip()
    # Try to extract the key entity — remove "Will ... win/be/reach/..."
    patterns = [
        r'^Will (.+?) win ',
        r'^Will (.+?) be ',
        r'^Will (.+?) reach ',
        r'^Will (.+?) become ',
        r'^Will (.+?) get ',
        r'^Will (.+?) have ',
        r'^Will (.+?) lose ',
        r'^Will (.+?) sign ',
        r'^Will (.+?) release ',
        r'^Will (.+?) announce ',
        r'^Will (.+?) hit ',
    ]
    for pat in patterns:
        m = re.match(pat, label, re.IGNORECASE)
        if m:
            return m.group(1).strip()

    # If label contains the event title, just return shortened version
    if len(label) > 40:
        return label[:38] + '…'
    return label


@st.cache_data(ttl=300, max_entries=5, show_spinner=False)
def fetch_markets(category='All', limit=30):
    import json as _json
    try:
        tag_id = CATEGORIES.get(category)
        params = {
            'limit': limit,
            'closed': 'false',
            'order': 'volume24hr',
            'ascending': 'false',
            'related_tags': 'true',
        }
        if tag_id:
            params['tag_id'] = tag_id

        r = requests.get(
            'https://gamma-api.polymarket.com/events',
            params=params,
            timeout=15,
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        r.raise_for_status()
        events = r.json()
        if not isinstance(events, list):
            events = events.get('data', [])

        results = []
        for ev in events:
            markets  = ev.get('markets', [])
            question = ev.get('title', ev.get('question', ''))
            slug     = ev.get('slug', ev.get('id', ''))
            vol      = float(ev.get('volume', 0) or 0)
            vol24    = float(ev.get('volume24hr', 0) or 0)
            vol1wk   = float(ev.get('volume1wk', 0) or 0)
            liq      = float(ev.get('liquidity', 0) or 0)
            oi       = float(ev.get('openInterest', 0) or 0)
            cat      = ev.get('category', '') or ev.get('subcategory', '') or ''
            subtitle = ev.get('subtitle', '') or ''

            # Sum OI from markets if event level is zero
            if oi == 0 and markets:
                oi = sum(float(m.get('openInterest', 0) or 0) for m in markets)

            end_date = ev.get('endDate', ev.get('end_date_iso', ''))
            expiry_str = ''
            if end_date:
                try:
                    dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                    expiry_str = dt.strftime('%d %b %Y')
                except Exception:
                    expiry_str = str(end_date)[:10]

            outcome_list = []
            is_multi = len(markets) > 1

            if is_multi:
                # Multi-outcome: each market = one outcome, use Yes price
                for m in markets:
                    mq = m.get('groupItemTitle', m.get('question', ''))
                    prices   = m.get('outcomePrices', '[]')
                    outcomes = m.get('outcomes', '[]')
                    if isinstance(prices, str):
                        try: prices = _json.loads(prices)
                        except Exception: prices = []
                    if isinstance(outcomes, str):
                        try: outcomes = _json.loads(outcomes)
                        except Exception: outcomes = []
                    yes_price = None
                    for o, p in zip(outcomes, prices):
                        if str(o).lower() == 'yes':
                            try: yes_price = round(float(p) * 100, 1)
                            except Exception: pass
                            break
                    if yes_price is None and prices:
                        try: yes_price = round(float(prices[0]) * 100, 1)
                        except Exception: pass
                    if mq:
                        outcome_list.append({'label': mq, 'pct': yes_price})
            else:
                # Binary: show Yes/No
                m = markets[0] if markets else {}
                outcomes = m.get('outcomes', '[]')
                prices   = m.get('outcomePrices', '[]')
                if isinstance(outcomes, str):
                    try: outcomes = _json.loads(outcomes)
                    except Exception: outcomes = []
                if isinstance(prices, str):
                    try: prices = _json.loads(prices)
                    except Exception: prices = []
                for o, p in zip(outcomes, prices):
                    try: pct = round(float(p) * 100, 1)
                    except Exception: pct = None
                    outcome_list.append({'label': str(o), 'pct': pct})

            outcome_list.sort(key=lambda x: x.get('pct') or 0, reverse=True)

            results.append({
                'question': question,
                'subtitle': subtitle,
                'outcomes': outcome_list[:3],  # max 3
                'volume': vol,
                'volume24': vol24,
                'vol1wk': vol1wk,
                'liquidity': liq,
                'open_interest': oi,
                'category': cat,
                'expiry': expiry_str,
                'url': f"https://polymarket.com/event/{slug}",
            })

        results.sort(key=lambda x: x['volume'], reverse=True)
        return results

    except Exception as e:
        logger.warning(f"Polymarket fetch error: {e}")
        return []


def _fmt_vol(v):
    if not v: return '—'
    if v >= 1e6: return f"${v/1e6:.1f}M"
    if v >= 1e3: return f"${v/1e3:.0f}K"
    return f"${v:.0f}"


def _pct_color(pct, pos_c, neg_c, mut):
    if pct is None: return mut
    if pct >= 70: return pos_c
    if pct <= 15: return neg_c
    return '#60a5fa'


def _build_table(markets, theme, sort_by='Volume'):
    bg2   = theme.get('bg2', '#0a0f1a')
    bg3   = theme.get('bg3', '#0f172a')
    bdr   = theme.get('border', '#1e293b')
    mut   = theme.get('muted', '#475569')
    txt2  = theme.get('text2', '#94a3b8')
    acc   = theme.get('accent', '#4ade80')
    pos_c = theme.get('pos', '#4ade80')
    neg_c = theme.get('neg', '#f59e0b')
    blue  = '#60a5fa'

    # Sort markets
    sort_map = {
        'Volume':    lambda x: x['volume'],
        '24H Vol':   lambda x: x['volume24'],
        '7D Vol':    lambda x: x['vol1wk'],
        'Liquidity': lambda x: x['liquidity'],
        'Expiry':    lambda x: x['expiry'] or 'zzz',
    }
    sort_fn = sort_map.get(sort_by, lambda x: x['volume'])
    rev = sort_by != 'Expiry'
    markets = sorted(markets, key=sort_fn, reverse=rev)

    HDR = "font-size:9px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#f8fafc"

    if not markets:
        return f"<div style='padding:40px;text-align:center;font-family:{FONTS};color:{mut};font-size:11px'>No markets found</div>"

    html = f"""<div style='font-family:{FONTS}'>
        <div style='display:flex;align-items:center;padding:5px 10px;
                    border-bottom:1px solid {bdr};gap:6px'>
            <div style='width:180px;flex-shrink:0;{HDR}'>QUESTION</div>
            <div style='flex:1;{HDR}'>OUTCOMES</div>
            <div style='width:65px;flex-shrink:0;{HDR};text-align:right'>VOLUME</div>
            <div style='width:55px;flex-shrink:0;{HDR};text-align:right'>24H</div>
            <div style='width:55px;flex-shrink:0;{HDR};text-align:right'>7D</div>
            <div style='width:65px;flex-shrink:0;{HDR};text-align:right'>LIQUIDITY</div>
            <div style='width:75px;flex-shrink:0;{HDR};text-align:center'>EXPIRY</div>
        </div>"""

    for i, m in enumerate(markets):
        row_bg = bg2 if i % 2 == 0 else bg3
        vol_str   = _fmt_vol(m['volume'])
        vol24_str = _fmt_vol(m['volume24'])
        vol1wk_str= _fmt_vol(m.get('vol1wk', 0))
        liq_str   = _fmt_vol(m.get('liquidity', 0))

        # Build outcomes rows — each on own line, consistent 3 lines
        outcomes_html = ''
        padded = [o for o in m['outcomes'][:3] if o]
        for o in padded:
            if o:
                color = _pct_color(o['pct'], pos_c, neg_c, mut)
                pct_str = f"{o['pct']:.0f}%" if o['pct'] is not None else '—'
                outcomes_html += (
                    f"<div style='display:flex;align-items:baseline;gap:8px;padding:2px 0'>"
                    f"<span style='font-size:10px;font-weight:700;color:{color};"
                    f"flex-shrink:0;min-width:38px'>{pct_str}</span>"
                    f"<span style='font-size:10px;color:#e2e8f0;line-height:1.3'>{o['label']}</span>"
                    f"</div>"
                )
            else:
                outcomes_html += "<div style='height:18px'></div>"

        # Category tag
        cat_html = ''
        if m.get('category'):
            cat_html = (
                f"<span style='font-size:8px;font-weight:600;color:{blue};"
                f"background:{blue}18;border-radius:2px;padding:1px 4px;"
                f"margin-top:2px;display:inline-block'>{m['category'].upper()}</span>"
            )

        html += f"""
        <a href='{m["url"]}' target='_blank' style='text-decoration:none;display:block'>
        <div style='display:flex;align-items:center;background:{row_bg};
                    padding:5px 10px;border-bottom:1px solid {bdr}12;gap:6px'>
            <div style='width:180px;flex-shrink:0'>
                <div style='font-size:11px;font-weight:600;color:#f8fafc;
                            line-height:1.3;overflow:hidden;
                            display:-webkit-box;-webkit-line-clamp:2;
                            -webkit-box-orient:vertical'>{m['question']}</div>
                {cat_html}
            </div>
            <div style='flex:1;overflow:hidden'>{outcomes_html}</div>
            <div style='width:65px;flex-shrink:0;font-size:10px;font-weight:600;
                        color:{acc};text-align:right'>{vol_str}</div>
            <div style='width:55px;flex-shrink:0;font-size:10px;
                        color:{txt2};text-align:right'>{vol24_str}</div>
            <div style='width:55px;flex-shrink:0;font-size:10px;
                        color:{txt2};text-align:right'>{vol1wk_str}</div>
            <div style='width:65px;flex-shrink:0;font-size:10px;
                        color:{blue};text-align:right'>{liq_str}</div>
            <div style='width:75px;flex-shrink:0;font-size:10px;
                        color:{txt2};text-align:center'>{m['expiry']}</div>
        </div>
        </a>"""

    html += "</div>"
    return html


def _wrap(body, height):
    t   = get_theme()
    bg2 = t.get('bg2', '#1a1a1a')
    bg3 = t.get('bg3', '#2a2a2a')
    bdr = t.get('border', '#333333')
    txt = t.get('text', '#ececec')
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<link href='https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap' rel='stylesheet'>"
        "<style>"
        "* { margin:0; padding:0; box-sizing:border-box; }"
        f"body {{ background:transparent; font-family:{FONTS}; color:{txt}; overflow:hidden; }}"
        f"::-webkit-scrollbar {{ width:4px; }}"
        f"::-webkit-scrollbar-track {{ background:{bg2}; }}"
        f"::-webkit-scrollbar-thumb {{ background:{bdr}; border-radius:2px; }}"
        f"a:hover > div {{ background:{bg3} !important; }}"
        "</style></head><body>"
        f"<div style='height:{height}px;overflow-y:auto;overflow-x:hidden'>{body}</div>"
        "</body></html>"
    )


def render_predictions_tab(is_mobile):
    t   = get_theme()
    mut = t.get('muted', '#475569')
    acc = t.get('accent', '#4ade80')
    sgt = pytz.timezone('Asia/Singapore')
    now_str = datetime.now(sgt).strftime('%d %b %Y %H:%M SGT')

    col_cat, col_sort, col_spacer, col_info = st.columns([1, 1, 2, 2])
    with col_cat:
        st.markdown(f"<div style='font-size:9px;font-weight:700;color:#e2e8f0;font-family:{FONTS};text-transform:uppercase;letter-spacing:0.08em;margin-bottom:-18px'>CATEGORY</div>", unsafe_allow_html=True)
        category = st.selectbox('Category', list(CATEGORIES.keys()), key='pred_category', label_visibility='collapsed')
    with col_sort:
        st.markdown(f"<div style='font-size:9px;font-weight:700;color:#e2e8f0;font-family:{FONTS};text-transform:uppercase;letter-spacing:0.08em;margin-bottom:-18px'>SORT BY</div>", unsafe_allow_html=True)
        sort_by = st.selectbox('Sort by', ['Volume', '24H Vol', '7D Vol', 'Liquidity', 'Expiry'], key='pred_sort_sel', label_visibility='collapsed')

    with st.spinner('Loading prediction markets...'):
        markets = fetch_markets(category)

    # Export to GitHub for Yurei (silent, background)
    _export_to_github(markets)

    n = len(markets)
    with col_info:
        st.markdown(
            f"<div style='font-family:{FONTS};text-align:right;padding:4px 0'>"
            f"<div style='font-size:9px;color:{mut}'>{n} markets · Powered by "
            f"<span style='color:{acc}'>Polymarket</span></div>"
            f"<div style='font-size:9px;color:#f8fafc'>Updated: {now_str}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    height = 740
    st_html(_wrap(_build_table(markets, t, sort_by), height), height=height)
