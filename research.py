"""
SANPO — Research tab
Minimal 2-line cards: title (white) + subtitle (tag accent color).
Auto-pulls from Substack RSS. No manual maintenance.
"""

import streamlit as st
import feedparser
import re
import logging
import urllib.request
from html import escape as html_escape, unescape as html_unescape
from config import get_theme, FONTS, st_html

logger = logging.getLogger(__name__)

SUBSTACK_URL = "https://sanporesearch.substack.com"
FEED_URL = f"{SUBSTACK_URL}/feed"
_UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'

# =============================================================================
# TAG → ACCENT COLOR
# =============================================================================

TAG_COLORS = {
    "geopolitics": "#60a5fa", "iran": "#f59e0b", "middle east": "#f59e0b",
    "macro": "#60a5fa", "gold": "#fbbf24", "oil": "#f59e0b",
    "energy": "#f59e0b", "crypto": "#fbbf24", "bitcoin": "#fbbf24",
    "digital assets": "#fbbf24", "portfolio": "#4ade80", "quant": "#c084fc",
    "singapore": "#4ade80", "rates": "#60a5fa", "fx": "#38bdf8",
    "china": "#ef4444", "markets": "#4ade80", "trade": "#60a5fa",
}

def _get_accent(tags):
    for t in tags:
        c = TAG_COLORS.get(t.lower().strip())
        if c:
            return c
    return "#f59e0b"

# =============================================================================
# RSS
# =============================================================================

def _clean(raw):
    if not raw:
        return ''
    t = re.sub(r'<[^>]+>', '', raw)
    return re.sub(r'\s+', ' ', html_unescape(t)).strip()

def _read_time(text):
    return f"{max(1, round(len(text.split()) / 250))} min"

@st.cache_data(ttl=1800, show_spinner=False)
def _fetch():
    try:
        req = urllib.request.Request(FEED_URL, headers={'User-Agent': _UA})
        resp = urllib.request.urlopen(req, timeout=15)
        feed = feedparser.parse(resp.read())
        out = []
        for e in feed.entries[:20]:
            title = _clean(getattr(e, 'title', ''))
            if not title:
                continue
            sub = _clean(getattr(e, 'summary', ''))
            content = ''
            if hasattr(e, 'content'):
                for c in e.content:
                    content += _clean(c.get('value', ''))
            tags = [getattr(t, 'term', '') for t in getattr(e, 'tags', []) if getattr(t, 'term', '')]
            dt = ''
            try:
                from email.utils import parsedate_to_datetime
                dt = parsedate_to_datetime(getattr(e, 'published', '')).strftime('%d %b %Y')
            except Exception:
                pass
            out.append({
                'title': html_escape(title), 'sub': html_escape(sub),
                'url': html_escape(getattr(e, 'link', ''), quote=True),
                'date': html_escape(dt), 'tags': tags,
                'rt': _read_time(content or sub),
            })
        return out
    except Exception as ex:
        logger.warning(f"RSS error: {ex}")
        return []

# =============================================================================
# RENDER
# =============================================================================

def render():
    t = get_theme()
    bg2 = t.get('bg2', '#0a0f1a')
    bg3 = t.get('bg3', '#0f172a'); bdr = t.get('border', '#1e293b')
    txt = t.get('text', '#e2e8f0')
    mut = t.get('muted', '#475569'); acc = t.get('accent', '#4ade80')

    articles = _fetch()
    n = len(articles)

    rows = ""
    for i, a in enumerate(articles):
        ac = _get_accent(a['tags'])
        rbg = bg2 if i % 2 == 0 else bg3
        rows += f"""
        <a href="{a['url']}" target="_blank" class="row">
            <div style="
                display:flex;align-items:center;gap:12px;
                border-left:3px solid {ac};
                background:{rbg};
                padding:8px 16px;
                border-bottom:1px solid {bdr}15;
            ">
                <div style="flex-shrink:0;width:90px">
                    <div style="font-size:9px;color:#f59e0b">{a['date']}</div>
                    <div style="font-size:8px;color:{mut};margin-top:1px">{a['rt']}</div>
                </div>
                <div style="flex:1;min-width:0">
                    <div style="font-size:12px;font-weight:600;color:{txt};
                                line-height:1.3;white-space:nowrap;overflow:hidden;
                                text-overflow:ellipsis">
                        {a['title']}
                    </div>
                    <div style="font-size:10px;color:#60a5fa;line-height:1.4;
                                margin-top:2px;white-space:nowrap;overflow:hidden;
                                text-overflow:ellipsis">
                        {a['sub']}
                    </div>
                </div>
            </div>
        </a>"""

    empty = f"""
        <div style="padding:40px;text-align:center">
            <div style="font-size:11px;color:{mut}">No articles yet — publish on Substack to populate</div>
        </div>"""

    list_h = 650
    body = f"""
    <div>
        <div style="display:flex;justify-content:space-between;align-items:center;
                    padding:4px 4px 8px 4px">
            <div style="display:flex;align-items:center;gap:8px">
                <span style="font-size:10px;font-weight:700;letter-spacing:0.12em;
                             text-transform:uppercase;color:{acc}">Research</span>
                <span style="font-size:9px;color:{mut}">· {n} article{'s' if n != 1 else ''}</span>
            </div>
            <a href="{SUBSTACK_URL}" target="_blank"
               style="font-size:8.5px;color:{mut};text-decoration:none;letter-spacing:0.06em;
                      text-transform:uppercase;font-weight:500">
                Substack ↗
            </a>
        </div>
        <div style="border:1px solid {bdr};border-radius:4px;overflow:hidden">
            <div style="max-height:{list_h}px;overflow-y:auto">
                {rows if articles else empty}
            </div>
        </div>
    </div>"""

    page = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<link href='https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap' rel='stylesheet'>"
        "<style>"
        "* { margin:0; padding:0; box-sizing:border-box; }"
        f"body {{ background:transparent; font-family:{FONTS}; color:{txt}; }}"
        "a.row { text-decoration:none; display:block; }"
        f"a.row:hover div:first-child {{ background:{bdr} !important; }}"
        f"::-webkit-scrollbar {{ width:4px; }}"
        f"::-webkit-scrollbar-track {{ background:{bg2}; }}"
        f"::-webkit-scrollbar-thumb {{ background:{bdr}; border-radius:2px; }}"
        "</style></head><body>"
        f"{body}"
        "</body></html>"
    )

    h = min(50 + max(n, 1) * 52, list_h + 50)
    st_html(page, height=h)
