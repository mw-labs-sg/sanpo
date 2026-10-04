import streamlit as st
import feedparser
import logging
import re
import urllib.request
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html import escape as html_escape, unescape as html_unescape
from config import FONTS, THEMES, st_html

logger = logging.getLogger(__name__)

_UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'

def get_theme():
    tn = st.session_state.get('theme', 'Dark')
    return THEMES.get(tn, THEMES['Dark'])

# ── Intelligence layer ──────────────────────────────────────
SOURCE_TIER = {
    # Tier 1 — primary market movers
    'bloomberg': 1, 'reuters': 1, 'financial times': 1, 'ft': 1,
    'wsj': 1, 'wall street journal': 1, "barron's": 1, 'barrons': 1,
    'ft.com': 1, 'marketwatch': 1,
    # Tier 2 — reliable secondary
    'cnbc': 2, 'investing.com': 2, 'forex.com': 2, 'seeking alpha': 2,
    'kitco': 2, 'tradingeconomics': 2, 'bbc': 2, 'associated press': 2,
    'ap news': 2, 'nikkei': 2,
    # Tier 3 — general / lower signal
    'yahoo finance': 3, 'motley fool': 3, 'the motley fool': 3,
    'benzinga': 3, 'msn': 3, 'aol': 3, 'newsweek': 3, 'fortune': 3,
    'usa today': 3, 'thestreet': 3,
}

def _source_tier(source_name):
    s = source_name.lower()
    for k, v in SOURCE_TIER.items():
        if k in s:
            return v
    return 2  # default mid-tier

def _recency_score(sort_key):
    """Return 0-3 recency score from an ISO-8601 publication timestamp."""
    if not sort_key: return 1
    try:
        dt = datetime.fromisoformat(sort_key)
    except (TypeError, ValueError):
        return 1
    now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
    minutes = (now - dt).total_seconds() / 60
    if minutes <= 120:  return 3   # < 2h
    if minutes <= 360:  return 2   # < 6h
    if minutes <= 1440: return 1   # < 24h
    return 0

def score_and_rank(items, top_n=8):
    """Score items by source tier + recency, deduplicate, return top N."""
    seen_titles = set()
    scored = []
    for item in items:
        title_key = re.sub(r'\s+', ' ', item.get('title','').lower())[:60]
        if title_key in seen_titles:
            continue
        seen_titles.add(title_key)
        tier  = _source_tier(item.get('source',''))
        rec   = _recency_score(item.get('sort_key',''))
        score = (4 - tier) * 10 + rec   # tier 1 = 30+rec, tier 2 = 20+rec, tier 3 = 10+rec
        scored.append({**item, '_score': score, '_tier': tier})
    scored.sort(key=lambda x: x['_score'], reverse=True)
    return scored[:top_n]

# ─────────────────────────────────────────────────────────────
def _clean(raw):
    if not raw: return ''
    t = re.sub(r'<[^>]+>', '', raw)
    return re.sub(r'\s+', ' ', html_unescape(t)).strip()

def _fetch_with_ua(url, timeout=10):
    """Fetch URL with browser user-agent. Returns raw bytes or None."""
    req = urllib.request.Request(url, headers={
        'User-Agent': _UA,
        'Accept': 'application/rss+xml, application/xml, text/xml, */*',
    })
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        return resp.read()
    except Exception:
        return None

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_rss_feed(name, url):
    try:
        # Try with browser user-agent first (needed for Nikkei, etc)
        raw = _fetch_with_ua(url)
        if raw:
            feed = feedparser.parse(raw)
        else:
            feed = feedparser.parse(url)
        items = []
        for entry in feed.entries[:20]:
            title = _clean(getattr(entry, 'title', ''))
            link = getattr(entry, 'link', '')
            pub = getattr(entry, 'published', getattr(entry, 'updated', ''))
            if not title or 'shareholders are encouraged' in title.lower():
                continue
            date_str = ''
            sort_key = ''
            # feedparser normalises RFC-2822 and ISO-8601 alike into
            # *_parsed, so prefer it. Parsing `published` by hand with
            # parsedate_to_datetime only understands RFC-2822, which left
            # ISO-8601 feeds (Business Insider) showing a raw timestamp and,
            # worse, an empty sort_key — so they never sorted or scored by age.
            parsed = (getattr(entry, 'published_parsed', None)
                      or getattr(entry, 'updated_parsed', None))
            if parsed:
                dt = datetime(*parsed[:6], tzinfo=timezone.utc)
                date_str = dt.strftime('%d %b')
                sort_key = dt.isoformat()
            elif pub:
                try:
                    from email.utils import parsedate_to_datetime
                    dt = parsedate_to_datetime(pub)
                    date_str = dt.strftime('%d %b')
                    sort_key = dt.isoformat()
                except Exception:
                    date_str = pub[:16]
                    sort_key = ''
            items.append({'title': html_escape(title), 'url': html_escape(link, quote=True),
                          'date': html_escape(date_str), 'sort_key': sort_key, 'source': name})
        return items
    except Exception as e:
        logger.warning(f"RSS error [{name}]: {e}")
        return []

_JSONLD_DATE = re.compile(r'"datePublished"\s*:\s*"([^"]+)"')


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_article_dates(urls):
    """Published dates for feeds that omit them, scraped from the article page.

    Nikkei Asia's RSS is RDF carrying only <title> and <link> — no pubDate,
    dc:date or any other date tag — so items from it have no date to show or
    sort by. The article pages do expose JSON-LD "datePublished".

    Returns {url: (date_str, iso_sort_key)}; missing/failed lookups are absent.
    Fetched in parallel and cached, so this costs one round of requests per
    refresh, not one per render.
    """
    def _one(url):
        try:
            raw = _fetch_with_ua(html_unescape(url), timeout=6)
            if not raw:
                return url, None
            m = _JSONLD_DATE.search(raw.decode('utf-8', 'replace'))
            if not m:
                return url, None
            dt = datetime.fromisoformat(m.group(1).replace('Z', '+00:00'))
            return url, (dt.strftime('%d %b'), dt.isoformat())
        except Exception:
            return url, None

    out = {}
    try:
        with ThreadPoolExecutor(max_workers=5) as ex:
            for url, got in ex.map(_one, list(urls)):
                if got:
                    out[url] = got
    except Exception as e:
        logger.warning(f"article date backfill failed: {e}")
    return out


def backfill_missing_dates(items):
    """Fill in dates for items whose feed supplied none. Returns new dicts."""
    need = [it['url'] for it in items if not it.get('sort_key') and it.get('url')]
    if not need:
        return items
    found = fetch_article_dates(tuple(need))
    out = []
    for it in items:
        it = dict(it)
        got = found.get(it.get('url'))
        if got and not it.get('sort_key'):
            it['date'], it['sort_key'] = html_escape(got[0]), got[1]
        out.append(it)
    return out


# ── MASTHEAD BOARD ───────────────────────────────────────────────────────────
# One box per outlet — no commingling, so each masthead gets its own slot and a
# busy wire can never crowd out a quiet one. This used to sit on PULSE; it is
# the news tab's job, and PULSE is for prices.

BOARD_SOURCES = [
    # (masthead, feed name, category, url) — ordered by category so the board
    # reads in blocks. Every feed here was checked to return items; two
    # candidates were dropped for returning nothing (Fierce Biotech, IMF Blog).
    ('STRAITS TIMES',    'ST',             'Singapore', 'https://www.straitstimes.com/news/business/rss.xml'),
    ('BUSINESS TIMES',   'BT',             'Singapore', 'https://www.businesstimes.com.sg/rss/top-stories'),
    ('CNA',              'CNA',            'Singapore', 'https://www.channelnewsasia.com/api/v1/rss-outbound-feed?_format=xml&category=6511'),

    ('SCMP',             'SCMP',           'Regional',  'https://www.scmp.com/rss/5/feed'),
    ('NIKKEI ASIA',      'Nikkei',         'Regional',  'https://asia.nikkei.com/rss/feed/nar'),
    ('MALAY MAIL',       'Malay Mail',     'Regional',  'https://www.malaymail.com/feed/rss/money'),

    ('BLOOMBERG',        'Bloomberg',      'World',     'https://feeds.bloomberg.com/markets/news.rss'),
    ('FT',               'FT',             'World',     'https://www.ft.com/rss/home'),
    # www.businessinsider.com/rss, not markets.businessinsider.com/rss/news —
    # the markets feed is mostly syndicated press releases.
    ('BUSINESS INSIDER', 'BI',             'World',     'https://www.businessinsider.com/rss'),

    ('ECONOMIST FIN',    'Economist Fin',  'Economy',   'https://www.economist.com/finance-and-economics/rss.xml'),
    ('ECONOMIST BIZ',    'Economist Biz',  'Economy',   'https://www.economist.com/business/rss.xml'),
    ('CNBC ECONOMY',     'CNBC',           'Economy',   'https://www.cnbc.com/id/20910258/device/rss/rss.html'),

    ('POLITICO',         'Politico',       'Politics',  'https://rss.politico.com/politics-news.xml'),
    ('THE HILL',         'The Hill',       'Politics',  'https://thehill.com/news/feed/'),
    ('BBC POLITICS',     'BBC Politics',   'Politics',  'https://feeds.bbci.co.uk/news/politics/rss.xml'),

    ('TECHCRUNCH',       'TechCrunch',     'Tech',      'https://techcrunch.com/feed/'),
    ('THE VERGE',        'The Verge',      'Tech',      'https://www.theverge.com/rss/index.xml'),
    ('ARS TECHNICA',     'Ars Technica',   'Tech',      'https://feeds.arstechnica.com/arstechnica/technology-lab'),

    ('TECHCRUNCH AI',    'TechCrunch AI',  'AI',        'https://techcrunch.com/category/artificial-intelligence/feed/'),
    ('MIT TECH REVIEW',  'MIT Tech Review', 'AI',       'https://www.technologyreview.com/feed/'),
    ('VENTUREBEAT AI',   'VentureBeat AI', 'AI',        'https://venturebeat.com/category/ai/feed/'),

    ('COINDESK',         'CoinDesk',       'Crypto',    'https://www.coindesk.com/arc/outboundfeeds/rss/'),
    ('COINTELEGRAPH',    'Cointelegraph',  'Crypto',    'https://cointelegraph.com/rss'),
    ('THE BLOCK',        'The Block',      'Crypto',    'https://www.theblock.co/rss.xml'),

    ('STAT NEWS',        'STAT',           'Health',    'https://www.statnews.com/feed/'),
    ('ENDPOINTS',        'Endpoints',      'Health',    'https://endpts.com/feed/'),
    ('BIOPHARMA DIVE',   'BioPharma Dive', 'Health',    'https://www.biopharmadive.com/feeds/news/'),
]

BOARD_PER_SOURCE = 5
BOARD_COLS = 3               # 3 across: the headline gets room to be read
_BOARD_ROW_H = 26            # measured row height, px
_BOARD_HEAD_H = 24           # box header
_BOARD_GAP = 6
_BOARD_CAT_H = 30            # category heading + its margin


def _board_surface():
    t = get_theme()
    is_light = t.get('mode') == 'light'
    return {
        'bg2': t.get('bg2', '#0a0f1a'),
        'border': t.get('border', '#1e293b'),
        'muted': t.get('muted', '#475569'),
        'link': '#334155' if is_light else t.get('text', '#e2e8f0'),
        'row_alt': '#f8fafc' if is_light else 'rgba(12,24,45,0.40)',
        # Category headings take the theme accent, like the tab underline and
        # the basket section headers, rather than a hardcoded teal of their own.
        'accent': t.get('accent', '#4ade80'),
    }


def _by_category():
    """Outlets grouped under their category, in the order they are listed."""
    out = OrderedDict()
    for heading, name, category, url in BOARD_SOURCES:
        out.setdefault(category, []).append((heading, name, url))
    return out


def board_height(cols=BOARD_COLS):
    """Exact height for the board — every row visible, no scroll.

    Each category is its own block, so the total is the sum of the blocks
    rather than one grid: a category with four outlets takes two rows on a
    three-wide board and the next heading has to clear them.
    """
    box = _BOARD_HEAD_H + BOARD_PER_SOURCE * _BOARD_ROW_H
    total = 0
    for members in _by_category().values():
        rows = -(-len(members) // max(1, cols))
        total += _BOARD_CAT_H + rows * box + _BOARD_GAP * (rows - 1) + _BOARD_GAP
    return total


def render_news_board(cols=BOARD_COLS):
    """A box per outlet, BOARD_PER_SOURCE headlines each, side by side.

    Height is derived from the content, so nothing needs scrolling to be read.
    """
    s = _board_surface()
    box_h = _BOARD_HEAD_H + BOARD_PER_SOURCE * _BOARD_ROW_H

    def _rows(items):
        out = ''
        for i, item in enumerate(items):
            bg = s['bg2'] if i % 2 == 0 else s['row_alt']
            # No source label: the box header already names the outlet, so the
            # space goes to the headline instead.
            out += (
                "<div style='padding:4px 10px;background:" + bg + ";border-bottom:1px solid " + s['border'] + "18;"
                "display:flex;align-items:baseline;gap:6px;font-family:" + FONTS + ";white-space:nowrap;overflow:hidden'>"
                "<span style='flex-shrink:0;width:34px;color:" + s['muted'] + ";font-size:9px'>"
                + item.get('date', '') + "</span>"
                "<a href='" + item.get('url', '#') + "' target='_blank' title='" + item.get('title', '') + "' "
                "style='color:" + s['link'] + ";text-decoration:none;flex:1;min-width:0;"
                "font-size:10.5px;font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap'>"
                + item.get('title', '') + "</a>"
                "</div>"
            )
        return out

    sections = ''
    rendered = 0
    for category, members in _by_category().items():
        boxes = ''
        for heading, name, url in members:
            items = fetch_rss_feed(name, url)
            items.sort(key=lambda x: x.get('sort_key', ''), reverse=True)
            items = items[:BOARD_PER_SOURCE]
            # Nikkei's feed carries no dates; scrape them off the article pages.
            # Done after the slice so it costs 5 lookups, not 20.
            items = backfill_missing_dates(items)
            items.sort(key=lambda x: x.get('sort_key', ''), reverse=True)
            rendered += len(items)

            body = _rows(items) if items else (
                "<div style='padding:10px;color:" + s['muted'] + ";font-size:10px;text-align:center'>Unavailable</div>"
            )
            boxes += (
                "<div style='background:" + s['bg2'] + ";border:1px solid " + s['border'] + ";border-radius:6px;"
                "overflow:hidden;display:flex;flex-direction:column;height:" + str(box_h) + "px'>"
                "<div style='padding:5px 10px;display:flex;justify-content:space-between;align-items:center;"
                "border-bottom:1px solid " + s['border'] + ";flex-shrink:0'>"
                "<span style='color:#f8fafc;font-size:9px;font-weight:600;letter-spacing:0.1em'>" + heading + "</span>"
                "<span style='color:" + s['muted'] + ";font-size:9px;font-weight:500'>"
                + str(len(items)) + "</span></div>"
                "<div style='overflow-y:auto;flex:1;min-height:0'>" + body + "</div>"
                "</div>"
            )

        sections += (
            "<div style='color:" + s['accent'] + ";font-size:10px;font-weight:700;"
            "letter-spacing:0.14em;text-transform:uppercase;margin:0 0 6px 2px;"
            "height:" + str(_BOARD_CAT_H - 6) + "px;line-height:" + str(_BOARD_CAT_H - 6) + "px'>"
            + category + "</div>"
            "<div style='display:grid;grid-template-columns:repeat(" + str(cols) + ",minmax(0,1fr));"
            "gap:" + str(_BOARD_GAP) + "px;margin-bottom:" + str(_BOARD_GAP) + "px'>" + boxes + "</div>"
        )

    if not rendered:
        return
    st_html(
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<style>*{margin:0;padding:0;box-sizing:border-box}"
        "body{background:transparent;font-family:" + FONTS + "}"
        "::-webkit-scrollbar{width:4px;height:4px}"
        "::-webkit-scrollbar-thumb{background:" + s['border'] + ";border-radius:2px}"
        "</style></head><body>"
        "<div style='font-family:" + FONTS + "'>" + sections + "</div>"
        "</body></html>",
        height=board_height(cols),
    )


def render_news_tab(is_mobile):
    """The masthead board, and nothing else.

    The category panels that used to sit underneath showed the same wires a
    second time, split by desk; the board already names each outlet's desk in
    its header.
    """
    render_news_board(cols=1 if is_mobile else BOARD_COLS)
