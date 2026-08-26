import streamlit as st
import feedparser
import logging
import re
import urllib.request
from datetime import datetime
from html import escape as html_escape, unescape as html_unescape
from config import FONTS, THEMES, st_html

logger = logging.getLogger(__name__)

_UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'

def get_theme():
    tn = st.session_state.get('theme', 'Dark')
    return THEMES.get(tn, THEMES['Dark'])

NEWS_FEEDS = {
    'Macro': [
        ('S&P 500', 'https://news.google.com/rss/search?q=S%26P+500+index&hl=en&gl=US&ceid=US:en'),
        ('Global Equities', 'https://news.google.com/rss/search?q=global+equities+MSCI+world&hl=en&gl=US&ceid=US:en'),
        ('Gold', 'https://news.google.com/rss/search?q=gold+price+XAU&hl=en&gl=US&ceid=US:en'),
        ('US T-Bills', 'https://news.google.com/rss/search?q=US+treasury+bills+fed+funds+rate&hl=en&gl=US&ceid=US:en'),
        ('Bitcoin', 'https://news.google.com/rss/search?q=bitcoin+BTC+price&hl=en&gl=US&ceid=US:en'),
    ],
    'Singapore': [
        ('SGX', 'https://news.google.com/rss/search?q=SGX+Singapore+Exchange&hl=en&gl=SG&ceid=SG:en'),
        ('STI Index', 'https://news.google.com/rss/search?q=Straits+Times+Index+STI&hl=en&gl=SG&ceid=SG:en'),
        ('Amova MBH', 'https://news.google.com/rss/search?q=Singapore+corporate+bonds+investment+grade+HDB+Temasek+UOB+LTA&hl=en&gl=SG&ceid=SG:en'),
        ('SG T-Bill', 'https://news.google.com/rss/search?q=Singapore+T-bill+rate+MAS+government+securities&hl=en&gl=SG&ceid=SG:en'),
    ],
    'CPF': [
        ('CPF', 'https://news.google.com/rss/search?q=CPF+Singapore+Central+Provident+Fund+interest+rate&hl=en&gl=SG&ceid=SG:en'),
    ],
    'Local': [
        ('CNA', 'https://www.channelnewsasia.com/api/v1/rss-outbound-feed?_format=xml&category=6511'),
        ('Straits Times', 'https://www.straitstimes.com/news/business/rss.xml'),
        ('Business Times', 'https://www.businesstimes.com.sg/rss/top-stories'),
    ],
    'Regional': [
        ('SCMP', 'https://www.scmp.com/rss/5/feed'),
        ('Nikkei Asia', 'https://asia.nikkei.com/rss/feed/nar'),
        ('Malay Mail', 'https://www.malaymail.com/feed/rss/money'),
        ('The Star', 'https://www.thestar.com.my/rss/Business'),
    ],
    'World': [
        ('Bloomberg', 'https://feeds.bloomberg.com/markets/news.rss'),
        ('FT', 'https://www.ft.com/rss/home'),
        ('CNBC', 'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258'),
        ('BBC Business', 'https://feeds.bbci.co.uk/news/business/rss.xml'),
    ],
    'Tech': [
        ('TechCrunch AI', 'https://techcrunch.com/category/artificial-intelligence/feed/'),
        ('The Verge', 'https://www.theverge.com/rss/index.xml'),
        ('Ars Technica', 'https://feeds.arstechnica.com/arstechnica/technology-lab'),
        ('CoinDesk', 'https://www.coindesk.com/arc/outboundfeeds/rss/'),
        ('STAT News', 'https://www.statnews.com/feed/'),
        ('Endpoints', 'https://endpts.com/feed/'),
        ('Longevity Tech', 'https://www.longevity.technology/feed/'),
    ],
}

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
            if pub:
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

def render_news_panel(region, feeds, max_items=20, height=600):
    """Render a ranked news panel — scores by source tier + recency."""
    t = get_theme(); pos_c = t['pos']
    _body_bg = t.get('bg2', '#0f1522')
    _bdr = t.get('border', '#1e293b')
    _mut = t.get('muted', '#4a5568')
    _link_c = t.get('text', '#c9d1d9')
    _row_alt = t.get('bg3', '#131b2e')
    _txt2 = t.get('text2', '#94a3b8')
    _accent = pos_c

    all_items = []
    for name, url in feeds:
        all_items.extend(fetch_rss_feed(name, url))
    # Intelligence layer: rank by source tier + recency, deduplicate
    all_items = score_and_rank(all_items, top_n=max_items)
    all_items.sort(key=lambda x: x.get('sort_key', ''), reverse=True)
    all_items = all_items[:max_items]

    rows = ''
    if not all_items:
        rows = f"<div style='padding:12px;color:{_mut};font-size:10px;text-align:center'>Feeds loading…</div>"
    else:
        for i, item in enumerate(all_items):
            bg = _body_bg if i % 2 == 0 else _row_alt
            rows += (
                "<div style='padding:4px 10px;background:" + bg + ";border-bottom:1px solid " + _bdr + "18;"
                "display:flex;align-items:baseline;gap:6px;font-family:" + FONTS + ";white-space:nowrap;overflow:hidden'>"
                "<span style='flex-shrink:0;width:115px;display:flex;gap:5px;align-items:baseline'>"
                "<span style='color:" + _accent + ";font-weight:600;font-size:9px'>" + item['source'] + "</span>"
                "<span style='color:" + _txt2 + ";font-size:9px'>" + item['date'] + "</span></span>"
                "<a href='" + item['url'] + "' target='_blank' title='" + item['title'] + "' "
                "style='color:" + _link_c + ";text-decoration:none;flex:1;min-width:0;"
                "font-size:10.5px;font-weight:500;overflow:hidden;text-overflow:ellipsis;"
                "white-space:nowrap'>" + item['title'] + "</a>"
                "</div>"
            )

    page = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<style>* { margin:0; padding:0; box-sizing:border-box; }"
        "body { background:transparent; overflow:hidden; }"
        "::-webkit-scrollbar { width:4px; }"
        "::-webkit-scrollbar-track { background:" + _body_bg + "; }"
        "::-webkit-scrollbar-thumb { background:" + _bdr + ";border-radius:2px; }"
        "</style></head><body>"
        "<div style='background:" + _body_bg + ";border:1px solid " + _bdr + ";border-radius:4px;"
        "height:" + str(height) + "px;overflow-y:auto'>"
        + rows +
        "</div></body></html>"
    )
    st_html(page, height=height)

def render_news_tab(is_mobile):
    # Left: general news tabs | Right: portfolio news tabs
    left, right = st.columns([1, 1])

    with left:
        general_tabs = ['Local', 'Regional', 'World', 'Tech']
        tabs = st.tabs(general_tabs)
        for tab, region in zip(tabs, general_tabs):
            with tab:
                render_news_panel(region, NEWS_FEEDS[region], height=580)

    with right:
        portfolio_tabs = ['Macro', 'Singapore', 'CPF']
        tabs = st.tabs(portfolio_tabs)
        for tab, region in zip(tabs, portfolio_tabs):
            with tab:
                render_news_panel(region, NEWS_FEEDS[region], height=580)
