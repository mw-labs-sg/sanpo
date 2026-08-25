# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SANPO is a financial market dashboard built with Streamlit. It provides real-time market data visualization across 12 tabs: PULSE, NEWS, WORLD, FX, PREDICT, PRIVATE, PORTFOLIO, SPREADS, CHARTS, OPTIONS, RATES, RESEARCH.

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app runs at http://localhost:8501. No build step, test suite, or linter is configured.

## Architecture

**Entry point**: `app.py` — sets up page config, injects theme CSS, renders the animated SVG logo, creates 12 Streamlit tabs, and wires auto-refresh at :00/:30 minute marks.

**Config**: `config.py` — central source of truth for symbol groups (18 asset classes, 130+ tickers), friendly name mappings (`SYMBOL_NAMES`), theme colors (`THEMES` dict), and helpers (`clean_symbol()`, `sym_name()`, `surface()`).

**Tab modules**: Each tab is a standalone Python file exporting `render_<tab>_tab(is_mobile)`. Modules import theme colors from `st.session_state`, fetch data via `@st.cache_data`-decorated functions (TTL 300s–3600s), compute metrics, and render Plotly charts or custom HTML tables via `st_html()`.

**Data fetch scripts** (run by GitHub Actions, not the app):
- `fetch_news.py` — crawls 20+ RSS feeds → `news.json`
- `fetch_private.py` — fetches yfinance data for private companies → `private.json`

**External APIs**: yfinance (stocks/options/crypto/FX), RSS feeds, Polymarket Gamma API, US Treasury XML, Singapore MAS API.

## Key Patterns

- **Dark theme only**. All colors come from the `THEMES` dict in `config.py`; CSS is injected in `app.py`.
- **No database**. JSON files (`news.json`, `private.json`, `predictions.json`) committed to the repo serve as data stores, updated by GitHub Actions.
- **Mobile awareness**. Every render function receives `is_mobile` and adjusts column layouts and font sizes accordingly.
- **Caching**. All yfinance calls use `@st.cache_data` with `show_spinner=False`. Never fetch without caching.
- **Session state keys**: `theme`, `symbol`, `sector`, `chart_type`, plus portfolio-specific keys (`port_preset_name`, `port_mode`, `port_score`, `port_rebal`, `port_period`).

## Adding a New Tab

1. Create `new_feature.py` with `def render_new_feature_tab(is_mobile):`.
2. Import and wire it in `app.py` within the tab list.

## GitHub Actions

- `fetch-news.yml` / `fetch-private.yml` — manual-trigger workflows that run fetch scripts and commit JSON data.
