"""Custom OpenAPI documentation UI — shared by the dashboard and the
inference server.

The visual design mirrors the `openapi-ui` VS Code extension
(https://github.com/jakubkozera/openapi-ui): a three-pane dark layout with
a tag-grouped sidebar on the left, a documentation column in the center
and a read-only "request preview" pane on the right.

This module exposes a single helper :func:`render_openapi_ui` that returns
a full HTML page. Both servers mount it (replacing their previous Swagger
UI page) and pass the URL of their `openapi.json` and a window title.

The page itself fetches the spec at runtime from `spec_url`, so it stays
in sync as new endpoints are added — no rebuild needed.
"""
from __future__ import annotations


def render_openapi_ui(*, title: str, spec_url: str,
                      swagger_url: str | None = None) -> str:
    """Return a self-contained HTML page that renders ``spec_url``.

    Parameters
    ----------
    title:
        Page title shown in the browser tab and in the sidebar header.
    spec_url:
        URL the page will fetch to load the OpenAPI/Swagger JSON spec.
        Both relative (``/openapi.json``) and absolute URLs work.
    swagger_url:
        Optional URL to a classic Swagger UI page. When provided, a small
        "Swagger UI ↗" link is rendered in the sidebar so power users can
        still drop into the original UI to actually run requests. The
        custom UI is read-only — it shows the request shape but does not
        execute calls (parity with the openapi-ui VS Code extension's
        documentation panel).
    """
    safe_title = (title or "API").replace("<", "&lt;").replace(">", "&gt;")
    swagger_link = ""
    if swagger_url:
        swagger_link = (
            f'<a class="oa-swagger-link" href="{swagger_url}" '
            'title="Open the classic Swagger UI in a new tab" '
            'target="_blank" rel="noopener">Swagger UI &#x2197;</a>'
        )

    # Inline everything: the page is a single HTML document with no
    # external runtime deps. CSS uses CSS variables for theme tokens, JS
    # is plain ES2020 (no bundler required).  Template parameters are
    # spliced in via str.replace below so we can keep this string a raw
    # triple-quoted block (no f-string brace escaping noise).
    return _TEMPLATE.replace("__PAGE_TITLE__", safe_title) \
                    .replace("__SPEC_URL__", spec_url) \
                    .replace("__SWAGGER_LINK__", swagger_link)


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------
#
# The ``__PAGE_TITLE__``, ``__SPEC_URL__`` and ``__SWAGGER_LINK__`` markers
# are replaced by :func:`render_openapi_ui`. They are deliberately ugly so
# the chance of them appearing inside the template itself is zero.
#
_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__PAGE_TITLE__</title>
  <style>
    /* ---------- design tokens ---------- */
    :root {
      color-scheme: dark;
      --bg-page: #0a0b0d;
      --bg-panel: #131418;
      --bg-panel-soft: #181a1f;
      --bg-elev: #1c1f25;
      --bg-input: #0d0e10;
      --bg-hover: #1f232b;
      --bg-selected: #2a3245;
      --border: #23262d;
      --border-strong: #2c3140;
      --accent: #7c87ff;
      --accent-soft: rgba(124, 135, 255, 0.18);
      --text: #e6e8ea;
      --text-muted: #b0b3b8;
      --text-faint: #7c8087;
      --type-chip-bg: #1d2a4a;
      --type-chip-fg: #8eb1ff;
      --code-bg: #0d0e10;
      --get: #10b981;
      --post: #3b82f6;
      --put: #f59e0b;
      --patch: #a855f7;
      --delete: #ef4444;
      --head: #6b7280;
      --options: #06b6d4;
      --trace: #94a3b8;
      --shadow-card: 0 1px 0 rgba(255,255,255,0.02), 0 12px 32px rgba(0,0,0,0.35);
      --radius: 8px;
      --radius-sm: 6px;
      --font-mono: ui-monospace, SFMono-Regular, "JetBrains Mono", Menlo, Consolas, monospace;
      --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto,
                   "Helvetica Neue", Arial, sans-serif;
    }
    * { box-sizing: border-box; }
    html, body {
      margin: 0; padding: 0;
      background: var(--bg-page);
      color: var(--text);
      font-family: var(--font-sans);
      font-size: 14px;
      line-height: 1.5;
      height: 100%;
      overflow: hidden;
    }
    a { color: var(--accent); text-decoration: none; }
    a:hover { text-decoration: underline; }

    /* ---------- layout ---------- */
    .oa-shell {
      display: grid;
      grid-template-columns: 280px 1fr 420px 56px;
      height: 100vh;
      width: 100vw;
    }

    /* ---------- left sidebar ---------- */
    .oa-sidebar {
      background: var(--bg-panel);
      border-right: 1px solid var(--border);
      display: flex; flex-direction: column;
      min-width: 0;
    }
    .oa-brand {
      display: flex; align-items: center; gap: 10px;
      padding: 18px 18px 14px;
      border-bottom: 1px solid var(--border);
    }
    .oa-brand-mark {
      display: inline-flex; align-items: center; justify-content: center;
      width: 30px; height: 30px;
      border: 1px solid var(--border-strong);
      border-radius: 6px;
      background: var(--bg-elev);
      font-family: var(--font-mono);
      font-weight: 600;
      color: var(--text);
      font-size: 13px;
      letter-spacing: -0.02em;
    }
    .oa-brand-title {
      font-weight: 600;
      font-size: 14px;
      color: var(--text);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .oa-search-row {
      padding: 12px 14px;
      border-bottom: 1px solid var(--border);
      position: relative;
    }
    .oa-search-input {
      width: 100%;
      background: var(--bg-input);
      border: 1px solid var(--border);
      color: var(--text);
      border-radius: var(--radius-sm);
      padding: 8px 10px 8px 30px;
      font-size: 13px;
      font-family: var(--font-sans);
      outline: none;
    }
    .oa-search-input:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 2px var(--accent-soft);
    }
    .oa-search-icon {
      position: absolute;
      left: 22px; top: 50%;
      transform: translateY(-50%);
      width: 14px; height: 14px;
      opacity: 0.55;
      pointer-events: none;
    }
    .oa-tag-list {
      flex: 1; overflow-y: auto;
      padding: 8px 6px 24px;
    }
    .oa-tag-group { margin-bottom: 2px; }
    .oa-tag-header {
      display: flex; align-items: center; gap: 8px;
      padding: 8px 10px;
      cursor: pointer;
      user-select: none;
      border-radius: var(--radius-sm);
      color: var(--text);
      font-weight: 500;
      font-size: 13.5px;
    }
    .oa-tag-header:hover { background: var(--bg-hover); }
    .oa-tag-chevron {
      display: inline-block;
      width: 14px;
      transition: transform 0.15s ease;
      color: var(--text-faint);
    }
    .oa-tag-group.collapsed .oa-tag-chevron { transform: rotate(-90deg); }
    .oa-tag-name { flex: 1; white-space: nowrap;
                   overflow: hidden; text-overflow: ellipsis; }
    .oa-tag-count {
      background: var(--bg-elev);
      color: var(--text-muted);
      border-radius: 999px;
      padding: 1px 8px;
      font-size: 11.5px;
      font-variant-numeric: tabular-nums;
    }
    .oa-tag-ops {
      list-style: none; padding: 4px 0 8px 28px; margin: 0;
    }
    .oa-tag-group.collapsed .oa-tag-ops { display: none; }
    .oa-tag-op {
      display: flex; align-items: center; gap: 8px;
      padding: 5px 8px;
      border-radius: var(--radius-sm);
      font-size: 13px;
      color: var(--text-muted);
      cursor: pointer;
    }
    .oa-tag-op:hover { background: var(--bg-hover); color: var(--text); }
    .oa-tag-op.active {
      background: var(--bg-selected);
      color: var(--text);
    }
    .oa-method-pill {
      font-family: var(--font-mono);
      font-size: 10.5px;
      font-weight: 700;
      letter-spacing: 0.04em;
      padding: 2px 6px;
      border-radius: 4px;
      text-transform: uppercase;
      min-width: 44px;
      text-align: center;
      flex-shrink: 0;
      color: #0a0b0d;
    }
    .m-get      { background: var(--get); }
    .m-post     { background: var(--post); color: white; }
    .m-put      { background: var(--put); }
    .m-patch    { background: var(--patch); color: white; }
    .m-delete   { background: var(--delete); color: white; }
    .m-head     { background: var(--head); color: white; }
    .m-options  { background: var(--options); }
    .m-trace    { background: var(--trace); color: white; }

    .oa-method-pill-outline {
      background: transparent;
      color: var(--method-color);
      border: 1px solid var(--method-color);
      padding: 3px 9px;
      border-radius: 5px;
      font-family: var(--font-mono);
      font-weight: 600;
      font-size: 11px;
    }

    .oa-op-path {
      font-family: var(--font-mono);
      font-size: 12px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .oa-sidebar-footer {
      padding: 10px 14px;
      border-top: 1px solid var(--border);
      font-size: 12px;
      color: var(--text-faint);
      display: flex; align-items: center; justify-content: space-between;
    }
    .oa-swagger-link {
      color: var(--text-muted);
      font-size: 12px;
    }
    .oa-swagger-link:hover { color: var(--accent); }

    /* ---------- center content ---------- */
    .oa-main {
      overflow-y: auto;
      padding: 28px 36px 64px;
      min-width: 0;
    }
    .oa-info {
      display: flex; flex-direction: column; gap: 4px;
      margin-bottom: 14px;
    }
    .oa-info h1 {
      margin: 0;
      font-size: 26px;
      font-weight: 600;
      letter-spacing: -0.01em;
      display: flex; align-items: center; gap: 10px;
      flex-wrap: wrap;
    }
    .oa-version-chip {
      background: var(--bg-elev);
      border: 1px solid var(--border-strong);
      color: var(--text-muted);
      border-radius: 4px;
      font-size: 12px;
      font-family: var(--font-mono);
      padding: 2px 8px;
      font-weight: 500;
    }
    .oa-endpoints-chip {
      background: var(--accent-soft);
      color: var(--accent);
      border-radius: 999px;
      padding: 2px 10px;
      font-size: 12px;
      font-weight: 600;
    }
    .oa-info-desc {
      color: var(--text-muted);
      max-width: 920px;
    }

    .oa-server-card {
      background: var(--bg-panel);
      border: 1px solid var(--accent-soft);
      border-radius: var(--radius);
      padding: 14px 16px;
      margin: 18px 0 28px;
      box-shadow: 0 0 0 1px rgba(124,135,255,0.04) inset;
    }
    .oa-server-head {
      display: flex; justify-content: space-between; align-items: center;
      margin-bottom: 10px;
    }
    .oa-server-head .label {
      font-weight: 600;
      font-size: 14px;
    }
    .oa-server-head .url {
      font-family: var(--font-mono);
      font-size: 12px;
      color: var(--text-muted);
    }
    .oa-server-row {
      display: grid;
      grid-template-columns: 110px 1fr;
      gap: 12px; align-items: center;
    }
    .oa-server-row label {
      color: var(--text-muted);
      font-size: 13px;
    }
    .oa-server-select {
      width: 100%;
      background: var(--bg-input);
      border: 1px solid var(--border);
      color: var(--text);
      border-radius: var(--radius-sm);
      padding: 8px 10px;
      font-family: var(--font-mono);
      font-size: 12.5px;
    }

    .oa-tag-section { margin: 0 0 28px; }
    .oa-tag-section-head {
      display: flex; align-items: center; gap: 10px;
      margin-bottom: 14px;
      cursor: pointer;
      user-select: none;
    }
    .oa-tag-section-head h2 {
      margin: 0; font-size: 18px; font-weight: 600;
    }
    .oa-tag-section-head .oa-tag-chevron-big {
      width: 14px;
      color: var(--text-faint);
      transition: transform 0.15s ease;
    }
    .oa-tag-section.collapsed .oa-tag-chevron-big {
      transform: rotate(-90deg);
    }
    .oa-tag-section-head .count-pill {
      background: var(--bg-elev);
      color: var(--text-muted);
      border-radius: 999px;
      padding: 1px 10px;
      font-size: 12px;
    }
    .oa-tag-section.collapsed .oa-op-card { display: none; }
    .oa-tag-section-desc {
      color: var(--text-muted);
      font-size: 13px;
      margin: -4px 0 12px;
    }

    .oa-op-card {
      background: var(--bg-panel);
      border: 1px solid var(--border);
      border-left: 3px solid var(--method-color, var(--accent));
      border-radius: var(--radius);
      padding: 16px 20px 18px;
      margin-bottom: 14px;
      box-shadow: var(--shadow-card);
      scroll-margin-top: 18px;
    }
    .oa-op-head {
      display: flex; align-items: center; gap: 12px;
      margin-bottom: 4px;
    }
    .oa-op-head .icons {
      display: flex; gap: 6px; color: var(--text-faint);
    }
    .oa-op-head .icons svg { width: 14px; height: 14px; }
    .oa-op-head .icons .heart:hover { color: #ef4444; cursor: pointer; }
    .oa-op-path-row {
      flex: 1; display: flex; align-items: center; gap: 10px;
      font-family: var(--font-mono);
      font-size: 14px;
    }
    .oa-op-method-btn {
      align-self: flex-start;
    }
    .oa-op-summary {
      color: var(--text);
      font-size: 14.5px;
      font-weight: 500;
      margin: 8px 0 2px;
    }
    .oa-op-desc {
      color: var(--text-muted);
      font-size: 13px;
      margin-bottom: 16px;
    }
    .oa-section-h {
      display: flex; align-items: center; gap: 8px;
      font-size: 14px; font-weight: 600;
      color: var(--text);
      margin: 18px 0 8px;
    }
    .oa-section-count {
      background: var(--bg-elev);
      color: var(--text-muted);
      border-radius: 999px;
      padding: 1px 8px;
      font-size: 11px;
      font-weight: 500;
    }
    .oa-param-table {
      display: flex; flex-direction: column; gap: 0;
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      overflow: hidden;
    }
    .oa-param-row {
      display: grid;
      grid-template-columns: 200px 1fr;
      padding: 12px 14px;
      gap: 10px;
      background: var(--bg-panel-soft);
      border-bottom: 1px solid var(--border);
    }
    .oa-param-row:last-child { border-bottom: none; }
    .oa-param-name {
      font-family: var(--font-mono);
      font-size: 13px;
      color: var(--text);
      display: flex; flex-direction: column; gap: 4px;
    }
    .oa-param-required {
      color: var(--delete);
      margin-left: 4px;
      font-weight: 700;
    }
    .oa-type-chip {
      display: inline-block;
      background: var(--type-chip-bg);
      color: var(--type-chip-fg);
      border-radius: 4px;
      padding: 1px 8px;
      font-family: var(--font-mono);
      font-size: 11px;
      font-weight: 600;
      margin-right: 6px;
    }
    .oa-param-desc {
      display: flex; flex-direction: column; gap: 6px;
      color: var(--text);
      font-size: 13px;
      min-width: 0;
    }
    .oa-param-desc .desc-text { color: var(--text); }
    .oa-validation-block {
      margin-top: 4px;
      font-size: 12px;
      color: var(--text-muted);
    }
    .oa-validation-block .label {
      color: var(--text-muted);
      font-weight: 500;
    }
    .oa-validation-block ul {
      margin: 4px 0 0; padding-left: 18px;
    }
    .oa-validation-block li { margin: 1px 0; }
    .oa-validation-block code {
      font-family: var(--font-mono);
      background: var(--code-bg);
      border: 1px solid var(--border);
      padding: 0 4px;
      border-radius: 3px;
      font-size: 11.5px;
    }

    .oa-resp-list { display: flex; flex-direction: column; gap: 8px; }
    .oa-resp-row {
      display: grid;
      grid-template-columns: 80px 1fr;
      align-items: center;
      gap: 12px;
      padding: 10px 14px;
      background: var(--bg-panel-soft);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
    }
    .oa-resp-status {
      font-family: var(--font-mono);
      font-weight: 600;
      font-size: 12.5px;
    }
    .status-2xx { color: var(--get); }
    .status-3xx { color: var(--put); }
    .status-4xx { color: var(--delete); }
    .status-5xx { color: var(--patch); }

    .oa-schema-block {
      margin-top: 8px;
      background: var(--code-bg);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 10px 12px;
      font-family: var(--font-mono);
      font-size: 12px;
      color: var(--text-muted);
      white-space: pre-wrap;
      overflow-x: auto;
      max-height: 320px;
    }

    /* ---------- right "request preview" pane ---------- */
    .oa-right {
      background: var(--bg-page);
      border-left: 1px solid var(--border);
      overflow-y: auto;
      padding: 18px 18px 32px;
      display: flex; flex-direction: column;
      gap: 14px;
      min-width: 0;
    }
    .oa-right-empty {
      color: var(--text-faint);
      font-size: 13px;
      padding: 40px 8px;
      text-align: center;
    }
    .oa-tryit-head {
      display: flex; align-items: center; gap: 10px;
      justify-content: space-between;
      flex-wrap: wrap;
    }
    .oa-tryit-method-row {
      display: flex; align-items: center; gap: 8px; min-width: 0; flex: 1;
    }
    .oa-tryit-method {
      padding: 6px 10px;
      border-radius: 5px;
      font-family: var(--font-mono);
      font-weight: 700;
      font-size: 12px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      flex-shrink: 0;
      color: #0a0b0d;
      display: inline-flex; align-items: center; gap: 6px;
    }
    .oa-tryit-path {
      font-family: var(--font-mono);
      font-size: 13px;
      color: var(--text);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .oa-add-btn {
      background: var(--bg-elev);
      border: 1px solid var(--border-strong);
      color: var(--text);
      border-radius: 6px;
      padding: 5px 10px;
      font-size: 12px;
      cursor: not-allowed;
      opacity: 0.7;
      display: inline-flex; align-items: center; gap: 6px;
    }
    .oa-tryit-section h4 {
      display: flex; align-items: center; gap: 8px;
      margin: 14px 0 8px;
      font-size: 13px; font-weight: 600;
      color: var(--text);
    }
    .oa-tryit-row {
      display: grid;
      grid-template-columns: 110px 80px 1fr;
      gap: 8px;
      align-items: center;
      margin-bottom: 6px;
    }
    .oa-tryit-row .name {
      font-family: var(--font-mono);
      font-size: 12.5px;
      color: var(--text);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .oa-tryit-row .type {
      justify-self: start;
    }
    .oa-tryit-input {
      background: var(--bg-input);
      border: 1px solid var(--border);
      color: var(--text);
      border-radius: 5px;
      padding: 6px 8px;
      font-family: var(--font-mono);
      font-size: 12px;
      width: 100%;
      outline: none;
    }
    .oa-tryit-input:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 2px var(--accent-soft);
    }
    .oa-tryit-input::placeholder { color: var(--text-faint); }

    .oa-tryit-body {
      background: var(--bg-input);
      border: 1px solid var(--border);
      color: var(--text);
      border-radius: 5px;
      padding: 8px 10px;
      font-family: var(--font-mono);
      font-size: 12px;
      width: 100%;
      min-height: 120px;
      resize: vertical;
      outline: none;
      white-space: pre;
      overflow: auto;
    }
    .oa-readonly-note {
      color: var(--text-faint);
      font-size: 11.5px;
      font-style: italic;
      margin-top: 2px;
    }

    /* ---------- far-right icon strip ---------- */
    .oa-rail {
      background: var(--bg-panel);
      border-left: 1px solid var(--border);
      display: flex; flex-direction: column; align-items: center;
      padding: 14px 0;
      gap: 4px;
    }
    .oa-rail button {
      background: transparent;
      border: none;
      color: var(--text-muted);
      width: 36px; height: 36px;
      border-radius: 6px;
      display: inline-flex; align-items: center; justify-content: center;
      cursor: pointer;
    }
    .oa-rail button:hover {
      background: var(--bg-hover);
      color: var(--text);
    }
    .oa-rail button.active {
      background: var(--accent-soft);
      color: var(--accent);
    }
    .oa-rail button svg { width: 18px; height: 18px; }
    .oa-rail .spacer { flex: 1; }

    /* ---------- loading + error states ---------- */
    .oa-loading, .oa-error {
      padding: 40px;
      text-align: center;
      color: var(--text-muted);
    }
    .oa-error {
      color: var(--delete);
      font-family: var(--font-mono);
      font-size: 13px;
      white-space: pre-wrap;
      text-align: left;
      max-width: 800px;
      margin: 60px auto;
      background: var(--bg-panel);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 24px;
    }

    /* ---------- scrollbar polish ---------- */
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-thumb {
      background: #2c3140;
      border-radius: 8px;
    }
    ::-webkit-scrollbar-thumb:hover { background: #3a4052; }
    ::-webkit-scrollbar-track { background: transparent; }

    @media (max-width: 1100px) {
      .oa-shell { grid-template-columns: 240px 1fr 56px; }
      .oa-right { display: none; }
    }
  </style>
</head>
<body>
  <div class="oa-shell">
    <!-- left sidebar -->
    <aside class="oa-sidebar">
      <div class="oa-brand">
        <span class="oa-brand-mark">{|}</span>
        <span class="oa-brand-title" id="oa-brand-title">__PAGE_TITLE__</span>
      </div>
      <div class="oa-search-row">
        <svg class="oa-search-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="11" cy="11" r="7" /><line x1="21" y1="21" x2="16.65" y2="16.65"/>
        </svg>
        <input class="oa-search-input" id="oa-search" type="text"
               placeholder="Search operations..." spellcheck="false" />
      </div>
      <nav class="oa-tag-list" id="oa-tag-list"></nav>
      <div class="oa-sidebar-footer">
        <span id="oa-version-foot">&nbsp;</span>
        __SWAGGER_LINK__
      </div>
    </aside>

    <!-- center docs -->
    <main class="oa-main" id="oa-main">
      <div class="oa-loading">Loading API specification…</div>
    </main>

    <!-- right read-only Try It pane -->
    <aside class="oa-right" id="oa-right">
      <div class="oa-right-empty">Select an operation on the left to preview its request shape.</div>
    </aside>

    <!-- far-right rail -->
    <aside class="oa-rail">
      <button title="Operations" class="active" aria-label="Operations">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
      </button>
      <button title="Authentication" aria-label="Authentication">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>
      </button>
      <button title="Variables" aria-label="Variables">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 4h6v6H5zM13 4h6v6h-6zM5 12h6v6H5zM13 12h6v6h-6z"/></svg>
      </button>
      <button title="Collections" aria-label="Collections">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>
      </button>
      <button title="Code samples" aria-label="Code samples">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>
      </button>
      <div class="spacer"></div>
    </aside>
  </div>

  <script>
  (() => {
    const SPEC_URL = "__SPEC_URL__";
    const METHOD_COLORS = {
      get: 'var(--get)', post: 'var(--post)', put: 'var(--put)',
      patch: 'var(--patch)', delete: 'var(--delete)', head: 'var(--head)',
      options: 'var(--options)', trace: 'var(--trace)',
    };
    const METHOD_ORDER = ['get', 'post', 'put', 'patch', 'delete', 'head', 'options', 'trace'];

    const $ = (sel, root = document) => root.querySelector(sel);
    const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
    const escapeHtml = s => String(s ?? '').replace(/[&<>"']/g, c => (
      {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]
    ));
    const slug = s => String(s).toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');

    let SPEC = null;
    let OPS = [];                  // flattened list of operations
    let SELECTED = null;           // current op id

    /* ---------- $ref resolver ---------- */
    function resolveRef(node, root) {
      if (!node || typeof node !== 'object') return node;
      if (node.$ref && typeof node.$ref === 'string' && node.$ref.startsWith('#/')) {
        const path = node.$ref.slice(2).split('/');
        let cur = root;
        for (const p of path) cur = cur && cur[p];
        return resolveRef(cur, root) || node;
      }
      return node;
    }

    /* ---------- spec flattening ---------- */
    function flattenOps(spec) {
      const ops = [];
      const paths = spec.paths || {};
      Object.keys(paths).sort().forEach(p => {
        const pathItem = paths[p] || {};
        const sharedParams = pathItem.parameters || [];
        METHOD_ORDER.forEach(method => {
          const op = pathItem[method];
          if (!op) return;
          const tags = (op.tags && op.tags.length) ? op.tags : ['default'];
          ops.push({
            id: slug(method + '-' + p),
            method, path: p, op,
            tag: tags[0],
            tags,
            sharedParams,
          });
        });
      });
      return ops;
    }

    function groupByTag(ops) {
      const groups = new Map();
      ops.forEach(o => {
        if (!groups.has(o.tag)) groups.set(o.tag, []);
        groups.get(o.tag).push(o);
      });
      // preserve original tag ordering from spec where possible
      const tagOrder = (SPEC.tags || []).map(t => t.name);
      const known = new Set();
      const ordered = [];
      tagOrder.forEach(t => { if (groups.has(t)) { ordered.push([t, groups.get(t)]); known.add(t); } });
      Array.from(groups.keys()).forEach(t => {
        if (!known.has(t)) ordered.push([t, groups.get(t)]);
      });
      return ordered;
    }

    /* ---------- sidebar render ---------- */
    function renderSidebar(grouped) {
      const root = $('#oa-tag-list');
      root.innerHTML = '';
      grouped.forEach(([tag, ops]) => {
        const group = document.createElement('div');
        group.className = 'oa-tag-group';
        group.dataset.tag = tag;
        group.innerHTML = `
          <div class="oa-tag-header" data-action="toggle-group">
            <span class="oa-tag-chevron">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                <polyline points="6 9 12 15 18 9"/></svg>
            </span>
            <span class="oa-tag-name">${escapeHtml(tag)}</span>
            <span class="oa-tag-count">${ops.length}</span>
          </div>
          <ul class="oa-tag-ops">
            ${ops.map(o => `
              <li class="oa-tag-op" data-op-id="${o.id}">
                <span class="oa-method-pill m-${o.method}">${o.method}</span>
                <span class="oa-op-path" title="${escapeHtml(o.path)}">${escapeHtml(o.path)}</span>
              </li>`).join('')}
          </ul>`;
        root.appendChild(group);
      });

      root.addEventListener('click', e => {
        const t = e.target.closest('[data-action="toggle-group"]');
        if (t) { t.parentElement.classList.toggle('collapsed'); return; }
        const opEl = e.target.closest('[data-op-id]');
        if (opEl) selectOp(opEl.dataset.opId);
      });
    }

    /* ---------- main content render ---------- */
    function renderInfo(spec) {
      const info = spec.info || {};
      const totalOps = OPS.length;
      const desc = info.description ? `<div class="oa-info-desc">${escapeHtml(info.description)}</div>` : '';
      const servers = (spec.servers && spec.servers.length) ? spec.servers : [{ url: window.location.origin }];
      const primary = servers[0];
      return `
        <div class="oa-info">
          <h1>
            ${escapeHtml(info.title || 'API')}
            ${info.version ? `<span class="oa-version-chip">${escapeHtml(info.version)}</span>` : ''}
            <span class="oa-endpoints-chip">${totalOps} endpoint${totalOps === 1 ? '' : 's'}</span>
          </h1>
        </div>
        <div class="oa-server-card">
          <div class="oa-server-head">
            <span class="label">API Server</span>
            <span class="url">${escapeHtml(primary.url || '')}</span>
          </div>
          <div class="oa-server-row">
            <label for="oa-server-select">Select Server:</label>
            <select class="oa-server-select" id="oa-server-select">
              ${servers.map(s => `<option value="${escapeHtml(s.url)}">${escapeHtml((s.description ? s.description + ' - ' : '') + s.url)}</option>`).join('')}
            </select>
          </div>
        </div>
        ${desc}
      `;
    }

    function paramRows(params, root) {
      if (!params || !params.length) return '';
      return `
        <div class="oa-param-table">
          ${params.map(raw => {
            const p = resolveRef(raw, root);
            const schema = resolveRef(p.schema || {}, root);
            const typeStr = formatType(schema);
            const validation = collectValidations(schema);
            const required = p.required ? '<span class="oa-param-required">*</span>' : '';
            return `
              <div class="oa-param-row">
                <div class="oa-param-name">
                  <span>${escapeHtml(p.name)}${required}</span>
                </div>
                <div class="oa-param-desc">
                  <div>
                    <span class="oa-type-chip">${escapeHtml(typeStr)}</span>
                  </div>
                  ${p.description ? `<div class="desc-text">${escapeHtml(p.description)}</div>` : ''}
                  ${validation.length ? `
                    <div class="oa-validation-block">
                      <span class="label">Validation:</span>
                      <ul>${validation.map(v => `<li>${v}</li>`).join('')}</ul>
                    </div>` : ''}
                </div>
              </div>`;
          }).join('')}
        </div>`;
    }

    function formatType(schema) {
      if (!schema) return 'any';
      if (schema.$ref) return schema.$ref.split('/').pop();
      if (schema.type === 'array') {
        const inner = schema.items ? formatType(schema.items) : 'any';
        return `array(${inner})`;
      }
      if (schema.format) return `${schema.type || 'any'}(${schema.format})`;
      if (schema.enum) return `${schema.type || 'string'}(enum)`;
      if (schema.oneOf) return 'oneOf';
      if (schema.anyOf) return 'anyOf';
      if (schema.allOf) return 'allOf';
      return schema.type || 'object';
    }

    function collectValidations(schema) {
      if (!schema) return [];
      const out = [];
      if (Array.isArray(schema.enum)) {
        out.push(`<span class="label">allowed values:</span> ${schema.enum.map(v => `<code>${escapeHtml(v)}</code>`).join(', ')}`);
      }
      if (schema.minimum !== undefined) out.push(`minimum: <code>${escapeHtml(schema.minimum)}</code>`);
      if (schema.maximum !== undefined) out.push(`maximum: <code>${escapeHtml(schema.maximum)}</code>`);
      if (schema.minLength !== undefined) out.push(`minLength: <code>${escapeHtml(schema.minLength)}</code>`);
      if (schema.maxLength !== undefined) out.push(`maxLength: <code>${escapeHtml(schema.maxLength)}</code>`);
      if (schema.pattern) out.push(`pattern: <code>${escapeHtml(schema.pattern)}</code>`);
      if (schema.default !== undefined) out.push(`default: <code>${escapeHtml(JSON.stringify(schema.default))}</code>`);
      return out;
    }

    function partitionParams(params, root) {
      const buckets = { query: [], header: [], path: [], cookie: [] };
      (params || []).forEach(raw => {
        const p = resolveRef(raw, root);
        if (!p || !p.in) return;
        if (buckets[p.in]) buckets[p.in].push(p);
      });
      return buckets;
    }

    function renderResponses(responses, root) {
      if (!responses) return '';
      const keys = Object.keys(responses);
      if (!keys.length) return '';
      return `
        <div class="oa-section-h">Responses <span class="oa-section-count">${keys.length}</span></div>
        <div class="oa-resp-list">
          ${keys.map(code => {
            const r = resolveRef(responses[code], root) || {};
            const cls = code.startsWith('2') ? 'status-2xx'
                      : code.startsWith('3') ? 'status-3xx'
                      : code.startsWith('4') ? 'status-4xx'
                      : code.startsWith('5') ? 'status-5xx' : '';
            return `
              <div class="oa-resp-row">
                <span class="oa-resp-status ${cls}">${escapeHtml(code)}</span>
                <span>${escapeHtml(r.description || '')}</span>
              </div>`;
          }).join('')}
        </div>`;
    }

    function renderRequestBody(rb, root) {
      if (!rb) return '';
      const r = resolveRef(rb, root);
      const content = r.content || {};
      const types = Object.keys(content);
      if (!types.length) return '';
      const t = types[0];
      const schema = resolveRef(content[t].schema || {}, root);
      const example = content[t].example
        ?? (content[t].examples && Object.values(content[t].examples)[0]?.value)
        ?? schema.example;
      let preview = '';
      if (example !== undefined) {
        try { preview = JSON.stringify(example, null, 2); }
        catch (_) { preview = String(example); }
      } else if (schema && Object.keys(schema).length) {
        try { preview = JSON.stringify(stripVerbose(schema), null, 2); }
        catch (_) { preview = ''; }
      }
      return `
        <div class="oa-section-h">Request Body <span class="oa-section-count">${escapeHtml(t)}</span></div>
        ${r.description ? `<div class="oa-param-desc"><div class="desc-text">${escapeHtml(r.description)}</div></div>` : ''}
        ${preview ? `<pre class="oa-schema-block">${escapeHtml(preview)}</pre>` : ''}
      `;
    }

    function stripVerbose(s) {
      // Remove $ref noise from a schema for human-readable preview.
      if (!s || typeof s !== 'object') return s;
      if (Array.isArray(s)) return s.map(stripVerbose);
      const out = {};
      for (const k of Object.keys(s)) {
        if (k === '$ref') { out['$ref'] = s[k].split('/').pop(); continue; }
        out[k] = stripVerbose(s[k]);
      }
      return out;
    }

    function renderOpCard(o) {
      const op = o.op;
      const allParams = (o.sharedParams || []).concat(op.parameters || []);
      const buckets = partitionParams(allParams, SPEC);
      const sections = [];
      if (buckets.path.length)   sections.push(['Path Parameters', buckets.path]);
      if (buckets.query.length)  sections.push(['Query Parameters', buckets.query]);
      if (buckets.header.length) sections.push(['Headers', buckets.header]);
      if (buckets.cookie.length) sections.push(['Cookies', buckets.cookie]);

      const sectionsHtml = sections.map(([label, ps]) => `
        <div class="oa-section-h">${label} <span class="oa-section-count">${ps.length}</span></div>
        ${paramRows(ps, SPEC)}
      `).join('');

      const methodColor = METHOD_COLORS[o.method] || 'var(--accent)';
      return `
        <article class="oa-op-card" id="op-${o.id}" data-op-id="${o.id}"
                 style="--method-color:${methodColor}">
          <div class="oa-op-head">
            <span class="icons" aria-hidden="true">
              <svg class="heart" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/></svg>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>
            </span>
            <span class="oa-op-path-row">
              <span class="oa-op-path">${escapeHtml(o.path)}</span>
            </span>
            <span class="oa-method-pill-outline oa-op-method-btn"
                  style="--method-color:${methodColor}">
              <span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:${methodColor};margin-right:4px"></span>
              ${o.method.toUpperCase()}
            </span>
          </div>
          ${op.summary ? `<div class="oa-op-summary">${escapeHtml(op.summary)}</div>` : ''}
          ${op.description ? `<div class="oa-op-desc">${escapeHtml(op.description)}</div>` : ''}
          ${sectionsHtml}
          ${renderRequestBody(op.requestBody, SPEC)}
          ${renderResponses(op.responses, SPEC)}
        </article>`;
    }

    function renderMain(spec, grouped) {
      const main = $('#oa-main');
      const sections = grouped.map(([tag, ops]) => {
        const tagDef = (spec.tags || []).find(t => t.name === tag) || {};
        return `
          <section class="oa-tag-section" data-tag="${escapeHtml(tag)}">
            <div class="oa-tag-section-head" data-action="toggle-section">
              <span class="oa-tag-chevron-big">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                  <polyline points="6 9 12 15 18 9"/></svg>
              </span>
              <h2>${escapeHtml(tag)}</h2>
              <span class="count-pill">${ops.length} endpoint${ops.length === 1 ? '' : 's'}</span>
            </div>
            ${tagDef.description ? `<div class="oa-tag-section-desc">${escapeHtml(tagDef.description)}</div>` : ''}
            ${ops.map(renderOpCard).join('')}
          </section>`;
      }).join('');
      main.innerHTML = renderInfo(spec) + sections;

      main.querySelectorAll('[data-action="toggle-section"]').forEach(h => {
        h.addEventListener('click', () => h.parentElement.classList.toggle('collapsed'));
      });
    }

    /* ---------- right "request preview" pane ---------- */
    function renderRightPane(o) {
      const right = $('#oa-right');
      if (!o) {
        right.innerHTML = '<div class="oa-right-empty">Select an operation on the left to preview its request shape.</div>';
        return;
      }
      const op = o.op;
      const allParams = (o.sharedParams || []).concat(op.parameters || []);
      const buckets = partitionParams(allParams, SPEC);
      const methodColor = METHOD_COLORS[o.method] || 'var(--accent)';

      const sectionFor = (label, list) => {
        if (!list.length) return '';
        return `
          <div class="oa-tryit-section">
            <h4>${label} <span class="oa-section-count">${list.length}</span></h4>
            ${list.map(p => {
              const schema = resolveRef(p.schema || {}, SPEC);
              const typeStr = formatType(schema);
              const placeholder = p.example != null ? p.example
                : (Array.isArray(schema.enum) && schema.enum.length) ? schema.enum[0]
                : (schema.default != null ? schema.default : (p.description || ''));
              const isEnum = Array.isArray(schema.enum) && schema.enum.length;
              const input = isEnum
                ? `<select class="oa-tryit-input">${schema.enum.map(v => `<option>${escapeHtml(v)}</option>`).join('')}</select>`
                : `<input class="oa-tryit-input" type="text" placeholder="${escapeHtml(placeholder ?? '')}" />`;
              return `
                <div class="oa-tryit-row">
                  <span class="name" title="${escapeHtml(p.name)}">${escapeHtml(p.name)}</span>
                  <span class="type"><span class="oa-type-chip">${escapeHtml(typeStr)}</span></span>
                  ${input}
                </div>`;
            }).join('')}
          </div>`;
      };

      let bodyHtml = '';
      if (op.requestBody) {
        const rb = resolveRef(op.requestBody, SPEC);
        const content = rb.content || {};
        const t = Object.keys(content)[0];
        if (t) {
          const schema = resolveRef(content[t].schema || {}, SPEC);
          const example = content[t].example
            ?? (content[t].examples && Object.values(content[t].examples)[0]?.value)
            ?? schema.example
            ?? stripVerbose(schema);
          let preview = '';
          try { preview = typeof example === 'string' ? example : JSON.stringify(example, null, 2); }
          catch (_) { preview = ''; }
          bodyHtml = `
            <div class="oa-tryit-section">
              <h4>Body <span class="oa-section-count">${escapeHtml(t)}</span></h4>
              <textarea class="oa-tryit-body" spellcheck="false">${escapeHtml(preview)}</textarea>
            </div>`;
        }
      }

      right.innerHTML = `
        <div class="oa-tryit-head">
          <div class="oa-tryit-method-row">
            <span class="oa-tryit-method" style="background:${methodColor}">
              <span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:#0a0b0d;margin-right:2px"></span>
              ${o.method.toUpperCase()}
            </span>
            <span class="oa-tryit-path" title="${escapeHtml(o.path)}">${escapeHtml(o.path)}</span>
          </div>
          <button class="oa-add-btn" disabled title="Read-only preview">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>
            Add
          </button>
        </div>
        ${sectionFor('Query Parameters', buckets.query)}
        ${sectionFor('Path Parameters', buckets.path)}
        ${sectionFor('Headers', buckets.header)}
        ${bodyHtml}
        <div class="oa-readonly-note">Read-only preview — use Swagger UI to actually run requests.</div>
      `;
    }

    /* ---------- selection / search / navigation ---------- */
    function selectOp(id) {
      SELECTED = id;
      $$('.oa-tag-op').forEach(el => el.classList.toggle('active', el.dataset.opId === id));
      const card = document.getElementById('op-' + id);
      if (card) card.scrollIntoView({ behavior: 'smooth', block: 'start' });
      const o = OPS.find(x => x.id === id);
      renderRightPane(o);
    }

    function applySearch(q) {
      q = q.trim().toLowerCase();
      $$('.oa-tag-group').forEach(g => {
        let any = false;
        $$('.oa-tag-op', g).forEach(li => {
          const text = li.textContent.toLowerCase();
          const visible = !q || text.includes(q);
          li.style.display = visible ? '' : 'none';
          if (visible) any = true;
        });
        g.style.display = any ? '' : 'none';
      });
      $$('.oa-tag-section').forEach(s => {
        const tag = s.dataset.tag.toLowerCase();
        let any = false;
        $$('.oa-op-card', s).forEach(c => {
          const text = c.textContent.toLowerCase();
          const visible = !q || text.includes(q) || tag.includes(q);
          c.style.display = visible ? '' : 'none';
          if (visible) any = true;
        });
        s.style.display = any ? '' : 'none';
      });
    }

    /* ---------- bootstrap ---------- */
    async function boot() {
      try {
        const res = await fetch(SPEC_URL, { headers: { 'Accept': 'application/json' } });
        if (!res.ok) throw new Error(`Failed to load spec: ${res.status} ${res.statusText}`);
        SPEC = await res.json();
      } catch (err) {
        $('#oa-main').innerHTML = `<div class="oa-error">Could not load OpenAPI spec from ${escapeHtml(SPEC_URL)}.\n\n${escapeHtml(err && err.message || err)}</div>`;
        return;
      }

      const info = SPEC.info || {};
      document.title = (info.title || 'API') + (info.version ? ' — ' + info.version : '');
      const brand = $('#oa-brand-title');
      if (brand && info.title) brand.textContent = info.title;
      const foot = $('#oa-version-foot');
      if (foot && info.version) foot.textContent = 'v' + info.version;

      OPS = flattenOps(SPEC);
      const grouped = groupByTag(OPS);
      renderSidebar(grouped);
      renderMain(SPEC, grouped);

      $('#oa-search').addEventListener('input', e => applySearch(e.target.value));

      // Highlight whichever op is closest to the top as user scrolls
      const main = $('#oa-main');
      const cards = $$('.oa-op-card', main);
      main.addEventListener('scroll', () => {
        const top = main.scrollTop + 24;
        let active = null;
        for (const c of cards) {
          if (c.offsetTop <= top) active = c.dataset.opId;
          else break;
        }
        if (active && active !== SELECTED) {
          SELECTED = active;
          $$('.oa-tag-op').forEach(el => el.classList.toggle('active', el.dataset.opId === active));
          const o = OPS.find(x => x.id === active);
          renderRightPane(o);
        }
      });

      // Auto-select first op so the right pane is never empty
      if (OPS.length) {
        const first = OPS[0];
        $$('.oa-tag-op').forEach(el => el.classList.toggle('active', el.dataset.opId === first.id));
        renderRightPane(first);
      }
    }
    boot();
  })();
  </script>
</body>
</html>"""
