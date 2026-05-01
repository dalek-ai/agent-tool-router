"""Crawl MCP server registries and emit a unified, enriched JSONL index.

Sources:
- modelcontextprotocol/servers (official) README
- punkpeye/awesome-mcp-servers (community) README

Output: data/mcp_servers.jsonl
Schema per line:
    {
      name, repo_url, description, source, last_seen,
      category,            # H2/H3 section the entry was found under
      official: bool,      # 🎖️ legend
      language: str|None,  # python|typescript|go|rust|csharp|java|c|ruby|None
      hosting: list[str],  # subset of {"cloud","local","embedded"}
      os_support: list[str], # subset of {"macos","windows","linux"}
    }

stdlib only. Run: python3 router/index/crawl_mcp.py
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = DATA / "mcp_servers.jsonl"

SOURCES = [
    {
        "name": "official",
        "url": "https://raw.githubusercontent.com/modelcontextprotocol/servers/main/README.md",
    },
    {
        "name": "awesome",
        "url": "https://raw.githubusercontent.com/punkpeye/awesome-mcp-servers/main/README.md",
    },
]

# Emoji legend (from awesome-mcp-servers README).
LANG_EMOJI = {
    "🐍": "python",
    "📇": "typescript",
    "🏎️": "go",
    "🏎": "go",
    "🦀": "rust",
    "#️⃣": "csharp",
    "#⃣": "csharp",
    "☕": "java",
    "🌊": "c",
    "💎": "ruby",
}
HOSTING_EMOJI = {
    "☁️": "cloud",
    "☁": "cloud",
    "🏠": "local",
    "📟": "embedded",
}
OS_EMOJI = {
    "🍎": "macos",
    "🪟": "windows",
    "🐧": "linux",
}
OFFICIAL_EMOJI = {"🎖️", "🎖"}

LINK_RE = re.compile(r"\[([^\]\n]+)\]\(([^)\s]+)\)")
BULLET_RE = re.compile(r"^\s*[-*+]\s")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
# Strips the [![badge](url)](href) markdown badges that pollute Glama-flavored entries.
BADGE_RE = re.compile(r"\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)")
WHITESPACE_RE = re.compile(r"\s+")


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "agent-tool-router-crawler/0.1"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def normalize_repo(url: str) -> str | None:
    if not url.startswith("http"):
        return None
    m = re.match(r"https?://github\.com/([^/\s#]+)/([^/\s#?]+)", url)
    if not m:
        return None
    owner, repo = m.group(1), m.group(2)
    repo = repo.removesuffix(".git")
    return f"https://github.com/{owner}/{repo}"


def extract_metadata(text: str) -> dict:
    """Extract emoji-encoded metadata + clean description text."""
    md = {
        "official": False,
        "language": None,
        "hosting": [],
        "os_support": [],
    }
    # Walk character-by-character to catch multi-codepoint emojis.
    # Cheap approach: scan each known emoji as a substring.
    for emoji in OFFICIAL_EMOJI:
        if emoji in text:
            md["official"] = True
            text = text.replace(emoji, "")
    for emoji, lang in LANG_EMOJI.items():
        if emoji in text:
            md["language"] = lang
            text = text.replace(emoji, "")
            break  # one language per server
    for emoji, host in HOSTING_EMOJI.items():
        if emoji in text:
            if host not in md["hosting"]:
                md["hosting"].append(host)
            text = text.replace(emoji, "")
    for emoji, osn in OS_EMOJI.items():
        if emoji in text:
            if osn not in md["os_support"]:
                md["os_support"].append(osn)
            text = text.replace(emoji, "")
    # Strip badges + whitespace + leading separators.
    text = BADGE_RE.sub("", text)
    text = WHITESPACE_RE.sub(" ", text).strip(" -—:|")
    md["clean_description"] = text
    return md


_HEADING_NOISE_RE = re.compile(r"<a\s+name=\"[^\"]*\"\s*>\s*</a>", re.IGNORECASE)
_LEADING_EMOJI_RE = re.compile(r"^[^\w(\[]*", flags=re.UNICODE)


def _clean_heading(title: str) -> str:
    title = _HEADING_NOISE_RE.sub("", title)
    title = title.strip()
    # Strip leading category emoji + whitespace.
    title = _LEADING_EMOJI_RE.sub("", title).strip()
    return title


def iter_lines_with_section(markdown: str) -> Iterator[tuple[str, str]]:
    """Yield (current_section, line) for every line, tracking H2/H3 sections.

    Section format: "h2:Title" or "h2:Parent / h3:Child" — joined when an H3 is
    inside an H2.
    """
    h2 = ""
    h3 = ""
    for line in markdown.splitlines():
        m = HEADING_RE.match(line)
        if m:
            level, title = len(m.group(1)), m.group(2).strip()
            title_clean = _clean_heading(title)
            if level == 2:
                h2 = title_clean
                h3 = ""
            elif level == 3:
                h3 = title_clean
        section = h2 if not h3 else f"{h2} / {h3}"
        yield section, line


def parse_servers(markdown: str, source: str) -> Iterable[dict]:
    """Extract servers, attaching the H2/H3 section as `category`."""
    seen_in_doc: set[str] = set()
    for section, raw_line in iter_lines_with_section(markdown):
        if not BULLET_RE.match(raw_line):
            continue
        # Skip lines that are TOC entries (typically in the very early sections).
        if section.lower() in {"", "server implementations"}:
            # Inside the master TOC ('Server Implementations'), bullets are
            # category links — not server entries — skip them.
            if section.lower() == "server implementations":
                continue
        line = raw_line.lstrip(" -*+\t")
        m = LINK_RE.search(line)
        if not m:
            continue
        name = m.group(1).strip()
        url = m.group(2).strip()
        repo = normalize_repo(url)
        if not repo:
            continue
        key = repo.lower()
        if key in seen_in_doc:
            continue
        seen_in_doc.add(key)
        rest = line[m.end():]
        meta = extract_metadata(rest)
        yield {
            "name": name,
            "repo_url": repo,
            "description": meta.pop("clean_description")[:500],
            "source": source,
            "last_seen": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "category": section,
            **meta,
        }


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    by_repo: dict[str, dict] = {}
    counts_per_source: dict[str, int] = {}

    for src in SOURCES:
        try:
            md = fetch(src["url"])
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] failed to fetch {src['name']}: {exc}", file=sys.stderr)
            continue
        n = 0
        for entry in parse_servers(md, src["name"]):
            key = entry["repo_url"].lower()
            existing = by_repo.get(key)
            if existing is None:
                by_repo[key] = entry
            else:
                # Merge: prefer richer fields.
                if not existing["description"] and entry["description"]:
                    existing["description"] = entry["description"]
                if existing["category"] in {"", None} and entry["category"]:
                    existing["category"] = entry["category"]
                existing["official"] = existing["official"] or entry["official"]
                if not existing["language"] and entry["language"]:
                    existing["language"] = entry["language"]
                for h in entry["hosting"]:
                    if h not in existing["hosting"]:
                        existing["hosting"].append(h)
                for o in entry["os_support"]:
                    if o not in existing["os_support"]:
                        existing["os_support"].append(o)
                if src["name"] not in existing["source"].split(","):
                    existing["source"] = existing["source"] + "," + src["name"]
                existing["last_seen"] = entry["last_seen"]
            n += 1
        counts_per_source[src["name"]] = n
        print(f"[ok] {src['name']}: {n} bullet-link entries parsed")

    with OUT.open("w", encoding="utf-8") as f:
        for entry in by_repo.values():
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Stats summary: language + category top distribution.
    lang_counts: dict[str, int] = {}
    cat_counts: dict[str, int] = {}
    host_counts: dict[str, int] = {}
    for e in by_repo.values():
        lang_counts[e["language"] or "unknown"] = lang_counts.get(e["language"] or "unknown", 0) + 1
        cat_counts[e["category"] or "unknown"] = cat_counts.get(e["category"] or "unknown", 0) + 1
        for h in (e["hosting"] or ["unknown"]):
            host_counts[h] = host_counts.get(h, 0) + 1

    print(f"[done] wrote {len(by_repo)} unique servers to {OUT}")
    print(f"[stats] per-source raw counts: {counts_per_source}")
    print(f"[stats] language: {dict(sorted(lang_counts.items(), key=lambda x: -x[1]))}")
    print(f"[stats] hosting: {dict(sorted(host_counts.items(), key=lambda x: -x[1]))}")
    print(f"[stats] top-15 categories:")
    for cat, n in sorted(cat_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"          {n:5d}  {cat}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
