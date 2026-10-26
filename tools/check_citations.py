"""Check every citation in the repo: arXiv IDs resolve to the paper the text claims, URLs respond,
relative links in Markdown point at real files.

    python tools/check_citations.py            # all _src/*.py and *.md
    python tools/check_citations.py --no-urls  # arXiv + local links only (fast)

arXiv titles come from the official API (https://info.arxiv.org/help/api/index.html). A citation
is flagged when fewer than half of the title's significant words appear within a few lines of the ID.
Flags are for a human to look at; many are fine (e.g. "CLIP arXiv:2103.00020").
"""
from __future__ import annotations

import concurrent.futures as cf
import re
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent.parent
SKIP = {".venv", "data", "practice", "solutions", ".ipynb_checkpoints"}
ARXIV_RE = re.compile(r"(?:arXiv:|arxiv\.org/(?:abs|pdf)/)(\d{4}\.\d{4,5})(?:v\d+)?", re.I)
URL_RE = re.compile(r"https?://(?:\([^\s()]*\)|[^\s()\]>`'\"|])+")  # allows balanced (...) as in DOIs
MDLINK_RE = re.compile(r"\]\(((?:\([^()\s]*\)|[^()#\s])+)(?:#[^)]*)?\)")
STOP = set("a an the of for and in on to with from via is are by at as its it we our into using toward towards".split())
UA = {"User-Agent": "Mozilla/5.0 (citation-check for a private study repo)"}


def files() -> list[Path]:
    out = []
    for p in REPO.rglob("*"):
        if any(part in SKIP for part in p.relative_to(REPO).parts):
            continue
        if p.name == "citation_report.md":
            continue
        if p.suffix == ".md" or (p.suffix == ".py" and "_src" in p.parts):
            out.append(p)
    return sorted(out)


def arxiv_titles(ids: list[str]) -> dict[str, tuple[str, str]]:
    titles = {}
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        for attempt in range(5):  # the arXiv API returns 503 under load; back off and retry
            r = requests.get("http://export.arxiv.org/api/query",
                             params={"id_list": ",".join(chunk), "max_results": len(chunk)}, timeout=60)
            if r.status_code != 503:
                break
            time.sleep(10 * (attempt + 1))
        r.raise_for_status()
        ns = {"a": "http://www.w3.org/2005/Atom"}
        for e in ET.fromstring(r.text).findall("a:entry", ns):
            aid = e.find("a:id", ns).text.rsplit("/", 1)[-1]
            aid = re.sub(r"v\d+$", "", aid)
            title = " ".join(e.find("a:title", ns).text.split())
            first = e.find("a:author/a:name", ns)
            titles[aid] = (title, first.text.split()[-1] if first is not None else "")
        time.sleep(3)  # arXiv API etiquette
    return titles


def words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP and len(w) > 2}


def check_url(url: str) -> tuple[str, str]:
    url = url.rstrip(".,;:")
    try:
        r = requests.head(url, allow_redirects=True, timeout=20, headers=UA)
        if r.status_code >= 400:
            r = requests.get(url, allow_redirects=True, timeout=20, headers=UA, stream=True)
        return url, str(r.status_code)
    except requests.RequestException as e:
        return url, type(e).__name__


def main(argv: list[str]) -> int:
    do_urls = "--no-urls" not in argv
    cites, urls, bad_links = [], {}, []
    for f in files():
        lines = f.read_text().splitlines()
        for i, line in enumerate(lines):
            ctx = " ".join(lines[max(0, i - 2): i + 2])
            for m in ARXIV_RE.finditer(line):
                cites.append((f, i + 1, m.group(1), ctx))
            for u in URL_RE.findall(line):
                urls.setdefault(u.rstrip(".,;:"), []).append(f"{f.relative_to(REPO)}:{i + 1}")
            if f.suffix == ".md":
                for target in MDLINK_RE.findall(line):
                    if target.startswith(("http", "mailto:")):
                        continue
                    if not (f.parent / urllib.parse.unquote(target)).exists():
                        bad_links.append(f"{f.relative_to(REPO)}:{i + 1} -> {target}")

    ids = sorted({c[2] for c in cites})
    titles = arxiv_titles(ids)
    report = ["# Citation check", "", f"{len(ids)} distinct arXiv IDs, {len(urls)} URLs.", "",
              "## arXiv IDs (title from the arXiv API)", "", "| ID | Title (first author) | Cited at | Flag |", "| --- | --- | --- | --- |"]
    flagged = 0
    for aid in ids:
        title, author = titles.get(aid, (None, ""))
        sites = [c for c in cites if c[2] == aid]
        flag = ""
        if title is None:
            flag = "NOT FOUND"
        else:
            tw = words(title)
            short = title.split(":")[0].lower() if ":" in title else None  # "LoRA: Low-Rank ..." -> "lora"

            def ok(ctx: str) -> bool:
                c = ctx.lower()
                return (len(tw & words(ctx)) / max(1, len(tw)) >= 0.5 or (short and short in c)
                        or (author and author.lower() in c))

            n_bad = sum(not ok(c[3]) for c in sites)
            if n_bad:
                flag = f"check context at {n_bad}/{len(sites)} sites (no title words, short name, or first author nearby)"
        flagged += bool(flag)
        where = ", ".join(sorted({f"{c[0].relative_to(REPO)}:{c[1]}" for c in sites})[:3])
        report.append(f"| {aid} | {title or '-'} ({author}) | {where} | {flag} |")

    report += ["", "## Broken relative links", ""] + ([f"- {b}" for b in bad_links] or ["none"])
    if do_urls:
        with cf.ThreadPoolExecutor(16) as ex:
            res = list(ex.map(check_url, urls))
        bad = [(u, s) for u, s in res if not s.isdigit() or int(s) >= 400]
        report += ["", "## URLs not returning 2xx/3xx (some sites block scripts; open these by hand)", ""]
        report += [f"- {s} {u} ({urls[u][0]})" for u, s in bad] or ["none"]
    out = REPO / "tools" / "citation_report.md"
    out.write_text("\n".join(report) + "\n")
    print(f"{len(ids)} arXiv IDs ({flagged} flagged), {len(bad_links)} broken local links"
          + (f", {len(bad)} URL problems" if do_urls else "") + f" -> {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
