#!/usr/bin/env python3
"""Audit paper/refs.bib entries against CrossRef.

Rules (deliberately tolerant where the project has prior decisions):
  * DOI entries: every entry must resolve; year must match ONE of CrossRef's
    issued / published-print / published-online / published / created years
    (karasiak2022spatial online-2021/print-2022 is legitimately "2022");
    first-author family name must match; title must match after normalization
    (exact, else SequenceMatcher >= 0.85 -> WARN, below -> FAIL);
    volume/pages compared when present in both (WARN on mismatch).
  * No-DOI entries are listed as a manual-check checklist (not fetched).

Exit code: 0 = no FAIL, 1 = at least one FAIL.
Run from repo root:  /home/lizhuo/venvs/junzhe-gpu/bin/python analysis/bib_audit.py
"""
from __future__ import annotations

import difflib
import json
import re
import sys
import time
import unicodedata
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIB = ROOT / "paper" / "refs.bib"
UA = "JunzheBibAudit/1.0 (mailto:noreply@example.org)"
SLEEP_S = 0.8


def parse_bib(text: str) -> list[dict]:
    entries = []
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text):
        etype, key = m.group(1), m.group(2)
        # brace-balanced body
        i = m.end() - 1  # at ',' before first field -> start after it
        depth = 1
        j = i
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        body = text[i : j - 1]
        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*\{", body):
            name = fm.group(1).lower()
            k = fm.end() - 1  # index of the opening '{'
            d = 1
            p = k + 1
            while p < len(body) and d > 0:
                if body[p] == "{":
                    d += 1
                elif body[p] == "}":
                    d -= 1
                p += 1
            # p is one past the matching '}' (or end of body)
            fields[name] = body[k + 1 : p - 1] if d == 0 else body[k + 1 :]
        entries.append({"type": etype, "key": key, "fields": fields})
    return entries


def strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


def clean_markup(s: str) -> str:
    # CrossRef inline markup (<scp>block</scp> <scp>CV</scp> renders as
    # "blockCV"); whitespace between adjacent tags is not rendered, so
    # collapse it before stripping the tags
    s = re.sub(r">\s+<", "><", s)
    return re.sub(r"<[^>]+>", "", s)


# CrossRef titles carry Unicode dashes/hyphens (e.g. U+2010 in "Leave-One-Out")
# that LaTeX sources write as ASCII "-"; fold them before comparing.
DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2015\u2212\u00ad"), "-")


def fold_dashes(s: str) -> str:
    return s.translate(DASHES)


def strip_tex_accents(s: str) -> str:
    """Drop LaTeX accent macros so "Mil\\`a" compares equal to CrossRef "Mila"."""
    s = re.sub(r"\\[\`'\"^~=.]\s*(?=\{?\w)", "", s)
    s = s.replace("{", "").replace("}", "")
    return s


def norm_title(s: str) -> str:
    s = clean_markup(s)
    s = strip_tex_accents(s)
    s = fold_dashes(s)
    s = strip_accents(s).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def first_author_family(bib_author: str) -> str:
    first = bib_author.split(" and ")[0]
    fam = first.split(",")[0].strip()
    return strip_accents(strip_tex_accents(fam)).casefold()


def cr_years(msg: dict) -> set[int]:
    years = set()
    for field in ("issued", "created", "published-print", "published-online", "published"):
        dp = (msg.get(field) or {}).get("date-parts") or [[None]]
        if dp and dp[0] and dp[0][0]:
            years.add(int(dp[0][0]))
    return years


def cr_first_family(msg: dict) -> str:
    auth = (msg.get("author") or [{}])[0]
    fam = auth.get("family") or auth.get("name") or ""
    return strip_accents(fam).casefold()


def fetch(doi: str) -> dict | None:
    url = f"https://api.crossref.org/works/{doi}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)["message"]
    except Exception as e:  # noqa: BLE001
        print(f"    FETCH ERROR {doi}: {e}")
        return None


def main() -> int:
    entries = parse_bib(BIB.read_text(encoding="utf-8"))
    doi_entries = [e for e in entries if "doi" in e["fields"]]
    nodoi = [e for e in entries if "doi" not in e["fields"]]
    print(f"parsed {len(entries)} entries: {len(doi_entries)} with DOI, "
          f"{len(nodoi)} without\n")

    fails = warns = 0
    for e in doi_entries:
        f = e["fields"]
        doi = f["doi"].strip()
        bib_year = int(re.search(r"\d{4}", f.get("year", "")).group(0))
        print(f"[{e['key']}] {doi}")
        msg = fetch(doi)
        time.sleep(SLEEP_S)
        if msg is None:
            fails += 1
            print("    FAIL: unresolved DOI")
            continue
        tlist = msg.get("title") or [""]
        raw_title = str(tlist[0] if tlist else "")
        # strip inline markup BEFORE whitespace collapse (see clean_markup)
        msg_title = " ".join(clean_markup(raw_title).split())
        problems, warns_here = [], []

        # year
        cr_y = cr_years(msg)
        if bib_year not in cr_y:
            problems.append(f"year bib={bib_year} crossref={sorted(cr_y)}")
        # first author (token containment either way: bib 'daudt' vs CR
        # 'caye daudt' is the same person; dirty given-names are common)
        bf = first_author_family(f.get("author", ""))
        cf = cr_first_family(msg)
        if bf != cf and not (set(bf.split()) <= set(cf.split())
                             or set(cf.split()) <= set(bf.split())):
            problems.append(f"first author bib={bf!r} crossref={cf!r}")
        # title
        bt, ct = norm_title(f.get("title", "")), norm_title(msg_title)
        if bt != ct:
            ratio = difflib.SequenceMatcher(None, bt, ct).ratio()
            if ratio >= 0.85:
                warns_here.append(f"title close (ratio={ratio:.2f}): "
                                  f"bib={bt[:60]!r} cr={ct[:60]!r}")
            else:
                problems.append(f"title mismatch (ratio={ratio:.2f}): "
                                f"bib={bt!r} cr={ct!r}")
        # volume / pages
        if f.get("volume") and msg.get("volume") and \
                f["volume"].strip() != str(msg["volume"]).strip():
            warns_here.append(f"volume bib={f['volume']} cr={msg['volume']}")
        if f.get("pages"):
            bp = re.sub(r"--|-", "-", f["pages"].strip())
            cp = re.sub(r"--|-", "-", str(msg.get("page") or "").strip())
            if cp and bp != cp:
                warns_here.append(f"pages bib={f['pages']} cr={msg.get('page')}")
        # venue sanity (warn only): first word of bib venue vs container-title.
        # For @inproceedings CrossRef's container-title is the book SERIES
        # (e.g. "Lecture Notes in Computer Science") while the bib stores the
        # conference name in booktitle and the series separately, so compare the
        # series when present instead of the conference name.
        venue = f.get("journal") or f.get("booktitle") or ""
        series = f.get("series") or ""
        clist = msg.get("container-title") or [""]
        cont = " ".join(str(clist[0] if clist else "").split())
        compare_venue = series if series else venue
        if venue and cont and \
                norm_title(compare_venue).split()[:1] != norm_title(cont).split()[:1]:
            warns_here.append(f"venue first-word differs: bib={venue[:40]!r} "
                              f"cr={cont[:40]!r}")

        for p in problems:
            print(f"    FAIL: {p}")
        for w in warns_here:
            print(f"    WARN: {w}")
        if problems:
            fails += 1
        elif warns_here:
            warns += 1
            print("    (pass with warnings)")
        else:
            print("    PASS")

    print(f"\n--- {len(doi_entries)} DOI entries: {fails} FAIL, {warns} WARN, "
          f"{len(doi_entries) - fails - warns} clean ---")
    print("--- manual-check list (no DOI, not fetched) ---")
    for e in nodoi:
        f = e["fields"]
        print(f"  {e['key']}: {f.get('title', '?')[:70]} | "
              f"{f.get('journal', f.get('howpublished', '?'))[:50]} | "
              f"year={f.get('year', '?')}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
